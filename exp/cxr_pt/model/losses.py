import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from open_clip.loss import ClipLoss, SigLipLoss


class OpenClipLoss(ClipLoss):
    # init logit scale from open_clip
    def __init__(self, init_logit_scale=np.log(1 / 0.07), **kwargs):
        super().__init__(**kwargs)
        self.logit_scale = nn.Parameter(torch.FloatTensor([init_logit_scale]))

    def forward(self, image_features, text_features):
        return super().forward(image_features, text_features, self.logit_scale.exp())


class OpenSigLipLoss(SigLipLoss):
    # init logit scale and bias from https://arxiv.org/abs/2303.15343
    def __init__(self, init_logit_scale=np.log(10), init_logit_bias=-10.0, **kwargs):
        super().__init__(**kwargs)
        self.logit_scale = nn.Parameter(torch.FloatTensor([init_logit_scale]))
        self.logit_bias = nn.Parameter(torch.FloatTensor([init_logit_bias]))

    def forward(self, image_features, text_features):
        return super().forward(
            image_features, text_features, self.logit_scale.exp(), self.logit_bias
        )


class RadZeroLoss(nn.Module):

    def __init__(
        self,
        hidden_dim=768,
        use_vision_cls_token=True,
        attn_temperature=None,
        loss_temperature=0.07,
        text_features_l2_norm=False,
        mpnce_row_sum=False,
        mpnce_col_sum=False,
        sim_op="dot",
        use_layer_norm=True,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else None

        self.use_vision_cls_token = use_vision_cls_token
        self.loss_temperature = nn.Parameter(
            torch.FloatTensor([np.log(loss_temperature)])
        )
        if attn_temperature is not None:
            self.attn_temperature = nn.Parameter(
                torch.FloatTensor([np.log(attn_temperature)])
            )
        else:
            self.attn_temperature = None
        self.text_features_l2_norm = text_features_l2_norm
        self.sim_op = sim_op

        self.similarity_logit = SimilarityLogit(sim_op)

        self.mpnce_row_sum = mpnce_row_sum
        self.mpnce_col_sum = mpnce_col_sum

    def forward(
        self,
        key_phrases,
        vision_tokens,
        forward_text_model,
        ddp_gather=True,
        need_attn_weights=False,
        compute_loss=True,
        **kwargs,
    ):
        outputs = {}

        text_features, group_map = self.compute_text_features(
            key_phrases, forward_text_model, ddp_gather
        )

        if ddp_gather and dist.is_initialized():
            vision_tokens = torch.cat(dist.nn.all_gather(vision_tokens), dim=0)

        if self.layer_norm is not None:
            vision_tokens = self.layer_norm(vision_tokens)

        vision_patch_tokens = vision_tokens[:, 1:]

        # text to image cross-attention
        if not self.use_vision_cls_token:
            vision_attn_tokens = vision_patch_tokens
        else:
            vision_attn_tokens = vision_tokens

        t2i_logits, t2i_attn_weights_list = self.compute_t2i_logits(
            text_features, vision_attn_tokens, need_attn_weights
        )
        outputs["t2i_logits"] = t2i_logits
        outputs["t2i_attn_weights"] = t2i_attn_weights_list

        if compute_loss:
            losses = {}
            loss = 0

            # compute t2i loss
            t2i_loss = multi_positive_nce_loss(
                t2i_logits,
                group_map,
                temperature=self.loss_temperature.exp(),
                row_sum=self.mpnce_row_sum,
                col_sum=self.mpnce_col_sum,
            )
            loss += t2i_loss
            losses["t2i_loss"] = t2i_loss

            losses["loss"] = loss
            outputs["losses"] = losses
        return outputs

    def compute_text_features(self, key_phrases, forward_text_model, ddp_gather=True):

        key_text_features_list = list()
        group_list = list()

        B_local = len(key_phrases)
        # Calculate offset by getting the rank of the current process when using DDP
        local_rank = dist.get_rank() if (ddp_gather and dist.is_initialized()) else 0

        for i, kp in enumerate(key_phrases):
            feats = forward_text_model(kp)

            # (N_i, D)
            if self.text_features_l2_norm:
                feat = feats["text_features"]
            else:
                feat = feats["text_features_wo_l2_norm"]

            if feat.shape[-1] == 2 * self.hidden_dim:
                feat = feat[:, self.hidden_dim :]

            key_text_features_list.append(feat)

            # Add local_rank * B_local offset to local index i
            global_index = i + local_rank * B_local
            group_list.extend([global_index] * feat.size(0))

        text_features = torch.cat(key_text_features_list, dim=0)
        group_map = torch.tensor(group_list, device=text_features.device)

        if ddp_gather and dist.is_initialized():
            # Gather text_features and image_features and group_map
            text_features = pad_and_gather(text_features)

            group_map = pad_and_gather(group_map)
            group_map = group_map.long()

        if self.layer_norm is not None:
            text_features = self.layer_norm(text_features)

        return text_features, group_map

    def compute_t2i_logits(
        self, text_features, vision_attn_tokens, need_attn_weights, repeat=True
    ):

        t2i_logits, t2i_attn_weights_list = self.similarity_logit(
            text_features,
            vision_attn_tokens,
            need_attn_weights,
            repeat=repeat,
            temperature=(
                self.attn_temperature.exp()
                if self.attn_temperature is not None
                else self.loss_temperature.exp()
            ),
        )

        return t2i_logits, t2i_attn_weights_list


class SimilarityLogit(nn.Module):
    def __init__(self, sim_op="dot", **kwargs):
        super().__init__()
        self.sim_op = sim_op

    def forward(
        self,
        queries: torch.Tensor,
        local_tokens: torch.Tensor,
        need_attn_weights: bool = False,
        repeat: bool = True,
        **kwargs,
    ):
        if repeat:
            query_attn_features = queries.unsqueeze(0).expand(
                local_tokens.shape[0], queries.shape[0], queries.shape[1]
            )
        else:
            assert queries.dim() == 3
            query_attn_features = queries

        if self.sim_op == "cos":
            temperature = kwargs.get("temperature")
            assert temperature is not None
            denominator = temperature
            query_attn_features = F.normalize(query_attn_features, p=2, dim=-1)
            local_tokens = F.normalize(local_tokens, p=2, dim=-1)
        elif self.sim_op == "dot":
            denominator = math.sqrt(local_tokens.size(-1))
        else:
            raise NotImplementedError

        scores = (
            torch.bmm(query_attn_features, local_tokens.permute(0, 2, 1)) / denominator
        )
        attn_weights = F.softmax(scores, dim=-1)

        aggregated = torch.matmul(attn_weights, local_tokens)

        query_attn_features = F.normalize(query_attn_features, p=2, dim=-1)
        aggregated = F.normalize(aggregated, p=2, dim=-1)

        logits = torch.matmul(
            query_attn_features.unsqueeze(2), aggregated.unsqueeze(-1)
        ).squeeze()

        logits = logits.T

        if need_attn_weights:
            attn_scores = [scores]
        else:
            attn_scores = None

        return logits, attn_scores


def multi_positive_nce_loss(
    logits: torch.Tensor,
    group_map: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
    row_sum: bool = False,
    col_sum: bool = False,
):
    """
    Args:
        logits: tensor of shape (N_total, B_global), each row is a logit between a key phrase and each candidate image.
        group_map: tensor of shape (N_total,), source image index of each key phrase.
        temperature: scaling factor.

    For each key phrase row i, the positive is the candidate image index == group_map[i],
    and the rest are treated as negatives.

    For each column j, each positive for image j is considered independently.

    Returns:
        loss: scalar tensor.
    """
    scaled_logits = torch.exp(logits / temperature)  # (N_total, B_global)

    pos_logits = scaled_logits[
        torch.arange(scaled_logits.size(0)), group_map
    ]  # (N_total,)

    row_loss = get_row_loss(
        scaled_logits,
        pos_logits,
        group_map,
        eps,
        row_sum,
    )

    neg_mask = torch.ones_like(scaled_logits)
    neg_mask[torch.arange(scaled_logits.size(0)), group_map] = 0  # (N_total, B_global)

    column_loss = get_col_loss(
        scaled_logits,
        pos_logits,
        neg_mask,
        group_map,
        eps,
        col_sum,
    )

    loss = (row_loss.mean() + column_loss.mean()) / 2

    return loss


def get_row_loss(
    logits: torch.Tensor,
    pos_logits: torch.Tensor,
    group_map: torch.Tensor,
    eps: float = 1e-8,
    row_sum: bool = False,
):
    if row_sum:
        # Create a tensor to hold the summed values
        row_sum_logits = torch.zeros(
            logits.shape[-1], device=logits.device
        )  # (B_global)
        row_pos_sum_logits = torch.zeros(
            logits.shape[-1], device=logits.device
        )  # (B_global)

        # Use scatter_add to sum values based on group_map
        row_sum_logits.scatter_add_(0, group_map, logits.sum(dim=1))  # (B_global)
        row_pos_sum_logits.scatter_add_(0, group_map, pos_logits)  # (B_global)
        p_row = row_pos_sum_logits / (row_sum_logits + eps)  # (B_global)
    else:
        row_sum_logits = logits.sum(dim=1)  # (N_total)
        p_row = pos_logits / (row_sum_logits + eps)  # (N_total)

    return -torch.log(p_row + eps)


def get_col_loss(
    logits: torch.Tensor,
    pos_logits: torch.Tensor,
    neg_mask: torch.Tensor,
    group_map: torch.Tensor,
    eps: float = 1e-8,
    col_sum: bool = False,
):
    if col_sum:
        # MIL-NCE loss
        column_sum_logits = logits.sum(dim=0)  # (B_global,)
        pos_mask = torch.ones_like(logits) - neg_mask  # (N_total, B_global)
        column_pos_logits = (logits * pos_mask).sum(dim=0)  # (B_global,)
        p_column = column_pos_logits / (column_sum_logits + eps)  # (B_global,)
    else:
        # MP-NCE loss (UniCLIP)
        neg_logits = logits * neg_mask  # (N_total, B_global)
        sum_neg_logits = neg_logits.sum(dim=0)  # (B_global,)
        sum_neg_logits = sum_neg_logits[group_map]  # (N_total)
        p_column = pos_logits / (pos_logits + sum_neg_logits + eps)  # (N_total)

    return -torch.log(p_column + eps)


def pad_keyphrase_features(features_list):
    """
    Pads a list of tensors (each of shape (N_i, D)) to a single tensor of shape (B, N_max, D)
    and returns a boolean mask indicating valid positions.

    Args:
        features_list (list of Tensors): List of length B, where each tensor is of shape (N_i, D)

    Returns:
        padded (Tensor): Padded tensor of shape (B, N_max, D)
        mask (Tensor): Boolean mask of shape (B, N_max), where True indicates a valid (non-padded) token.
    """
    # B = len(features_list)
    D = features_list[0].size(1)
    N_max = max(f.size(0) for f in features_list)
    padded_list = []
    mask_list = []
    for f in features_list:
        N_i = f.size(0)
        pad_len = N_max - N_i
        if pad_len > 0:
            pad = f.new_zeros(pad_len, D)
            f_padded = torch.cat([f, pad], dim=0)
            m = torch.cat(
                [
                    torch.ones(N_i, dtype=torch.bool, device=f.device),
                    torch.zeros(pad_len, dtype=torch.bool, device=f.device),
                ]
            )
        else:
            f_padded = f
            m = torch.ones(N_i, dtype=torch.bool, device=f.device)
        padded_list.append(f_padded.unsqueeze(0))
        mask_list.append(m.unsqueeze(0))
    padded = torch.cat(padded_list, dim=0)  # (B, N_max, D)
    mask = torch.cat(mask_list, dim=0)  # (B, N_max)
    return padded, mask


def pad_and_gather(tensor):
    # Determine the size of the tensor
    local_size = torch.tensor(tensor.size(), device=tensor.device)

    # Gather all sizes
    all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)

    # Determine the maximum size
    max_size = torch.stack(all_sizes).max(dim=0)[0]

    # Pad the tensor to the maximum size
    padded_tensor = torch.zeros(max_size.tolist(), device=tensor.device)
    padded_tensor[: local_size[0]] = tensor

    # Gather all padded tensors
    gathered_tensors = dist.nn.all_gather(padded_tensor)

    # Trim the gathered tensors to their original sizes
    gathered_tensors = [g[: s[0]] for g, s in zip(gathered_tensors, all_sizes)]

    gathered_tensors = torch.cat(gathered_tensors, dim=0)

    return gathered_tensors


def pad_and_gather_2nd_dim(tensor, mask):
    # Determine the size of the tensor
    local_size = torch.tensor(tensor.size(), device=tensor.device)

    # Gather all sizes
    all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)

    # Determine the maximum size
    max_len = max([i[1].item() for i in all_sizes])

    # Pad the tensor to the maximum size
    padded_tensor = torch.zeros(
        (local_size[0], max_len, local_size[2]), device=tensor.device
    )
    padded_tensor[:, : local_size[1]] = tensor

    padded_mask = torch.zeros((local_size[0], max_len), device=mask.device)
    padded_mask[:, : local_size[1]] = mask

    # Gather all padded tensors
    gathered_tensors = dist.nn.all_gather(padded_tensor)
    gathered_masks = dist.nn.all_gather(padded_mask)

    gathered_tensors = torch.cat(gathered_tensors, dim=0)
    gathered_masks = torch.cat(gathered_masks, dim=0)

    return gathered_tensors, gathered_masks
