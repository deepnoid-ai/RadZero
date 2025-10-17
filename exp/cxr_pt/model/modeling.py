import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel
from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.mpnet.modeling_mpnet import MPNetModel

from common.trainer import logger
from exp.cxr_pt.model.common_layers import BasePreTrainedModel

from .align_transformers import build_align_transformer
from .configuration import CxrAlignConfig
from .losses import (
    RadZeroLoss,
    OpenClipLoss,
    OpenSigLipLoss,
)
from .text_encoders import aggregate_tokens, build_text_encoder
from .vision_encoders import MRM, Dinov2Model, build_vision_encoder


class CxrAlignModel(BasePreTrainedModel):

    config_class = CxrAlignConfig

    def build_vision_model(self, config: CxrAlignConfig):
        vision_config = config.vision_config
        vision_config.pretrained_dir = config.pretrained_dir
        vision_model = build_vision_encoder(vision_config)
        return vision_model

    def build_text_model(self, config: CxrAlignConfig):
        text_config = config.text_config
        text_model = build_text_encoder(text_config)

        if text_config.model_type == "bioclinicalmpbert":
            self.tokenizer = AutoTokenizer.from_pretrained(
                text_config.pretrained_tokenizer_name_or_path
            )
            self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        return text_model

    def build_align_transformer_model(self, config: CxrAlignConfig):
        align_transformer_config = config.align_transformer_config
        align_transformer = build_align_transformer(align_transformer_config)

        return align_transformer

    def __init__(self, config: CxrAlignConfig):
        super().__init__(config)

        logger.info("Build vision model ...")
        self.vision_model = self.build_vision_model(config)

        logger.info("Build text model ...")
        self.text_model = self.build_text_model(config)

        if (
            isinstance(self.text_model, CLIPTextModel)
            or isinstance(self.text_model, MPNetModel)
            or isinstance(self.text_model, BertModel)
        ):
            text_dim = self.text_model.config.hidden_size

        self.hidden_size = config.align_transformer_config.hidden_size

        if config.text_config.use_text_projection:
            self.text_projector = nn.Linear(text_dim, 2 * self.hidden_size)
        else:
            self.text_projector = None

        logger.info("Build align transformer model ...")
        self.align_transformer = self.build_align_transformer_model(config)

        logger.info("Build loss functions ...")
        loss_cfg = config.kwargs["loss"]
        self.loss_ratio = dict()
        self.loss_fns = nn.ModuleDict()
        for loss_type, ratio in zip(loss_cfg["apply"], loss_cfg["ratio"]):
            logger.info(f"Build {loss_type} loss function ...")
            if loss_cfg[loss_type] is None:
                loss_cfg[loss_type] = dict()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                loss_cfg[loss_type]["rank"] = torch.distributed.get_rank()
                loss_cfg[loss_type]["world_size"] = torch.distributed.get_world_size()
            self.loss_fns[loss_type] = eval(loss_type)(**loss_cfg[loss_type])
            self.loss_ratio[loss_type] = ratio

        self.compute_logits_type = config.kwargs.get("compute_logits_type")
        self.use_negative_logits = config.kwargs.get("use_negative_logits")

        self.module_to_update = config.kwargs.get("module_to_update")

    def forward_vision_model(self, pixel_values):

        if isinstance(self.vision_model, Dinov2Model):
            vision_tokens = self.vision_model(pixel_values)["last_hidden_state"]
        elif isinstance(self.vision_model, MRM):
            img_emb_g, img_emb_l = self.vision_model(pixel_values)
            img_emb_g = img_emb_g.unsqueeze(1)
            img_emb_l = img_emb_l.view(img_emb_l.size(0), img_emb_l.size(1), -1)
            img_emb_l = img_emb_l.permute(0, 2, 1)

            vision_tokens = torch.cat([img_emb_g, img_emb_l], dim=1)
        else:
            raise NotImplementedError

        vision_tokens = self.align_transformer(vision_tokens)

        cls_token = vision_tokens[:, 0]
        patch_tokens = vision_tokens[:, 1:]
        image_features = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        image_features = F.normalize(image_features, p=2, dim=1)

        outputs = {}
        outputs["vision_tokens"] = vision_tokens
        outputs["image_cls_token"] = cls_token
        outputs["image_patch_tokens"] = patch_tokens
        outputs["image_features"] = image_features

        return outputs

    def forward_text_model(self, encoded_input):
        text_outputs = {}

        if isinstance(self.text_model, MPNetModel):
            model_output = self.text_model(
                input_ids=encoded_input["input_ids"],
                attention_mask=encoded_input["attention_mask"],
            )

            token_embeddings = model_output[
                0
            ]  # First element of model_output contains all token embeddings

            # text embedding projection
            if self.text_projector is not None:
                token_embeddings = self.text_projector(token_embeddings)

            # token_embeddings = self.text_projector(token_embeddings)
            if self.config.text_config.use_cls_token:
                text_features = token_embeddings[:, 0, :]

            else:
                # mean pooling
                input_mask_expanded = (
                    encoded_input["attention_mask"]
                    .unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float()
                )
                text_features = torch.sum(
                    token_embeddings * input_mask_expanded, 1
                ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        elif isinstance(self.text_model, BertModel):
            # BioClinicalMPBERT

            model_output = self.text_model(
                input_ids=encoded_input["input_ids"],
                attention_mask=encoded_input["attention_mask"],
                token_type_ids=encoded_input.get("token_type_ids", None),
            )

            if self.config.text_config.use_cls_token:
                text_features = model_output.last_hidden_state[:, 0, :]

            elif self.config.text_config.use_aggregate_tokens:

                all_embeddings = model_output[2]
                embeddings = torch.stack(
                    all_embeddings[-self.config.text_config.last_n_layers :]
                )
                embeddings = embeddings.permute(1, 0, 2, 3)

                embeddings, sents = aggregate_tokens(
                    embeddings, encoded_input["input_ids"], self.idxtoword
                )
                sent_embeddings = embeddings.mean(axis=2)

                if self.config.text_config.aggregate_method == "sum":
                    word_embeddings = embeddings.sum(axis=1)
                    sent_embeddings = sent_embeddings.sum(axis=1)
                elif self.config.text_config.aggregate_method == "mean":
                    word_embeddings = embeddings.mean(axis=1)
                    sent_embeddings = sent_embeddings.mean(axis=1)

                word_embeddings = word_embeddings.permute(0, 2, 1)

                text_features = sent_embeddings
                text_outputs["word_embeddings"] = word_embeddings

            else:
                text_features = model_output.last_hidden_state
                mask = encoded_input["attention_mask"].unsqueeze(-1).float()
                text_features = torch.sum(text_features * mask, dim=1) / torch.clamp(
                    mask.sum(dim=1), min=1e-9
                )

            if self.text_projector is not None:
                text_features = self.text_projector(text_features)

        else:
            raise NotImplementedError

        text_outputs["text_features_wo_l2_norm"] = text_features
        text_outputs["text_features"] = F.normalize(text_features, p=2, dim=1)

        return text_outputs

    def forward(
        self,
        pixel_values,
        encoded_findings=None,
        encoded_key_phrases=None,
        encoded_negative_phrases=None,
        encoded_random_key_phrases=None,
        return_loss=True,
        **kwargs,
    ):
        vision_outputs = self.forward_vision_model(pixel_values)

        outputs = {}
        outputs.update(vision_outputs)

        # Trainer's self.can_return_loss is True if 'return_loss' is in model's forward function
        if return_loss:
            loss = 0
            losses = {}

            for loss_type, loss_fn in self.loss_fns.items():
                if isinstance(loss_fn, OpenClipLoss):
                    text_outputs = self.forward_text_model(encoded_random_key_phrases)
                    outputs.update(text_outputs)

                    clip_loss = loss_fn(
                        outputs["image_features"],
                        outputs["text_features"],
                    )
                    losses["clip_loss"] = clip_loss
                    loop_loss = clip_loss
                elif isinstance(loss_fn, OpenSigLipLoss):
                    text_outputs = self.forward_text_model(encoded_random_key_phrases)
                    outputs.update(text_outputs)

                    siglip_loss = loss_fn(
                        outputs["image_features"],
                        outputs["text_features"],
                    )
                    losses["siglip_loss"] = siglip_loss
                    loop_loss = siglip_loss
                elif isinstance(loss_fn, RadZeroLoss):
                    loss_outputs = loss_fn(
                        encoded_key_phrases,
                        outputs["vision_tokens"],
                        self.forward_text_model,
                    )
                    radzero_losses = loss_outputs["losses"]
                    losses["radzero_loss"] = (
                        radzero_losses.pop("loss")
                    )
                    for loss_name, loss_value in radzero_losses.items():
                        losses[loss_name] = loss_value
                    loop_loss = losses["radzero_loss"]
                else:
                    raise NotImplementedError

                loss += loop_loss * self.loss_ratio[loss_type]

            losses["loss"] = loss

            outputs["losses"] = losses

        return outputs

    def compute_logits(
        self,
        pixel_values,
        encoded_key_phrases,
        **kwargs,
    ):
        vision_outputs = self.forward_vision_model(pixel_values)

        outputs = {}

        if self.compute_logits_type == "radzero":

            splited_key_phrases = [
                {
                    "input_ids": encoded_key_phrases[0]["input_ids"][i : i + 1],
                    "attention_mask": encoded_key_phrases[0]["attention_mask"][
                        i : i + 1
                    ],
                }
                for i in range(encoded_key_phrases[0]["input_ids"].size(0))
            ]

            loss_outputs = self.loss_fns["RadZeroLoss"](
                splited_key_phrases,
                vision_outputs["vision_tokens"],
                self.forward_text_model,
                ddp_gather=False,
                need_attn_weights=True,
                compute_loss=False,
            )
            outputs.update(loss_outputs)

            # mean attention weights from all layers
            outputs["similarity_scores"] = torch.mean(
                torch.stack(loss_outputs["t2i_attn_weights"]), dim=0
            )

            # remove attention score for cls token
            if self.loss_fns["RadZeroLoss"].use_vision_cls_token:
                outputs["similarity_scores"] = outputs["similarity_scores"][:, :, 1:]

            # compute logits
            if self.loss_fns["RadZeroLoss"].compute_i2t_loss:
                logits = (loss_outputs["t2i_logits"] + loss_outputs["i2t_logits"]) / 2
            else:
                logits = loss_outputs["t2i_logits"]
            logits = logits.T

            logits = (
                logits / self.loss_fns["RadZeroLoss"].loss_temperature.exp()
            )

        elif self.compute_logits_type == "cls_alignment":

            features_list = []
            for kp in encoded_key_phrases:
                feat = self.forward_text_model(kp)["text_features"]  # (N_i, D)
                features_list.append(feat)

            key_features = torch.cat(features_list, dim=0)  # (N_total, D)
            logits = vision_outputs["image_cls_token"] @ key_features.T

        elif self.compute_logits_type == "global_alignment":
            features_list = []
            for kp in encoded_key_phrases:
                feat = self.forward_text_model(kp)["text_features"]  # (N_i, D)
                features_list.append(feat)

            key_features = torch.cat(features_list, dim=0)  # (N_total, D)
            logits = vision_outputs["image_features"] @ key_features.T
            similarity_scores = torch.einsum(
                "ind,jd->ijn",
                vision_outputs["image_patch_tokens"],
                key_features[:, self.hidden_size :],
            )
            outputs["similarity_scores"] = similarity_scores  # (B, N_total, L)

        outputs["logits"] = logits
        return outputs
