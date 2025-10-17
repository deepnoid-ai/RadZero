import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPVisionModel,
    Dinov2Model,
    PreTrainedModel,
    SiglipVisionModel,
    CLIPModel,
)
import open_clip
from common.trainer import logger

from external.CARZero.CARZero.models.transformer_backbones import MRM, load_weight

from functools import partial
import os
import math




def build_vision_encoder(config):
    if config.model_type == "siglip":
        model = SiglipVisionModel.from_pretrained(config.pretrained_name_or_path)
    elif config.model_type == "clip":
        model = CLIPVisionModel.from_pretrained(config.pretrained_name_or_path)
    elif config.model_type == "dinov2":
        model = Dinov2Model.from_pretrained(config.pretrained_name_or_path)
    elif config.model_type == "biomedclip":
        model = BioMedClipVisionEncoder(config)
    elif config.model_type == "xrayclip":
        model = CLIPVisionEncoder(config)
    elif config.model_type == "m3ae":
        model = mrm_vit_b16(
            ckpt_path=os.path.join(
                config.pretrained_dir, config.pretrained_name_or_path
            )
        )
    else:
        raise NotImplementedError()

    return model


# copy from CARZero
def mrm_vit_b16(ckpt_path, **kwargs):
    model = MRM(
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

    model = load_weight(model, ckpt_path)  # train from M3AE

    return model

class BioMedClipVisionEncoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # load pretrained model
        # reference: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/main/biomed_clip_example.ipynb
        clip_model, preprocess = open_clip.create_model_from_pretrained(
            config.pretrained_name_or_path
        )
        self.visual_model = clip_model.visual.trunk

        # use sequence
        self.visual_model.global_pool = None

        self.img_size = config.to_dict().get("img_size")

        # default is 224
        if self.img_size is not None and self.img_size != 224:
            logger.info(
                f"apply high resolution size (biomedclip): 224 -> {self.img_size}"
            )
            self.visual_model.patch_embed.img_size = (
                self.img_size,
                self.img_size,
            )

            self.visual_model.pos_embed = self.resample_pos_embed(
                self.visual_model.pos_embed,
                self.img_size,
                self.visual_model.patch_embed.proj.kernel_size[0],
            )

    def resample_pos_embed(self, posemb, hr_size, kernel_size, num_prefix_tokens=1):
        new_size = (hr_size // kernel_size, hr_size // kernel_size)

        num_pos_tokens = posemb.shape[1] - num_prefix_tokens
        old_size = int(math.sqrt(num_pos_tokens))
        bs = posemb.shape[0]

        if num_prefix_tokens:
            posemb_prefix, posemb = (
                posemb[:, :num_prefix_tokens],
                posemb[:, num_prefix_tokens:],
            )
        else:
            posemb_prefix, posemb = None, posemb

        embed_dim = posemb.shape[-1]
        orig_dtype = posemb.dtype
        posemb = posemb.float()
        posemb = posemb.reshape(bs, old_size, old_size, -1).permute(0, 3, 1, 2)
        posemb = F.interpolate(posemb, size=new_size, mode="bicubic", antialias=True)
        posemb = posemb.permute(0, 2, 3, 1).reshape(bs, -1, embed_dim)
        posemb = posemb.to(dtype=orig_dtype)

        if posemb_prefix is not None:
            posemb = torch.cat([posemb_prefix, posemb], 1)

        posemb = nn.Parameter(posemb)
        return posemb

    def forward(self, pixel_values: torch.FloatTensor, **kwargs):
        outputs = self.visual_model(pixel_values)

        return {"last_hidden_state": outputs}


# copy from transformers.models.clip_modeling_clip.py
class CLIPVisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = CLIPModel.from_pretrained(
            config.pretrained_name_or_path
        ).vision_model

        self.img_size = config.to_dict().get("img_size")

        # default is 512
        if self.img_size is not None and self.img_size != 512:
            logger.info(
                f"apply high resolution size (XrayCLIP): 512 -> {self.img_size}"
            )

            self.patch_size = config.vision_config["patch_size"]

            self.num_patches = (self.img_size // self.patch_size) ** 2
            self.num_positions = self.num_patches + 1
            self.vision_model.embeddings.position_ids = torch.arange(
                self.num_positions
            ).expand((1, -1))

            self.vision_model.embeddings.position_embedding = self.resample_pos_embed(
                self.vision_model.embeddings.position_embedding.weight,
                self.img_size,
                self.patch_size,
                num_prefix_tokens=1,
            )

    def resample_pos_embed(self, posemb, hr_size, kernel_size, num_prefix_tokens=1):
        new_size = (hr_size // kernel_size, hr_size // kernel_size)

        num_pos_tokens = posemb.shape[0] - num_prefix_tokens
        old_size = int(math.sqrt(num_pos_tokens))

        if num_prefix_tokens:
            posemb_prefix, posemb = (
                posemb[:num_prefix_tokens],
                posemb[num_prefix_tokens:],
            )
        else:
            posemb_prefix, posemb = None, posemb

        embed_dim = posemb.shape[-1]
        orig_dtype = posemb.dtype
        posemb = posemb.float()
        posemb = posemb.reshape(old_size, old_size, -1).permute(2, 0, 1).unsqueeze(0)

        posemb = F.interpolate(posemb, size=new_size, mode="bicubic", antialias=True)
        posemb = posemb.permute(0, 2, 3, 1).reshape(-1, embed_dim)
        posemb = posemb.to(dtype=orig_dtype)

        if posemb_prefix is not None:
            posemb = torch.cat([posemb_prefix, posemb], dim=0)

        posemb = nn.Embedding.from_pretrained(posemb)
        return posemb

    def forward(self, *args, **kwargs):
        return self.vision_model(*args, **kwargs)