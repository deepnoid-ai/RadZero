import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder
from .configuration import AlignTransformerConfig


def build_align_transformer(config):
    if config.model_type == "align_transformer":
        model = AlignTransformer(config)
    elif config.model_type == "identity":
        model = Identity()
    elif config.model_type == "mlp":
        model = MLP()
    elif config.model_type == "linear":
        model = Linear()
    else:
        raise NotImplementedError()

    return model


class AlignTransformer(PreTrainedModel):
    def __init__(self, config: AlignTransformerConfig):
        super().__init__(config)

        if config.num_hidden_layers:
            self.transformer_layers = Dinov2Encoder(config)
        else:
            self.transformer_layers = None

        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.layer_norm = None

    def forward(self, vision_tokens):

        if self.transformer_layers is not None:
            vision_tokens = self.transformer_layers(vision_tokens)["last_hidden_state"]

        if self.layer_norm is not None:
            vision_tokens = self.layer_norm(vision_tokens)

        return vision_tokens


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Linear(nn.Module):
    def __init__(self, in_features=768, out_features=768):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        dropout = 0.1
        self.mlp_layer = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 768),
        )

    def forward(self, x):
        return self.mlp_layer(x)
