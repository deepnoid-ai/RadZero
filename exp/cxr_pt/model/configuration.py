from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config


class VisionConfig(PretrainedConfig):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @staticmethod
    def from_exp_config(vision_config: dict):

        model_type = vision_config["model_type"]

        if model_type in [
            "siglip_vision_model",
            "clip_vision_model",
            "dinov2",
            "sam",
            "raddino",
        ]:
            config = AutoConfig.from_pretrained(
                vision_config["pretrained_name_or_path"]
            )
            config = config.to_dict()
            vision_config.update(config)
        elif model_type == "xrayclip":
            config = AutoConfig.from_pretrained(
                vision_config["pretrained_name_or_path"]
            )
            config = config.to_dict()
            config["model_type"] = "xrayclip"
            vision_config.update(config)
        elif model_type == "biomedclip":
            pass
        elif model_type == "m3ae":
            pass

        else:
            raise NotImplementedError()

        vision_config = VisionConfig(**vision_config)

        return vision_config


class TextConfig(PretrainedConfig):
    def __init__(
        self,
        model_type,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = model_type

    @staticmethod
    def from_exp_config(
        text_config: dict,
    ):
        model_type = text_config["model_type"]

        if model_type in [
            "siglip_text_model",
            "clip_text_model",
            "mpnet",
            "biomedclip",
            "bioclinicalmpbert",
        ]:
            text_config = TextConfig(**text_config)
        else:
            raise NotImplementedError()

        return text_config


class AlignTransformerConfig(PretrainedConfig):
    def __init__(
        self,
        model_type: str = "align_transformer",
        projector_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.projector_config = projector_config

    @staticmethod
    def from_exp_config(
        align_transformer_config: dict,
    ):
        projector_config = align_transformer_config.pop("projector_config", None)

        config = Dinov2Config(**align_transformer_config)
        config = config.to_dict()

        align_transformer_config = AlignTransformerConfig(
            **(config | align_transformer_config),
            projector_config=projector_config,
        )

        return align_transformer_config


class CxrAlignConfig(PretrainedConfig):
    is_composition = True

    def __init__(
        self,
        vision_config: dict,
        text_config: dict,
        align_transformer_config: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Vision config
        self.vision_config = VisionConfig.from_exp_config(vision_config)

        # text config
        self.text_config = TextConfig.from_exp_config(text_config)

        self.align_transformer_config = AlignTransformerConfig.from_exp_config(
            align_transformer_config
        )

        self.kwargs = kwargs
