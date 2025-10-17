import os
from typing import List

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from common.trainer import logger

from .configuration import CxrAlignConfig
from .modeling import CxrAlignModel
from .processing import load_processor


def load_model(config, **kwargs):
    dtype = config["dtype"]
    if type(dtype) is str:
        dtype = eval(dtype)

    model_class = eval(config["model_class_name"])
    config_class = eval(config["config_class_name"])

    # prepare model
    logger.info("Build model...")

    # load pretrained model
    if config["pretrained_ckpt"]:
        logger.info(f"Load PreTrainedModel : {config['pretrained_ckpt']}")
        model = model_class.from_pretrained(
            config["pretrained_ckpt"], torch_dtype=dtype
        )

    else:
        model_config = config_class(
            module_to_update=config["module_to_update"], **config["model_config"]
        )
        model = model_class(model_config)

        # set dtype of model
        model.to(dtype)

    # load adapter & trained parameters
    if config["adapter_ckpt"]:
        model = PeftModel.from_pretrained(
            model, config["adapter_ckpt"], is_trainable=True
        )

    output = {"model": model}

    output.update(load_processor(config))

    output["model"] = apply_params_setting(
        config, output["model"], output_dir=kwargs["output_dir"]
    )

    return output


def set_trainable_parameters(model, module_to_update: List[str] = None):

    for param in model.parameters():
        param.requires_grad = False

    if not module_to_update:
        logger.info("All parameters are frozen.")
        return model

    for model_name in module_to_update:

        if not hasattr(model, model_name) or not getattr(model, model_name):
            logger.info(f"{model_name} model does not exist in model.")
            continue

        if type(model.__getattr__(model_name)) == torch.nn.parameter.Parameter:
            model.__getattr__(model_name).requires_grad = True
        else:
            for param in model.__getattr__(model_name).parameters():
                param.requires_grad = True

        logger.info(f"{model_name} model's parameters are trainable.")


def apply_params_setting(config, model, output_dir=None):

    # apply lora if requested
    if config["lora_config"]["use_lora"]:
        assert (
            type(model) != PeftModel
        ), "model is already PeftModel with adapter. Please check the model configs"

        # LoRA
        peft_config = LoraConfig(
            target_modules=r"{}".format(config["lora_config"]["target_modules"]),
            inference_mode=config["lora_config"]["inference_mode"],
            r=config["lora_config"]["lora_r"],
            lora_alpha=config["lora_config"]["lora_alpha"],
            lora_dropout=config["lora_config"]["lora_dropout"],
            # additional trainable parameters
            modules_to_save=config["module_to_update"],
        )

        # save base model for load base model and adapter
        if not config["pretrained_ckpt"] and output_dir is not None:
            # save checkpoint as base model
            model.save_pretrained(os.path.join(output_dir, "base"))

        # obtain peft-applied model and set the base_model_name_or_path
        model = get_peft_model(model, peft_config)
    else:
        # Full training

        # set trainable parameters
        set_trainable_parameters(model, config["module_to_update"])

    return model
