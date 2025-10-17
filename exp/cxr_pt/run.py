import argparse
import os
from functools import partial

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import EarlyStoppingCallback, TrainingArguments

from common.code_snapshot import code_snapshot
from common.trainer import logger
from common.utils import Config, output_directory_setting, str2bool
from exp.cxr_pt.dataset import collate_fn, load_datasets
from exp.cxr_pt.inference.inference import Inference
from exp.cxr_pt.model import load_model
from exp.cxr_pt.trainer import CXRPreTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument(
        "--cfg-path",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="path to configuration file.",
    )
    parser.add_argument(
        "--add_cfg_list",
        default=[],
        type=str,
        nargs="+",
        help="List of YAML files. The cfg will be overwritten in the given order.",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="User. This argument, when provided, overrides any previous settings.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of experiment. This argument, when provided, overrides any previous settings.",
    )
    parser.add_argument(
        "--train",
        default=True,
        type=str2bool,
        choices=[True, False],
        help="Set the train mode to True or False",
    )
    parser.add_argument(
        "--inference",
        default=True,
        type=str2bool,
        choices=[True, False],
        help="Set the inference mode to True or False",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="directory for saving inference results",
    )
    parser.add_argument(
        "--compute_metric",
        default=True,
        type=str2bool,
        choices=[True, False],
        help="Compute metrics with inference results. Set to True or False",
    )
    parser.add_argument("--no_report", action="store_true", help="Skip report to wandb")
    args = parser.parse_args()

    return args


@record
def main():
    args = parse_args()
    cfg = Config(args).config

    output_directory_setting(cfg, logger)
    snapshot_dir = code_snapshot(
        save_dir=os.path.join(cfg["train"]["output_dir"], "snapshot"), cfg=cfg
    )

    deepspeed = dict(cfg["deepspeed"]) if cfg.get("deepspeed") else None

    training_args = TrainingArguments(**cfg["train"], deepspeed=deepspeed)
    models = load_model(cfg["model"], output_dir=cfg["train"]["output_dir"])

    dataset = load_datasets(
        cfg["dataset"],
        train=cfg["args"]["train"],
        inference=False,
    )

    trainer = CXRPreTrainer(
        model=models["model"],
        args=training_args,
        train_dataset=dataset.get("train"),
        eval_dataset=dataset.get("eval"),
        data_collator=partial(
            collate_fn,
            tokenizer=models["tokenizer"],
            image_processor=models["image_processor"],
        ),
        cfg=cfg,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["experiment"]["early_stopping_patience"]
            )
        ],
        tokenizer=models["tokenizer"],
    )

    if cfg["args"]["train"]:
        logger.info("run train")
        trainer.train(
            resume_from_checkpoint=cfg["experiment"]["resume_from_checkpoint"]
        )

    if cfg["args"]["inference"]:

        # checkpoint path
        if trainer.state.best_model_checkpoint is not None:
            checkpoint = trainer.state.best_model_checkpoint
        else:
            checkpoint = cfg["experiment"]["resume_from_checkpoint"]

        resume_from_checkpoint = trainer.inference_load_from_checkpoint(
            resume_from_checkpoint=checkpoint
        )

        if trainer.is_world_process_zero():
            models["model"].to(trainer.args.device)
            models["model"].to(torch.float32)

            data_root_dir = cfg["dataset"]["data_root"]
            save_root_dir = os.path.join(
                cfg["train"]["output_dir"],
                "inference",
                os.path.basename(snapshot_dir),
                os.path.basename(resume_from_checkpoint),
            )

            inference = Inference(
                cls_dataset=cfg["inference"]["cls_dataset"],
                det_dataset=cfg["inference"]["det_dataset"],
                seg_dataset=cfg["inference"]["seg_dataset"],
                data_root_dir=data_root_dir,
                batch_size=cfg["inference"]["batch_size"],
                num_workers=cfg["inference"]["num_workers"],
            )

            inference.classification(
                **models, save_root_dir=os.path.join(save_root_dir, "classification")
            )
            inference.grounding(
                **models, save_root_dir=os.path.join(save_root_dir, "grounding")
            )
            inference.segmentation(
                **models,
                save_root_dir=os.path.join(save_root_dir, "segmentation"),
                compute_pixel_level_auroc=cfg["inference"]["compute_pixel_level_auroc"],
            )

        else:
            pass


if __name__ == "__main__":
    main()
