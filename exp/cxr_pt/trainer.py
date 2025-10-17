import os
from typing import Optional

from common.trainer import BaseTrainer, get_last_checkpoint, logger


class CXRPreTrainer(BaseTrainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        cfg=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            cfg=cfg,
            **kwargs,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        losses = outputs.pop("losses")
        loss = losses.pop("loss")

        outputs = dict(outputs)
        outputs["losses"] = losses

        return (loss, outputs) if return_outputs else loss

    def inference_step(self, model, inputs, ignore_keys, **kwargs):
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []
        outputs = {}

        # TODO: implement inference code

        # remove outputs to ignore
        for k in ignore_keys:
            if k in outputs:
                outputs.pop(k)

        return outputs

    def save_inference_outputs(
        self,
        outputs,
        output_dir: Optional[str] = None,
        checkpoint_name: Optional[str] = None,
        **kwargs,
    ):
        if output_dir is None:
            output_dir = os.path.join(self.args.output_dir, "inference")
        os.makedirs(output_dir, exist_ok=True)
        # save_path = os.path.join(output_dir, f"{checkpoint_name}.json")

        results = {}

        # TODO: save outputs

        return results

    def inference_load_from_checkpoint(self, resume_from_checkpoint=None):

        # skip resume_from_checkpoint after train (self._load_best_model())
        if self.args.load_best_model_at_end and (
            resume_from_checkpoint == self.state.best_model_checkpoint
        ):
            logger.info(
                "Since best model has already been loaded, skip resume from checkpoint"
            )
            pass

        else:
            if resume_from_checkpoint is False:
                resume_from_checkpoint = None

            # Load potential model checkpoint
            if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
                resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
                if resume_from_checkpoint is None:
                    raise ValueError(
                        f"No valid checkpoint found in output directory ({self.args.output_dir})"
                    )

            if resume_from_checkpoint is not None:
                self._load_from_checkpoint(resume_from_checkpoint)

        return resume_from_checkpoint
