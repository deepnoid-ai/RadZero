from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import AUROC, Accuracy


class SSLFineTuner(LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        model_name: str = "resnet_50",
        in_features: int = 2048,
        num_classes: int = 5,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-6,
        task: str = "multilabel",
        *args,
        **kwargs
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.task = task
        if self.task == "multilabel" or self.task == "binary":
            self.train_auc = AUROC(task=self.task, num_labels=num_classes)
            self.val_auc = AUROC(task=self.task, num_labels=num_classes)
            self.test_auc = AUROC(task=self.task, num_labels=num_classes)
        else:
            self.train_acc = Accuracy(task=self.task, num_classes=num_classes, top_k=1)
            self.val_acc = Accuracy(task=self.task, num_classes=num_classes, top_k=1)
            self.test_acc = Accuracy(task=self.task, num_classes=num_classes, top_k=1)

        self.save_hyperparameters(ignore=["backbone"])

        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.model_name = model_name

        self.linear_layer = SSLEvaluator(
            n_input=in_features, n_classes=num_classes, p=dropout, n_hidden=hidden_dim
        )

        self.concat_2D = kwargs["concat_2D"]  # bool

    def on_train_batch_start(self, batch, batch_idx) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        if self.task == "multilabel" or self.task == "binary":
            auc = self.train_auc(torch.sigmoid(logits).float(), y.long())
            self.log("train_auc_step", auc, prog_bar=True, sync_dist=True)
            self.log("train_auc_epoch", self.train_auc, prog_bar=True, sync_dist=True)
        else:
            acc = self.train_acc(F.softmax(logits, dim=-1).float(), y.long())
            self.log("train_acc_step", acc, prog_bar=True, sync_dist=True)
            self.log("train_acc_epoch", self.train_acc, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        if self.task == "multilabel" or self.task == "binary":
            self.val_auc(torch.sigmoid(logits).float(), y.long())
            self.log(
                "val_auc", self.val_auc, on_epoch=True, prog_bar=True, sync_dist=True
            )
        else:
            self.val_acc(F.softmax(logits, dim=-1).float(), y.long())
            self.log(
                "val_acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True
            )

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.log("test_loss", loss, sync_dist=True)
        if self.task == "multilabel" or self.task == "binary":
            self.test_auc(torch.sigmoid(logits).float(), y.long())
            # TODO: save probs and labels
            self.log("test_auc", self.test_auc, on_epoch=True)
        else:
            self.test_acc(F.softmax(logits, dim=-1).float(), y.long())
            self.log("test_acc", self.test_acc, on_epoch=True)

        return loss

    def shared_step(self, batch):
        x, y = batch
        # For multi-class
        with torch.no_grad():
            if self.model_name == "xraydino_vit":
                feats = self.backbone(x).last_hidden_state[:, 0]
                seq_tokens = self.backbone(x).last_hidden_state[:, 1:]
            else:
                feats, seq_tokens = self.backbone(x)

            if self.concat_2D:
                gap = nn.AdaptiveAvgPool1d(1)
                seq_tokens_t = seq_tokens.transpose(1, 2)
                seq_pooled = gap(seq_tokens_t).squeeze(-1)
                feats = torch.cat([feats, seq_pooled], dim=1)

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        if self.task == "multilabel" or self.task == "binary":
            loss = F.binary_cross_entropy_with_logits(logits.float(), y.float())
        else:
            y = y.squeeze()
            loss = F.cross_entropy(logits.float(), y.long())

        return loss, logits, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.linear_layer.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )

        # lr_scheduler = CosineAnnealingWarmupRestarts(
        #     optimizer,
        #     first_cycle_steps=self.training_steps,
        #     cycle_mult=1.0,
        #     max_lr=self.hparams.learning_rate,
        #     min_lr=0.0,
        #     warmup_steps=int(self.training_steps * 0.4)
        # )
        # scheduler = {
        #     "scheduler": lr_scheduler,
        #     "interval": "step",
        #     "frequency": 1
        # }
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        effective_batch_size = trainer.accumulate_grad_batches * trainer.num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=None, p=0.1) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if self.n_hidden is None:
            self.block_forward = nn.Sequential(
                Flatten(), nn.Dropout(p=p), nn.Linear(n_input, n_classes)
            )
        else:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes),
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class Flatten(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)
