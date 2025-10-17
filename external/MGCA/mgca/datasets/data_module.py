import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BlipImageProcessor


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        data_root,
        dataset,
        collate_fn,
        transforms,
        data_pct,
        batch_size,
        num_workers,
        crop_size=224,
    ):
        super().__init__()
        self.config = config
        self.data_root = data_root
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.transforms = transforms
        self.data_pct = data_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size

    def train_dataloader(self):
        if self.transforms:
            if isinstance(self.transforms, BlipImageProcessor):
                transform = self.transforms
            else:
                transform = self.transforms(True, self.crop_size)
        else:
            transform = None

        dataset = self.dataset(
            self.config,
            self.data_root,
            split="train",
            transform=transform,
            data_pct=self.data_pct,
        )

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.transforms:
            if isinstance(self.transforms, BlipImageProcessor):
                transform = self.transforms
            else:
                transform = self.transforms(True, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            self.config,
            self.data_root,
            split="valid",
            transform=transform,
            data_pct=self.data_pct,
        )
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if self.transforms:
            if isinstance(self.transforms, BlipImageProcessor):
                transform = self.transforms
            else:
                transform = self.transforms(True, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            self.config,
            self.data_root,
            split="test",
            transform=transform,
            data_pct=self.data_pct,
        )
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
