import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from external.MGCA.mgca.constants import (
    CHEXPERT_COMPETITION_TASKS,
    CHEXPERT_DATA_DIR,
    CHEXPERT_PATH_COL,
    CHEXPERT_TEST_CSV,
    CHEXPERT_TRAIN_CSV,
    CHEXPERT_UNCERTAIN_MAPPINGS,
    CHEXPERT_VALID_CSV,
    CHEXPERT_VIEW_COL,
    COVIDX_DATA_DIR,
    COVIDX_TEST_CSV,
    COVIDX_TRAIN_CSV,
    COVIDX_VALID_CSV,
    MIMIC_CXR_DATA_DIR,
    MIMIC_CXR_PATH_COL,
    MIMIC_CXR_TEST_CSV,
    MIMIC_CXR_TRAIN_CSV,
    MIMIC_CXR_VALID_CSV,
    MIMIC_CXR_VIEW_COL,
    RSNA_DATA_DIR,
    RSNA_IMG_DIR,
    RSNA_TEST_CSV,
    RSNA_TRAIN_CSV,
    RSNA_VALID_CSV,
)
from external.MGCA.mgca.datasets.utils import get_imgs, read_from_dicom

np.random.seed(42)


class BaseImageDataset(Dataset):
    def __init__(self, split="train", transform=None) -> None:
        super().__init__()

        self.split = split
        self.transform = transform

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class CheXpertImageDataset(BaseImageDataset):
    def __init__(
        self,
        config,
        data_root,
        split="train",
        transform=None,
        img_type="Frontal",
        data_pct=0.01,
        imsize=256,
    ):
        super().__init__(split=split, transform=transform)

        data_dir = os.path.join(data_root, config["path"])

        if not os.path.exists(data_dir):
            raise RuntimeError(f"{data_dir} does not exist!")

        self.imsize = imsize

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(os.path.join(data_root, config["train"]))
        elif split == "valid":
            self.df = pd.read_csv(os.path.join(data_root, config["eval"]))
        elif split == "test":
            self.df = pd.read_csv(os.path.join(data_root, config["test"]))
        else:
            raise NotImplementedError(f"split {split} is not implemented!")

        # filter image type
        if img_type != "All":
            self.df = self.df[self.df["Frontal/Lateral"] == img_type]

        # sample data
        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)

        # get path
        self.df["Path"] = self.df["Path"].apply(
            lambda x: os.path.join(data_dir, "/".join(x.split("/")[1:]))
        )

        # fill na with 0s
        self.df = self.df.fillna(0)

        # replace uncertains
        uncertain_mask = {k: -1 for k in config["task"]}
        self.df = self.df.replace(uncertain_mask, config["uncertain_map"])

        self.path = self.df["Path"].values
        self.labels = self.df.loc[:, config["task"]].values

    def __getitem__(self, index):
        # get image
        img_path = self.path[index]
        x = get_imgs(img_path, self.imsize, self.transform)

        # get labels
        y = self.labels[index]
        y = torch.tensor(y)

        return x, y

    def __len__(self):
        return len(self.df)


class MIMICImageDataset(BaseImageDataset):
    def __init__(
        self,
        split="train",
        transform=None,
        data_pct=1.0,
        img_type="Frontal",
        imsize=256,
    ):
        super().__init__(split, transform)
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(
                "MIMIC CXR data directory %s does not exist!" % MIMIC_CXR_DATA_DIR
            )

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(MIMIC_CXR_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(MIMIC_CXR_VALID_CSV)
        else:
            self.df = pd.read_csv(MIMIC_CXR_TEST_CSV)

        # filter image type
        if img_type != "All":
            self.df = self.df[self.df[MIMIC_CXR_VIEW_COL].isin(["PA", "AP"])]

        # get a fraction of dataset
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
            # print(self.df)

        # get path
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:]))
        )

        # fill na with 0s
        self.df = self.df.fillna(0)

        # replace uncertains
        uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
        self.df = self.df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)

        self.imsize = imsize

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # get image
        img_path = row["Path"]
        x = get_imgs(img_path, self.imsize, self.transform)

        # get labels
        y = list(row[CHEXPERT_COMPETITION_TASKS])
        y = torch.tensor(y)

        return x, y, img_path

    def __len__(self):
        return len(self.df)


class RSNAImageDataset(BaseImageDataset):
    def __init__(
        self,
        config,
        data_root,
        split="train",
        transform=None,
        phase="classification",
        data_pct=0.01,
        imsize=256,
    ) -> None:
        super().__init__(split=split, transform=transform)

        data_dir = os.path.join(data_root, config["path"])

        if not os.path.exists(data_dir):
            raise RuntimeError(f"{data_dir} does not exist!")

        if self.split == "train":
            self.df = pd.read_csv(os.path.join(data_root, config["train"]))
        elif self.split == "valid":
            self.df = pd.read_csv(os.path.join(data_root, config["eval"]))
        elif self.split == "test":
            self.df = pd.read_csv(os.path.join(data_root, config["test"]))
        else:
            raise ValueError(f"split {split} does not exist!")

        if phase == "detection":
            self.df = self.df[self.df["Target"] == 1]

        self.df["Path"] = self.df["patientId"].apply(
            lambda x: os.path.join(data_dir, config["img_dir"], (x + ".dcm"))
        )

        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)

        self.imsize = imsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get image
        img_path = row["Path"]
        x = read_from_dicom(img_path, self.imsize, self.transform)
        y = float(row["Target"])
        y = torch.tensor([y])

        return x, y


class COVIDXImageDataset(BaseImageDataset):
    def __init__(
        self,
        config,
        data_root,
        split="train",
        transform=None,
        data_pct=0.01,
        imsize=256,
    ) -> None:
        super().__init__(split=split, transform=transform)

        data_dir = os.path.join(data_root, config["path"])

        if not os.path.exists(data_dir):
            raise RuntimeError(f"{data_dir} does not exist!")

        if self.split == "train":
            self.df = pd.read_csv(os.path.join(data_root, config["train"]))
            self.df["filename"] = self.df["filename"].apply(
                lambda x: os.path.join(data_dir, f"train/{x}")
            )
        elif self.split == "valid":
            self.df = pd.read_csv(os.path.join(data_root, config["eval"]))
            self.df["filename"] = self.df["filename"].apply(
                lambda x: os.path.join(data_dir, f"train/{x}")
            )
        elif self.split == "test":
            self.df = pd.read_csv(os.path.join(data_root, config["test"]))
            self.df["filename"] = self.df["filename"].apply(
                lambda x: os.path.join(data_dir, f"test/{x}")
            )
        else:
            raise ValueError(f"split {split} does not exist!")

        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)

        self.imsize = imsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get image
        img_path = row["filename"]
        x = get_imgs(img_path, self.imsize, self.transform)
        y = float(row["labels"])
        y = torch.tensor([y])

        return x, y
