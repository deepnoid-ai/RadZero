import os
from typing import List, Optional

from datasets import Dataset, DatasetDict, DatasetInfo, Features, Image, NamedSplit
from tqdm import tqdm

from common.trainer import logger
from common.utils import load_json


class WithMissingValueDataset(Dataset):
    @classmethod
    def from_list(
        cls,
        mapping: List[dict],
        features: Optional[Features] = None,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
    ) -> "Dataset":
        """
        Convert a list of dicts to a `pyarrow.Table` to create a [`Dataset`]`.

        Note that the keys of the first entry will be used to determine the dataset columns,
        regardless of what is passed to features.

        Args:
            mapping (`List[dict]`): A list of mappings of strings to row values.
            features (`Features`, optional): Dataset features.
            info (`DatasetInfo`, optional): Dataset information, like description, citation, etc.
            split (`NamedSplit`, optional): Name of the dataset split.

        Returns:
            [`Dataset`]
        """

        # (modified) get all of column names
        column_names = set()
        for i in mapping:
            column_names.update(i.keys())

        # for simplicity and consistency wrt OptimizedTypedSequence we do not use InMemoryTable.from_pylist here
        mapping = (
            {k: [r.get(k) for r in mapping] for k in column_names} if mapping else {}
        )

        return cls.from_dict(mapping, features, info, split)


def input_json_file_load(json_path, train_flag, data_root, **kwargs):
    logger.info(f"load dataset: {json_path}")
    input_json = load_json(os.path.join(data_root, json_path))

    # add data_root path
    for index, v in enumerate(tqdm(input_json)):
        v["image"] = os.path.join(data_root, v["image"])
        v["train"] = train_flag

    return input_json


def transform(batch):
    # TODO: implement data prerpocessing

    # TODO: implement data augmentation

    return batch


def load_datasets(cfg, train, inference):
    dataset = {}

    if train:
        # train dataset
        train_dataset = []
        for i in cfg["train"]:
            train_dataset += input_json_file_load(cfg[i], train_flag=True, **cfg)
        train_dataset = WithMissingValueDataset.from_list(train_dataset)

        # eval dataset
        eval_dataset = []
        for i in cfg["eval"]:
            eval_dataset += input_json_file_load(cfg[i], train_flag=False, **cfg)
        eval_dataset = WithMissingValueDataset.from_list(eval_dataset)

        dataset.update({"train": train_dataset, "eval": eval_dataset})

    if inference:
        # test dataset
        test_dataset = []
        for i in cfg["test"]:
            test_dataset += input_json_file_load(cfg[i], train_flag=False, **cfg)
        test_dataset = WithMissingValueDataset.from_list(test_dataset)

        dataset.update({"test": test_dataset})

    dataset = DatasetDict(dataset)

    dataset.cleanup_cache_files()
    dataset = dataset.cast_column("image", Image())
    dataset.set_transform(transform)

    return dataset


def collate_fn(batch):
    outputs = {}

    outputs["image"] = [i["image"].convert("RGB") for i in batch]

    # TODO: apply image processor

    # TODO: apply text tokenizer

    # TODO: convert inputs, labels to torch.tensor

    # TODO: batch level preprocessing (ex: padding for same sequence length)

    return outputs