import os
import random

import numpy as np
import PIL
import torch
from datasets import DatasetDict, Image
from torchvision.transforms import Compose
from tqdm import tqdm
from transformers import BertTokenizerFast, MPNetTokenizerFast

from common.dataset import WithMissingValueDataset
from common.trainer import logger
from common.utils import load_json
from exp.cxr_pt.model.processing import M3AEImageProcessor


def input_json_file_load(json_path, data_root, train_flag, **kwargs):
    logger.info(f"load dataset: {json_path}")
    input_json = load_json(os.path.join(data_root, json_path))

    use_frontal_view_only = kwargs.get("use_frontal_view_only", False)
    dataset_name = json_path.split("/")[0]

    data_list = list()
    for data in tqdm(input_json):
        entry = dict()

        if dataset_name == "MIMIC-CXR":
            view_position = data.get("view_position", "")
            # There are cases where it is treated as 'Nan'
            view_position = (
                str(view_position).lower()
                if isinstance(view_position, str) and view_position.strip()
                else ""
            )

            if use_frontal_view_only and view_position not in ["pa", "ap", ""]:
                continue

            entry = {}
            image_path = os.path.join(
                data_root, "MIMIC-CXR", "images", data["dicom_id"]
            )
            entry["image"] = image_path

            if data.get("key_phrases"):
                entry["key_phrases"] = [i for i in data["key_phrases"] if i.strip()]
            else:
                continue

            entry["train"] = train_flag

            data_list.append(entry)

    # remove MS-CXR from the training dataset
    if kwargs.get("rm_mscxr") and train_flag:
        ms_cxr_test_json = load_json(os.path.join(data_root, kwargs.get("MS_CXR_test")))
        ms_cxr_image_path_set = {os.path.basename(i["image"]) for i in ms_cxr_test_json}

        filtered_data_list = [
            i
            for i in data_list
            if os.path.basename(i["image"]) not in ms_cxr_image_path_set
        ]
        logger.info(
            f"number of instances and MS CXR removed from the training dataset: {len(data_list) - len(filtered_data_list)}"
        )
        data_list = filtered_data_list

    # log dataset name and number of instances, 1 line
    logger.info(f"dataset name: {dataset_name}, number of instances: {len(data_list)}")

    return data_list


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

    return dataset


def transform_fn(batch, transforms):
    if batch["train"][0]:
        for i in range(len(batch["image"])):
            batch["image"][i] = PIL.Image.fromarray(
                transforms(image=np.array(batch["image"][i]))["image"]
            )
    return batch


def collate_fn(batch, tokenizer, image_processor):
    output_batch = {}

    # TODO : Check PIL.Image dtype

    if isinstance(image_processor, Compose):
        processor_outputs = torch.stack(
            [image_processor(item["image"].convert("RGB")) for item in batch]
        )
    elif isinstance(image_processor, M3AEImageProcessor):
        processor_outputs = image_processor(
            [item["image"] for item in batch], train=batch[0]["train"]
        )
    else:
        processor_outputs = image_processor(
            [item["image"].convert("RGB") for item in batch]
        )
    if torch.is_tensor(processor_outputs):
        output_batch["pixel_values"] = processor_outputs
    else:
        output_batch["pixel_values"] = torch.FloatTensor(
            np.array(processor_outputs["pixel_values"])
        )

    # text tokenize
    if isinstance(tokenizer, MPNetTokenizerFast):
        output_batch.update(tokenize_batch(tokenizer, batch))
    elif isinstance(tokenizer, BertTokenizerFast):
        output_batch.update(
            tokenize_batch(
                tokenizer, batch, truncation=True, padding="max_length", max_length=97
            )
        )

    return output_batch


# text tokenize function
def tokenize_batch(tokenizer, batch, truncation=True, padding=True, max_length=None):
    outputs = {}

    if batch[0].get("key_phrases"):
        outputs["encoded_random_key_phrases"] = tokenizer(
            [random.choice(i["key_phrases"]) for i in batch],
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )

        outputs["encoded_key_phrases"] = [
            tokenizer(
                i["key_phrases"],
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors="pt",
            )
            for i in batch
        ]

    return outputs

