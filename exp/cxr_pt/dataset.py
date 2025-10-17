import os
import random
from functools import partial

import cv2
import numpy as np
import PIL
import torch
from datasets import DatasetDict, Image
from nltk.tokenize import sent_tokenize
from torchvision.transforms import Compose
from tqdm import tqdm
from transformers import BertTokenizerFast, MPNetTokenizerFast

from common.dataset import WithMissingValueDataset
from common.trainer import logger
from common.utils import load_json
from exp.augmentation import augmemtation_transforms
from exp.cxr_pt.model.processing import M3AEImageProcessor


def input_json_file_load(json_path, data_root, train_flag, **kwargs):
    logger.info(f"load dataset: {json_path}")
    input_json = load_json(os.path.join(data_root, json_path))

    use_frontal_view_only = kwargs.get("use_frontal_view_only", False)
    use_radgrpah_key_phrase = kwargs.get("use_radgrpah_key_phrase", False)
    use_dicom_path = kwargs.get("use_dicom_path", False)
    dataset_name = json_path.split("/")[0]

    if "carzero" in json_path:
        dataset_name = "MIMIC-CXR-CARZERO"

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
            if use_dicom_path:
                if data.get("original_dicom_id"):
                    image_path = os.path.join(data_root, data["original_dicom_id"])
                elif data.get("dicom_path"):
                    image_path = os.path.join(data_root, data["dicom_path"])
                else:
                    raise NotImplementedError
            else:
                image_path = os.path.join(
                    data_root, "MIMIC-CXR", "images", data["dicom_id"]
                )
            entry["image"] = image_path

            # if data.get("findings"):
            #     entry["sentences"] = sent_tokenize(data["findings"])
            #     entry["findings"] = data["findings"]

            # if data.get("findings_clean"):
            #     # entry["sentences_clean"] = sent_tokenize(data["findings_clean"])
            #     entry["findings_clean"] = data["findings_clean"]
            # else:
            #     continue

            if data.get("key_phrases"):
                if use_radgrpah_key_phrase:
                    entry["key_phrases"] = [
                        i for i in data["radgraph_key_phrase"] if i.strip()
                    ]
                else:
                    entry["key_phrases"] = [i for i in data["key_phrases"] if i.strip()]
            else:
                continue

            if data.get("negative_existence"):
                entry["negative_existence"] = [
                    i for i in data["negative_existence"] if i.strip()
                ]
            else:
                entry["negative_existence"] = []

            if data.get("negative_position"):
                entry["negative_position"] = [
                    i for i in data["negative_position"] if i.strip()
                ]
            else:
                entry["negative_position"] = []

            if data.get("negative_finding"):
                entry["negative_finding"] = [
                    i for i in data["negative_finding"] if i.strip()
                ]
            else:
                entry["negative_finding"] = []

            if kwargs.get("use_negative_phrases"):
                if not (
                    entry["negative_existence"]
                    + entry["negative_position"]
                    + entry["negative_finding"]
                ):
                    continue

            entry["train"] = train_flag

            data_list.append(entry)

        elif dataset_name == "MIMIC-CXR-CARZERO":
            entry = {}
            if use_dicom_path:
                image_path = os.path.join(data_root, data["original_dicom_id"])
            else:
                image_path = os.path.join(
                    data_root, "MIMIC-CXR", "images", data["dicom_id"]
                )
            entry["image"] = image_path

            entry["key_phrases"] = [i for i in data["key_phrases"] if i.strip()]

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

    if cfg["augmentation"]["apply"]:
        logger.info(f"augmentation: {cfg['augmentation']}")
        transforms = augmemtation_transforms(cfg["augmentation"])
        _transform_fn = partial(transform_fn, transforms=transforms)
        dataset.set_transform(_transform_fn)

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

    # # add "image_path" in input_json_file_load if necessary
    # output_batch["image_path"] = [item["image_path"] for item in batch]

    return output_batch


# text tokenize function
def tokenize_batch(tokenizer, batch, truncation=True, padding=True, max_length=None):
    outputs = {}

    if batch[0].get("findings_clean"):
        outputs["encoded_findings"] = tokenizer(
            [i["findings_clean"] for i in batch],
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )

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

    if batch[0].get("negative_existence"):
        outputs["encoded_negative_phrases"] = [
            tokenizer(
                i["negative_existence"]
                + i["negative_position"]
                + i["negative_finding"],
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors="pt",
            )
            for i in batch
        ]

    return outputs


if __name__ == "__main__":
    import yaml

    cfg_path = "exp/cxr_pt/pt/config.yaml"

    with open(cfg_path, "r") as file:
        cfg = yaml.safe_load(file)

    dataset = load_datasets(cfg["dataset"], train=True, inference=False)

    batch_size = 4
    processed_batches = []
    for split, dataset_split in dataset.items():
        print(f"split: {split}")
        for i in tqdm(range(0, len(dataset_split), batch_size)):
            batch = dataset_split[i : i + batch_size]
            # processed_batch = transform(batch, cfg["dataset"])

            # processed_batches.append(processed_batch)
