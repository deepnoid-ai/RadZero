import csv
import os
from collections import defaultdict

import torch

from common.utils import save_json


def csv_file_read(csv_path):
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
    return data


def multi_task_format(ms_cxr_dataset, medrpg_splits, save_output_dir):
    os.makedirs(save_output_dir, exist_ok=True)

    split_set = defaultdict(set)
    for split, data in medrpg_splits.items():
        for i in data:
            split_set[split].add((os.path.basename(i[3]).replace(".jpg", ""), i[-1]))

    data_list = defaultdict(list)
    for split in split_set.keys():
        for data in ms_cxr_dataset:
            if (data["dicom_id"], data["label_text"]) in split_set[split]:
                tmp_dict = {}
                image_path = os.path.join(
                    "MIMIC-CXR", "images", data["dicom_id"] + ".jpg"
                )
                tmp_dict["image"] = image_path
                tmp_dict["det"] = [
                    {
                        "name": data["label_text"],
                        "label": [
                            [
                                eval(data["x"]),
                                eval(data["y"]),
                                eval(data["x"]) + eval(data["w"]),
                                eval(data["y"]) + eval(data["h"]),
                            ]
                        ],
                    }
                ]
                data_list[split].append(tmp_dict)

    # save outputs
    for split in ["train", "val", "test"]:
        save_json(data_list[split], os.path.join(save_output_dir, f"{split}.json"))
        print(f"dataset length : {split}, {len(data_list[split])}")


if __name__ == "__main__":
    data_root = "datasets/"
    version = "v2.0"
    save_output_dir = os.path.join(data_root, "MS-CXR-0.1", "preprocess", version)
    csv_path = os.path.join(data_root, "MS-CXR-0.1/MS_CXR_Local_Alignment_v1.0.0.csv")

    ms_cxr_dataset = csv_file_read(csv_path)

    splits = ["train", "val", "test"]
    data_dir = os.path.join(data_root, "MS-CXR-0.1/MedRPG/data/MS_CXR")

    # medrpg split from https://github.com/eraserNut/MedRPG/tree/master/data/MS_CXR

    medrpg_splits = {}
    for split in splits:
        data_path = os.path.join(data_dir, f"MS_CXR_{split}.pth")
        medrpg_splits[split] = torch.load(data_path)

    multi_task_format(ms_cxr_dataset, medrpg_splits, data_root, save_output_dir)
