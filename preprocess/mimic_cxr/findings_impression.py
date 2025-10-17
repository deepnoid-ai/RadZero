import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dicom_preprocess import make_folder_path_dict
from tqdm import tqdm

from common.utils import save_json

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mimic_cxr_sectioned_path",
    default="/data/advanced_tech/datasets/MIMIC-CXR/preprocess/sections/mimic_cxr_sectioned.csv",
)
parser.add_argument(
    "--mimic_cxr_split_path",
    default="/data/advanced_tech/datasets/MIMIC-CXR/original_data_info/mimic-cxr-2.0.0-split.csv",
)
parser.add_argument(
    "--mimic_cxr_metadata_path",
    default="/data/advanced_tech/datasets/MIMIC-CXR/original_data_info/mimic-cxr-2.0.0-metadata.csv",
)
parser.add_argument(
    "--save_dir",
    default="/data/advanced_tech/datasets/MIMIC-CXR/preprocess/v12.0",
)


def full_report(args, folder_path_dict):
    args = parser.parse_args(args)

    mimic_cxr_sectioned_path = Path(args.mimic_cxr_sectioned_path)
    mimic_cxr_split_path = Path(args.mimic_cxr_split_path)
    mimic_cxr_metadata_path = Path(args.mimic_cxr_metadata_path)
    save_dir = Path(args.save_dir)

    os.makedirs(save_dir, exist_ok=True)

    sectioned = pd.read_csv(mimic_cxr_sectioned_path)
    split = pd.read_csv(mimic_cxr_split_path)
    metadata = pd.read_csv(mimic_cxr_metadata_path)

    # study to split
    study_split = {}
    for i in tqdm(range(len(split))):
        study_split[str(split.iloc[i]["study_id"])] = split.iloc[i]["split"]

    # sectioned findings
    sectioned_findings = {}
    for i in tqdm(range(len(sectioned))):
        study_id = sectioned.iloc[i]["study"]
        findings = sectioned.iloc[i]["findings"]
        impression = sectioned.iloc[i]["impression"]
        sectioned_findings[study_id.replace("s", "")] = {
            "findings": findings,
            "impression": impression,
        }

    # metadata to study
    study_dict = defaultdict(dict)
    for i in tqdm(range(len(metadata))):
        row = metadata.iloc[i]

        study_id = str(row["study_id"])
        if study_id not in study_dict:

            study_dict[study_id]["study_id"] = study_id
            study_dict[study_id]["subject_id"] = str(row["subject_id"])
            study_dict[study_id]["dicom_id"] = []
            study_dict[study_id]["view_position"] = []
            study_dict[study_id]["split"] = []

        study_dict[study_id]["dicom_id"].append(row["dicom_id"] + ".jpg")
        study_dict[study_id]["view_position"].append(row["ViewPosition"])
        study_dict[study_id]["split"].append(study_split[str(study_id)])

        if study_id in sectioned_findings:
            report_dict = sectioned_findings[study_id]
            findings = report_dict["findings"]
            impression = report_dict["impression"]
            if type(findings) is str and findings:
                study_dict[study_id]["findings"] = findings
            else:
                study_dict[study_id]["findings"] = ""
            if type(impression) is str and impression:
                study_dict[study_id]["impression"] = impression
            else:
                study_dict[study_id]["impression"] = ""
        else:
            study_dict[study_id]["findings"] = ""
            study_dict[study_id]["impression"] = ""

    # check split
    for study_id, study in study_dict.items():
        split_set = set(study["split"])
        assert len(split_set) == 1
        study["split"] = split_set.pop()

    dicom_level_data = defaultdict(list)
    for study_id, study in tqdm(study_dict.items()):
        split = study["split"]

        for dicom_id, view in zip(study["dicom_id"], study["view_position"]):
            dicom_level = {}
            dicom_level["study_id"] = study_id
            dicom_level["subject_id"] = study["subject_id"]
            dicom_level["dicom_id"] = dicom_id
            dicom_level["view_position"] = view
            dicom_level["split"] = study["split"]
            dicom_level["findings"] = study["findings"]
            dicom_level["impression"] = study["impression"]

            # dicom_path
            assert dicom_id in folder_path_dict
            dicom_level["dicom_path"] = folder_path_dict[dicom_id]

            dicom_level_data[split].append(dicom_level)

    for split in dicom_level_data.keys():
        save_json(dicom_level_data[split], os.path.join(save_dir, split + ".json"))

        print(f"{split} : {len(dicom_level_data[split])}")


if __name__ == "__main__":

    data_root = "/data/advanced_tech/datasets"
    folder_path_dict = make_folder_path_dict(data_root)

    full_report(sys.argv[1:], folder_path_dict)
