import os
import shutil
from datetime import datetime
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common.utils import load_json
from exp.cxr_pt.inference.segmentation_utils import interpolate_similarity_scores
from exp.cxr_pt.inference.utils import load_pretrained_model
from exp.cxr_pt.inference.visualization.qualitative_assessment_utils import (
    visualize_disease_segmap,
)


def extract_similarity_map(image_path, text, model, image_processor, tokenizer):
    image = Image.open(image_path)

    (width, height) = image.size

    image_size = (height, width)

    image_processor_outputs = image_processor(image)

    processed_image = torch.FloatTensor(
        np.array(image_processor_outputs["pixel_values"])
    ).to(model.device)

    tokenized_text = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    output = model.compute_logits(processed_image, [tokenized_text])
    similarity_scores = output["similarity_scores"]
    similarity_scores = similarity_scores.view(-1)

    similarity_scores = interpolate_similarity_scores(
        similarity_scores, image_size, image_processor
    )

    similarity_scores = similarity_scores.squeeze(0)

    return similarity_scores


def process_and_disease_visualize_segmentation(
    image_path,
    text_list,
    model,
    image_processor,
    tokenizer,
    dpi,
    overlay_alpha,
    bbox=None,
    colors=None,
    save_dir=None,
    duplication=False,
):

    probability_map_list = []
    for text in text_list:
        similarity_map = extract_similarity_map(
            image_path, text, model, image_processor, tokenizer
        )
        similarity_map = torch.sigmoid(similarity_map)

        similarity_map[similarity_map < threshold] = -1
        probability_map_list.append(similarity_map)

    probability_map_list.append(torch.zeros_like(similarity_map))
    text_list.append("background")

    probability_map = torch.stack(probability_map_list, dim=0)
    visualize_disease_segmap(
        image_path,
        probability_map,
        class_names=text_list,
        dpi=dpi,
        overlay_alpha=overlay_alpha,
        bbox=bbox,
        colors=colors,
        save_dir=save_dir,
        duplication=duplication,
    )


def input_fileter_gt_data(gt_json):
    gt_dict = {}
    for gt in gt_json:
        image_id = gt["file_name"]
        syms = gt["syms"]
        boxes = gt["boxes"]

        unique_syms = list(set(syms))

        if len(unique_syms) != len(syms) or len(unique_syms) < 1:
            continue
        else:
            syms = syms

        gt_dict[image_id] = (syms, boxes)

    return gt_dict


def input_gt_data(gt_json):
    gt_dict = {}
    for gt in gt_json:
        image_id = gt["file_name"]
        syms = gt["syms"]
        boxes = gt["boxes"]
        if len(syms) < 2:
            continue

        gt_dict[image_id] = (syms, boxes)

    return gt_dict


if __name__ == "__main__":
    data_root = "/data/advanced_tech/datasets"
    experiment_root = "/data/advanced_tech/experiments"

    dpi = 100
    overlay_alpha = 0.8

    colors = [
        "#33FF99",  # 민트 (밝은 그린-블루 계열)
        "#FF4444",  # 레드
        "#66FFFF",  # 아쿠아 (밝은 시안)
        "#FFCC33",  # 골드 (밝은 옐로우-오렌지)
        "#99FFCC",  # 진한 파스텔 에메랄드
        "#B266FF",  # 퍼플 (보라색 계열)
        "#FFA366",  # 오렌지 (밝은 오렌지)
        "#D98B8B",  # 진한 파스텔 브라운
        "#6699FF",  # 네이비 블루 (선명한 블루)
        "#FF77A3",  # 진한 파스텔 핑크
    ]

    finding_mapping = {
        "Atelectasis": "Atelectasis",
        "Tissue Calcification": "Calcification",
        "Pulmonary Consolidation": "Consolidation",
        "Pleural Effusion": "Effusion",
        "Pulmonary Emphysema": "Emphysema",
        "Fibrosis": "Fibrosis",
        "Bone Fracture": "Fracture",
        "Pulmonary Mass": "Mass",
        "Lung Nodule": "Nodule",
        "Pneumothorax": "Pneumothorax",
    }

    reverse_finding_mapping = {v: k for k, v in finding_mapping.items()}

    colors_dict = {
        f"There is {k}": colors[i] for i, k in enumerate(finding_mapping.keys())
    }
    colors_dict["background"] = "#000000"

    checkpoint_dir = os.path.join(experiment_root, "125_batch_256/checkpoint-17927")

    model, image_processor, tokenizer = load_pretrained_model(
        checkpoint_dir,
        # load config from latest snapshot
        # TODO: modify whean needed
        config_path=sorted(
            glob(
                os.path.join(
                    os.path.dirname(checkpoint_dir), "snapshot", "*", "config.yaml"
                )
            )
        )[-1],
    )

    gt_json_path = os.path.join(data_root, "ChestXDet10/test.json")
    gt_json = load_json(gt_json_path)

    # input_data = input_fileter_gt_data(gt_json)
    input_data = input_gt_data(gt_json)

    duplication = True
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    threshold = 0.7

    save_dir = os.path.join(
        checkpoint_dir,
        "..",
        "qualitative_assessment",
        f"{current_time}",
        f"disease_segmap_{threshold}",
    )
    os.makedirs(save_dir, exist_ok=True)

    for image_id, (syms, boxes) in tqdm(list(input_data.items())):
        image_path = os.path.join(
            data_root, "CarZero/images/ChestXDet10/test_data", image_id
        )
        shutil.copy(image_path, os.path.join(save_dir, image_id))

        syms_list = []
        for sym in syms:
            syms_list.append(f"There is {reverse_finding_mapping[sym]}")

        process_and_disease_visualize_segmentation(
            image_path,
            syms_list,
            model,
            image_processor,
            tokenizer,
            dpi,
            overlay_alpha,
            bbox=boxes,
            colors=colors_dict,
            save_dir=save_dir,
            duplication=duplication,
        )
