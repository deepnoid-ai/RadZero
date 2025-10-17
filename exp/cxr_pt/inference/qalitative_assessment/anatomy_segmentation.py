import os
import shutil
from datetime import datetime
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from exp.cxr_pt.inference.qalitative_assessment.qualitative_assessment_utils import (
    visualize_segmap,
)
from exp.cxr_pt.inference.segmentation_utils import interpolate_similarity_scores
from exp.cxr_pt.inference.utils import load_pretrained_model


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


def process_and_visualize_segmentation(
    image_path,
    text_list,
    model,
    image_processor,
    tokenizer,
    k,
    dpi,
    overlay_alpha,
    bbox=None,
    colors=None,
    save_dir=None,
):
    """
    This function calculates similarity maps for the given image and text list,
    filters the top k percent, and performs visualization.

    Parameters:
        image_path (str): Path to the image file to be processed.
        text_list (list of str): List of texts (classes).
        model: Model to be used for extracting similarity maps.
        image_processor: Image preprocessing object.
        tokenizer: Text tokenizer.
        k (float): Value used to filter the top k percent.
    """
    probability_map_list = []
    for text in text_list:
        similarity_map = extract_similarity_map(
            image_path, text, model, image_processor, tokenizer
        )
        similarity_map = torch.sigmoid(similarity_map)

        similarity_map[similarity_map < threshold] = -1
        probability_map_list.append(similarity_map)

    # Add background class: add a zero map of the same size
    probability_map_list.append(torch.zeros_like(similarity_map))
    text_list.append("background")

    # Stack probability maps and visualize
    probability_map = torch.stack(probability_map_list, dim=0)
    visualize_segmap(
        image_path,
        probability_map,
        class_names=text_list,
        dpi=dpi,
        overlay_alpha=overlay_alpha,
        bbox=bbox,
        colors=colors,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    data_root = "/data/advanced_tech/datasets"
    experiment_root = "/data/advanced_tech/experiments"

    view_type = "frontal"  # lateral
    dpi = 100
    overlay_alpha = 0.8

    checkpoint_dir = os.path.join(
        experiment_root, "125_batch_256/checkpoint-17927"
    )

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

    dir_path = os.path.join(
        data_root,
        "iu_xray/iu_xray_kaggle/images/images_normalized",
    )

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    cmap = None
    threshold = 0.4

    save_dir = os.path.join(
        checkpoint_dir,
        "..",
        "qualitative_assessment",
        f"{current_time}",
        f"{threshold}_{view_type}",
    )
    os.makedirs(save_dir, exist_ok=True)

    image_list = []

    if view_type == "frontal":
        cmap = "tab20"
        frontal_position_list = [
            "right upper lung zone",
            "left upper lung zone",
            "right mid lung zone",
            "left mid lung zone",
            "right lower lung zone",
            "left lower lung zone",
            "right costophrenic angle",
            "left costophrenic angle",
            "right hemidiaphragm",
            "left hemidiaphragm",
            "right clavicle",
            "left clavicle",
            "cardiac silhouette",
            "spine",
            "abdomen",
        ]

        for file in os.listdir(dir_path):
            if file.endswith(".png"):
                if file.split("-")[-1][0] == "1":
                    image_list.append(os.path.join(dir_path, file))

    elif view_type == "lateral":
        cmap = "tab10"

        lateral_position_list = [
            "cardiac silhouette",
            "trachea",
            "upper lobe",
            "middle lobe",
            "lower lobe",
            "abdomen",
            "hemidiaphragm",
            "spine",
        ]

        for file in os.listdir(dir_path):
            if file.endswith(".png"):
                if file.split("-")[-1][0] == "2":
                    image_list.append(os.path.join(dir_path, file))

    for image_path in tqdm(image_list):
        if view_type == "frontal":
            position_list = frontal_position_list
        elif view_type == "lateral":
            position_list = lateral_position_list

        position_list = [f"There is {pos}" for pos in position_list]
        shutil.copy(
            image_path, os.path.join(save_dir, os.path.basename(image_path))
        )
        process_and_visualize_segmentation(
            image_path,
            position_list.copy(),
            model,
            image_processor,
            tokenizer,
            dpi,
            overlay_alpha,
            bbox=None,
            colors=cmap,
            save_dir=save_dir,
        )
