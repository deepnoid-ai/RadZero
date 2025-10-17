import os
import random
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from tqdm import tqdm

from common.utils import load_json
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
):

    probability_map_list = []
    for text in text_list:
        similarity_map = extract_similarity_map(
            image_path, text, model, image_processor, tokenizer
        )
        similarity_map = torch.sigmoid(similarity_map)

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
    )


def process_and_visualize_map(
    image_path,
    text_list,
    model,
    image_processor,
    tokenizer,
    bbox=None,
    bbox_color="red",
    alpha=None,
    save_dir=None,
    width=None,
    image_key=None,
):
    text = text_list[0]
    text_list_str = "_".join(text_list)
    image_id = image_key
    similarity_map = extract_similarity_map(
        image_path, text, model, image_processor, tokenizer
    )
    similarity_map = torch.sigmoid(similarity_map)

    # Convert similarity map to numpy array
    sim_map_np = similarity_map.detach().cpu().numpy().squeeze()

    # Open the original image
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)

    sim_map_resized = cv2.resize(sim_map_np, (img_np.shape[1], img_np.shape[0]))

    sim_map_blurred = cv2.GaussianBlur(sim_map_resized, (5, 5), sigmaX=0)

    # cmap = plt.get_cmap("jet")
    cmap = plt.get_cmap("inferno")
    sim_map_colored = cmap(sim_map_blurred)

    sim_map_colored[..., 3] = sim_map_blurred * alpha

    sim_map_img = Image.fromarray((sim_map_colored[..., :3] * 255).astype(np.uint8))

    blended_image = Image.blend(image, sim_map_img, alpha=alpha)

    nobox_image_id = "nobox_" + text_list_str + "_" + image_id
    nobox_save_path = os.path.join(save_dir, nobox_image_id)
    blended_image.save(nobox_save_path)

    if bbox is not None:
        draw = ImageDraw.Draw(blended_image)

        if isinstance(bbox[0], list):

            for box in bbox:
                left, top, right, bottom = box
                draw.rectangle(
                    [left, top, right, bottom],
                    outline=bbox_color,
                    width=width,
                )
        else:
            left, top, right, bottom = bbox
            draw.rectangle([left, top, right, bottom], outline=bbox_color, width=width)

    # Draw box on the original image
    if bbox is not None:
        draw_orig = ImageDraw.Draw(image)

        if isinstance(bbox[0], list):

            for box in bbox:
                left, top, right, bottom = box
                draw_orig.rectangle(
                    [left, top, right, bottom],
                    outline=bbox_color,
                    width=width,
                )
        else:
            left, top, right, bottom = bbox
            draw_orig.rectangle(
                [left, top, right, bottom], outline=bbox_color, width=width
            )

    # Save the original image with the box drawn
    orig_image_id = "orig_" + image_key
    orig_save_path = os.path.join(save_dir, orig_image_id)
    image.save(orig_save_path)

    save_image_id = text_list_str + "_" + image_key
    save_path = os.path.join(save_dir, save_image_id)
    blended_image.save(save_path)


if __name__ == "__main__":
    data_root = "/data/advanced_tech/datasets"
    experiment_root = "/data/advanced_tech/experiments"

    color = "red"
    alpha = 0.2
    width = 3

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

    gt_json_path = os.path.join(data_root, "CarZero/preprocess/ChestXDet10/test.json")
    gt_json = load_json(gt_json_path)

    save_dir = f"125model/disease_attention_map_final/{type}_"

    os.makedirs(save_dir, exist_ok=True)

    gt_dict = {}
    for gt in gt_json:
        image_id = gt["file_name"]
        syms = gt["syms"]
        boxes = gt["boxes"]

    #     if syms == []:
    #         continue

    #     if type == "duplication_false":
    #         # Create a list with duplicates removed
    #         unique_syms = list(set(syms))

    #         # Extract only those with no duplicates and at least one unique lesion
    #         if len(unique_syms) != len(syms) or len(unique_syms) < 1:
    #             continue
    #         else:
    #             syms = syms
    #             boxes = boxes

    #     elif type == "only_one":
    #         if len(syms) != 1:
    #             continue
    #         else:
    #             syms = syms
    #             boxes = boxes

    #     else:
    #         raise NotImplementedError
    #         # filtered_syms_boxes = [
    #         #     (sym, box) for sym, box in zip(syms, boxes) if syms.count(sym) > 1
    #         # ]
    #         # syms, boxes = zip(*filtered_syms_boxes) if filtered_syms_boxes else ([], [])

    #         # if syms == []:
    #         #     continue
    #         # else:
    #         #     syms = [syms[0]]
    #         #     boxes = [boxes[0]]

    #     gt_dict[image_id] = (syms, boxes)

    # for image_id, (syms, boxes) in tqdm(gt_dict.items()):

    #     image_path = os.path.join(
    #         data_root, "CarZero/images/ChestXDet10/test_data", image_id
    #     )
    #     syms_list = []
    #     for sym in syms:
    #         syms_list.append(f"There is {sym}")

    #     process_and_visualize_map(
    #         image_path,
    #         syms_list,
    #         model,
    #         image_processor,
    #         tokenizer,
    #         boxes,
    #         color,
    #         alpha,
    #         save_dir,
    #         width=width,
    #         type=type,
    #     )

    # Save the original file name when creating gt_dict
    gt_dict = {}

    for entry in gt_json:
        file_name = entry["file_name"]
        syms = entry["syms"]
        boxes = entry["boxes"]

        # Skip if empty
        if not syms:
            continue

        # Merge lesions within the same image (grouping)
        lesion_groups = {}  # key: lesion name, value: list of corresponding boxes
        order = []  # list to maintain order

        for sym, box in zip(syms, boxes):
            if sym in lesion_groups:
                lesion_groups[sym].append(box)
            else:
                lesion_groups[sym] = [box]
                order.append(sym)

        # Save to gt_dict by group (add suffix if there are multiple lesions)
        count = 0
        for sym in order:
            # The key of gt_dict is temporarily generated, but the original file name is saved separately.
            key = file_name if file_name not in gt_dict else f"{count}_{file_name}"
            while key in gt_dict:
                count += 1
                key = f"{count}_{file_name}"
            # Save as a tuple (original file name, lesion name, list of corresponding boxes)
            gt_dict[key] = (file_name, sym, lesion_groups[sym])

    for image_key, (orig_file, sym, boxes) in tqdm(gt_dict.items()):

        # Use a for loop to get one sym per image
        image_path = os.path.join(
            data_root, "CarZero/images/ChestXDet10/test_data", orig_file
        )
        syms_list = [f"There is {sym}"]

        process_and_visualize_map(
            image_path,
            syms_list,
            model,
            image_processor,
            tokenizer,
            boxes,
            color,
            alpha,
            save_dir,
            width=width,
            image_key=image_key,
        )
