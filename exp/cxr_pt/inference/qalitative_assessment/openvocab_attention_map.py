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
):

    text = text_list[0]

    similarity_map = extract_similarity_map(
        image_path, text, model, image_processor, tokenizer
    )
    similarity_map = torch.sigmoid(similarity_map)

    sim_map_np = similarity_map.detach().cpu().numpy().squeeze()

    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)

    sim_map_resized = cv2.resize(sim_map_np, (img_np.shape[1], img_np.shape[0]))

    sim_map_blurred = cv2.GaussianBlur(sim_map_resized, (5, 5), sigmaX=0)

    cmap = plt.get_cmap("inferno") 
    
    sim_map_colored = cmap(sim_map_blurred)

    sim_map_colored[..., 3] = sim_map_blurred * alpha

    sim_map_img = Image.fromarray((sim_map_colored[..., :3] * 255).astype(np.uint8))

    blended_image = Image.blend(image, sim_map_img, alpha=alpha)

    # 컬러바만 따로 저장
    plt.figure(figsize=(2, 5))
    plt.imshow([[0, 1]], cmap=cmap)
    plt.gca().set_visible(False)
    cbar = plt.colorbar(orientation="vertical")
    cbar.ax.tick_params(labelsize=10)

    # 컬러바 경로
    colorbar_image_id = "colorbar_" + image_path.split("/")[-1]
    colorbar_save_path = os.path.join(save_dir, colorbar_image_id)

    plt.savefig(colorbar_save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

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

    orig_image_id = "orig_" + image_path.split("/")[-1]
    orig_save_path = os.path.join(save_dir, orig_image_id)
    image.save(orig_save_path)

    image_id = image_path.split("/")[-1]

    text_list_str = "_".join(text_list)

    image_id = text_list_str + "_" + image_id
    save_path = os.path.join(save_dir, image_id)
    blended_image.save(save_path)

    smoothed_map = cv2.GaussianBlur(sim_map_blurred, (7, 7), sigmaX=0)

    mask_img = Image.fromarray((smoothed_map * 255).astype(np.uint8))

    # if bbox is not None:
    #     mask_img = mask_img.convert("RGB")
    #     draw = ImageDraw.Draw(mask_img)
    #     if isinstance(bbox[0], list):
    #         for box in bbox:
    #             left, top, right, bottom = box
    #             draw.rectangle(
    #                 [left, top, right, bottom], outline=bbox_color, width=width
    #             )
    #     else:
    #         left, top, right, bottom = bbox
    #         draw.rectangle([left, top, right, bottom], outline=bbox_color, width=width)

    mask_image_id = "mask_" + image_id
    save_path = os.path.join(save_dir, mask_image_id)
    mask_img.save(save_path)


def process_openvocab_prompt(gt_json, save_dir, type, syms_list):
    gt_dict = {}
    for gt in gt_json:
        image_id = gt["file_name"]
        syms = gt["syms"]
        boxes = gt["boxes"]

        if type == "abnormal_data":
            if len(syms) != 1:
                continue
        elif type == "normal_data":
            if syms != []:
                continue

        gt_dict[image_id] = (syms, boxes)

    for image_id, (syms, boxes) in tqdm(gt_dict.items()):

        image_path = os.path.join(
            data_root, "CarZero/images/ChestXDet10/test_data", image_id
        )

        if type == "normal_data":
            boxes = None
            for i in range(len(syms_list)):
                boxes = boxes
                process_and_visualize_map(
                    image_path,
                    [syms_list[i]],
                    model,
                    image_processor,
                    tokenizer,
                    boxes,
                    color,
                    alpha,
                    save_dir,
                    width=width,
                )
        elif type == "abnormal_data":
            current_syms_list = []
            for sym in syms:
                sym = sym.lower()
                current_syms_list.append(f"There is {sym}")
                current_syms_list.append(f"There is no {sym}")
                current_syms_list.extend(syms_list)

            for i in range(len(current_syms_list)):

                boxes = boxes
                process_and_visualize_map(
                    image_path,
                    [current_syms_list[i]],
                    model,
                    image_processor,
                    tokenizer,
                    boxes,
                    color,
                    alpha,
                    save_dir,
                    width=width,
                )


# left right prompt
def process_left_right_prompt(position_template_list, gt_json, save_dir):
    for position in position_template_list:
        gt_dict = {}
        for gt in gt_json:
            image_id = gt["file_name"]
            syms = gt["syms"]
            boxes = gt["boxes"]

            if syms == []:
                continue
            else:
                if len(syms) != 1:
                    continue
                else:
                    gt_dict[image_id] = (syms, boxes)

        for image_id, (syms, boxes) in tqdm(gt_dict.items()):

            image_path = os.path.join(
                data_root, "CarZero/images/ChestXDet10/test_data", image_id
            )
            syms_list = []
            for sym in syms:
                if sym in position_sentence:
                    syms_list.append(position_sentence[sym][position])

            if not syms_list:
                continue

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
            )


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

    position_sentence = {
        "Atelectasis": {
            "left": "There is atelectasis in the left lung.",
            "right": "There is atelectasis in the right lung.",
            "left upper": "There is atelectasis in the left upper lung.",
            "left mid": "There is atelectasis in the left mid lung.",
            "left lower": "There is atelectasis in the left lower lung.",
            "right upper": "There is atelectasis in the right upper lung.",
            "right mid": "There is atelectasis in the right mid lung.",
            "right lower": "There is atelectasis in the right lower lung.",
        },
        "Calcification": {
            "left": "There is calcification in the left lung.",
            "right": "There is calcification in the right lung.",
            "left upper": "There is calcification in the left upper lung.",
            "left mid": "There is calcification in the left mid lung.",
            "left lower": "There is calcification in the left lower lung.",
            "right upper": "There is calcification in the right upper lung.",
            "right mid": "There is calcification in the right mid lung.",
            "right lower": "There is calcification in the right lower lung.",
        },
        "Consolidation": {
            "left": "There is consolidation in the left lung.",
            "right": "There is consolidation in the right lung.",
            "left upper": "There is consolidation in the left upper lung.",
            "left mid": "There is consolidation in the left mid lung.",
            "left lower": "There is consolidation in the left lower lung.",
            "right upper": "There is consolidation in the right upper lung.",
            "right mid": "There is consolidation in the right mid lung.",
            "right lower": "There is consolidation in the right lower lung.",
        },
        "Effusion": {
            "left": "There is effusion in the left lung.",
            "right": "There is effusion in the right lung.",
            "left upper": "There is effusion in the left upper lung.",
            "left mid": "There is effusion in the left mid lung.",
            "left lower": "There is effusion in the left lower lung.",
            "right upper": "There is effusion in the right upper lung.",
            "right mid": "There is effusion in the right mid lung.",
            "right lower": "There is effusion in the right lower lung.",
        },
        "Fibrosis": {
            "left": "There is fibrosis in the left lung.",
            "right": "There is fibrosis in the right lung.",
            "left upper": "There is fibrosis in the left upper lung.",
            "left mid": "There is fibrosis in the left mid lung.",
            "left lower": "There is fibrosis in the left lower lung.",
            "right upper": "There is fibrosis in the right upper lung.",
            "right mid": "There is fibrosis in the right mid lung.",
            "right lower": "There is fibrosis in the right lower lung.",
        },
        "Mass": {
            "left": "There is a mass in the left lung.",
            "right": "There is a mass in the right lung.",
            "left upper": "There is a mass in the left upper lung.",
            "left mid": "There is a mass in the left mid lung.",
            "left lower": "There is a mass in the left lower lung.",
            "right upper": "There is a mass in the right upper lung.",
            "right mid": "There is a mass in the right mid lung.",
            "right lower": "There is a mass in the right lower lung.",
        },
        "Nodule": {
            "left": "There is a nodule in the left lung.",
            "right": "There is a nodule in the right lung.",
            "left upper": "There is a nodule in the left upper lung.",
            "left mid": "There is a nodule in the left mid lung.",
            "left lower": "There is a nodule in the left lower lung.",
            "right upper": "There is a nodule in the right upper lung.",
            "right mid": "There is a nodule in the right mid lung.",
            "right lower": "There is a nodule in the right lower lung.",
        },
        "Pneumothorax": {
            "left": "There is pneumothorax in the left lung.",
            "right": "There is pneumothorax in the right lung.",
            "left upper": "There is pneumothorax in the left upper lung.",
            "left mid": "There is pneumothorax in the left mid lung.",
            "left lower": "There is pneumothorax in the left lower lung.",
            "right upper": "There is pneumothorax in the right upper lung.",
            "right mid": "There is pneumothorax in the right mid lung.",
            "right lower": "There is pneumothorax in the right lower lung.",
        },
    }

    position_template_list = [
        "left",
        "right",
        "left upper",
        "left mid",
        "left lower",
        "right upper",
        "right mid",
        "right lower",
    ]

    gt_json_path = os.path.join(data_root, "CarZero/preprocess/ChestXDet10/test.json")
    gt_json = load_json(gt_json_path)

    abnormal_type = "abnormal_data"  
    type_abnormal_syms_list = [
        "This is a normal case.",
        "The lungs are clear.",
        "No abnormalities detected.",
        "There are no signs of disease.",
    ]

    normal_type = "normal_data"
    type_normal_syms_list = [
        "This is a normal case.",
        "The lungs are clear.",
        "No abnormalities detected.",
        "There are no signs of disease.",
        "There is no consolidation",
        "There is no atelectasis",
        "There is a nodule",
        "There is consolidation",
        "There is no pneumorhorax",
        "There is pneumothorax",
        "There is no atelectasis",
        "There is atelectasis",
    ]

    openvocab_abnormal_save_dir = (
        f"125model/openvocab/{abnormal_type}"
    )
    openvocab_normal_save_dir = (
        f"125model/openvocab/{normal_type}"
    )

    os.makedirs(openvocab_abnormal_save_dir, exist_ok=True)
    os.makedirs(openvocab_normal_save_dir, exist_ok=True)

    process_openvocab_prompt(
        gt_json,
        save_dir=openvocab_abnormal_save_dir,
        type=abnormal_type,
        syms_list=type_abnormal_syms_list,
    )

    process_openvocab_prompt(
        gt_json,
        save_dir=openvocab_normal_save_dir,
        type=normal_type,
        syms_list=type_normal_syms_list,
    )

    """
    """
    # left right
    left_right_save_dir = (
        "125model/left_right_attention"
    )
    os.makedirs(left_right_save_dir, exist_ok=True)

    process_left_right_prompt(
        position_template_list, gt_json, save_dir=left_right_save_dir
    )
