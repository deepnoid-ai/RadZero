import os
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BitImageProcessor, BlipImageProcessor

from common.utils import load_json
from exp.cxr_pt.inference.dataset import InferDataset, collate_fn
from exp.cxr_pt.inference.utils import process_class_prompts
from exp.cxr_pt.model.processing import (
    AspectRatioBlipImageProcessor,
    M3AEImageProcessor,
)


def split_list(lst, chunk_size):
    result = []
    for i in range(0, len(lst), chunk_size):
        chunk = lst[i : i + chunk_size]
        result.append(chunk)
    return result


def get_similarity_scores(
    image_paths,
    text_batch,
    model,
    image_processor,
    batch_size,
    num_workers,
    data_root_dir,
):
    with torch.no_grad():
        dataset = InferDataset(image_processor, image_paths, data_root_dir)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=partial(
                collate_fn,
                image_processor=image_processor,
            ),
        )

        similarity_scores = []

        for images in tqdm(loader):
            images = images.to(model.device)

            outputs = model.compute_logits(
                pixel_values=images,
                encoded_key_phrases=[text_batch["encoded_key_phrases"]],
                use_negative_logits=False,
            )
            similarity_scores.append(outputs["similarity_scores"])

        similarity_scores = torch.cat(similarity_scores, dim=0)
        return similarity_scores


def chestXDet10_eval_grounding(
    model,
    image_processor,
    tokenizer,
    batch_size,
    num_workers,
    image_path,
    text_path,
    test_path,
    data_root_dir,
):
    grounding_results = defaultdict(list)

    image = pd.read_csv(image_path)
    text_prompt = load_json(text_path)

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

    finding_classes = [
        finding_mapping[text[0].replace("There is ", "")]
        for k, text in text_prompt.items()
    ]

    finding_indices = {
        finding_class: i for i, finding_class in enumerate(finding_classes)
    }

    image_list = image["Path"].tolist()
    text_batch = process_class_prompts(text_prompt, tokenizer, model)

    similarity_scores = get_similarity_scores(
        image_list,
        text_batch,
        model,
        image_processor,
        batch_size,
        num_workers,
        data_root_dir,
    )

    image_size_list = []
    for image_path in image_list:
        image = Image.open(os.path.join(data_root_dir, image_path))
        # care about the order of width and height
        (width, height) = image.size
        image_size = (height, width)
        image_size_list.append(image_size)

    bbox_labels = load_json(test_path)

    for bbox_label, similarity_score, image_size in zip(
        bbox_labels, similarity_scores, image_size_list
    ):

        syms = bbox_label["syms"]
        boxes = bbox_label["boxes"]

        image_finding_box = defaultdict(list)
        for sym, box in zip(syms, boxes):
            image_finding_box[sym].append(box)

        for finding_class, _boxes in image_finding_box.items():
            finding_index = finding_indices[finding_class]
            finding_similarity_score = similarity_score[finding_index, :]

            (x_index, y_index) = get_grounding_point(
                finding_similarity_score, image_size, image_processor
            )

            grounding_results[finding_class].append(
                is_point_in_bbox(_boxes, (x_index, y_index))
            )

    result = {}
    for finding_class, results in grounding_results.items():
        result[finding_class] = np.mean(results)

    result["mean_pointing_score"] = np.mean(list(result.values()))

    # print result with text decription
    for finding_class, pointing_score in result.items():
        print(f"{finding_class}: {pointing_score:.4f}")

    return result


def get_grounding_point(similarity_score, image_size, image_processor):
    # (height, width)
    (height, width) = image_size

    similarity_score = similarity_score.view(
        int(np.sqrt(len(similarity_score))), int(np.sqrt(len(similarity_score)))
    )
    if isinstance(image_processor, AspectRatioBlipImageProcessor):

        padded_size = max(height, width)

        similarity_score = F.interpolate(
            similarity_score.unsqueeze(0).unsqueeze(0),
            size=(padded_size, padded_size),
            mode="bilinear",
            align_corners=False,
        )

        # calculate the area where the original image is located
        pad_left = (padded_size - width) // 2
        pad_top = (padded_size - height) // 2

        # Crop only the calculated area to restore the original image size
        similarity_score = similarity_score[
            :, :, pad_top : pad_top + height, pad_left : pad_left + width
        ].contiguous()

    elif isinstance(image_processor, BlipImageProcessor):
        similarity_score = F.interpolate(
            similarity_score.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    elif isinstance(image_processor, BitImageProcessor):
        shortest = min(height, width)

        similarity_score = F.interpolate(
            similarity_score.unsqueeze(0).unsqueeze(0),
            size=(shortest, shortest),
            mode="bilinear",
            align_corners=False,
        )

        cropped_left = (width - shortest) // 2
        cropped_top = (height - shortest) // 2

        original_size_map = torch.ones(height, width) * -999
        original_size_map[
            cropped_top : cropped_top + shortest, cropped_left : cropped_left + shortest
        ] = similarity_score.view(shortest, shortest)

        similarity_score = original_size_map
    elif isinstance(image_processor, M3AEImageProcessor):

        padded_size = max(height, width)

        padded_size_map = torch.ones(padded_size, padded_size) * -999

        cropped_size = int(padded_size * 224 / 256)

        similarity_score = F.interpolate(
            similarity_score.unsqueeze(0).unsqueeze(0),
            size=(cropped_size, cropped_size),
            mode="bilinear",
            align_corners=False,
        )

        left = (padded_size - cropped_size) // 2
        top = (padded_size - cropped_size) // 2

        padded_size_map[top : top + cropped_size, left : left + cropped_size] = (
            similarity_score.view(cropped_size, cropped_size)
        )

        # calculate the area where the original image is located
        pad_left = (padded_size - width) // 2
        pad_top = (padded_size - height) // 2

        similarity_score = padded_size_map[
            pad_top : pad_top + height, pad_left : pad_left + width
        ].contiguous()

    else:
        raise NotImplementedError(
            f"Image processor {type(image_processor)} is not supported"
        )

    values, indices = similarity_score.view(-1).max(dim=0)

    h_index, w_index = torch.unravel_index(indices, (height, width))

    x_index = w_index.item()
    y_index = h_index.item()

    return (x_index, y_index)


def is_point_in_bbox(bbox_list, point):
    """
    Returns True if the point is inside any of the bounding boxes in bbox_list, otherwise False.

    Parameters:
        bbox_list (list): A list where each bbox is in the format [x_min, y_min, x_max, y_max]
        point (tuple): A point coordinate in the format (x, y)

    Returns:
        bool: True if the point is inside any of the bounding boxes, otherwise False
    """
    x, y = point
    for bbox in bbox_list:
        x_min, y_min, x_max, y_max = bbox
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
    return False


def eval_refer_grounding(
    model,
    image_processor,
    tokenizer,
    data,
    data_root_dir,
):
    prediction_results = []
    for d in tqdm(data):
        image_path = d["image"]

        image = Image.open(os.path.join(data_root_dir, image_path))
        text = d["det"][0]["name"]
        label = d["det"][0]["label"][0]
        # (width, height)
        (width, height) = image.size
        image_size = (height, width)

        pixel_values = image_processor(image)
        pixel_values = torch.FloatTensor(np.array(pixel_values["pixel_values"])).to(
            model.device
        )

        encoded_key_phrases = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.compute_logits(pixel_values, [encoded_key_phrases])
        similarity_scores = outputs["similarity_scores"]

        (x_index, y_index) = get_grounding_point(
            similarity_scores.view(-1), image_size, image_processor
        )

        prediction_results.append(is_point_in_bbox([label], (x_index, y_index)))

    accuracy = sum(prediction_results) / len(prediction_results)

    print("Accuracy of MS-CXR: {}".format(accuracy))

    return accuracy
