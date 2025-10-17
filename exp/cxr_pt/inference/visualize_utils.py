import math
import os
import random
from collections import defaultdict
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.utils import save_json
from exp.cxr_pt.inference.dataset import InferDataset, collate_fn
from exp.cxr_pt.inference.segmentation_utils import (
    interpolate_similarity_scores,
    read_from_dicom,
    rle2mask,
)
from exp.cxr_pt.inference.utils import process_class_prompts


def get_attention_weights(
    model,
    image_processor,
    tokenizer,
    batch_size,
    num_workers,
    data_root_dir,
    image_paths,
    text_prompts,
):
    text_batch = process_class_prompts(
        {str(i): [text] for i, text in enumerate(text_prompts)}, tokenizer, model
    )["encoded_key_phrases"]

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

        attention_weights = defaultdict(list)

        for image in tqdm(loader):
            image = image.to(model.device)
            attn_weight = model.compute_logits(image, [text_batch])["t2i_attn_weights"]
            for layer, weight in enumerate(attn_weight):
                flat_weight = weight[:, :, 1:]
                sqrt_d = int(math.sqrt(flat_weight.size(-1)))
                weight_2d = flat_weight.view(
                    weight.size(0),
                    weight.size(1),
                    sqrt_d,
                    sqrt_d,
                )
                attention_weights[layer].append(weight_2d)

        return {
            layer: torch.cat(weights, dim=0)
            for layer, weights in attention_weights.items()
        }


def save_attention_map(
    attention_weights,
    image_paths,
    data_dir,
    save_dir,
    image_processor,
    bboxes=None,
    masks=None,
):
    for layer, attn_weights in attention_weights.items():
        for i, (img_file, attn_weight) in enumerate(zip(image_paths, attn_weights)):

            # load image file
            img_path = os.path.join(data_dir, img_file)
            if img_path.endswith("dcm"):
                orig_img = read_from_dicom(img_path)
            else:
                orig_img = Image.open(img_path)
            (width, height) = orig_img.size

            # save original image
            img_filename = os.path.basename(img_path)
            filename, ext = os.path.splitext(img_filename)
            if ext == ".dcm":
                ext = ".png"
            orig_img.save(os.path.join(save_dir, f"{filename}{ext}"))

            # save attention map for each prompt
            for prompt_id in range(attn_weight.size(0)):
                new_file = os.path.join(
                    save_dir, f"{filename}_layer{layer}_prompt{prompt_id}{ext}"
                )

                _attn_weight = attn_weight[prompt_id : prompt_id + 1]
                _attn_weight = _attn_weight.view(_attn_weight.size(0), -1)

                # interpolate attention map
                interpolated = interpolate_similarity_scores(
                    _attn_weight,
                    (height, width),
                    image_processor,
                )

                # normalize attention map
                normalized = 255 * F.sigmoid(interpolated)
                image_array = normalized[0].cpu().numpy().astype(np.uint8)

                # convert to image
                img = Image.fromarray(image_array).convert("RGB")

                # draw bbox
                if bboxes is not None:
                    bbox = bboxes[prompt_id]
                    draw = ImageDraw.Draw(img)
                    for b in bbox:
                        if isinstance(b, np.ndarray):
                            assert b.shape == (
                                4,
                            ), f"Expected bbox of shape (4,), but got {b.shape}"
                            b = b.tolist()
                        if sum(b) > 0:
                            draw.rectangle(b, outline="red", width=3)

                # draw mask
                if masks is not None:
                    mask = (255 * masks[prompt_id]).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    img_array = np.array(img)
                    for contour in contours:
                        cv2.drawContours(
                            img_array, [contour], -1, (255, 0, 0), 2
                        )  # red line (255, 0, 0)
                    img = Image.fromarray(img_array)

                img.save(new_file)


def visualize_chestXDet10(
    inference,
    data_list,
    image_list,
    model,
    image_processor,
    tokenizer,
    data_root_dir,
    save_root_dir,
    prompt_templates,
):

    save_data = list()
    for data, image in zip(data_list, image_list):
        assert os.path.basename(image) == data["file_name"]
        image_paths = [os.path.join(data_root_dir, image)]
        syms = data["syms"]
        boxes = data["boxes"]

        text_prompts = list()
        bbox_dict = defaultdict(list)
        if len(syms) == 0:
            syms = random.sample(
                ["Effusion", "Consolidation", "Fibrosis", "Emphysema"], 1
            )
            bboxes = None
        else:
            bboxes = list()
            for sym, bbox in zip(syms, boxes):
                bbox_dict[sym].append(bbox)
        for sym in sorted(list(set(syms))):
            text_prompts.extend([t % sym for t in prompt_templates])
            if bboxes is not None:
                bboxes.extend(len(prompt_templates) * [bbox_dict[sym]])

        inference.visualization(
            model,
            image_processor,
            tokenizer,
            save_root_dir,
            text_prompts,
            image_paths,
            bboxes=bboxes,
        )

        save_data.append(
            {
                "image": image,
                "prompts": text_prompts,
            }
        )

    save_json(save_data, os.path.join(save_root_dir, "prompt.json"))


def visualize_SIIM(
    inference,
    siim_df,
    model,
    image_processor,
    tokenizer,
    data_root_dir,
    save_root_dir,
    prompt_templates,
):
    save_data = list()
    for _, row in siim_df.iterrows():
        image_paths = [os.path.join(data_root_dir, row["ours_path"])]
        encoded_labels = row[" EncodedPixels"]
        mask = np.zeros([1024, 1024])
        if encoded_labels != " -1":
            mask += rle2mask(encoded_labels, 1024, 1024)
        mask = (mask >= 1).astype("float32")

        text_prompts = [t % "pneumothorax" for t in prompt_templates]
        inference.visualization(
            model,
            image_processor,
            tokenizer,
            save_root_dir,
            text_prompts,
            image_paths,
            masks=len(prompt_templates) * [mask],
        )

        save_data.append(
            {
                "image": os.path.basename(row["ours_path"]),
                "prompts": text_prompts,
            }
        )

    save_json(save_data, os.path.join(save_root_dir, "prompt.json"))


def visualize_RSNA(
    inference,
    rsna_data,
    model,
    image_processor,
    tokenizer,
    data_root_dir,
    save_root_dir,
    prompt_templates,
):
    save_data = list()
    for image_path, boxes in zip(rsna_data[0], rsna_data[1]):
        image_paths = [os.path.join(data_root_dir, image_path)]

        text_prompts = [t % "pneumonia" for t in prompt_templates]
        inference.visualization(
            model,
            image_processor,
            tokenizer,
            save_root_dir,
            text_prompts,
            image_paths,
            bboxes=len(prompt_templates) * [boxes],
        )

        save_data.append(
            {
                "image": os.path.basename(image_path),
                "prompts": text_prompts,
            }
        )

    save_json(save_data, os.path.join(save_root_dir, "prompt.json"))
