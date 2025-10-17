import os

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchmetrics.segmentation import DiceScore
from tqdm import tqdm
from transformers import BitImageProcessor, BlipImageProcessor

from exp.cxr_pt.model.processing import (
    AspectRatioBlipImageProcessor,
    M3AEImageProcessor,
)


def rle2mask(rle, width, height):
    """Run length encoding to segmentation mask"""
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position : current_position + lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height).T


def interpolate_similarity_scores(similarity_scores, origin_size, image_processor):
    (height, width) = origin_size
    patch_size = int(similarity_scores.shape[-1] ** 0.5)
    scores = similarity_scores.view(1, 1, patch_size, patch_size)

    if isinstance(image_processor, AspectRatioBlipImageProcessor):
        # Keep aspect ratio
        padded_size = max(height, width)
        interpolated_scores = F.interpolate(
            scores,
            size=(padded_size, padded_size),
            mode="bilinear",
            align_corners=False,
        )

        # calculate the area where the original image is located
        pad_left = (padded_size - width) // 2
        pad_top = (padded_size - height) // 2

        # Crop only the calculated area to restore the original image size
        interpolated_scores = interpolated_scores[
            :, :, pad_top : pad_top + height, pad_left : pad_left + width
        ].contiguous()

        interpolated_scores = interpolated_scores.squeeze(1)

    elif isinstance(image_processor, BlipImageProcessor):
        # XrayDINOv2
        interpolated_scores = F.interpolate(
            scores,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        interpolated_scores = interpolated_scores.squeeze(1)

    elif isinstance(image_processor, BitImageProcessor):
        shortest = min(height, width)

        interpolated_scores = F.interpolate(
            scores,
            size=(shortest, shortest),
            mode="bilinear",
            align_corners=False,
        )

        cropped_left = (width - shortest) // 2
        cropped_top = (height - shortest) // 2

        original_size_map = torch.ones(height, width) * -999
        original_size_map[
            cropped_top : cropped_top + shortest, cropped_left : cropped_left + shortest
        ] = interpolated_scores.view(shortest, shortest)

        interpolated_scores = original_size_map
        interpolated_scores = interpolated_scores.unsqueeze(0)
    elif isinstance(image_processor, M3AEImageProcessor):

        padded_size = max(height, width)

        padded_size_map = torch.ones(padded_size, padded_size) * -999

        cropped_size = int(padded_size * 224 / 256)

        interpolated_scores = F.interpolate(
            scores,
            size=(cropped_size, cropped_size),
            mode="bilinear",
            align_corners=False,
        )

        left = (padded_size - cropped_size) // 2
        top = (padded_size - cropped_size) // 2

        padded_size_map[top : top + cropped_size, left : left + cropped_size] = (
            interpolated_scores.view(cropped_size, cropped_size)
        )

        # calculate the area where the original image is located
        pad_left = (padded_size - width) // 2
        pad_top = (padded_size - height) // 2

        interpolated_scores = padded_size_map[
            pad_top : pad_top + height, pad_left : pad_left + width
        ]
        interpolated_scores = interpolated_scores.unsqueeze(0)
    return interpolated_scores


def read_from_dicom(img_path):
    dcm = pydicom.read_file(img_path)
    x = dcm.pixel_array
    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))

    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    return Image.fromarray(x).convert("RGB")


def compute_specificity(negative_probs, threshold):
    """
    Calculate image-level specificity.

    Specificity is the proportion of true negatives (images correctly identified as negative)
    out of the total number of negatives.

    Args:
        negative_probs (torch.Tensor): Tensor of predicted probabilities for negative images.
        threshold (float): Threshold to classify the predictions.

    Returns:
        float: Specificity value.
    """
    # Check if all pixels in the negative images are below the threshold
    true_negatives = (
        (negative_probs.squeeze(1) > threshold).long().sum(-1).sum(-1) == 0
    ).sum()

    # Calculate specificity as the ratio of true negatives to the total number of negative images
    spec = (true_negatives / len(negative_probs)).item()

    return spec


@torch.no_grad()
def eval_segmentation_siim(
    model,
    image_processor,
    tokenizer,
    data,
    text,
    data_root_dir,
    compute_pixel_level_auroc,
):
    data["class"] = data[" EncodedPixels"].apply(lambda x: x != " -1")
    imgids = data.ImageId.unique().tolist()

    prob_list = []
    mask_list = []

    logits = []
    labels = []

    positive_probs = []
    positive_masks = []

    negative_probs = []
    negative_masks = []

    for imgid in tqdm(imgids):
        imgid_df = data.groupby("ImageId").get_group(imgid)

        # image
        image = read_from_dicom(
            os.path.join(data_root_dir, imgid_df["ours_path"].tolist()[0])
        )
        (width, height) = image.size

        # mask
        encoded_labels = imgid_df[" EncodedPixels"].tolist()
        mask = np.zeros([height, width])
        if encoded_labels[0] != " -1":
            for encoded_label in encoded_labels:
                mask += rle2mask(encoded_label, height, width)
        mask = torch.LongTensor(mask > 0).unsqueeze(0)
        labels.append((mask.sum() > 0).long())

        processed_image = image_processor(image)
        processed_image = torch.FloatTensor(
            np.array(processed_image["pixel_values"])
        ).to(model.device)

        # text
        tokenized_text = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        output = model.compute_logits(processed_image, [tokenized_text])
        similarity_scores = output["similarity_scores"]
        logits.append(output["logits"])

        # resize
        recon_feature = interpolate_similarity_scores(
            similarity_scores, (height, width), image_processor
        )
        prob = torch.sigmoid(recon_feature)
        prob = prob.detach().cpu()

        prob_list.append(prob)
        mask_list.append(mask)

        if mask.sum() > 0:
            positive_probs.append(prob)
            positive_masks.append(mask)
        else:
            negative_probs.append(prob)
            negative_masks.append(mask)

    result = {}

    # classification auc
    labels = torch.stack(labels).numpy()
    logits = torch.stack(logits).cpu().numpy()
    classification_auc = roc_auc_score(labels, logits)

    print(f"classification auc: {classification_auc}")
    result["auc"] = classification_auc

    # dice score
    positive_probs = torch.stack(positive_probs).cpu()
    positive_masks = torch.stack(positive_masks).cpu()
    negative_probs = torch.stack(negative_probs).cpu()
    negative_masks = torch.stack(negative_masks).cpu()

    best_dice = 0.0
    for t in tqdm(np.arange(0, 1.01, 0.01)):
        dice = DiceScore(num_classes=1)

        cur_dice = dice((positive_probs > t).long(), positive_masks)
        if cur_dice > best_dice:
            best_dice = cur_dice
            best_threshold = t

    dice_score = best_dice.item()

    print(f"dice score (positive only): {dice_score}")
    print(f"best threshold: {best_threshold}")
    result["dice"] = dice_score
    result["best_threshold"] = best_threshold

    # specificity
    specificity = compute_specificity(negative_probs, best_threshold)
    print(f"specificity: {specificity}")
    result["specificity"] = specificity

    if compute_pixel_level_auroc:
        # compute pixel-level auroc
        all_probs = torch.stack(prob_list).view(-1).cpu().numpy()
        all_masks = torch.stack(mask_list).long().view(-1).cpu().numpy()

        pixel_level_auroc = roc_auc_score(all_masks, all_probs)
        print(f"pixel-level auroc: {pixel_level_auroc}")
        result["pixel_level_auroc"] = pixel_level_auroc

    print("SIIM Result")
    print(result)

    return result


@torch.no_grad()
def eval_segmentation_rsna_medklip(
    model,
    image_processor,
    tokenizer,
    data,
    text,
    data_root_dir,
    compute_pixel_level_auroc,
):

    image_paths = data["img_path"].tolist()
    bboxs = data["boxes"].tolist()

    image_paths = np.array(image_paths)

    prob_list = []
    mask_list = []

    logits = []
    labels = []

    positive_probs = []
    positive_masks = []

    negative_probs = []
    negative_masks = []

    labels = []
    for img_path, bbox in tqdm(zip(image_paths, bboxs), total=len(image_paths)):
        # image
        image = read_from_dicom(os.path.join(data_root_dir, img_path))
        (width, height) = image.size

        # mask
        mask = np.zeros([height, width])

        if not pd.isna(bbox) and bbox != "nan":
            boxes = bbox.split("|")
            for box in boxes:
                cc = box.split(";")
                mask[
                    int(float(cc[1])) : (int(float(cc[1])) + int(float(cc[3]))),
                    int(float(cc[0])) : (int(float(cc[0])) + int(float(cc[2]))),
                ] = 1

        mask = torch.LongTensor(mask).unsqueeze(0)
        labels.append((mask.sum() > 0).long())

        processed_image = image_processor(image)
        processed_image = torch.FloatTensor(
            np.array(processed_image["pixel_values"])
        ).to(model.device)

        # text
        tokenized_text = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        output = model.compute_logits(processed_image, [tokenized_text])
        similarity_scores = output["similarity_scores"]
        logits.append(output["logits"])

        # resize
        recon_feature = interpolate_similarity_scores(
            similarity_scores, (height, width), image_processor
        )
        prob = torch.sigmoid(recon_feature)
        prob = prob.detach().cpu()

        prob_list.append(prob)
        mask_list.append(mask)

        if mask.sum() > 0:
            positive_probs.append(prob)
            positive_masks.append(mask)

        else:
            negative_probs.append(prob)
            negative_masks.append(mask)

    result = {}

    # classification auc
    labels = torch.stack(labels).numpy()
    logits = torch.stack(logits).cpu().numpy()
    classification_auc = roc_auc_score(labels, logits)

    print(f"classification auc: {classification_auc}")
    result["auc"] = classification_auc

    # dice score
    positive_probs = torch.stack(positive_probs).cpu()
    positive_masks = torch.stack(positive_masks).cpu()
    negative_probs = torch.stack(negative_probs).cpu()
    negative_masks = torch.stack(negative_masks).cpu()

    best_dice = 0.0
    for t in tqdm(np.arange(0, 1.01, 0.01)):
        # use cpu to avoid gpu memory error
        dice = DiceScore(num_classes=1)

        cur_dice = dice((positive_probs > t).long(), positive_masks)
        if cur_dice > best_dice:
            best_dice = cur_dice
            best_threshold = t

    dice_score = best_dice.item()

    print(f"dice score (positive only): {dice_score}")
    print(f"best threshold: {best_threshold}")
    result["dice"] = dice_score
    result["best_threshold"] = best_threshold

    # specificity
    specificity = compute_specificity(negative_probs, best_threshold)
    print(f"specificity: {specificity}")
    result["specificity"] = specificity

    if compute_pixel_level_auroc:
        # compute pixel-level auroc
        all_probs = torch.stack(prob_list).view(-1).cpu().numpy()
        all_masks = torch.stack(mask_list).long().view(-1).cpu().numpy()

        pixel_level_auroc = roc_auc_score(all_masks, all_probs)
        print(f"pixel-level auroc: {pixel_level_auroc}")

        result["pixel_level_auroc"] = pixel_level_auroc

    print("RSNA Result")
    print(result)

    return result
