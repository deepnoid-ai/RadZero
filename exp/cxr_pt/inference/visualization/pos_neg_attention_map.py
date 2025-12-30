import os
import random
from glob import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from exp.cxr_pt.inference.segmentation_utils import (
    interpolate_similarity_scores,
    read_from_dicom,
)
from exp.cxr_pt.inference.utils import load_pretrained_model


def extract_similarity_map(image, text, model, image_processor, tokenizer):
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


def visualize_map(
    image_path,
    text,
    model,
    image_processor,
    tokenizer,
):
    if image_path.endswith("dcm"):
        image = read_from_dicom(image_path)
    else:
        image = Image.open(image_path)

    similarity_map = extract_similarity_map(
        image, text, model, image_processor, tokenizer
    )

    return torch.sigmoid(similarity_map), image


if __name__ == "__main__":
    data_root = "datasets"
    experiment_root = "experiments"
    test_csv = "RSNA-pneumonia/medklip/test_ours.csv"
    seed = 0

    checkpoint_dir = os.path.join(
        experiment_root, "119_prompt_align_data_hr/checkpoint-22056"
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

    random.seed(seed)
    rsna_data = pd.read_csv(
        os.path.join(data_root, "RSNA-pneumonia/medklip/test_ours.csv")
    )
    pos_data = rsna_data[rsna_data["classes"] == 1]
    neg_data = rsna_data[rsna_data["classes"] == 0]
    # # match same image with CARZero by ID
    # pos_data = rsna_data[rsna_data["ID"] == "bc87d74c-9d22-4046-ae17-5891adfa0b41"]
    # neg_data = rsna_data[rsna_data["ID"] == "6e62a24f-ac15-421e-b1c7-1a3db1b7cdc0"]
    pos_row = pos_data.sample(n=1)
    neg_row = neg_data.sample(n=1)

    save_dir = "result/attn_map"
    os.makedirs(save_dir, exist_ok=True)
    text = "There is pneumonia"

    pos_map, pos_img = visualize_map(
        os.path.join(
            data_root,
            pos_row["img_path"].values[0],
        ),
        text,
        model,
        image_processor,
        tokenizer,
    )
    neg_map, neg_img = visualize_map(
        os.path.join(
            data_root,
            neg_row["img_path"].values[0],
        ),
        text,
        model,
        image_processor,
        tokenizer,
    )

    min_val = torch.cat([pos_map, neg_map]).min()
    max_val = torch.cat([pos_map, neg_map]).max()

    Image.fromarray(
        # (255 * pos_map)
        (255 * (pos_map - min_val) / (max_val - min_val))
        .detach()
        .cpu()
        .round()
        .int()
        .numpy()
    ).convert("RGB").save(
        os.path.join(
            save_dir,
            "seed%02d_pos_%s.jpg" % (seed, pos_row["ID"].values[0]),
        )
    )
    pos_img.save(
        os.path.join(
            save_dir,
            "gt_pos_%s.jpg" % pos_row["ID"].values[0],
        )
    )

    Image.fromarray(
        # (255 * neg_map)
        (255 * (neg_map - min_val) / (max_val - min_val))
        .detach()
        .cpu()
        .round()
        .int()
        .numpy()
    ).convert("RGB").save(
        os.path.join(
            save_dir,
            "seed%02d_neg_%s.jpg" % (seed, neg_row["ID"].values[0]),
        )
    )
    neg_img.save(
        os.path.join(
            save_dir,
            "gt_neg_%s.jpg" % neg_row["ID"].values[0],
        )
    )
