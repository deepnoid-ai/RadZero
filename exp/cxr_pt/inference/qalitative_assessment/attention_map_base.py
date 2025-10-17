import os
from glob import glob

import numpy as np
import torch
from PIL import Image

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


def visualize_map(
    image_path,
    text,
    model,
    image_processor,
    tokenizer,
):

    similarity_map = extract_similarity_map(
        image_path, text, model, image_processor, tokenizer
    )

    similarity_map = torch.sigmoid(similarity_map)


if __name__ == "__main__":
    data_root = "datasets"
    experiment_root = "experiments"

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

    save_dir = "debug"
    image_path = "external_models/CARZero/test.jpg"
    text = "There is pneumonia in the left lung"

    visualize_map(image_path, text, model, image_processor, tokenizer)
