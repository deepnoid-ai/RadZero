import datetime
import os
import pickle
import random
from glob import glob

import pandas as pd

from common.utils import load_json
from exp.cxr_pt.inference.inference import Inference
from exp.cxr_pt.inference.utils import load_pretrained_model
from exp.cxr_pt.inference.visualize_utils import (
    visualize_chestXDet10,
    visualize_RSNA,
    visualize_SIIM,
)

if __name__ == "__main__":
    checkpoint_dir = "085_ds_cos_hr/checkpoint-22168"
    data_root_dir = "datasets"
    now = datetime.datetime.now()
    save_root_dir = os.path.join(
        os.path.dirname(checkpoint_dir),
        "attention_map",
        os.path.basename(checkpoint_dir),
        "run-%s" % now.strftime("%m%d-%H%M%S"),
    )

    batch_size = 1
    num_workers = 1
    num_samples = 5
    prompt_templates = [
        "There is %s",
        "There is %s of the right side",
        "There is %s of the left side",
    ]

    vis_dataset = [
        "ChestXDet10",
        "SIIM",
        "RSNA",
    ]

    model, image_processor, tokenizer = load_pretrained_model(
        checkpoint_dir,
        # load config from latest snapshot
        # TODO: modify when needed
        config_path=sorted(
            glob(
                os.path.join(
                    os.path.dirname(checkpoint_dir), "snapshot", "*", "config.yaml"
                )
            )
        )[-1],
    )

    inference = Inference(
        cls_dataset=[],
        det_dataset=[],
        seg_dataset=[],
        data_root_dir=data_root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if "ChestXDet10" in vis_dataset:
        # TODO: use get_infer_dirs
        image_csv = os.path.join(
            data_root_dir,
            "ChestXDet10/chestXDet10_test_image.csv",
        )
        text_path = os.path.join(data_root_dir, "ChestXDet10/test.json")

        image = pd.read_csv(image_csv)
        image_list = image["Path"].tolist()
        cxdet10_data = load_json(text_path)
        random_ids = random.sample(range(len(image_list)), num_samples)

        visualize_chestXDet10(
            inference,
            [cxdet10_data[i] for i in random_ids],
            [image_list[i] for i in random_ids],
            model,
            image_processor,
            tokenizer,
            data_root_dir,
            os.path.join(save_root_dir, "ChestXDet10"),
            prompt_templates,
        )
    if "SIIM" in vis_dataset:
        # TODO: use get_infer_dirs
        image_csv = os.path.join(data_root_dir, "external/MGCA/siim/test_ours_path.csv")
        siim_data = pd.read_csv(image_csv)
        siim_data["class"] = siim_data[" EncodedPixels"].apply(lambda x: x != " -1")
        pos_data = siim_data[siim_data["class"]]
        random_ids = random.sample(list(pos_data["ImageId"]), num_samples)

        df = siim_data[siim_data["ImageId"].isin(random_ids)]
        visualize_SIIM(
            inference,
            df,
            model,
            image_processor,
            tokenizer,
            data_root_dir,
            os.path.join(save_root_dir, "SIIM"),
            prompt_templates,
        )
    if "RSNA" in vis_dataset:
        # TODO: use get_infer_dirs
        data_path = os.path.join(data_root_dir, "external/MGCA/rsna/test_ours.pkl")
        with open(data_path, "rb") as f:
            rsna_data = pickle.load(f)
        pos_ids = [i for i in range(rsna_data[0].size) if rsna_data[1][i].sum() > 0]
        random_ids = random.sample(pos_ids, num_samples)
        data = [
            [rsna_data[0][i] for i in random_ids],
            [rsna_data[1][i] for i in random_ids],
        ]
        visualize_RSNA(
            inference,
            data,
            model,
            image_processor,
            tokenizer,
            data_root_dir,
            os.path.join(save_root_dir, "RSNA"),
            prompt_templates,
        )
