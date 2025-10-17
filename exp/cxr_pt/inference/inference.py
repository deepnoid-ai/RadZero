import os
import traceback

import pandas as pd
import torch

from common.utils import load_json, save_json
from exp.cxr_pt.inference.grounding_utils import (
    chestXDet10_eval_grounding,
    eval_refer_grounding,
)
from exp.cxr_pt.inference.segmentation_utils import (
    eval_segmentation_rsna_medklip,
    eval_segmentation_siim,
)
from exp.cxr_pt.inference.utils import eval_classification, get_infer_dirs
from exp.cxr_pt.inference.visualize_utils import (
    get_attention_weights,
    save_attention_map,
)


class Inference:
    def __init__(
        self,
        cls_dataset,
        det_dataset,
        seg_dataset,
        data_root_dir,
        batch_size,
        num_workers,
    ):
        self.data_root_dir = data_root_dir
        self.cls_dataset = cls_dataset
        self.det_dataset = det_dataset
        self.seg_dataset = seg_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    @torch.no_grad()
    def classification(self, model, image_processor, tokenizer, save_root_dir):
        try:
            os.makedirs(save_root_dir, exist_ok=True)
            dir_dict = get_infer_dirs(self.data_root_dir)

            image_paths, text_paths = [], []
            for cls_d in self.cls_dataset:
                cls_d_dirs = dir_dict[cls_d]

                image_paths.append(cls_d_dirs["image_path"])
                text_paths.append(cls_d_dirs["text_path"])

            performances = eval_classification(
                model,
                image_processor,
                tokenizer,
                self.cls_dataset,
                self.batch_size,
                self.num_workers,
                self.data_root_dir,
                save_root_dir,
                image_paths,
                text_paths,
            )

            save_json(performances, os.path.join(save_root_dir, "result.json"))
        except Exception as e:
            print(f"Error in zero shot classification: {e}\n{traceback.format_exc()}")

    @torch.no_grad()
    def grounding(self, model, image_processor, tokenizer, save_root_dir):
        try:
            os.makedirs(save_root_dir, exist_ok=True)
            dir_dict = get_infer_dirs(self.data_root_dir)

            result = {}
            for det_d in self.det_dataset:
                if det_d == "ChestXDet10":
                    det_d_dirs = dir_dict[det_d]
                    image_path = det_d_dirs["image_path"]
                    text_path = det_d_dirs["text_path"]
                    test_path = os.path.join(
                        self.data_root_dir, "CarZero/preprocess/ChestXDet10/test.json"
                    )

                    result["ChestXDet10"] = chestXDet10_eval_grounding(
                        model,
                        image_processor,
                        tokenizer,
                        self.batch_size,
                        self.num_workers,
                        image_path,
                        text_path,
                        test_path,
                        self.data_root_dir,
                    )

                elif det_d == "MS-CXR":
                    det_d_dirs = dir_dict[det_d]
                    data_path = det_d_dirs["data_path"]

                    data = load_json(data_path)

                    result["MS-CXR"] = eval_refer_grounding(
                        model,
                        image_processor,
                        tokenizer,
                        data,
                        self.data_root_dir,
                    )

            # save result
            save_json(result, os.path.join(save_root_dir, "result.json"))

        except Exception as e:
            print(f"Error in zero shot grounding: {e}\n{traceback.format_exc()}")

    @torch.no_grad()
    def segmentation(
        self,
        model,
        image_processor,
        tokenizer,
        save_root_dir,
        compute_pixel_level_auroc,
    ):
        try:
            os.makedirs(save_root_dir, exist_ok=True)
            dir_dict = get_infer_dirs(self.data_root_dir)

            result = {}
            for seg_d in self.seg_dataset:
                if seg_d == "SIIM":
                    seg_d_dirs = dir_dict[seg_d]
                    data_path = seg_d_dirs["data_path"]

                    data = pd.read_csv(data_path)
                    text = "There is Pneumothorax"

                    # SIIM dataset split from MGCA
                    result["SIIM"] = eval_segmentation_siim(
                        model,
                        image_processor,
                        tokenizer,
                        data,
                        text,
                        self.data_root_dir,
                        compute_pixel_level_auroc,
                    )

                elif seg_d == "RSNA":
                    seg_d_dirs = dir_dict[seg_d]
                    data_path = seg_d_dirs["data_path"]

                    medklip_data = pd.read_csv(data_path)

                    text = "There is Pneumonia"

                    # RSNA dataset split from Medklip
                    result["RSNA"] = eval_segmentation_rsna_medklip(
                        model,
                        image_processor,
                        tokenizer,
                        medklip_data,
                        text,
                        self.data_root_dir,
                        compute_pixel_level_auroc,
                    )

            save_json(result, os.path.join(save_root_dir, "result.json"))

        except Exception as e:

            print(f"Error in zero shot segmentation: {e}\n{traceback.format_exc()}")

    def visualization(
        self,
        model,
        image_processor,
        tokenizer,
        save_root_dir,
        text_prompts,
        image_paths=None,
        bboxes=None,
        masks=None,
    ):
        os.makedirs(save_root_dir, exist_ok=True)
        if image_paths is None:
            image_paths = list()
            dir_dict = get_infer_dirs(self.data_root_dir)
            for sel_d in self.det_dataset:
                sel_d_dirs = dir_dict[sel_d]
                df = pd.read_csv(sel_d_dirs["image_path"])
                image_paths.extend(df["Path"].tolist())

        attention_weights = get_attention_weights(
            model,
            image_processor,
            tokenizer,
            self.batch_size,
            self.num_workers,
            self.data_root_dir,
            image_paths,
            text_prompts,
        )

        save_attention_map(
            attention_weights,
            image_paths,
            self.data_root_dir,
            save_root_dir,
            image_processor,
            bboxes,
            masks,
        )


if __name__ == "__main__":
    pass
