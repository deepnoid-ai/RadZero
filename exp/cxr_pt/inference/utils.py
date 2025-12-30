import json
import os
from functools import partial

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer

from exp.cxr_pt.inference.dataset import InferDataset, collate_fn
from exp.cxr_pt.model import load_model
from external.CARZero.inference import (
    triple_ChestXDet10_result,
    triple_Chexpert5_result,
    triple_chexpert5x200_result,
    triple_Chexpert14_result,
    tripple_openi_rusult_merge,
    tripple_padchest_rusult_merge,
)


def load_pretrained_model(
    resume_from_checkpoint,
    config_path=os.path.join("exp/cxr_pt/pt/configs/align_model.yaml"),
):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    models = load_model(config["model"], output_dir=None)
    trainer = Trainer(model=models["model"])
    trainer._load_from_checkpoint(resume_from_checkpoint)

    model = trainer.model
    model.eval()
    model.to(torch.float32)

    return model, models["image_processor"], models["tokenizer"]


def process_class_prompts(text_prompt, tokenizer, model):
    text_prompt_list = []
    neg_text_prompt_list = []
    for i in range(len(text_prompt)):
        text_prompt_list.append(text_prompt[str(i)][0])
        neg_text_prompt_list.append(
            text_prompt[str(i)][0].replace("There is", "There is no")
        )

    text_batch = tokenizer(
        text_prompt_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    neg_text_batch = tokenizer(
        neg_text_prompt_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    return {
        "encoded_key_phrases": text_batch,
        "encoded_negative_phrases": neg_text_batch,
    }


def calculate_similarities(
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

        logit_list = []

        for image in tqdm(loader):
            image = image.to(model.device)
            logits = model.compute_logits(
                pixel_values=image,
                encoded_key_phrases=[text_batch["encoded_key_phrases"]],
                encoded_negative_phrases=[text_batch["encoded_negative_phrases"]],
            )["logits"]
            logit_list.append(logits)

    class_similarities = torch.cat(logit_list, dim=0)
    class_similarities = class_similarities.float().detach().cpu().numpy()

    return class_similarities


def get_infer_dirs(data_root_dir):
    dir_dict = {
        "OpenI": {
            "image_path": os.path.join(
                data_root_dir,
                "CarZero/preprocess/OpenI/openi_multi_label_image.csv",
            ),
            "text_path": os.path.join(
                data_root_dir, "CarZero/preprocess/OpenI/openi_multi_label_text.json"
            ),
        },
        "PadChest": {
            "image_path": os.path.join(
                data_root_dir,
                "CarZero/preprocess/PadChest/padchest_multi_label_image.csv",
            ),
            "text_path": os.path.join(
                data_root_dir,
                "CarZero/preprocess/PadChest/padchest_multi_label_text.json",
            ),
        },
        "ChestXray14": {
            "image_path": os.path.join(
                data_root_dir,
                "CarZero/preprocess/ChestXray14/chestxray14_test_image.csv",
            ),
            "text_path": os.path.join(
                data_root_dir,
                "CarZero/preprocess/ChestXray14/chestxray14_test_text.json",
            ),
        },
        "Chexpert": {
            "image_path": os.path.join(
                data_root_dir,
                "CarZero/preprocess/Chexpert/chexpert5_test_image.csv",
            ),
            "text_path": os.path.join(
                data_root_dir, "CarZero/preprocess/Chexpert/chexpert5_test_text.json"
            ),
        },
        "ChestXDet10": {
            "image_path": os.path.join(
                data_root_dir,
                "CarZero/preprocess/ChestXDet10/chestXDet10_test_image.csv",
            ),
            "text_path": os.path.join(
                data_root_dir,
                "CarZero/preprocess/ChestXDet10/chestXDet10_test_text.json",
            ),
        },
        "MS-CXR": {
            "data_path": os.path.join(
                data_root_dir,
                "MS-CXR-0.1/preprocess/test.json",
            )
        },
        "SIIM": {
            "data_path": os.path.join(
                data_root_dir,
                "external/MGCA/siim/test.csv",
            ),
        },
        "RSNA": {
            "data_path": os.path.join(
                data_root_dir,
                "RSNA-pneumonia/medklip/test.csv",
            ),
        },
    }
    return dir_dict


def eval_classification(
    model,
    image_processor,
    tokenizer,
    sel_datasets,
    batch_size,
    num_workers,
    data_root_dir,
    save_root_dir,
    image_paths,
    text_paths,
):
    performances = {}
    for image_path, text_path, sel_dataset in zip(
        image_paths, text_paths, sel_datasets
    ):
        df = pd.read_csv(image_path)
        image_set = df["Path"].tolist()

        with open(text_path, "r") as f:
            text_prompt = json.load(f)

        text_batch = process_class_prompts(text_prompt, tokenizer, model)

        similarities = calculate_similarities(
            image_set,
            text_batch,
            model,
            image_processor,
            batch_size,
            num_workers,
            data_root_dir,
        )
        sim_df = pd.DataFrame(similarities)
        sim_df.to_csv(os.path.join(save_root_dir, sel_dataset) + ".csv", index=False)

        performance = cal_performance(similarities, data_root_dir, sel_dataset)
        performances.update(performance)

    return performances


def cal_performance(similarities, data_root_dir, sel_dataset):
    performance = {}

    if "OpenI" == sel_dataset:
        print("OpenI")
        (
            head_macro_auc,
            medium_macro_auc,
            tail_macro_auc,
            total_macro_auc,
            micro_prc,
            macro_prc,
        ) = tripple_openi_rusult_merge(
            similarities,
            os.path.join(data_root_dir, "CarZero/preprocess/OpenI/custom.csv"),
        )
        performance["OpenI"] = {
            "Head AUC": head_macro_auc,
            "Medium AUC": medium_macro_auc,
            "Tail AUC": tail_macro_auc,
            "Total AUC": total_macro_auc,
            "Micro AUPRC": micro_prc,
            "Macro AUPRC": macro_prc,
        }

    if "PadChest" == sel_dataset:
        print("PadChest")
        (
            head_macro_auc,
            medium_macro_auc,
            tail_macro_auc,
            total_macro_auc,
            micro_prc,
            macro_prc,
            auc_scores_20,
            macro_auprc_20,
        ) = tripple_padchest_rusult_merge(
            similarities,
            os.path.join(
                data_root_dir, "CarZero/preprocess/PadChest/manual_image.json"
            ),
        )
        performance["PadChest"] = {
            "Head AUC": head_macro_auc,
            "Medium AUC": medium_macro_auc,
            "Tail AUC": tail_macro_auc,
            "Total AUC": total_macro_auc,
            "Micro AUPRC": micro_prc,
            "Macro AUPRC": macro_prc,
            "Padhcest20 AUROC": auc_scores_20,
            "Padhcest20 AUPRC": macro_auprc_20,
        }

    if "ChestXray14" == sel_dataset:
        print("ChestXray14")
        total_macro_auc, micro_prc, macro_prc = triple_Chexpert14_result(
            similarities,
            os.path.join(
                data_root_dir, "CarZero/preprocess/ChestXray14/test_list_ours.txt"
            ),
        )
        performance["ChestXray14"] = {
            "Total AUC": total_macro_auc,
            "Micro AUPRC": micro_prc,
            "Macro AUPRC": macro_prc,
        }

    if "Chexpert" == sel_dataset:
        print("Chexpert")
        total_macro_auc, micro_prc, macro_prc = triple_Chexpert5_result(
            similarities,
            os.path.join(
                data_root_dir, "CarZero/preprocess/Chexpert/test_labels_ours.csv"
            ),
        )
        performance["Chexpert"] = {
            "Total AUC": total_macro_auc,
            "Micro AUPRC": micro_prc,
            "Macro AUPRC": macro_prc,
        }

    if "ChestXDet10" == sel_dataset:
        print("ChestXDet10")
        total_macro_auc, micro_prc, macro_prc = triple_ChestXDet10_result(
            similarities,
            os.path.join(data_root_dir, "CarZero/preprocess/ChestXDet10/test.json"),
        )
        performance["ChestXDet10"] = {
            "Total AUC": total_macro_auc,
            "Micro AUPRC": micro_prc,
            "Macro AUPRC": macro_prc,
        }

    # print("chexpert5x200")
    # triple_chexpert5x200_result(
    #     save_csv_dir[5],
    #     os.path.join(
    #         data_root_dir, "Chexpert_5x200", "chexpert_5x200_newpath_ours.csv"
    #     ),
    # )

    return performance
