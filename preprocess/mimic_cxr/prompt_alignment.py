import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_json(file_path, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as f:
        return json.load(f)


def save_json(data, file_path, encoding="utf-8", indent=2):
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, indent=indent)


KEY_PHRASES = """
You are an expert medical assistant AI specializing in understanding and analyzing chest x-ray radiology reports.

Your task is to extract the medically significant and meaningful findings from the given chest x-ray report, focusing on identifying phrases or expressions that describe notable conditions or abnormalities.
Note that the report may reference previous studies, but we only need an interpretation based on the current chest x-ray. Therefore, remove and rewrite terms like  "new", "improved", "unchanged", "worsened"  or "consistent" to reflect the current status in a way that indicates the condition exists as observed in this image, without implying any comparison to prior images or studies.

The template format includes:
"There is [finding] of [location]."
"There may be [finding] of [location]."
"There is no [finding] of [location]."

[finding] represents the extracted key findings from the radiology report, and [location] represents the anatomical location mentioned in the report. If no location is provided, do not include it in the output.

Adhere strictly to the following JSON format for the final output, using examples as a guideline for the desired analysis structure. Do not provide any explanations; output only in JSON format.
If the report does not contain any findings, output an empty list (example: {"key_phrases": []}).

[Example]
INPUT:
Cardiomegaly is accompanied by improving pulmonary vascular congestion and decreasing pulmonary edema. Left retrocardiac opacity has substantially improved, likely a combination of atelectasis and effusion. A more confluent opacity at the right lung base persists, and could be due to asymmetrically resolving edema, but pneumonia should be considered in the appropriate clinical setting. Small right pleural effusion is likely unchanged, with pigtail pleural catheter remaining in place and no visible pneumothorax.

OUTPUT:
{
  "key_phrase" : [
    "There is cardiomegaly with pulmonary vascular congestion",
    "There is pulmonary edema",
    "There is left retrocardiac opacity",
    "There may be atelectasis",
    "There may be effusion",
    "There is right lung base opacity",
    "There is right lung base opacity suggestive of possible pneumonia",
    "There may be small right pleural effusion",
    "There is pigtail pleural catheter in place",
    "There is no pneumothorax",
    ]
}
"""


def generate_batch_response(system_prompt, llm, batch):

    prompts = []
    for data in batch:
        inputs = data["findings"] + " " + data["impression"]
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"INPUT:\n{inputs}\n\nOUTPUT:"},
        ]
        prompts.append(prompt)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    prompts = [
        tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]

    # default generation config: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/generation_config.json
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=512)

    outputs = llm.generate(prompts, sampling_params)

    max_count = 5
    text_outputs = []
    for i, data in enumerate(batch):
        text_outputs.append(outputs[i].outputs[0].text)

    success_json_file = []
    failed_json_file = []
    for i, data in enumerate(batch):
        generated_text = text_outputs[i]
        attempts = 0
        result = None
        while attempts < max_count:
            try:
                result = eval(generated_text)
                data["key_phrases"] = result["key_phrases"]
                break
            except Exception:
                if not (data["findings"] + data["impression"]).strip():
                    print("empty findings and impression")
                    break

                attempts += 1
                # Regenerate output
                print(
                    f"Regenerate output because it is not in JSON format! : {attempts}"
                )
                outputs = llm.generate([prompts[i]], sampling_params)
                generated_text = outputs[0].outputs[0].text

        if attempts == max_count:
            failed_json_file.append(data)
            result = None

        if result:
            success_json_file.append(data)

    return success_json_file, failed_json_file


if __name__ == "__main__":

    llm = LLM(
        model="meta-llama/Llama-3.3-70B-Instruct",
        tensor_parallel_size=4,
        dtype=torch.bfloat16,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        gpu_memory_utilization=0.70,
    )

    version = "v12.2"

    data_root = "datasets"

    split_list = ["validate", "test", "train"]

    batch_size = 32768

    for split in tqdm(split_list):

        save_dir = os.path.join(
            data_root,
            f"MIMIC-CXR/preprocess/{version}",
        )

        os.makedirs(save_dir, exist_ok=True)

        split_save_dir = os.path.join(save_dir, split)
        os.makedirs(split_save_dir, exist_ok=True)

        # result from preprocess.mimic_cxr.findings_impression.py
        file_path = f"MIMIC-CXR/preprocess/v12.0/{split}.json"
        input_list = load_json(os.path.join(data_root, file_path))

        # process all result
        all_results = []

        # failed file
        failed_json_result_list = []

        for i in tqdm(range(0, len(input_list), batch_size)):

            # save batch
            gpt_assisted_result_save_path = os.path.join(
                split_save_dir, f"{split}_{i//batch_size}.json"
            )

            # For things that already exist, continue
            if os.path.exists(gpt_assisted_result_save_path):
                all_results += load_json(gpt_assisted_result_save_path)
                if os.path.exists(
                    gpt_assisted_result_save_path.replace(".json", "_failed.json")
                ):
                    failed_json_result_list += load_json(
                        gpt_assisted_result_save_path.replace(".json", "_failed.json")
                    )
                continue

            batch = input_list[i : i + batch_size]

            gpt_assisted_result, failed_json_result = generate_batch_response(
                KEY_PHRASES, llm, batch
            )

            # add failed files
            if failed_json_result:
                save_json(
                    failed_json_result,
                    gpt_assisted_result_save_path.replace(".json", "_failed.json"),
                )
                failed_json_result_list += failed_json_result

            save_json(gpt_assisted_result, gpt_assisted_result_save_path)

            all_results.extend(gpt_assisted_result)

        if len(failed_json_result_list) > 0:
            save_json(
                failed_json_result_list,
                os.path.join(save_dir, f"{split}_failed.json"),
            )

        # save all result, Add negative findings key value at the end if negative position
        overall_save_path = os.path.join(save_dir, f"{split}.json")
        save_json(all_results, overall_save_path)

        print(f"Finished processing for argument: {split}")
