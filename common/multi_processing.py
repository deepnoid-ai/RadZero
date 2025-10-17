import json
import multiprocessing
import os
import time
from functools import partial

import PIL.Image
from tqdm import tqdm


def func_with_multiprocessing(func, input_list, num_processes):
    results = []

    if num_processes > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            for result in tqdm(pool.imap(func, input_list), total=len(input_list)):
                if result is not None:
                    results.append(result)
    else:
        for i in tqdm(input_list):
            result = func(i)
            if result is not None:
                results.append(result)
    return results


def open_json(path):
    with open(path, "r") as json_file:
        data = json.load(json_file)
    return data


def image_load_check(image_path):
    try:
        _ = PIL.Image.open(image_path).convert("RGB")
    except Exception:
        print(f"image file load error: {image_path}")


if __name__ == "__main__":

    input_list = [
        "/path/to/your/file1.txt",
        "/path/to/your/file2.txt",
        "/path/to/your/file3.txt",
    ] * 10

    # example

    def preprocess_file(file_path):
        # 예시: 파일 경로 출력 및 간단한 대기 시간 추가
        print(f"Processing {file_path}")
        time.sleep(1)

    func_with_multiprocessing(preprocess_file, input_list, 8)

    # example, partial

    def preprocess_2_file(file_path, my_name):
        # 예시: 파일 경로 출력 및 간단한 대기 시간 추가
        print(f"Processing {file_path}")
        print(my_name)
        time.sleep(1)

    partial_func = partial(preprocess_2_file, my_name="jg")

    func_with_multiprocessing(partial_func, input_list, 6)
