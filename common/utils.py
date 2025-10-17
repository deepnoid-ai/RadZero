import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime

from dotenv import load_dotenv
import numpy as np
import torch.distributed as dist
from omegaconf import OmegaConf


def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")[:-1]


def update_nested_dict(original, updates):
    for key, value in updates.items():
        if key in original:
            if isinstance(value, dict) and isinstance(original[key], dict):
                update_nested_dict(original[key], value)
            else:
                original[key] = value
        else:
            original[key] = value


class Config:
    def __init__(self, args):
        self.config = OmegaConf.load(args.cfg_path)

        # convert to dict
        self.config = OmegaConf.to_container(self.config, resolve=True)

        # apply additional cfg list
        for cfg_name in args.add_cfg_list:
            add_cfg_path = os.path.join(
                os.path.dirname(args.cfg_path), "configs", cfg_name
            )

            # add .yaml
            if not cfg_name.endswith(".yaml"):
                add_cfg_path += ".yaml"

            # config load
            add_cfg = OmegaConf.load(add_cfg_path)
            add_cfg = OmegaConf.to_container(add_cfg, resolve=True)

            update_nested_dict(self.config, add_cfg)

        self.config.update({"args": vars(args)})

        # override user, name with args
        if self.config["args"].get("user"):
            self.config["experiment"]["user"] = self.config["args"].get("user")

        if self.config["args"].get("name"):
            self.config["experiment"]["name"] = self.config["args"].get("name")


class CustomPrefixFilter(logging.Filter):
    def filter(self, record):
        # 현재 시간을 'YYYY-MM-DD HH:MM:SS' 형식으로 포맷
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 로그 레벨 이름을 가져옴
        log_level = record.levelname
        # 메시지에 현재 시간과 로그 레벨을 prefix로 추가
        record.msg = f"[{current_time} {log_level}] {record.msg}"
        return True


# 메인 프로세스 필터
class MainProcessFilter(logging.Filter):
    def filter(self, record):
        # DDP가 초기화되었는지 확인하고, 초기화되었다면 rank를 체크
        if dist.is_initialized():
            # 메인 프로세스(rank 0)일 경우에만 로그를 출력
            return dist.get_rank() == 0
        else:
            # DDP가 초기화되지 않은 경우 모든 로그를 출력
            return True


def load_logger(logger=None, level=logging.INFO):
    if logger is None:
        logger = logging.getLogger("myLogger")

    logger.setLevel(level)
    # 로거 전파 비활성화
    logger.propagate = False

    # 콘솔에 로그 출력을 위한 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))  # 커스텀 메시지 포맷
    logger.addHandler(console_handler)

    # 커스텀 필터 추가
    logger.addFilter(CustomPrefixFilter())

    # DDP 환경인 경우에만 메인 프로세스 필터를 추가
    logger.addFilter(MainProcessFilter())
    return logger


def set_logger_file(filepath, logger):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # set logging file
    logger.addHandler(logging.FileHandler(filepath, mode="a"))

    # TODO: sys.stdout write file


def load_json(file_path, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as f:
        return json.load(f)


def save_json(data, file_path, encoding="utf-8", indent=2):
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, indent=indent)


def save_evaluation_result(result_dict, save_path, datatset_name):
    result = {f"{datatset_name}": result_dict}
    save_json(result, save_path)


def output_directory_setting(cfg, logger):
    # output directory settings
    cfg["train"]["output_dir"] = os.path.join(
        cfg["experiment"]["output_root_dir"],
        cfg["experiment"]["project"],
        cfg["experiment"]["user"],
        cfg["experiment"]["name"],
    )
    set_logger_file(os.path.join(cfg["train"]["output_dir"], "output.log"), logger)
    logger.info(f"experiment output directory : {cfg['train']['output_dir']}")

    # skip report in debug mode
    if (cfg["args"]["no_report"]) or cfg["experiment"]["user"] == "debug":
        logger.info("skip report to wandb")
        cfg["train"]["report_to"] = "none"
    else:
        # wandb setting
        if cfg["train"]["report_to"] == "wandb":
            cfg["train"]["run_name"] = os.path.join(
                cfg["experiment"]["user"], cfg["experiment"]["name"]
            )
            load_dotenv()
            os.environ["WANDB_PROJECT"] = cfg["experiment"]["project"]
            os.environ["WANDB_DIR"] = cfg["train"]["output_dir"]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def serialize(string):
    return bytearray(pickle.dumps(string))


def deserialize(serialized_data):
    if type(serialized_data) is np.ndarray:
        serialized_data = serialized_data.tolist()
    return pickle.loads(bytes(serialized_data))
