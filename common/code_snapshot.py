import datetime
import json
import os

from git import Repo
from omegaconf import OmegaConf


def code_snapshot(save_dir, cfg=None):
    now = datetime.datetime.now()
    now_dir = "run-%s" % now.strftime("%m%d-%H%M%S")
    save_dir = os.path.join(save_dir, now_dir)
    os.makedirs(save_dir, exist_ok=True)

    # load git repo
    repo = Repo(os.getcwd())

    # save diff
    diff = repo.git.diff()
    with open(os.path.join(save_dir, "diff.diff"), "w") as f:
        f.write(diff)
        f.write("\n")

    # save last commit information
    last_commit = list(repo.iter_commits())[0]
    last_commit_info = dict()
    last_commit_info["id"] = last_commit.hexsha
    last_commit_info["committed_datetime"] = str(last_commit.committed_datetime)
    last_commit_info["summary"] = last_commit.summary
    last_commit_info["name_rev"] = last_commit.name_rev
    with open(os.path.join(save_dir, "last_commit.json"), "w") as f:
        json.dump(last_commit_info, f, indent=2)

    # save arguments
    if cfg:
        OmegaConf.save(cfg, f=os.path.join(save_dir, "config.yaml"))
    return save_dir
