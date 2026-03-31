import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.backends import cudnn


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def official_root() -> Path:
    return project_root() / "KDD2023-DCdetector"


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(array[idx - 1])


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Deterministic DCdetector entrypoint.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--win_size", type=int, default=100)
    parser.add_argument("--patch_size", nargs="+", type=int, default=[5])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss_fuc", type=str, default="MSE")
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--rec_timeseries", action="store_true", default=True)

    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=True)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    parser.add_argument("--index", type=int, default=137)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--input_c", type=int, default=9)
    parser.add_argument("--output_c", type=int, default=9)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="credit")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--data_path", type=str, default="./dataset/creditcard_ts.csv")
    parser.add_argument("--model_save_path", type=str, default="checkpoints")
    parser.add_argument("--anormly_ratio", type=float, default=4.0)

    return parser.parse_args()


def adjust_batch_size(config):
    if config.dataset == "UCR":
        buffer = [2, 4, 8, 16, 32, 64, 128, 256]
        data_len = np.load("dataset/" + config.data_path + f"/UCR_{config.index}_train.npy").shape[0]
        config.batch_size = find_nearest(buffer, data_len / config.win_size)
    elif config.dataset == "UCR_AUG":
        buffer = [2, 4, 8, 16, 32, 64, 128, 256]
        data_len = np.load("dataset/" + config.data_path + f"/UCR_AUG_{config.index}_train.npy").shape[0]
        config.batch_size = find_nearest(buffer, data_len / config.win_size)
    elif config.dataset == "SMD_Ori":
        buffer = [2, 4, 8, 16, 32, 64, 128, 256, 512]
        data_len = np.load("dataset/" + config.data_path + f"/SMD_Ori_{config.index}_train.npy").shape[0]
        config.batch_size = find_nearest(buffer, data_len / config.win_size)


def main():
    config = parse_args()
    set_seed(config.seed)

    root = official_root()
    if not root.exists():
        raise FileNotFoundError(f"Official repo not found: {root}")

    sys.path.insert(0, str(root))
    from solver import Solver  # noqa: PLC0415
    from utils.utils import mkdir  # noqa: PLC0415

    os.chdir(root)
    adjust_batch_size(config)

    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False
    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(" ", "")
        device_ids = config.devices.split(",")
        config.device_ids = [int(device_id) for device_id in device_ids]
        config.gpu = config.device_ids[0]

    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)

    print(f"seed={config.seed}")
    print(f"cudnn.deterministic={cudnn.deterministic}")
    print(f"cudnn.benchmark={cudnn.benchmark}")
    print(f"dataset={config.dataset}")
    print(f"mode={config.mode}")
    print(f"data_path={config.data_path}")

    solver = Solver(vars(config))
    if config.mode == "train":
        solver.train()
    else:
        solver.test()


if __name__ == "__main__":
    main()
