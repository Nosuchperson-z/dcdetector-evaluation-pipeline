import argparse
import gc
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = ROOT / "data_processed"


DEFAULT_WINDOWS = {
    "SMAP": {"win_size": 105, "train_stride": 1, "test_stride": 1},
    "MSL": {"win_size": 90, "train_stride": 1, "test_stride": 1},
    "HAI21.03": {"win_size": 100, "train_stride": 1, "test_stride": 1},
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_name(dataset_name: str) -> str:
    mapping = {
        "smap": "SMAP",
        "msl": "MSL",
        "hai": "HAI21.03",
        "hai21.03": "HAI21.03",
        "hai21_03": "HAI21.03",
    }
    key = dataset_name.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return mapping[key]


def make_windows(array: np.ndarray, win_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    if win_size <= 0 or stride <= 0:
        raise ValueError("win_size and stride must be positive")
    if array.shape[0] < win_size:
        raise ValueError("array is shorter than win_size")

    starts = range(0, array.shape[0] - win_size + 1, stride)
    windows = np.stack([array[start:start + win_size] for start in starts]).astype(np.float32)
    end_indices = np.asarray([start + win_size - 1 for start in starts], dtype=np.int64)
    return windows, end_indices


def write_windows_memmap(array: np.ndarray, win_size: int, stride: int, output_path: Path) -> np.ndarray:
    if win_size <= 0 or stride <= 0:
        raise ValueError("win_size and stride must be positive")
    if array.shape[0] < win_size:
        raise ValueError("array is shorter than win_size")

    num_windows = ((array.shape[0] - win_size) // stride) + 1
    end_indices = np.empty(num_windows, dtype=np.int64)
    memmap = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_windows, win_size, array.shape[1]),
    )

    for idx, start in enumerate(range(0, array.shape[0] - win_size + 1, stride)):
        end = start + win_size
        memmap[idx] = array[start:end]
        end_indices[idx] = end - 1

    del memmap
    return end_indices


def build_for_dataset(dataset_name: str, win_size: int, train_stride: int, test_stride: int) -> None:
    dataset_dir = PROCESSED_ROOT / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_dir}")

    train = np.load(dataset_dir / "train.npy")
    val = np.load(dataset_dir / "val.npy")
    test = np.load(dataset_dir / "test.npy")
    label = np.load(dataset_dir / "label.npy")

    window_dir = dataset_dir / "windows"
    ensure_dir(window_dir)

    train_end = write_windows_memmap(train, win_size, train_stride, window_dir / "train_windows.npy")
    val_end = write_windows_memmap(val, win_size, test_stride, window_dir / "val_windows.npy")
    test_end = write_windows_memmap(test, win_size, test_stride, window_dir / "test_windows.npy")
    np.save(window_dir / "train_window_end_indices.npy", train_end)
    np.save(window_dir / "val_window_end_indices.npy", val_end)
    np.save(window_dir / "test_window_end_indices.npy", test_end)
    np.save(window_dir / "test_point_label.npy", label.astype(np.int64))

    train_windows_shape = ((train.shape[0] - win_size) // train_stride + 1, win_size, train.shape[1])
    val_windows_shape = ((val.shape[0] - win_size) // test_stride + 1, win_size, val.shape[1])
    test_windows_shape = ((test.shape[0] - win_size) // test_stride + 1, win_size, test.shape[1])

    meta_path = dataset_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["windowing"] = {
        "win_size": win_size,
        "train_stride": train_stride,
        "test_stride": test_stride,
        "train_windows": int(train_windows_shape[0]),
        "val_windows": int(val_windows_shape[0]),
        "test_windows": int(test_windows_shape[0]),
        "train_window_shape": list(train_windows_shape),
        "val_window_shape": list(val_windows_shape),
        "test_window_shape": list(test_windows_shape),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Build windowed arrays from processed datasets.")
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument("--win_size", type=int, default=None)
    parser.add_argument("--train_stride", type=int, default=None)
    parser.add_argument("--test_stride", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if any(name.lower() == "all" for name in args.dataset):
        datasets = ["SMAP", "MSL", "HAI21.03"]
    else:
        datasets = [normalize_name(name) for name in args.dataset]

    for dataset_name in datasets:
        defaults = DEFAULT_WINDOWS[dataset_name]
        win_size = args.win_size if args.win_size is not None else defaults["win_size"]
        train_stride = args.train_stride if args.train_stride is not None else defaults["train_stride"]
        test_stride = args.test_stride if args.test_stride is not None else defaults["test_stride"]
        print(
            f"windowing {dataset_name} win_size={win_size} train_stride={train_stride} test_stride={test_stride}"
        )
        build_for_dataset(dataset_name, win_size, train_stride, test_stride)


if __name__ == "__main__":
    main()
