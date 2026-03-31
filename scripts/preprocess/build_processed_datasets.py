import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data_raw"
PROCESSED_ROOT = ROOT / "data_processed"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_numpy(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(path, array)


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


def split_train_val(train: np.ndarray, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    split_index = int(train.shape[0] * (1 - val_ratio))
    split_index = max(1, min(split_index, train.shape[0] - 1))
    return train[:split_index], train[split_index:]


def drop_full_empty_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    full_empty = [col for col in df.columns if df[col].isna().all()]
    return df.drop(columns=full_empty), full_empty


def numeric_feature_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude]


def clean_dataframe(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    data = df.loc[:, feature_columns].copy()
    data, dropped_full_empty = drop_full_empty_columns(data)

    missing_before = [col for col in data.columns if data[col].isna().any()]
    if missing_before:
        data = data.ffill().bfill()

    constant_cols = [col for col in data.columns if data[col].nunique(dropna=False) <= 1]
    if constant_cols:
        data = data.drop(columns=constant_cols)

    return data, dropped_full_empty, constant_cols


def fit_transform_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    scaler = StandardScaler()
    scaler.fit(train_df.to_numpy(dtype=np.float64))
    train = scaler.transform(train_df.to_numpy(dtype=np.float64)).astype(np.float32)
    val = scaler.transform(val_df.to_numpy(dtype=np.float64)).astype(np.float32)
    test = scaler.transform(test_df.to_numpy(dtype=np.float64)).astype(np.float32)
    return train, val, test, scaler


def save_processed_dataset(
    dataset_name: str,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    label: np.ndarray,
    meta: dict,
) -> None:
    dataset_dir = PROCESSED_ROOT / dataset_name
    ensure_dir(dataset_dir)

    save_numpy(dataset_dir / "train.npy", train)
    save_numpy(dataset_dir / "val.npy", val)
    save_numpy(dataset_dir / "test.npy", test)
    save_numpy(dataset_dir / "label.npy", label.astype(np.int64))

    with (dataset_dir / "meta.json").open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)


def build_smap_or_msl(dataset_name: str, val_ratio: float) -> None:
    raw_dir = RAW_ROOT / dataset_name
    train_raw = np.load(raw_dir / f"{dataset_name}_train.npy").astype(np.float64)
    test_raw = np.load(raw_dir / f"{dataset_name}_test.npy").astype(np.float64)
    label = np.load(raw_dir / f"{dataset_name}_test_label.npy").astype(np.int64)

    constant_mask = np.var(train_raw, axis=0) > 0
    dropped_constant_indices = [int(idx) for idx, keep in enumerate(constant_mask.tolist()) if not keep]
    train_raw = train_raw[:, constant_mask]
    test_raw = test_raw[:, constant_mask]

    train_raw, val_raw = split_train_val(train_raw, val_ratio)

    scaler = StandardScaler()
    scaler.fit(train_raw)
    train = scaler.transform(train_raw).astype(np.float32)
    val = scaler.transform(val_raw).astype(np.float32)
    test = scaler.transform(test_raw).astype(np.float32)

    meta = {
        "dataset": dataset_name,
        "source_type": "preprocessed_numpy",
        "train_shape_raw": list(train_raw.shape),
        "val_shape_raw": list(val_raw.shape),
        "test_shape_raw": list(test_raw.shape),
        "label_shape": list(label.shape),
        "feature_count": int(train.shape[1]),
        "label_semantics": "point-level anomaly label for test only",
        "time_column": None,
        "dropped_constant_feature_indices": dropped_constant_indices,
        "validation_rule": f"last_{int(val_ratio * 100)}pct_of_train",
        "standardization": "fit_on_train_only",
    }
    save_processed_dataset(dataset_name, train, val, test, label, meta)


def build_hai(val_ratio: float) -> None:
    dataset_name = "HAI21.03"
    raw_dir = RAW_ROOT / "hai" / "hai-21.03"
    train_files = ["train1.csv.gz", "train2.csv.gz", "train3.csv.gz"]
    test_files = ["test1.csv.gz", "test2.csv.gz", "test3.csv.gz", "test4.csv.gz", "test5.csv.gz"]

    train_frames = [pd.read_csv(raw_dir / name, compression="gzip", low_memory=False) for name in train_files]
    test_frames = [pd.read_csv(raw_dir / name, compression="gzip", low_memory=False) for name in test_files]

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    label = test_df["attack"].to_numpy(dtype=np.int64)
    train_timestamps = train_df["time"].tolist()
    test_timestamps = test_df["time"].tolist()

    exclude = {"attack", "attack_P1", "attack_P2", "attack_P3"}
    feature_cols = numeric_feature_columns(train_df, exclude)

    train_features, dropped_empty_train, constant_train = clean_dataframe(train_df, feature_cols)
    test_features = test_df.loc[:, train_features.columns].copy()

    train_split_df, val_split_df = split_train_val(train_features.to_numpy(dtype=np.float64), val_ratio)
    train_split_df = pd.DataFrame(train_split_df, columns=train_features.columns)
    val_split_df = pd.DataFrame(val_split_df, columns=train_features.columns)

    train, val, test, _ = fit_transform_splits(train_split_df, val_split_df, test_features)

    meta = {
        "dataset": dataset_name,
        "source_type": "multi_file_csv_gz",
        "train_files": train_files,
        "test_files": test_files,
        "feature_count": int(train.shape[1]),
        "time_column": "time",
        "label_column": "attack",
        "aux_label_columns": ["attack_P1", "attack_P2", "attack_P3"],
        "train_time_start": train_timestamps[0],
        "train_time_end": train_timestamps[-1],
        "test_time_start": test_timestamps[0],
        "test_time_end": test_timestamps[-1],
        "dropped_full_empty_columns": dropped_empty_train,
        "dropped_constant_columns": constant_train,
        "validation_rule": f"last_{int(val_ratio * 100)}pct_of_concatenated_train",
        "standardization": "fit_on_train_only",
        "segment_lengths": {
            "train": {name: int(frame.shape[0]) for name, frame in zip(train_files, train_frames)},
            "test": {name: int(frame.shape[0]) for name, frame in zip(test_files, test_frames)},
        },
    }
    save_processed_dataset(dataset_name, train, val, test, label, meta)


def make_windows(array: np.ndarray, win_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    if win_size <= 0 or stride <= 0:
        raise ValueError("win_size and stride must be positive")
    if array.shape[0] < win_size:
        raise ValueError("array is shorter than win_size")

    windows = []
    end_indices = []
    for start in range(0, array.shape[0] - win_size + 1, stride):
        end = start + win_size
        windows.append(array[start:end])
        end_indices.append(end - 1)
    return np.asarray(windows, dtype=np.float32), np.asarray(end_indices, dtype=np.int64)


def parse_args():
    parser = argparse.ArgumentParser(description="Build unified processed datasets.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["SMAP", "MSL", "HAI21.03"],
        help="Datasets to process. Supported: SMAP, MSL, HAI21.03, all",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    requested = args.dataset
    if any(name.lower() == "all" for name in requested):
        datasets = ["SMAP", "MSL", "HAI21.03"]
    else:
        datasets = [normalize_name(name) for name in requested]

    for dataset_name in datasets:
        print(f"processing {dataset_name}")
        if dataset_name in {"SMAP", "MSL"}:
            build_smap_or_msl(dataset_name, args.val_ratio)
        elif dataset_name == "HAI21.03":
            build_hai(args.val_ratio)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")


if __name__ == "__main__":
    main()
