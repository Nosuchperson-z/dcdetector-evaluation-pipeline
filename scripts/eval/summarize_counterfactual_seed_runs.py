import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLE_ROOT = ROOT / "outputs" / "tables"

MODE_CONFIG = {
    "v1": {
        "pattern": "results_counterfactual_v1_seed*.csv",
        "output": "outputs/tables/results_counterfactual_v1_seed_summary.csv",
        "group_cols": [
            "Dataset",
            "Score Source",
            "Selection Mode",
            "Threshold Selection Length",
            "Evaluation Offset",
        ],
    },
    "v2": {
        "pattern": "results_counterfactual_v2_seed*.csv",
        "output": "outputs/tables/results_counterfactual_v2_seed_summary.csv",
        "group_cols": [
            "Dataset",
            "Score Source",
            "Selection Mode",
            "Threshold Selection Length",
            "Evaluation Offset",
        ],
    },
    "v3": {
        "pattern": "results_counterfactual_v3_seed*.csv",
        "output": "outputs/tables/results_counterfactual_v3_seed_summary.csv",
        "group_cols": [
            "Ablation",
            "Dataset",
            "Contribution Method",
            "Random Baseline Modes",
            "Deletion Method",
            "Selection Mode",
            "Threshold Selection Length",
            "Evaluation Offset",
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize multi-seed counterfactual result CSVs.")
    parser.add_argument("--mode", choices=sorted(MODE_CONFIG.keys()), required=True)
    parser.add_argument("--input-pattern", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def infer_seed_from_name(path: Path) -> int | None:
    match = re.search(r"_seed(\d+)", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def load_seed_runs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype=str)
        if frame.empty:
            continue
        if "Event Sampling Seed" not in frame.columns:
            seed = infer_seed_from_name(path)
            frame["Event Sampling Seed"] = "" if seed is None else str(seed)
        frame["Source File"] = path.name
        frames.append(frame)
    if not frames:
        raise RuntimeError("No non-empty result file was found.")
    return pd.concat(frames, ignore_index=True)


def normalize_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.replace(
            {
                "NA": np.nan,
                "inf": np.nan,
                "-inf": np.nan,
                "": np.nan,
            }
        ),
        errors="coerce",
    )


def summarize(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    cfg = MODE_CONFIG[mode]
    group_cols = [col for col in cfg["group_cols"] if col in df.columns]
    protected_cols = set(group_cols) | {"Event Sampling Seed", "Source File"}

    numeric_cols = []
    for col in df.columns:
        if col in protected_cols:
            continue
        converted = normalize_numeric(df[col])
        if converted.notna().any():
            df[col] = converted
            numeric_cols.append(col)

    grouped_rows = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        seeds = sorted({str(seed) for seed in group["Event Sampling Seed"].dropna().tolist() if str(seed) != ""})
        row["Run Count"] = int(len(group))
        row["Seed List"] = ",".join(seeds)
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col} Mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{col} Std"] = float(values.std(ddof=0)) if values.notna().any() else np.nan
        grouped_rows.append(row)
    if not grouped_rows:
        raise RuntimeError("No grouped summary row was produced.")
    return pd.DataFrame(grouped_rows)


def main():
    args = parse_args()
    cfg = MODE_CONFIG[args.mode]
    pattern = args.input_pattern or cfg["pattern"]
    output_path = ROOT / (args.output or cfg["output"])
    input_paths = [
        path
        for path in sorted(TABLE_ROOT.glob(pattern))
        if path.name != output_path.name and not path.name.endswith("_seed_summary.csv")
    ]
    if not input_paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    summary = summarize(load_seed_runs(input_paths), args.mode)
    summary.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
