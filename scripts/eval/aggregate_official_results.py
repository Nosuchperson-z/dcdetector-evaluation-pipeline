import csv
import re
from pathlib import Path

import numpy as np

from unified_evaluator import (
    aggregate_window_point_scores_to_points,
    apply_smoothing,
    evaluate_point_scores,
    threshold_scores,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "KDD2023-DCdetector" / "result"
TABLE_DIR = ROOT / "outputs" / "tables"
PROCESSED_ROOT = ROOT / "data_processed"
SCORE_ROOT = ROOT / "outputs" / "scores"

DATASET_ORDER = ["SMAP", "MSL", "HAI21.03"]
CSV_COLUMNS = [
    "index",
    "pa_accuracy",
    "pa_precision",
    "pa_recall",
    "pa_f_score",
    "mcc_score",
    "affiliation_precision",
    "affiliation_recall",
    "r_auc_roc",
    "r_auc_pr",
    "vus_roc",
    "vus_pr",
]
LOG_PATTERNS = {
    "threshold": re.compile(r"Threshold\s*:\s*([0-9eE.+-]+)"),
    "anormly_ratio": re.compile(r"anormly_ratio:\s*([0-9eE.+-]+)"),
    "epoch_seconds": re.compile(r"Epoch:\s*\d+,\s*Cost time:\s*([0-9eE.+-]+)s"),
}


def read_last_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = [row for row in csv.reader(fp) if row]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    last = rows[-1]
    if len(last) != len(CSV_COLUMNS):
        raise ValueError(f"Unexpected column count in {path}: {len(last)}")
    return dict(zip(CSV_COLUMNS, last))


def parse_last_match(text: str, pattern: re.Pattern[str]) -> str | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    return matches[-1]


def parse_log_metrics(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    parsed = {}
    for key, pattern in LOG_PATTERNS.items():
        parsed[key] = parse_last_match(text, pattern)
    return parsed


def format_metric(value: str | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    value_lower = value.lower()
    if value_lower == "nan":
        return "NA"
    return f"{float(value):.{digits}f}"


def build_rows(dataset: str) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    csv_metrics = read_last_csv_row(RESULT_DIR / f"{dataset}.csv")
    log_metrics = parse_log_metrics(RESULT_DIR / f"{dataset}.log")
    unified_metrics = compute_unified_metrics(dataset)

    threshold_type = "NA"
    if log_metrics["anormly_ratio"] is not None:
        percentile = 100.0 - float(log_metrics["anormly_ratio"])
        threshold_type = f"energy_percentile_{percentile:.2f}"

    raw_row = {
        "Dataset": dataset,
        "Threshold": log_metrics["threshold"] or "NA",
        "PA-Accuracy": csv_metrics["pa_accuracy"],
        "PA-Precision": csv_metrics["pa_precision"],
        "PA-Recall": csv_metrics["pa_recall"],
        "PA-F1": csv_metrics["pa_f_score"],
        "MCC": csv_metrics["mcc_score"],
        "Affiliation precision": csv_metrics["affiliation_precision"],
        "Affiliation recall": csv_metrics["affiliation_recall"],
        "R-AUC-ROC": csv_metrics["r_auc_roc"],
        "R-AUC-PR": csv_metrics["r_auc_pr"],
        "VUS-ROC": csv_metrics["vus_roc"],
        "VUS-PR": csv_metrics["vus_pr"],
        "Anormly ratio": log_metrics["anormly_ratio"] or "NA",
        "Epoch seconds": log_metrics["epoch_seconds"] or "NA",
        "Unified Threshold": unified_metrics["threshold"],
        "Unified F1": unified_metrics["f1"],
        "Unified Fc1": unified_metrics["fc1"],
        "Unified Delay": unified_metrics["delay"],
    }

    main_row = {
        "Dataset": dataset,
        "AUPR": format_metric(csv_metrics["r_auc_pr"]),
        "AUROC": format_metric(csv_metrics["r_auc_roc"]),
        "F1": format_metric(csv_metrics["pa_f_score"]),
        "Fc1": format_metric(unified_metrics["fc1"]),
        "Delay": format_metric(unified_metrics["delay"]),
        "VUS-PR": format_metric(csv_metrics["vus_pr"]),
        "VUS-ROC": format_metric(csv_metrics["vus_roc"]),
    }

    appendix_row = {
        "Dataset": dataset,
        "PA-F1": format_metric(csv_metrics["pa_f_score"]),
        "Best Threshold": format_metric(log_metrics["threshold"], digits=6),
        "Threshold Type": threshold_type,
        "Smoothing": "none",
    }
    return raw_row, main_row, appendix_row


def resolve_score_file(dataset: str, filename: str) -> Path:
    dataset_dir = SCORE_ROOT / dataset
    direct = dataset_dir / filename
    if direct.exists():
        return direct
    matches = sorted(dataset_dir.glob(filename.replace(".npy", "*.npy")))
    if not matches:
        raise FileNotFoundError(f"Missing score file for {dataset}: {filename}")
    return matches[0]


def compute_unified_metrics(dataset: str) -> dict[str, str]:
    labels = np.load(PROCESSED_ROOT / dataset / "label.npy").astype(np.int64)
    window_point_scores = np.load(resolve_score_file(dataset, "test_window_point_scores.npy")).astype(np.float32)
    window_end_indices = np.load(resolve_score_file(dataset, "test_window_end_indices.npy")).astype(np.int64).reshape(-1)

    effective_length = min(window_point_scores.shape[0], window_end_indices.shape[0])
    window_point_scores = window_point_scores[:effective_length]
    window_end_indices = window_end_indices[:effective_length]

    point_scores = aggregate_window_point_scores_to_points(
        window_point_scores=window_point_scores,
        window_end_indices=window_end_indices,
        total_length=len(labels),
        method="mean",
    )
    point_scores = apply_smoothing(point_scores, "none", 0.0)
    threshold = threshold_scores(point_scores, "best_f1", labels=labels, param=None)
    results = evaluate_point_scores(point_scores, labels, threshold)
    return {
        "threshold": format_metric(str(results["threshold"]), digits=6),
        "f1": format_metric(str(results["f1"])),
        "fc1": format_metric(str(results["fc1"])),
        "delay": format_metric(str(results["delay"])),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    raw_rows = []
    main_rows = []
    appendix_rows = []
    for dataset in DATASET_ORDER:
        raw_row, main_row, appendix_row = build_rows(dataset)
        raw_rows.append(raw_row)
        main_rows.append(main_row)
        appendix_rows.append(appendix_row)

    write_csv(
        TABLE_DIR / "official_raw_metrics.csv",
        list(raw_rows[0].keys()),
        raw_rows,
    )
    write_csv(
        TABLE_DIR / "baseline_main_table.csv",
        ["Dataset", "AUPR", "AUROC", "F1", "Fc1", "Delay", "VUS-PR", "VUS-ROC"],
        main_rows,
    )
    write_csv(
        TABLE_DIR / "baseline_appendix_table.csv",
        ["Dataset", "PA-F1", "Best Threshold", "Threshold Type", "Smoothing"],
        appendix_rows,
    )
    print(TABLE_DIR / "official_raw_metrics.csv")
    print(TABLE_DIR / "baseline_main_table.csv")
    print(TABLE_DIR / "baseline_appendix_table.csv")


if __name__ == "__main__":
    main()
