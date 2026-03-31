import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
KDD_ROOT = ROOT / "KDD2023-DCdetector"
DOC_PATH = ROOT / "docs" / "score_coverage_fix_report.md"
OLD_SCORE_ROOT = ROOT / "outputs" / "scores"
NEW_SCORE_ROOT = ROOT / "outputs" / "scores_full_coverage"
PROCESSED_ROOT = ROOT / "data_processed"

if str(KDD_ROOT) not in sys.path:
    sys.path.insert(0, str(KDD_ROOT))

from solver import Solver  # noqa: E402


DATASET_CONFIGS = {
    "SMAP": {
        "index": 137,
        "dataset": "SMAP",
        "data_path": "SMAP",
        "win_size": 105,
        "patch_size": [3, 5, 7],
        "batch_size": 8,
        "input_c": 25,
        "output_c": 25,
        "anormly_ratio": 0.85,
        "loss_fuc": "MSE",
    },
    "MSL": {
        "index": 137,
        "dataset": "MSL",
        "data_path": "MSL",
        "win_size": 90,
        "patch_size": [3, 5],
        "batch_size": 8,
        "input_c": 55,
        "output_c": 55,
        "anormly_ratio": 1.0,
        "loss_fuc": "MSE",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export full-coverage DCdetector score artifacts.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["SMAP", "MSL"],
        choices=sorted(DATASET_CONFIGS.keys()),
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        default=True,
        help="Write coverage fix markdown report after export.",
    )
    return parser.parse_args()


def build_solver_config(dataset: str) -> dict:
    base = DATASET_CONFIGS[dataset].copy()
    base.update(
        {
            "lr": 1e-4,
            "n_heads": 1,
            "e_layers": 3,
            "d_model": 256,
            "rec_timeseries": True,
            "use_gpu": True,
            "gpu": 0,
            "use_multi_gpu": True,
            "devices": "0,1,2,3",
            "num_epochs": 1,
            "mode": "test",
            "model_save_path": "checkpoints",
            "train_stride": 1,
            "test_stride": 1,
            "score_output_dir": "../outputs/scores",
            "export_full_coverage": True,
            "full_coverage_score_output_dir": "../outputs/scores_full_coverage",
            "k": 3,
        }
    )
    return base


def expected_window_count(label_length: int, win_size: int, step: int) -> int:
    return (label_length - win_size) // step + 1


def covered_length_from_count(window_count: int, win_size: int, step: int, label_length: int) -> int:
    if window_count <= 0:
        return 0
    return int(min(win_size + (window_count - 1) * step, label_length))


def load_score_meta(score_root: Path, dataset: str, label_length: int, win_size: int, step: int) -> dict:
    meta_path = score_root / dataset / "score_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["label_length"] = int(meta.get("label_length", label_length))
    meta["covered_length"] = int(
        meta.get(
            "covered_length",
            covered_length_from_count(
                int(meta["window_count"]),
                win_size=win_size,
                step=step,
                label_length=label_length,
            ),
        )
    )
    return meta


def sample_window_rows(dataset: str, score_root: Path, labels: np.ndarray, win_size: int) -> list[dict]:
    end_indices = np.load(score_root / dataset / "test_window_end_indices.npy").astype(np.int64).reshape(-1)
    if end_indices.size == 0:
        return []
    sample_indices = sorted({0, int(end_indices.size / 2), int(end_indices.size - 1)})
    rows = []
    for window_idx in sample_indices:
        end_idx = int(end_indices[window_idx])
        start_idx = end_idx - win_size + 1
        label_slice = labels[start_idx:end_idx + 1]
        rows.append(
            {
                "window_index": window_idx,
                "end_index": end_idx,
                "point_interval": f"[{start_idx}, {end_idx}]",
                "label_interval": f"[{start_idx}, {end_idx}]",
                "label_positive_count": int(label_slice.sum()),
                "label_length": int(label_slice.shape[0]),
            }
        )
    return rows


def build_dataset_report(dataset: str) -> str:
    cfg = DATASET_CONFIGS[dataset]
    labels = np.load(PROCESSED_ROOT / dataset / "label.npy").astype(np.int64)
    label_length = int(labels.shape[0])
    step = 1
    expected_windows = expected_window_count(label_length, cfg["win_size"], step)
    old_meta = load_score_meta(OLD_SCORE_ROOT, dataset, label_length, cfg["win_size"], step)
    new_meta = load_score_meta(NEW_SCORE_ROOT, dataset, label_length, cfg["win_size"], step)
    new_point_scores = np.load(NEW_SCORE_ROOT / dataset / "test_point_scores.npy").astype(np.float32)
    new_end_indices = np.load(NEW_SCORE_ROOT / dataset / "test_window_end_indices.npy").astype(np.int64).reshape(-1)
    strict_alignment = (
        new_point_scores.shape[0] == label_length
        and new_meta["covered_length"] == label_length
        and int(new_end_indices[-1]) == label_length - 1
    )

    lines = [
        f"## {dataset}",
        "",
        f"- label length: `{label_length}`",
        f"- expected overlapping windows: `{expected_windows}`",
        f"- coverage before fix: `{old_meta['covered_length']} / {label_length}`",
        f"- coverage after fix: `{new_meta['covered_length']} / {label_length}`",
        f"- window count before fix: `{old_meta['window_count']}`",
        f"- window count after fix: `{new_meta['window_count']}`",
        f"- point score length after fix: `{new_point_scores.shape[0]}`",
        f"- last end index after fix: `{int(new_end_indices[-1])}`",
        f"- strict point-label alignment after fix: `{'yes' if strict_alignment else 'no'}`",
        "",
        "### Sample windows after fix",
        "",
        "| window index | end index | point interval | label interval | label positive count | label length |",
        "| --- | ---: | --- | --- | ---: | ---: |",
    ]
    for row in sample_window_rows(dataset, NEW_SCORE_ROOT, labels, cfg["win_size"]):
        lines.append(
            f"| {row['window_index']} | {row['end_index']} | {row['point_interval']} | "
            f"{row['label_interval']} | {row['label_positive_count']} | {row['label_length']} |"
        )
    lines.extend(
        [
            "",
            "- 问题源头：官方 `get_loader_segment(..., drop_last=True)` 让 `test_loader` 和 `thre_loader` 在尾部不足一个 batch 时直接丢弃最后一批窗口。",
            "- 修复方式：新增 `export_full_coverage=True` 选项，仅为 full-coverage 导出构造 `drop_last=False` 的 `full_coverage_test_loader`，旧 baseline 路径保持不变。",
            "- 对 baseline / v1 / v2 的影响：旧结果仍然可复现，但其 `SMAP / MSL` 统一评测使用的是 covered-prefix 口径；今后若要做统一横向比较，推荐优先使用 `outputs/scores_full_coverage/SMAP` 和 `outputs/scores_full_coverage/MSL`。",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(datasets: list[str]) -> str:
    lines = [
        "# Score Coverage Fix Report",
        "",
        "本报告记录 `SMAP / MSL` 在 full-coverage score export 修复前后的覆盖率差异，以及该问题对统一评测的影响。",
        "",
        "## 结论摘要",
        "",
        "- 修复前：`test_loader` / `thre_loader` 由于 `drop_last=True` 丢弃尾部不足一个 batch 的窗口，导致 `SMAP / MSL` 的最后一小段 label 没有对应 score。",
        "- 修复后：新增 `full_coverage_test_loader(drop_last=False)` 并导出到 `outputs/scores_full_coverage/`，`point score length == label length`，尾部窗口完整覆盖。",
        "- 可比性建议：保留旧 baseline / v1 / v2 作为历史结果；后续若继续做统一评测和解释分析，推荐统一切换到 full-coverage score 源。",
        "",
    ]
    for dataset in datasets:
        lines.append(build_dataset_report(dataset))
    return "\n".join(lines)


def export_dataset(dataset: str) -> None:
    cfg = build_solver_config(dataset)
    solver = Solver(cfg)
    solver.export_full_coverage_scores()


def main() -> None:
    args = parse_args()
    os.chdir(KDD_ROOT)
    for dataset in args.datasets:
        print(f"exporting {dataset}")
        export_dataset(dataset)
    if args.write_report:
        DOC_PATH.write_text(build_report(args.datasets), encoding="utf-8")
        print(DOC_PATH)


if __name__ == "__main__":
    main()
