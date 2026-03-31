import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
KDD_ROOT = ROOT / "KDD2023-DCdetector"
PROCESSED_ROOT = ROOT / "data_processed"
RAW_ROOT = ROOT / "data_raw"
TABLE_ROOT = ROOT / "outputs" / "tables"

if str(KDD_ROOT) not in sys.path:
    sys.path.insert(0, str(KDD_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from solver import my_kl_loss  # noqa: E402
from model.DCdetector import DCdetector  # noqa: E402
from run_event_aware_v2 import (  # noqa: E402
    apply_event_aware_thresholds,
    build_point_scores,
    build_threshold_views,
    format_value,
    select_event_aware_v2_thresholds,
    write_csv,
    write_xlsx,
)
from unified_evaluator import (  # noqa: E402
    aggregate_window_point_scores_to_points,
    event_score,
    points_to_events,
)


MODEL_CONFIGS = {
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
    "HAI21.03": {
        "index": 137,
        "dataset": "HAI21.03",
        "data_path": "HAI21.03",
        "win_size": 100,
        "patch_size": [2, 4, 5, 10, 20, 25, 50],
        "batch_size": 4,
        "input_c": 57,
        "output_c": 57,
        "anormly_ratio": 1.0,
        "loss_fuc": "MSE",
    },
}


@dataclass
class AnalysisEvent:
    event: tuple[int, int]
    true_events: list[tuple[int, int]]
    source: str


@dataclass
class ModelBundle:
    model: torch.nn.Module
    device: torch.device
    win_size: int


def clip_event_to_local_segment(
    event: tuple[int, int],
    point_scores: np.ndarray,
    max_length: int,
) -> tuple[int, int]:
    start, end = event
    length = end - start + 1
    max_length = max(1, int(max_length))
    if length <= max_length:
        return event
    event_scores = point_scores[start:end + 1]
    peak_offset = int(np.argmax(event_scores))
    peak_index = start + peak_offset
    new_start = max(start, peak_index - max_length // 2)
    new_end = new_start + max_length - 1
    if new_end > end:
        new_end = end
        new_start = end - max_length + 1
    return int(new_start), int(new_end)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Counterfactual v1 analysis on saved score artifacts.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval" / "counterfactual_v1.json"))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--num-random-trials", type=int, default=None)
    parser.add_argument("--event-sampling-seed", type=int, default=None)
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--save-case-figures", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def resolve_output_tag(output_tag: str | None, cli_seed: int | None) -> str | None:
    if output_tag is not None and output_tag.strip():
        return output_tag.strip()
    if cli_seed is None:
        return None
    return f"seed{int(cli_seed)}"


def tagged_output_path(path: Path, output_tag: str | None) -> Path:
    if not output_tag:
        return path
    return path.with_name(f"{path.stem}_{output_tag}{path.suffix}")


def tagged_output_dir(path: Path, output_tag: str | None) -> Path:
    if not output_tag:
        return path
    return path.parent / f"{path.name}_{output_tag}"


def load_csv_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    return {row["Dataset"]: row for row in rows}


def resolve_score_root(dataset: str, config: dict) -> Path:
    override = config["score_source_overrides"].get(dataset)
    if override is None:
        raise KeyError(f"Missing score source override for {dataset}")
    return ROOT / override / dataset


def coverage_length(end_indices: np.ndarray, label_length: int) -> int:
    if end_indices.size == 0:
        return 0
    return int(min(int(end_indices.max()) + 1, label_length))


def load_score_inputs(dataset: str, config: dict) -> dict:
    score_root = resolve_score_root(dataset, config)
    labels = np.load(PROCESSED_ROOT / dataset / "label.npy").astype(np.int64)
    window_point_scores = np.load(score_root / "test_window_point_scores.npy").astype(np.float32)
    window_end_indices = np.load(score_root / "test_window_end_indices.npy").astype(np.int64).reshape(-1)
    effective = min(window_point_scores.shape[0], window_end_indices.shape[0])
    window_point_scores = window_point_scores[:effective]
    window_end_indices = window_end_indices[:effective]
    covered = coverage_length(window_end_indices, len(labels))
    labels_aligned = labels[:covered] if config["trim_to_coverage"] else labels
    return {
        "score_root": score_root,
        "labels": labels_aligned,
        "full_labels": labels,
        "window_point_scores": window_point_scores,
        "window_end_indices": window_end_indices,
        "covered_length": covered,
    }


def load_feature_names(dataset: str) -> list[str]:
    if dataset in {"SMAP", "MSL"}:
        raw_test = np.load(KDD_ROOT / "dataset" / dataset / f"{dataset}_test.npy", mmap_mode="r")
        feature_count = int(raw_test.shape[1])
        return [f"feature_{idx}" for idx in range(feature_count)]

    meta = load_json(PROCESSED_ROOT / dataset / "meta.json")
    feature_count = int(meta["feature_count"])
    if dataset == "HAI21.03":
        sample_df = pd.read_csv(RAW_ROOT / "hai" / "hai-21.03" / "train1.csv.gz", compression="gzip", nrows=1, low_memory=False)
        exclude = {"time", "attack", "attack_P1", "attack_P2", "attack_P3"}
        cols = [col for col in sample_df.columns if col not in exclude]
        dropped = set(meta.get("dropped_full_empty_columns", [])) | set(meta.get("dropped_constant_columns", []))
        cols = [col for col in cols if col not in dropped]
        if len(cols) == feature_count:
            return cols

    return [f"feature_{idx}" for idx in range(feature_count)]


def load_model_space_arrays(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset in {"SMAP", "MSL"}:
        train_data = np.load(KDD_ROOT / "dataset" / dataset / f"{dataset}_train.npy", mmap_mode="r")
        test_data = np.load(KDD_ROOT / "dataset" / dataset / f"{dataset}_test.npy", mmap_mode="r")
        return train_data, test_data
    train_data = np.load(PROCESSED_ROOT / dataset / "train.npy", mmap_mode="r")
    test_data = np.load(PROCESSED_ROOT / dataset / "test.npy", mmap_mode="r")
    return train_data, test_data


def compute_train_stats(train_data: np.ndarray) -> dict[str, np.ndarray]:
    train_array = np.asarray(train_data, dtype=np.float32)
    mean = train_array.mean(axis=0).astype(np.float32)
    std = np.clip(train_array.std(axis=0).astype(np.float32), 1e-6, None)
    median = np.median(train_array, axis=0).astype(np.float32)
    mad = np.median(np.abs(train_array - median.reshape(1, -1)), axis=0).astype(np.float32)
    mad = np.clip(mad, 1e-6, None)
    return {"mean": mean, "std": std, "median": median, "mad": mad}


def overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def select_analysis_events(
    pred_events: list[tuple[int, int]],
    true_events: list[tuple[int, int]],
    point_scores: np.ndarray,
    max_events: int | None,
    mode: str = "all_predicted",
    num_strata: int = 4,
    random_seed: int = 20260322,
) -> list[AnalysisEvent]:
    selected = []
    ranked_pred_events = sorted(pred_events, key=lambda event: event_score(point_scores, event, pooling="mean"), reverse=True)
    if mode == "all_predicted":
        for event in ranked_pred_events:
            overlapping_true = [true_event for true_event in true_events if overlap(event, true_event)]
            source = "predicted_overlap" if overlapping_true else "predicted_no_overlap"
            selected.append(AnalysisEvent(event=event, true_events=overlapping_true, source=source))
    elif mode == "matched_predicted_only":
        for event in ranked_pred_events:
            overlapping_true = [true_event for true_event in true_events if overlap(event, true_event)]
            if overlapping_true:
                selected.append(AnalysisEvent(event=event, true_events=overlapping_true, source="predicted_overlap"))
    elif mode == "oracle_fallback_true_event":
        for event in ranked_pred_events:
            overlapping_true = [true_event for true_event in true_events if overlap(event, true_event)]
            if overlapping_true:
                selected.append(AnalysisEvent(event=event, true_events=overlapping_true, source="predicted_overlap"))
        if not selected and true_events:
            fallback = max(true_events, key=lambda event: event_score(point_scores, event, pooling="mean"))
            selected.append(AnalysisEvent(event=fallback, true_events=[fallback], source="fallback_true_event"))
    elif mode == "score_stratified":
        for event in ranked_pred_events:
            overlapping_true = [true_event for true_event in true_events if overlap(event, true_event)]
            source = "predicted_overlap" if overlapping_true else "predicted_no_overlap"
            selected.append(AnalysisEvent(event=event, true_events=overlapping_true, source=source))
    else:
        raise ValueError(f"Unsupported event selection mode: {mode}")

    if max_events is None:
        return selected
    max_events = max(0, int(max_events))
    if len(selected) <= max_events:
        return selected
    if mode != "score_stratified":
        return selected[:max_events]

    num_strata = max(1, min(int(num_strata), len(selected), max_events))
    bounds = np.linspace(0, len(selected), num=num_strata + 1, dtype=np.int64)
    rng = np.random.default_rng(int(random_seed))
    chosen_indices = []
    remaining_budget = max_events

    for stratum_idx in range(num_strata):
        start = int(bounds[stratum_idx])
        end = int(bounds[stratum_idx + 1])
        candidates = np.arange(start, end, dtype=np.int64)
        if candidates.size == 0:
            continue
        remaining_strata = num_strata - stratum_idx
        target = max(1, remaining_budget // remaining_strata)
        target = min(target, int(candidates.size), remaining_budget)
        sampled = rng.choice(candidates, size=target, replace=False)
        chosen_indices.extend(int(idx) for idx in sampled.tolist())
        remaining_budget -= target
        if remaining_budget <= 0:
            break

    if remaining_budget > 0:
        already = set(chosen_indices)
        leftovers = [idx for idx in range(len(selected)) if idx not in already]
        if leftovers:
            extra = rng.choice(np.asarray(leftovers, dtype=np.int64), size=min(remaining_budget, len(leftovers)), replace=False)
            chosen_indices.extend(int(idx) for idx in extra.tolist())

    chosen_indices = sorted(set(chosen_indices))
    return [selected[idx] for idx in chosen_indices[:max_events]]


def restrict_events_to_range(
    events: list[tuple[int, int]],
    start_idx: int,
    end_idx: int,
) -> list[tuple[int, int]]:
    restricted = []
    for start, end in events:
        if end < start_idx or start > end_idx:
            continue
        restricted.append((max(start, start_idx), min(end, end_idx)))
    return restricted


def contribution_scores(
    event_segment: np.ndarray,
    event_point_scores: np.ndarray,
    train_stats: dict[str, np.ndarray],
    method: str = "score_weighted_abs_deviation_sum",
) -> np.ndarray:
    segment = np.asarray(event_segment, dtype=np.float64)
    if method == "score_weighted_abs_deviation_sum":
        deviations = np.abs(segment - train_stats["mean"].reshape(1, -1))
        weights = np.clip(np.asarray(event_point_scores, dtype=np.float64), a_min=1e-6, a_max=None)
        return (deviations * weights.reshape(-1, 1)).sum(axis=0)
    if method == "score_weighted_zscore_sum":
        deviations = np.abs((segment - train_stats["mean"].reshape(1, -1)) / train_stats["std"].reshape(1, -1))
        weights = np.clip(np.asarray(event_point_scores, dtype=np.float64), a_min=1e-6, a_max=None)
        return (deviations * weights.reshape(-1, 1)).sum(axis=0)
    if method == "score_weighted_robust_zscore_sum":
        deviations = np.abs((segment - train_stats["median"].reshape(1, -1)) / train_stats["mad"].reshape(1, -1))
        weights = np.clip(np.asarray(event_point_scores, dtype=np.float64), a_min=1e-6, a_max=None)
        return (deviations * weights.reshape(-1, 1)).sum(axis=0)
    if method == "tail_score_weighted_abs_deviation_sum":
        deviations = np.abs(segment - train_stats["mean"].reshape(1, -1))
        scores = np.asarray(event_point_scores, dtype=np.float64)
        tail_start = float(np.quantile(scores, 0.75)) if scores.size else 0.0
        weights = np.clip(scores - tail_start, a_min=0.0, a_max=None) ** 2
        if not np.any(weights > 0):
            weights = np.clip(scores - float(np.min(scores)), a_min=0.0, a_max=None)
        if not np.any(weights > 0):
            weights = np.ones_like(scores, dtype=np.float64)
        weights = np.clip(weights, a_min=1e-6, a_max=None)
        return (deviations * weights.reshape(-1, 1)).sum(axis=0)
    if method in {"plain_abs_deviation_sum", "abs_deviation_sum"}:
        deviations = np.abs(segment - train_stats["mean"].reshape(1, -1))
        return deviations.sum(axis=0)
    raise ValueError(f"Unsupported contribution method: {method}")


def exact_single_var_cf_contributions(
    bundle: ModelBundle,
    local_original: np.ndarray,
    local_event: tuple[int, int],
    event_segment: np.ndarray,
    reference_segment: np.ndarray,
    score_before: float,
    batch_size: int,
    pooling: str,
) -> np.ndarray:
    num_features = int(event_segment.shape[1])
    gains = np.zeros(num_features, dtype=np.float64)
    event_slice = slice(local_event[0], local_event[1] + 1)
    for var_idx in range(num_features):
        repaired_segment = repair_segment(event_segment, reference_segment, np.asarray([var_idx], dtype=np.int64))
        local_repaired = local_original.copy()
        local_repaired[event_slice] = repaired_segment
        score_after = compute_local_event_score(
            bundle=bundle,
            local_series=local_repaired,
            local_event=local_event,
            batch_size=batch_size,
            pooling=pooling,
        )
        gains[var_idx] = cf_gain(score_before, score_after)
    return gains


def topk_indices(contributions: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), int(contributions.shape[0])))
    return np.argsort(contributions)[::-1][:k]


def jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def topk_stability(
    event_segment: np.ndarray,
    event_point_scores: np.ndarray,
    train_stats: dict[str, np.ndarray],
    contribution_method: str,
    topk: int,
    local_window: int,
) -> float:
    if contribution_method == "exact_single_var_cf_gain":
        return float("nan")
    length = event_segment.shape[0]
    if length <= 1:
        return 1.0
    local_window = max(1, min(local_window, length))
    starts = list(range(0, max(length - local_window + 1, 1), max(1, local_window // 2)))
    last_start = max(0, length - local_window)
    if last_start not in starts:
        starts.append(last_start)
    starts = sorted(set(starts))

    top_sets = []
    for start in starts:
        end = start + local_window
        contrib = contribution_scores(
            event_segment[start:end],
            event_point_scores[start:end],
            train_stats,
            method=contribution_method,
        )
        top_sets.append(set(topk_indices(contrib, topk).tolist()))

    if len(top_sets) <= 1:
        return 1.0
    return float(np.mean([jaccard(lhs, rhs) for lhs, rhs in zip(top_sets[:-1], top_sets[1:])]))


def candidate_starts(train_length: int, window_length: int, stride: int, max_candidates: int) -> np.ndarray:
    if train_length < window_length:
        raise ValueError("train sequence is shorter than event length")
    starts = np.arange(0, train_length - window_length + 1, max(1, stride), dtype=np.int64)
    last_start = train_length - window_length
    if starts.size == 0 or starts[-1] != last_start:
        starts = np.append(starts, last_start)
    if starts.size > max_candidates:
        indices = np.linspace(0, starts.size - 1, num=max_candidates, dtype=np.int64)
        starts = starts[indices]
    return np.unique(starts)


def nearest_reference_window(
    train_data: np.ndarray,
    event_segment: np.ndarray,
    stride: int,
    max_candidates: int,
) -> tuple[np.ndarray, int, float]:
    window_length = event_segment.shape[0]
    starts = candidate_starts(train_data.shape[0], window_length, stride, max_candidates)
    best_start = 0
    best_dist = None
    for start in starts:
        candidate = train_data[start:start + window_length]
        dist = float(np.mean((candidate - event_segment) ** 2))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_start = int(start)
    return train_data[best_start:best_start + window_length].copy(), best_start, float(best_dist)


def make_overlapping_windows(sequence: np.ndarray, win_size: int) -> tuple[np.ndarray, np.ndarray]:
    if sequence.shape[0] < win_size:
        raise ValueError("sequence is shorter than win_size")
    starts = np.arange(0, sequence.shape[0] - win_size + 1, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(sequence, window_shape=win_size, axis=0)
    windows = np.moveaxis(windows, -1, 1).astype(np.float32, copy=False)
    end_indices = starts + win_size - 1
    return windows, end_indices


def compute_window_point_scores(model, device, windows: np.ndarray, win_size: int, batch_size: int) -> np.ndarray:
    temperature = 50
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, windows.shape[0], batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size]).float().to(device)
            series, prior = model(batch)
            series_loss = 0.0
            prior_loss = 0.0
            for level in range(len(prior)):
                normalized_prior = prior[level] / torch.unsqueeze(torch.sum(prior[level], dim=-1), dim=-1).repeat(
                    1, 1, 1, win_size
                )
                if level == 0:
                    series_loss = my_kl_loss(series[level], normalized_prior.detach()) * temperature
                    prior_loss = my_kl_loss(normalized_prior, series[level].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[level], normalized_prior.detach()) * temperature
                    prior_loss += my_kl_loss(normalized_prior, series[level].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            outputs.append(metric.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def local_context_bounds(event: tuple[int, int], total_length: int, win_size: int) -> tuple[int, int]:
    start, end = event
    context_start = max(0, start - (win_size - 1))
    context_end = min(total_length - 1, end + (win_size - 1))
    if context_end - context_start + 1 < win_size:
        context_start = max(0, context_end - win_size + 1)
        context_end = min(total_length - 1, context_start + win_size - 1)
    return context_start, context_end


def compute_local_event_score(
    bundle: ModelBundle,
    local_series: np.ndarray,
    local_event: tuple[int, int],
    batch_size: int,
    pooling: str,
) -> float:
    windows, end_indices = make_overlapping_windows(local_series, bundle.win_size)
    current_batch_size = max(1, int(batch_size))
    while True:
        try:
            window_point_scores = compute_window_point_scores(
                bundle.model,
                bundle.device,
                windows=windows,
                win_size=bundle.win_size,
                batch_size=current_batch_size,
            )
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            torch.cuda.empty_cache()
    local_point_scores = aggregate_window_point_scores_to_points(
        window_point_scores=window_point_scores,
        window_end_indices=end_indices,
        total_length=local_series.shape[0],
        method="mean",
    )
    start, end = local_event
    return float(event_score(local_point_scores, (start, end), pooling=pooling))


def repair_segment(event_segment: np.ndarray, reference_segment: np.ndarray, variable_indices: np.ndarray) -> np.ndarray:
    repaired = event_segment.copy()
    repaired[:, variable_indices] = reference_segment[:, variable_indices]
    return repaired


def cf_gain(score_before: float, score_after: float, epsilon: float = 1e-6) -> float:
    return float((score_before - score_after) / max(score_before, epsilon))


def build_model_bundle(dataset: str, requested_device: str = "auto") -> ModelBundle:
    cfg = MODEL_CONFIGS[dataset].copy()
    if requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DCdetector(
        win_size=cfg["win_size"],
        enc_in=cfg["input_c"],
        c_out=cfg["output_c"],
        n_heads=1,
        d_model=256,
        e_layers=3,
        patch_size=cfg["patch_size"],
        channel=cfg["input_c"],
    ).to(device)
    checkpoint = torch.load(KDD_ROOT / "checkpoints" / f"{dataset}_checkpoint.pth", map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return ModelBundle(model=model, device=device, win_size=int(cfg["win_size"]))


def plot_case_figure(
    path: Path,
    dataset: str,
    point_scores: np.ndarray,
    labels: np.ndarray,
    analysis_event: AnalysisEvent,
    case_row: dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    event_start, event_end = analysis_event.event
    margin = max(50, event_end - event_start + 1)
    plot_start = max(0, event_start - margin)
    plot_end = min(len(point_scores) - 1, event_end + margin)
    x = np.arange(plot_start, plot_end + 1)
    local_scores = point_scores[plot_start:plot_end + 1]
    local_labels = labels[plot_start:plot_end + 1]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [2.2, 1.6, 1.6]})
    axes[0].plot(x, local_scores, color="#1f4e79", linewidth=1.0, label="Point score")
    axes[0].axvspan(event_start, event_end, color="#b22222", alpha=0.15, label="Selected event")
    in_true = False
    true_start = 0
    for idx, value in zip(x, local_labels):
        if value == 1 and not in_true:
            true_start = idx
            in_true = True
        elif value == 0 and in_true:
            axes[0].axvspan(true_start, idx - 1, color="#f4c542", alpha=0.2)
            in_true = False
    if in_true:
        axes[0].axvspan(true_start, plot_end, color="#f4c542", alpha=0.2)
    axes[0].set_title(f"{dataset} counterfactual case")
    axes[0].set_xlabel("Time index")
    axes[0].set_ylabel("Score")
    axes[0].grid(alpha=0.2, linestyle="--")
    axes[0].legend(loc="upper right")

    axes[1].barh(np.arange(len(case_row["top5_variable_names"])), case_row["top5_contributions"], color="#2a7f62")
    axes[1].set_yticks(np.arange(len(case_row["top5_variable_names"])), labels=case_row["top5_variable_names"])
    axes[1].invert_yaxis()
    axes[1].set_title("Top suspicious variables")
    axes[1].set_xlabel("Contribution")
    axes[1].grid(alpha=0.2, axis="x", linestyle="--")

    bar_labels = ["before", "top1", "top3", "top5", "rand1", "rand3", "rand5"]
    bar_values = [
        case_row["score_before"],
        case_row["score_after_top1"],
        case_row["score_after_top3"],
        case_row["score_after_top5"],
        case_row["score_after_rand1"],
        case_row["score_after_rand3"],
        case_row["score_after_rand5"],
    ]
    axes[2].bar(np.arange(len(bar_labels)), bar_values, color=["#7f7f7f", "#1f4e79", "#1f4e79", "#1f4e79", "#b22222", "#b22222", "#b22222"])
    axes[2].set_xticks(np.arange(len(bar_labels)), labels=bar_labels)
    axes[2].set_title("Repair score comparison")
    axes[2].set_ylabel("Event score")
    axes[2].grid(alpha=0.2, axis="y", linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_dataset(
    dataset: str,
    cf_config: dict,
    v2_config: dict,
    bundle: ModelBundle,
    baseline_rows: dict[str, dict[str, str]],
    v2_rows: dict[str, dict[str, str]],
) -> tuple[dict, list[dict], dict | None]:
    loaded = load_score_inputs(dataset, cf_config)
    labels = loaded["labels"]
    window_point_scores = loaded["window_point_scores"]
    window_end_indices = loaded["window_end_indices"]
    score_root = loaded["score_root"]

    v2_cfg = v2_config["event_aware_v2"]
    point_scores = build_point_scores(
        labels,
        window_point_scores,
        window_end_indices,
        aggregation=v2_cfg["aggregation"],
        smoothing=v2_cfg["smoothing"],
        smoothing_param=v2_cfg["smoothing_param"],
    )
    threshold_views = build_threshold_views(
        dataset,
        labels,
        point_scores,
        v2_config.get("threshold_calibration", {"enabled": False}),
        v2_cfg,
        requested_device=str(bundle.device.type),
    )
    best_v2, _ = select_event_aware_v2_thresholds(
        threshold_views["selection_scores"],
        threshold_views["selection_labels"],
        v2_cfg,
        v2_config.get("threshold_calibration", {"enabled": False}),
    )
    applied_v2 = apply_event_aware_thresholds(
        point_scores=point_scores,
        total_length=len(labels),
        cfg=v2_cfg,
        point_threshold=best_v2["point_threshold"],
        event_threshold=best_v2["event_threshold"],
    )
    evaluation_offset = int(threshold_views["evaluation_offset"])
    evaluation_end = len(labels) - 1
    final_events = restrict_events_to_range(applied_v2["processed"]["final_events"], evaluation_offset, evaluation_end)
    true_events = restrict_events_to_range(points_to_events(labels), evaluation_offset, evaluation_end)
    analysis_events = select_analysis_events(
        pred_events=final_events,
        true_events=true_events,
        point_scores=point_scores,
        max_events=cf_config.get("max_events_per_dataset"),
        mode=cf_config.get("event_selection_mode", "all_predicted"),
        num_strata=int(cf_config.get("event_sampling_num_strata", 4)),
        random_seed=int(cf_config.get("event_sampling_seed", 20260322)),
    )

    train_data, test_data = load_model_space_arrays(dataset)
    if len(labels) < test_data.shape[0]:
        test_data = test_data[: len(labels)]
    train_stats = compute_train_stats(train_data)
    feature_names = load_feature_names(dataset)
    local_batch_size = int(cf_config["local_score_batch_size"])
    topk_list = [int(k) for k in cf_config["topk_list"]]

    detail_rows = []
    case_choice = None

    for idx, analysis_event in enumerate(analysis_events):
        original_event_start, original_event_end = analysis_event.event
        event_start, event_end = clip_event_to_local_segment(
            analysis_event.event,
            point_scores=point_scores,
            max_length=int(cf_config["max_event_analysis_length"]),
        )
        event_segment = np.asarray(test_data[event_start:event_end + 1], dtype=np.float32)
        event_point_scores = point_scores[event_start:event_end + 1]
        contribution_method = str(cf_config.get("contribution_method", "score_weighted_abs_deviation_sum"))
        reference_segment, reference_start, reference_dist = nearest_reference_window(
            train_data=train_data,
            event_segment=event_segment,
            stride=int(cf_config["candidate_window_stride"]),
            max_candidates=int(cf_config["max_reference_candidates"]),
        )

        context_start, context_end = local_context_bounds(analysis_event.event, len(test_data), bundle.win_size)
        local_original = np.asarray(test_data[context_start:context_end + 1], dtype=np.float32).copy()
        local_event = (event_start - context_start, event_end - context_start)
        score_before = compute_local_event_score(
            bundle=bundle,
            local_series=local_original,
            local_event=local_event,
            batch_size=local_batch_size,
            pooling=cf_config["counterfactual_score_pooling"],
        )
        if contribution_method == "exact_single_var_cf_gain":
            contrib = exact_single_var_cf_contributions(
                bundle=bundle,
                local_original=local_original,
                local_event=local_event,
                event_segment=event_segment,
                reference_segment=reference_segment,
                score_before=score_before,
                batch_size=local_batch_size,
                pooling=cf_config["counterfactual_score_pooling"],
            )
        else:
            contrib = contribution_scores(event_segment, event_point_scores, train_stats, method=contribution_method)

        gains = {}
        random_gains = {}
        after_scores = {}
        random_after_scores = {}
        for k in topk_list:
            vars_idx = topk_indices(contrib, k)
            repaired_segment = repair_segment(event_segment, reference_segment, vars_idx)
            local_repaired = local_original.copy()
            local_repaired[local_event[0]:local_event[1] + 1] = repaired_segment
            score_after = compute_local_event_score(
                bundle=bundle,
                local_series=local_repaired,
                local_event=local_event,
                batch_size=local_batch_size,
                pooling=cf_config["counterfactual_score_pooling"],
            )
            after_scores[k] = float(score_after)
            gains[k] = cf_gain(score_before, score_after)

            trial_gains = []
            trial_scores = []
            for trial in range(int(cf_config["num_random_trials"])):
                random_state = np.random.default_rng(seed=20260320 + idx * 97 + k * 13 + trial)
                rand_idx = np.sort(random_state.choice(event_segment.shape[1], size=min(k, event_segment.shape[1]), replace=False))
                rand_repaired_segment = repair_segment(event_segment, reference_segment, rand_idx)
                local_rand = local_original.copy()
                local_rand[local_event[0]:local_event[1] + 1] = rand_repaired_segment
                rand_score_after = compute_local_event_score(
                    bundle=bundle,
                    local_series=local_rand,
                    local_event=local_event,
                    batch_size=local_batch_size,
                    pooling=cf_config["counterfactual_score_pooling"],
                )
                trial_scores.append(float(rand_score_after))
                trial_gains.append(cf_gain(score_before, rand_score_after))
            random_gains[k] = float(np.mean(trial_gains))
            random_after_scores[k] = float(np.mean(trial_scores))

        stability = topk_stability(
            event_segment=event_segment,
            event_point_scores=event_point_scores,
            train_stats=train_stats,
            contribution_method=contribution_method,
            topk=int(cf_config["stability_topk"]),
            local_window=min(bundle.win_size, event_segment.shape[0]),
        )

        order = np.argsort(contrib)[::-1]
        top5_idx = order[: min(5, len(order))]
        row = {
            "Dataset": dataset,
            "Event Index": idx,
            "Original Event Start": original_event_start,
            "Original Event End": original_event_end,
            "Original Event Length": int(original_event_end - original_event_start + 1),
            "Analyzed Event Start": event_start,
            "Analyzed Event End": event_end,
            "Analyzed Event Length": int(event_end - event_start + 1),
            "Matched True Events": "; ".join(f"[{start},{end}]" for start, end in analysis_event.true_events),
            "Event Source": analysis_event.source,
            "Reference Start": int(reference_start),
            "Reference Distance": float(reference_dist),
            "Contribution Method": contribution_method,
            "Score Before": float(score_before),
            "Top1 Gain": float(gains.get(1, np.nan)),
            "Top3 Gain": float(gains.get(3, np.nan)),
            "Top5 Gain": float(gains.get(5, np.nan)),
            "Top1 Random Gain": float(random_gains.get(1, np.nan)),
            "Top3 Random Gain": float(random_gains.get(3, np.nan)),
            "Top5 Random Gain": float(random_gains.get(5, np.nan)),
            "Top3 Stability": float(stability),
            "Top5 Variables": ", ".join(feature_names[int(var_idx)] for var_idx in top5_idx),
        }
        detail_rows.append(row)

        case_row = {
            "dataset": dataset,
            "event": analysis_event.event,
            "score_before": float(score_before),
            "score_after_top1": float(after_scores.get(1, np.nan)),
            "score_after_top3": float(after_scores.get(3, np.nan)),
            "score_after_top5": float(after_scores.get(5, np.nan)),
            "score_after_rand1": float(random_after_scores.get(1, np.nan)),
            "score_after_rand3": float(random_after_scores.get(3, np.nan)),
            "score_after_rand5": float(random_after_scores.get(5, np.nan)),
            "top5_variable_names": [feature_names[int(var_idx)] for var_idx in top5_idx],
            "top5_contributions": [float(contrib[int(var_idx)]) for var_idx in top5_idx],
            "gain_margin": float(gains.get(3, np.nan) - random_gains.get(3, np.nan)),
        }
        if case_choice is None or case_row["gain_margin"] > case_choice["gain_margin"]:
            case_choice = case_row

        torch.cuda.empty_cache()

    top1_gain = float(np.mean([row["Top1 Gain"] for row in detail_rows])) if detail_rows else float("nan")
    top3_gain = float(np.mean([row["Top3 Gain"] for row in detail_rows])) if detail_rows else float("nan")
    top5_gain = float(np.mean([row["Top5 Gain"] for row in detail_rows])) if detail_rows else float("nan")
    top1_random_gain = float(np.mean([row["Top1 Random Gain"] for row in detail_rows])) if detail_rows else float("nan")
    top3_random_gain = float(np.mean([row["Top3 Random Gain"] for row in detail_rows])) if detail_rows else float("nan")
    top5_random_gain = float(np.mean([row["Top5 Random Gain"] for row in detail_rows])) if detail_rows else float("nan")
    top3_stability = float(np.mean([row["Top3 Stability"] for row in detail_rows])) if detail_rows else float("nan")

    baseline_row = baseline_rows[dataset]
    v2_row = v2_rows[dataset]
    summary_row = {
        "Dataset": dataset,
        "Score Source": str(score_root.relative_to(ROOT)),
        "Total Predicted Event Count": len(final_events),
        "Analyzed Event Count": len(detail_rows),
        "Selection Mode": cf_config.get("event_selection_mode", "all_predicted"),
        "Contribution Method": str(cf_config.get("contribution_method", "score_weighted_abs_deviation_sum")),
        "Threshold Selection Length": int(threshold_views["selection_length"]),
        "Evaluation Offset": evaluation_offset,
        "Top1 CF Gain": top1_gain,
        "Top3 CF Gain": top3_gain,
        "Top5 CF Gain": top5_gain,
        "Top1 Random Gain": top1_random_gain,
        "Top3 Random Gain": top3_random_gain,
        "Top5 Random Gain": top5_random_gain,
        "Top3 Stability": top3_stability,
        "Baseline Unified F1": float(baseline_row["baseline_unified_f1"]),
        "Baseline Unified Fc1": float(baseline_row["baseline_unified_fc1"]),
        "Baseline Unified Delay": float(baseline_row["baseline_unified_delay"]),
        "Event-aware v2 Unified F1": float(v2_row["v2_unified_f1"]),
        "Event-aware v2 Unified Fc1": float(v2_row["v2_unified_fc1"]),
        "Event-aware v2 Unified Delay": float(v2_row["v2_unified_delay"]),
    }
    return summary_row, detail_rows, case_choice


def main():
    args = parse_args()
    cf_config = load_json(args.config)
    if args.datasets:
        cf_config["datasets"] = args.datasets
    if args.max_events is not None:
        cf_config["max_events_per_dataset"] = int(args.max_events)
    if args.num_random_trials is not None:
        cf_config["num_random_trials"] = int(args.num_random_trials)
    if args.event_sampling_seed is not None:
        cf_config["event_sampling_seed"] = int(args.event_sampling_seed)
    save_case_figures_override = parse_optional_bool(args.save_case_figures)
    if save_case_figures_override is not None:
        cf_config["save_case_figures"] = save_case_figures_override
    output_tag = resolve_output_tag(args.output_tag, args.event_sampling_seed)
    v2_config = load_json(ROOT / cf_config["event_config"])
    baseline_rows = load_csv_rows(TABLE_ROOT / "event_aware_v2_comparison.csv")
    v2_rows = load_csv_rows(TABLE_ROOT / "event_aware_v2_comparison.csv")

    analysis_dir = tagged_output_dir(ROOT / cf_config["analysis_dir"], output_tag)
    results_csv = tagged_output_path(ROOT / cf_config["results_csv"], output_tag)
    comparison_csv = tagged_output_path(ROOT / cf_config["comparison_csv"], output_tag)
    results_xlsx = tagged_output_path(ROOT / cf_config["results_xlsx"], output_tag)
    event_details_csv = tagged_output_path(ROOT / cf_config["event_details_csv"], output_tag)

    summary_rows = []
    comparison_rows = []
    detail_rows = []

    for dataset in cf_config["datasets"]:
        print(f"processing {dataset}")
        bundle = build_model_bundle(dataset, requested_device=args.device)
        summary_row, dataset_detail_rows, case_choice = analyze_dataset(
            dataset=dataset,
            cf_config=cf_config,
            v2_config=v2_config,
            bundle=bundle,
            baseline_rows=baseline_rows,
            v2_rows=v2_rows,
        )
        summary_rows.append(
            {
                "Dataset": dataset,
                "Score Source": summary_row["Score Source"],
                "Total Predicted Event Count": summary_row["Total Predicted Event Count"],
                "Analyzed Event Count": summary_row["Analyzed Event Count"],
                "Selection Mode": summary_row["Selection Mode"],
                "Contribution Method": summary_row["Contribution Method"],
                "Event Sampling Seed": int(cf_config.get("event_sampling_seed", 20260322)),
                "Threshold Selection Length": summary_row["Threshold Selection Length"],
                "Evaluation Offset": summary_row["Evaluation Offset"],
                "Top1 CF Gain": format_value(summary_row["Top1 CF Gain"]),
                "Top3 CF Gain": format_value(summary_row["Top3 CF Gain"]),
                "Top5 CF Gain": format_value(summary_row["Top5 CF Gain"]),
                "Top1 Random Gain": format_value(summary_row["Top1 Random Gain"]),
                "Top3 Random Gain": format_value(summary_row["Top3 Random Gain"]),
                "Top5 Random Gain": format_value(summary_row["Top5 Random Gain"]),
                "Top3 Stability": format_value(summary_row["Top3 Stability"]),
            }
        )
        comparison_rows.append(
            {
                "Dataset": dataset,
                "baseline_unified_f1": format_value(summary_row["Baseline Unified F1"]),
                "event_aware_v2_unified_f1": format_value(summary_row["Event-aware v2 Unified F1"]),
                "baseline_unified_fc1": format_value(summary_row["Baseline Unified Fc1"]),
                "event_aware_v2_unified_fc1": format_value(summary_row["Event-aware v2 Unified Fc1"]),
                "baseline_unified_delay": format_value(summary_row["Baseline Unified Delay"]),
                "event_aware_v2_unified_delay": format_value(summary_row["Event-aware v2 Unified Delay"]),
                "total_predicted_event_count": summary_row["Total Predicted Event Count"],
                "analyzed_event_count": summary_row["Analyzed Event Count"],
                "selection_mode": summary_row["Selection Mode"],
                "contribution_method": summary_row["Contribution Method"],
                "event_sampling_seed": int(cf_config.get("event_sampling_seed", 20260322)),
                "top1_cf_gain": format_value(summary_row["Top1 CF Gain"]),
                "top3_cf_gain": format_value(summary_row["Top3 CF Gain"]),
                "top5_cf_gain": format_value(summary_row["Top5 CF Gain"]),
                "top3_random_gain": format_value(summary_row["Top3 Random Gain"]),
                "topk_stability": format_value(summary_row["Top3 Stability"]),
            }
        )
        for row in dataset_detail_rows:
            row["Event Sampling Seed"] = int(cf_config.get("event_sampling_seed", 20260322))
        detail_rows.extend(dataset_detail_rows)

        if case_choice is not None and cf_config["save_case_figures"]:
            loaded = load_score_inputs(dataset, cf_config)
            point_scores = build_point_scores(
                loaded["labels"],
                loaded["window_point_scores"],
                loaded["window_end_indices"],
                aggregation=v2_config["event_aware_v2"]["aggregation"],
                smoothing=v2_config["event_aware_v2"]["smoothing"],
                smoothing_param=v2_config["event_aware_v2"]["smoothing_param"],
            )
            analysis_event = AnalysisEvent(event=case_choice["event"], true_events=[], source="case")
            plot_case_figure(
                analysis_dir / f"{dataset}_case_study.png",
                dataset=dataset,
                point_scores=point_scores,
                labels=loaded["labels"],
                analysis_event=analysis_event,
                case_row=case_choice,
            )
        print(f"finished {dataset}")
        del bundle
        torch.cuda.empty_cache()

    write_csv(results_csv, list(summary_rows[0].keys()), summary_rows)
    write_csv(comparison_csv, list(comparison_rows[0].keys()), comparison_rows)
    detail_fieldnames = list(detail_rows[0].keys()) if detail_rows else [
        "Dataset",
        "Event Sampling Seed",
        "Event Index",
        "Original Event Start",
        "Original Event End",
        "Original Event Length",
        "Analyzed Event Start",
        "Analyzed Event End",
        "Analyzed Event Length",
        "Matched True Events",
        "Event Source",
        "Reference Start",
        "Reference Distance",
        "Score Before",
        "Top1 Gain",
        "Top3 Gain",
        "Top5 Gain",
        "Top1 Random Gain",
        "Top3 Random Gain",
        "Top5 Random Gain",
        "Top3 Stability",
        "Top5 Variables",
    ]
    write_csv(event_details_csv, detail_fieldnames, detail_rows)
    write_xlsx(
        results_xlsx,
        [
            ("counterfactual_v1", summary_rows),
            ("comparison", comparison_rows),
            ("event_details", detail_rows),
        ],
    )
    print(results_csv)
    print(comparison_csv)
    print(event_details_csv)
    print(results_xlsx)
    print(analysis_dir)


if __name__ == "__main__":
    main()
