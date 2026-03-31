import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_counterfactual_v1 import (
    AnalysisEvent,
    build_model_bundle,
    candidate_starts,
    cf_gain,
    clip_event_to_local_segment,
    compute_local_event_score,
    load_csv_rows,
    load_feature_names,
    load_json,
    load_model_space_arrays,
    load_score_inputs,
    local_context_bounds,
    parse_optional_bool,
    repair_segment,
    restrict_events_to_range,
    select_analysis_events,
    resolve_output_tag,
    tagged_output_dir,
    tagged_output_path,
)
from run_event_aware_v2 import (
    apply_event_aware_thresholds,
    build_point_scores,
    build_threshold_views,
    format_value,
    select_event_aware_v2_thresholds,
    write_csv,
    write_xlsx,
)
from unified_evaluator import points_to_events


ROOT = Path(__file__).resolve().parents[2]
TABLE_ROOT = ROOT / "outputs" / "tables"


CF_V1_REFERENCE = {
    "SMAP": {
        "top1_cf_gain_v1": -0.0006,
        "top3_cf_gain_v1": 0.0102,
        "top5_cf_gain_v1": 0.0225,
        "top3_random_gain_v1": 0.0045,
        "topk_stability_v1": 1.0000,
    },
    "MSL": {
        "top1_cf_gain_v1": 0.0018,
        "top3_cf_gain_v1": 0.0053,
        "top5_cf_gain_v1": 0.0077,
        "top3_random_gain_v1": 0.0006,
        "topk_stability_v1": 1.0000,
    },
    "HAI21.03": {
        "top1_cf_gain_v1": -0.0002,
        "top3_cf_gain_v1": 0.0018,
        "top5_cf_gain_v1": 0.0040,
        "top3_random_gain_v1": 0.0002,
        "topk_stability_v1": 0.8075,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Counterfactual v2 analysis on saved score artifacts.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval" / "counterfactual_v2.json"))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--num-random-trials", type=int, default=None)
    parser.add_argument("--event-sampling-seed", type=int, default=None)
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--save-case-figures", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_cf_v1_reference() -> dict[str, dict[str, float]]:
    rows = {dataset: values.copy() for dataset, values in CF_V1_REFERENCE.items()}
    path = TABLE_ROOT / "counterfactual_v1_comparison.csv"
    if not path.exists():
        return rows
    for raw_row in load_csv_rows(path).values():
        dataset = raw_row["Dataset"]
        if dataset not in rows:
            rows[dataset] = {}
        row = rows[dataset]
        for key in (
            "top1_cf_gain_v1",
            "top3_cf_gain_v1",
            "top5_cf_gain_v1",
            "top3_random_gain_v1",
            "topk_stability_v1",
        ):
            source_key = {
                "top1_cf_gain_v1": "top1_cf_gain",
                "top3_cf_gain_v1": "top3_cf_gain",
                "top5_cf_gain_v1": "top5_cf_gain",
                "top3_random_gain_v1": "top3_random_gain",
                "topk_stability_v1": "topk_stability",
            }[key]
            value = raw_row.get(source_key)
            if value not in (None, ""):
                row[key] = float(value)
    return rows


def compute_train_stats(train_data: np.ndarray) -> dict[str, np.ndarray]:
    train_array = np.asarray(train_data, dtype=np.float32)
    mean = train_array.mean(axis=0).astype(np.float32)
    std = np.clip(train_array.std(axis=0).astype(np.float32), 1e-6, None)
    median = np.median(train_array, axis=0).astype(np.float32)
    mad = np.median(np.abs(train_array - median.reshape(1, -1)), axis=0).astype(np.float32)
    mad = np.clip(mad, 1e-6, None)
    return {"mean": mean, "std": std, "median": median, "mad": mad}


def contribution_scores_v2(
    event_segment: np.ndarray,
    event_point_scores: np.ndarray,
    stats: dict[str, np.ndarray],
    method: str,
) -> np.ndarray:
    segment = np.asarray(event_segment, dtype=np.float64)
    weights = np.clip(np.asarray(event_point_scores, dtype=np.float64), a_min=1e-6, a_max=None).reshape(-1, 1)
    if method == "score_weighted_abs_deviation_sum":
        deviations = np.abs(segment - stats["mean"].reshape(1, -1))
    elif method == "score_weighted_zscore_sum":
        deviations = np.abs((segment - stats["mean"].reshape(1, -1)) / stats["std"].reshape(1, -1))
    elif method == "score_weighted_robust_zscore_sum":
        deviations = np.abs((segment - stats["median"].reshape(1, -1)) / stats["mad"].reshape(1, -1))
    else:
        raise ValueError(f"Unsupported contribution method: {method}")
    return (deviations * weights).sum(axis=0)


def topk_indices(contributions: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), int(contributions.shape[0])))
    return np.argsort(contributions)[::-1][:k]


def jaccard(lhs: set[int], rhs: set[int]) -> float:
    union = lhs | rhs
    if not union:
        return 1.0
    return len(lhs & rhs) / len(union)


def topk_stability_v2(
    event_segment: np.ndarray,
    event_point_scores: np.ndarray,
    stats: dict[str, np.ndarray],
    method: str,
    topk: int,
    local_window: int,
) -> float:
    length = int(event_segment.shape[0])
    if length <= 1:
        return 1.0
    local_window = max(1, min(int(local_window), length))
    stride = max(1, local_window // 2)
    starts = list(range(0, max(length - local_window + 1, 1), stride))
    last_start = max(0, length - local_window)
    if last_start not in starts:
        starts.append(last_start)
    starts = sorted(set(starts))
    top_sets = []
    for start in starts:
        end = start + local_window
        contrib = contribution_scores_v2(event_segment[start:end], event_point_scores[start:end], stats, method)
        top_sets.append(set(topk_indices(contrib, topk).tolist()))
    if len(top_sets) <= 1:
        return 1.0
    return float(np.mean([jaccard(lhs, rhs) for lhs, rhs in zip(top_sets[:-1], top_sets[1:])]))


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
        candidate = np.asarray(train_data[start:start + window_length], dtype=np.float32)
        dist = float(np.mean((candidate - event_segment) ** 2))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_start = int(start)
    best_segment = np.asarray(train_data[best_start:best_start + window_length], dtype=np.float32).copy()
    return best_segment, best_start, float(best_dist)


def random_reference_window(
    train_data: np.ndarray,
    window_length: int,
    stride: int,
    max_candidates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    starts = candidate_starts(train_data.shape[0], window_length, stride, max_candidates)
    choice = int(rng.choice(starts))
    segment = np.asarray(train_data[choice:choice + window_length], dtype=np.float32).copy()
    return segment, choice


def delete_segment(
    event_segment: np.ndarray,
    variable_indices: np.ndarray,
    replacement_vector: np.ndarray,
) -> np.ndarray:
    deleted = event_segment.copy()
    deleted[:, variable_indices] = replacement_vector[variable_indices].reshape(1, -1)
    return deleted


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def median_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(values))


def positive_gain_ratio(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr > 0))


def win_rate(lhs: list[float], rhs: list[float]) -> float:
    if not lhs or not rhs:
        return float("nan")
    length = min(len(lhs), len(rhs))
    if length == 0:
        return float("nan")
    lhs_arr = np.asarray(lhs[:length], dtype=np.float64)
    rhs_arr = np.asarray(rhs[:length], dtype=np.float64)
    return float(np.mean(lhs_arr > rhs_arr))


def plot_case_figure_v2(
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

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [2.2, 1.6, 1.8]})

    axes[0].plot(x, local_scores, color="#1f4e79", linewidth=1.0, label="Point score")
    axes[0].axvspan(event_start, event_end, color="#b22222", alpha=0.16, label="Selected event")
    in_true = False
    true_start = 0
    for idx, value in zip(x, local_labels):
        if value == 1 and not in_true:
            true_start = idx
            in_true = True
        elif value == 0 and in_true:
            axes[0].axvspan(true_start, idx - 1, color="#f4c542", alpha=0.18)
            in_true = False
    if in_true:
        axes[0].axvspan(true_start, plot_end, color="#f4c542", alpha=0.18)
    axes[0].set_title(f"{dataset} counterfactual v2 case")
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

    labels_bar = ["before", "cf1", "cf3", "cf5", "randA3", "randB3", "del3"]
    values_bar = [
        case_row["score_before"],
        case_row["score_after_top1"],
        case_row["score_after_top3"],
        case_row["score_after_top5"],
        case_row["score_after_rand_a_top3"],
        case_row["score_after_rand_b_top3"],
        case_row["score_after_delete_top3"],
    ]
    colors = ["#7f7f7f", "#1f4e79", "#1f4e79", "#1f4e79", "#b22222", "#db7c26", "#5a5a5a"]
    axes[2].bar(np.arange(len(labels_bar)), values_bar, color=colors)
    axes[2].set_xticks(np.arange(len(labels_bar)), labels=labels_bar)
    axes[2].set_title("Repair and deletion score comparison")
    axes[2].set_ylabel("Event score")
    axes[2].grid(alpha=0.2, axis="y", linestyle="--")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_dataset(
    dataset: str,
    cf_config: dict,
    v2_config: dict,
    bundle,
    baseline_rows: dict[str, dict[str, str]],
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
    stats = compute_train_stats(train_data)
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
        contrib = contribution_scores_v2(
            event_segment=event_segment,
            event_point_scores=event_point_scores,
            stats=stats,
            method=cf_config["contribution_method"],
        )
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

        gains = {}
        gains_random_a = {}
        gains_random_b = {}
        gains_delete = {}
        after_scores = {}
        random_a_after_scores = {}
        random_b_after_scores = {}
        delete_after_scores = {}

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

            trial_gains_a = []
            trial_scores_a = []
            trial_gains_b = []
            trial_scores_b = []
            for trial in range(int(cf_config["num_random_trials"])):
                rng = np.random.default_rng(seed=20260321 + idx * 97 + k * 31 + trial)
                rand_idx = np.sort(rng.choice(event_segment.shape[1], size=min(k, event_segment.shape[1]), replace=False))

                if "random_variables_same_reference" in cf_config["random_baseline_modes"]:
                    rand_segment_a = repair_segment(event_segment, reference_segment, rand_idx)
                    local_rand_a = local_original.copy()
                    local_rand_a[local_event[0]:local_event[1] + 1] = rand_segment_a
                    rand_score_a = compute_local_event_score(
                        bundle=bundle,
                        local_series=local_rand_a,
                        local_event=local_event,
                        batch_size=local_batch_size,
                        pooling=cf_config["counterfactual_score_pooling"],
                    )
                    trial_scores_a.append(float(rand_score_a))
                    trial_gains_a.append(cf_gain(score_before, rand_score_a))

                if "random_variables_random_reference" in cf_config["random_baseline_modes"]:
                    random_reference, _ = random_reference_window(
                        train_data=train_data,
                        window_length=event_segment.shape[0],
                        stride=int(cf_config["candidate_window_stride"]),
                        max_candidates=int(cf_config["max_reference_candidates"]),
                        rng=rng,
                    )
                    rand_segment_b = repair_segment(event_segment, random_reference, rand_idx)
                    local_rand_b = local_original.copy()
                    local_rand_b[local_event[0]:local_event[1] + 1] = rand_segment_b
                    rand_score_b = compute_local_event_score(
                        bundle=bundle,
                        local_series=local_rand_b,
                        local_event=local_event,
                        batch_size=local_batch_size,
                        pooling=cf_config["counterfactual_score_pooling"],
                    )
                    trial_scores_b.append(float(rand_score_b))
                    trial_gains_b.append(cf_gain(score_before, rand_score_b))

            gains_random_a[k] = mean_or_nan(trial_gains_a)
            gains_random_b[k] = mean_or_nan(trial_gains_b)
            random_a_after_scores[k] = mean_or_nan(trial_scores_a)
            random_b_after_scores[k] = mean_or_nan(trial_scores_b)

            if cf_config["use_deletion_test"]:
                if cf_config["deletion_method"] == "train_median_replace":
                    replacement = stats["median"]
                elif cf_config["deletion_method"] == "train_mean_replace":
                    replacement = stats["mean"]
                elif cf_config["deletion_method"] == "reference_replace":
                    replacement = reference_segment.mean(axis=0)
                else:
                    raise ValueError(f"Unsupported deletion method: {cf_config['deletion_method']}")
                deleted_segment = delete_segment(event_segment, vars_idx, np.asarray(replacement, dtype=np.float32))
                local_deleted = local_original.copy()
                local_deleted[local_event[0]:local_event[1] + 1] = deleted_segment
                delete_score_after = compute_local_event_score(
                    bundle=bundle,
                    local_series=local_deleted,
                    local_event=local_event,
                    batch_size=local_batch_size,
                    pooling=cf_config["counterfactual_score_pooling"],
                )
                delete_after_scores[k] = float(delete_score_after)
                gains_delete[k] = cf_gain(score_before, delete_score_after)

        stability = topk_stability_v2(
            event_segment=event_segment,
            event_point_scores=event_point_scores,
            stats=stats,
            method=cf_config["contribution_method"],
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
            "Score Before": float(score_before),
            "Top1 Gain": float(gains.get(1, np.nan)),
            "Top3 Gain": float(gains.get(3, np.nan)),
            "Top5 Gain": float(gains.get(5, np.nan)),
            "Top1 Random A Gain": float(gains_random_a.get(1, np.nan)),
            "Top3 Random A Gain": float(gains_random_a.get(3, np.nan)),
            "Top5 Random A Gain": float(gains_random_a.get(5, np.nan)),
            "Top1 Random B Gain": float(gains_random_b.get(1, np.nan)),
            "Top3 Random B Gain": float(gains_random_b.get(3, np.nan)),
            "Top5 Random B Gain": float(gains_random_b.get(5, np.nan)),
            "Top1 Deletion Gain": float(gains_delete.get(1, np.nan)),
            "Top3 Deletion Gain": float(gains_delete.get(3, np.nan)),
            "Top5 Deletion Gain": float(gains_delete.get(5, np.nan)),
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
            "score_after_rand_a_top3": float(random_a_after_scores.get(3, np.nan)),
            "score_after_rand_b_top3": float(random_b_after_scores.get(3, np.nan)),
            "score_after_delete_top3": float(delete_after_scores.get(3, np.nan)),
            "top5_variable_names": [feature_names[int(var_idx)] for var_idx in top5_idx],
            "top5_contributions": [float(contrib[int(var_idx)]) for var_idx in top5_idx],
            "gain_margin": float(gains.get(3, np.nan) - max(gains_random_a.get(3, -np.inf), gains_random_b.get(3, -np.inf))),
        }
        if case_choice is None or case_row["gain_margin"] > case_choice["gain_margin"]:
            case_choice = case_row

        torch.cuda.empty_cache()

    top1_cf_gains = [row["Top1 Gain"] for row in detail_rows]
    top3_cf_gains = [row["Top3 Gain"] for row in detail_rows]
    top5_cf_gains = [row["Top5 Gain"] for row in detail_rows]
    top1_random_a = [row["Top1 Random A Gain"] for row in detail_rows]
    top3_random_a = [row["Top3 Random A Gain"] for row in detail_rows]
    top5_random_a = [row["Top5 Random A Gain"] for row in detail_rows]
    top1_random_b = [row["Top1 Random B Gain"] for row in detail_rows]
    top3_random_b = [row["Top3 Random B Gain"] for row in detail_rows]
    top5_random_b = [row["Top5 Random B Gain"] for row in detail_rows]
    top1_delete = [row["Top1 Deletion Gain"] for row in detail_rows]
    top3_delete = [row["Top3 Deletion Gain"] for row in detail_rows]
    top5_delete = [row["Top5 Deletion Gain"] for row in detail_rows]
    top3_stability = [row["Top3 Stability"] for row in detail_rows]

    baseline_row = baseline_rows[dataset]
    summary_row = {
        "Dataset": dataset,
        "Score Source": str(score_root.relative_to(ROOT)).replace("\\", "/"),
        "Total Predicted Event Count": len(final_events),
        "Analyzed Event Count": len(detail_rows),
        "Selection Mode": cf_config.get("event_selection_mode", "all_predicted"),
        "Threshold Selection Length": int(threshold_views["selection_length"]),
        "Evaluation Offset": evaluation_offset,
        "Top1 CF Gain Mean": mean_or_nan(top1_cf_gains),
        "Top1 CF Gain Median": median_or_nan(top1_cf_gains),
        "Top1 Positive Gain Ratio": positive_gain_ratio(top1_cf_gains),
        "Top1 Random A Gain Mean": mean_or_nan(top1_random_a),
        "Top1 Random B Gain Mean": mean_or_nan(top1_random_b),
        "Top1 Win Rate vs Random A": win_rate(top1_cf_gains, top1_random_a),
        "Top1 Win Rate vs Random B": win_rate(top1_cf_gains, top1_random_b),
        "Top1 Deletion Gain Mean": mean_or_nan(top1_delete),
        "Top3 CF Gain Mean": mean_or_nan(top3_cf_gains),
        "Top3 CF Gain Median": median_or_nan(top3_cf_gains),
        "Top3 Positive Gain Ratio": positive_gain_ratio(top3_cf_gains),
        "Top3 Random A Gain Mean": mean_or_nan(top3_random_a),
        "Top3 Random B Gain Mean": mean_or_nan(top3_random_b),
        "Top3 Win Rate vs Random A": win_rate(top3_cf_gains, top3_random_a),
        "Top3 Win Rate vs Random B": win_rate(top3_cf_gains, top3_random_b),
        "Top3 Deletion Gain Mean": mean_or_nan(top3_delete),
        "Top5 CF Gain Mean": mean_or_nan(top5_cf_gains),
        "Top5 CF Gain Median": median_or_nan(top5_cf_gains),
        "Top5 Positive Gain Ratio": positive_gain_ratio(top5_cf_gains),
        "Top5 Random A Gain Mean": mean_or_nan(top5_random_a),
        "Top5 Random B Gain Mean": mean_or_nan(top5_random_b),
        "Top5 Win Rate vs Random A": win_rate(top5_cf_gains, top5_random_a),
        "Top5 Win Rate vs Random B": win_rate(top5_cf_gains, top5_random_b),
        "Top5 Deletion Gain Mean": mean_or_nan(top5_delete),
        "Top3 Stability": mean_or_nan(top3_stability),
        "Baseline Unified F1": float(baseline_row["baseline_unified_f1"]),
        "Baseline Unified Fc1": float(baseline_row["baseline_unified_fc1"]),
        "Baseline Unified Delay": float(baseline_row["baseline_unified_delay"]),
        "Detector Unified F1": float(baseline_row["v2_unified_f1"]),
        "Detector Unified Fc1": float(baseline_row["v2_unified_fc1"]),
        "Detector Unified Delay": float(baseline_row["v2_unified_delay"]),
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
    cf_v1_rows = load_cf_v1_reference()

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
        )
        summary_rows.append(
            {
                "Dataset": dataset,
                "Score Source": summary_row["Score Source"],
                "Total Predicted Event Count": summary_row["Total Predicted Event Count"],
                "Analyzed Event Count": summary_row["Analyzed Event Count"],
                "Selection Mode": summary_row["Selection Mode"],
                "Event Sampling Seed": int(cf_config.get("event_sampling_seed", 20260322)),
                "Threshold Selection Length": summary_row["Threshold Selection Length"],
                "Evaluation Offset": summary_row["Evaluation Offset"],
                "Detector Unified F1 (from Event-aware v2)": format_value(summary_row["Detector Unified F1"]),
                "Detector Unified Fc1 (from Event-aware v2)": format_value(summary_row["Detector Unified Fc1"]),
                "Detector Unified Delay (from Event-aware v2)": format_value(summary_row["Detector Unified Delay"]),
                "Top1 CF Gain Mean": format_value(summary_row["Top1 CF Gain Mean"]),
                "Top1 CF Gain Median": format_value(summary_row["Top1 CF Gain Median"]),
                "Top1 Positive Gain Ratio": format_value(summary_row["Top1 Positive Gain Ratio"]),
                "Top1 Random A Gain Mean": format_value(summary_row["Top1 Random A Gain Mean"]),
                "Top1 Random B Gain Mean": format_value(summary_row["Top1 Random B Gain Mean"]),
                "Top1 Win Rate vs Random A": format_value(summary_row["Top1 Win Rate vs Random A"]),
                "Top1 Win Rate vs Random B": format_value(summary_row["Top1 Win Rate vs Random B"]),
                "Top1 Deletion Gain Mean": format_value(summary_row["Top1 Deletion Gain Mean"]),
                "Top3 CF Gain Mean": format_value(summary_row["Top3 CF Gain Mean"]),
                "Top3 CF Gain Median": format_value(summary_row["Top3 CF Gain Median"]),
                "Top3 Positive Gain Ratio": format_value(summary_row["Top3 Positive Gain Ratio"]),
                "Top3 Random A Gain Mean": format_value(summary_row["Top3 Random A Gain Mean"]),
                "Top3 Random B Gain Mean": format_value(summary_row["Top3 Random B Gain Mean"]),
                "Top3 Win Rate vs Random A": format_value(summary_row["Top3 Win Rate vs Random A"]),
                "Top3 Win Rate vs Random B": format_value(summary_row["Top3 Win Rate vs Random B"]),
                "Top3 Deletion Gain Mean": format_value(summary_row["Top3 Deletion Gain Mean"]),
                "Top5 CF Gain Mean": format_value(summary_row["Top5 CF Gain Mean"]),
                "Top5 CF Gain Median": format_value(summary_row["Top5 CF Gain Median"]),
                "Top5 Positive Gain Ratio": format_value(summary_row["Top5 Positive Gain Ratio"]),
                "Top5 Random A Gain Mean": format_value(summary_row["Top5 Random A Gain Mean"]),
                "Top5 Random B Gain Mean": format_value(summary_row["Top5 Random B Gain Mean"]),
                "Top5 Win Rate vs Random A": format_value(summary_row["Top5 Win Rate vs Random A"]),
                "Top5 Win Rate vs Random B": format_value(summary_row["Top5 Win Rate vs Random B"]),
                "Top5 Deletion Gain Mean": format_value(summary_row["Top5 Deletion Gain Mean"]),
                "Top3 Stability": format_value(summary_row["Top3 Stability"]),
            }
        )

        cf_v1_ref = cf_v1_rows[dataset]
        comparison_rows.append(
            {
                "Dataset": dataset,
                "baseline_unified_f1": format_value(summary_row["Baseline Unified F1"]),
                "event_aware_v2_unified_f1": format_value(summary_row["Detector Unified F1"]),
                "baseline_unified_fc1": format_value(summary_row["Baseline Unified Fc1"]),
                "event_aware_v2_unified_fc1": format_value(summary_row["Detector Unified Fc1"]),
                "baseline_unified_delay": format_value(summary_row["Baseline Unified Delay"]),
                "event_aware_v2_unified_delay": format_value(summary_row["Detector Unified Delay"]),
                "total_predicted_event_count": summary_row["Total Predicted Event Count"],
                "analyzed_event_count": summary_row["Analyzed Event Count"],
                "selection_mode": summary_row["Selection Mode"],
                "event_sampling_seed": int(cf_config.get("event_sampling_seed", 20260322)),
                "top1_cf_gain_v1": format_value(cf_v1_ref["top1_cf_gain_v1"]),
                "top1_cf_gain_v2": format_value(summary_row["Top1 CF Gain Mean"]),
                "top3_cf_gain_v1": format_value(cf_v1_ref["top3_cf_gain_v1"]),
                "top3_cf_gain_v2": format_value(summary_row["Top3 CF Gain Mean"]),
                "top3_random_gain_v1": format_value(cf_v1_ref["top3_random_gain_v1"]),
                "top3_random_a_gain_v2": format_value(summary_row["Top3 Random A Gain Mean"]),
                "top3_random_b_gain_v2": format_value(summary_row["Top3 Random B Gain Mean"]),
                "top3_deletion_gain_v2": format_value(summary_row["Top3 Deletion Gain Mean"]),
                "topk_stability_v1": format_value(cf_v1_ref["topk_stability_v1"]),
                "topk_stability_v2": format_value(summary_row["Top3 Stability"]),
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
            plot_case_figure_v2(
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

    if not summary_rows:
        raise RuntimeError("No dataset was processed.")

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
        "Top1 Random A Gain",
        "Top3 Random A Gain",
        "Top5 Random A Gain",
        "Top1 Random B Gain",
        "Top3 Random B Gain",
        "Top5 Random B Gain",
        "Top1 Deletion Gain",
        "Top3 Deletion Gain",
        "Top5 Deletion Gain",
        "Top3 Stability",
        "Top5 Variables",
    ]
    write_csv(event_details_csv, detail_fieldnames, detail_rows)
    write_xlsx(
        results_xlsx,
        [
            ("counterfactual_v2", summary_rows),
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
