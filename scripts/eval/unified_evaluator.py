import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = ROOT / "data_processed"


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


def aggregate_window_scores_to_points(
    window_scores: np.ndarray,
    window_end_indices: np.ndarray,
    total_length: int,
    win_size: int,
    method: str = "mean",
) -> np.ndarray:
    point_scores = np.zeros(total_length, dtype=np.float64)
    counts = np.zeros(total_length, dtype=np.float64)

    for score, end_idx in zip(window_scores, window_end_indices):
        start_idx = int(end_idx) - win_size + 1
        end_idx = int(end_idx)
        if method == "mean":
            point_scores[start_idx:end_idx + 1] += score
            counts[start_idx:end_idx + 1] += 1
        elif method == "max":
            point_scores[start_idx:end_idx + 1] = np.maximum(point_scores[start_idx:end_idx + 1], score)
            counts[start_idx:end_idx + 1] = 1
        elif method == "last":
            point_scores[end_idx] = score
            counts[end_idx] = 1
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

    if method == "mean":
        valid = counts > 0
        point_scores[valid] = point_scores[valid] / counts[valid]
    return point_scores.astype(np.float32)


def aggregate_window_point_scores_to_points(
    window_point_scores: np.ndarray,
    window_end_indices: np.ndarray,
    total_length: int,
    method: str = "mean",
) -> np.ndarray:
    point_scores = np.zeros(total_length, dtype=np.float64)
    counts = np.zeros(total_length, dtype=np.float64)

    for row_scores, end_idx in zip(window_point_scores, window_end_indices):
        row_scores = np.asarray(row_scores, dtype=np.float64).reshape(-1)
        end_idx = int(end_idx)
        start_idx = end_idx - row_scores.shape[0] + 1
        if method == "mean":
            point_scores[start_idx:end_idx + 1] += row_scores
            counts[start_idx:end_idx + 1] += 1
        elif method == "max":
            point_scores[start_idx:end_idx + 1] = np.maximum(point_scores[start_idx:end_idx + 1], row_scores)
            counts[start_idx:end_idx + 1] = 1
        elif method == "last":
            point_scores[start_idx:end_idx + 1] = row_scores
            counts[start_idx:end_idx + 1] = 1
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

    if method == "mean":
        valid = counts > 0
        point_scores[valid] = point_scores[valid] / counts[valid]
    return point_scores.astype(np.float32)


def moving_average(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return scores.astype(np.float32)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(scores, kernel, mode="same").astype(np.float32)


def gaussian_smoothing(scores: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return scores.astype(np.float32)
    radius = max(1, int(3 * sigma))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return np.convolve(scores, kernel, mode="same").astype(np.float32)


def apply_smoothing(scores: np.ndarray, mode: str, param: float) -> np.ndarray:
    if mode == "none":
        return scores.astype(np.float32)
    if mode == "moving_average":
        return moving_average(scores, int(param))
    if mode == "gaussian":
        return gaussian_smoothing(scores, float(param))
    raise ValueError(f"Unsupported smoothing mode: {mode}")


def threshold_by_quantile(scores: np.ndarray, quantile: float) -> float:
    return float(np.quantile(scores, quantile))


def threshold_by_best_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return float(scores.mean())

    f1_values = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index])


def threshold_candidates(scores: np.ndarray, steps: int = 401) -> np.ndarray:
    if steps <= 1:
        return np.asarray([float(scores.mean())], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    return np.unique(np.quantile(scores, quantiles)).astype(np.float64)


def evaluate_threshold_curve(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    curve = []
    for threshold in thresholds:
        preds = (scores >= threshold).astype(np.int64)
        f1 = float(f1_score(labels, preds, zero_division=0))
        fc1 = float(composite_event_f1(labels, preds))
        delay = float(detection_delay(labels, preds))
        curve.append(
            {
                "threshold": float(threshold),
                "f1": f1,
                "fc1": fc1,
                "delay": delay,
            }
        )
    return curve


def threshold_by_best_fc1(scores: np.ndarray, labels: np.ndarray, steps: int = 401) -> float:
    candidates = threshold_candidates(scores, steps=steps)
    curve = evaluate_threshold_curve(scores, labels, candidates)
    best = max(
        curve,
        key=lambda row: (
            row["fc1"],
            row["f1"],
            -row["delay"] if np.isfinite(row["delay"]) else float("-inf"),
            -row["threshold"],
        ),
    )
    return float(best["threshold"])


def threshold_dynamic(scores: np.ndarray, z: float = 3.0) -> float:
    mean = float(scores.mean())
    std = float(scores.std())
    return mean + z * std


def threshold_scores(scores: np.ndarray, method: str, labels: np.ndarray | None = None, param: float | None = None) -> float:
    if method == "quantile":
        quantile = 0.995 if param is None else float(param)
        return threshold_by_quantile(scores, quantile)
    if method == "best_f1":
        if labels is None:
            raise ValueError("labels are required for best_f1 thresholding")
        return threshold_by_best_f1(scores, labels)
    if method == "best_fc1":
        if labels is None:
            raise ValueError("labels are required for best_fc1 thresholding")
        steps = 401 if param is None else int(param)
        return threshold_by_best_fc1(scores, labels, steps=steps)
    if method == "dynamic":
        z = 3.0 if param is None else float(param)
        return threshold_dynamic(scores, z=z)
    raise ValueError(f"Unsupported threshold method: {method}")


def points_to_events(binary_points: np.ndarray) -> list[tuple[int, int]]:
    events = []
    start = None
    for idx, value in enumerate(binary_points.astype(int)):
        if value == 1 and start is None:
            start = idx
        elif value == 0 and start is not None:
            events.append((start, idx - 1))
            start = None
    if start is not None:
        events.append((start, len(binary_points) - 1))
    return events


def events_to_points(events: list[tuple[int, int]], total_length: int) -> np.ndarray:
    preds = np.zeros(total_length, dtype=np.int64)
    for start, end in events:
        preds[start:end + 1] = 1
    return preds


def merge_close_events(events: list[tuple[int, int]], gap_size: int) -> list[tuple[int, int]]:
    if not events or gap_size <= 0:
        return list(events)
    merged = [events[0]]
    for start, end in events[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= gap_size:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def filter_short_events(events: list[tuple[int, int]], min_length: int) -> list[tuple[int, int]]:
    if min_length <= 1:
        return list(events)
    return [(start, end) for start, end in events if (end - start + 1) >= min_length]


def event_score(point_scores: np.ndarray, event: tuple[int, int], pooling: str = "mean") -> float:
    start, end = event
    window = point_scores[start:end + 1]
    if pooling == "mean":
        return float(np.mean(window))
    if pooling == "max":
        return float(np.max(window))
    if pooling == "area":
        return float(np.sum(window))
    raise ValueError(f"Unsupported event score pooling: {pooling}")


def event_scores(point_scores: np.ndarray, events: list[tuple[int, int]], pooling: str = "mean") -> np.ndarray:
    if not events:
        return np.zeros(0, dtype=np.float32)
    return np.asarray([event_score(point_scores, event, pooling=pooling) for event in events], dtype=np.float32)


def filter_events_by_score(
    point_scores: np.ndarray,
    events: list[tuple[int, int]],
    threshold: float,
    pooling: str = "mean",
) -> list[tuple[int, int]]:
    if not events:
        return []
    pooled = event_scores(point_scores, events, pooling=pooling)
    return [event for event, score in zip(events, pooled) if float(score) >= float(threshold)]


def covered_prefix_length(window_end_indices: np.ndarray, total_length: int) -> int:
    if window_end_indices.size == 0:
        return 0
    return min(int(window_end_indices.max()) + 1, int(total_length))


def trim_labels_to_coverage(labels: np.ndarray, window_end_indices: np.ndarray) -> tuple[np.ndarray, int]:
    usable_length = covered_prefix_length(window_end_indices, len(labels))
    return labels[:usable_length].astype(np.int64), usable_length


def composite_event_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    true_events = points_to_events(labels)
    if not true_events:
        return 0.0
    tp = sum(int(preds[start:end + 1].any()) for start, end in true_events)
    fn = len(true_events) - tp
    rec_e = tp / max(tp + fn, 1)
    prec_t = float(((preds == 1) & (labels == 1)).sum() / max((preds == 1).sum(), 1))
    if prec_t == 0 and rec_e == 0:
        return 0.0
    return float(2 * prec_t * rec_e / max(prec_t + rec_e, 1e-12))


def event_detection_delays(labels: np.ndarray, preds: np.ndarray) -> list[float]:
    true_events = points_to_events(labels)
    delays = []
    for start, end in true_events:
        pred_hits = np.where(preds[start:end + 1] == 1)[0]
        if pred_hits.size:
            delays.append(float(pred_hits[0]))
    return delays


def detection_delay(labels: np.ndarray, preds: np.ndarray) -> float:
    delays = event_detection_delays(labels, preds)
    if not delays:
        return float("inf")
    return float(np.mean(delays))


def mean_true_event_length(labels: np.ndarray) -> float:
    true_events = points_to_events(labels)
    if not true_events:
        return 1.0
    lengths = [(end - start + 1) for start, end in true_events]
    return float(np.mean(lengths))


def normalized_delay(labels: np.ndarray, preds: np.ndarray) -> float:
    delay = detection_delay(labels, preds)
    if not np.isfinite(delay):
        return 1.0
    scale = max(mean_true_event_length(labels), 1.0)
    return float(min(delay / scale, 1.0))


def evaluate_point_scores(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(np.int64)

    results = {
        "threshold": float(threshold),
        "aupr": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, preds)),
        "fc1": float(composite_event_f1(labels, preds)),
        "delay": float(detection_delay(labels, preds)),
        "predicted_events": points_to_events(preds),
        "true_events": points_to_events(labels),
    }
    if len(np.unique(labels)) > 1:
        results["auroc"] = float(roc_auc_score(labels, scores))
    else:
        results["auroc"] = float("nan")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Unified evaluator for processed/windowed datasets.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--window_scores", required=True, help="Path to per-window scores .npy")
    parser.add_argument("--window_end_indices", default=None, help="Optional override path for test window end indices .npy")
    parser.add_argument("--aggregation", default="mean", choices=["mean", "max", "last"])
    parser.add_argument("--smoothing", default="none", choices=["none", "moving_average", "gaussian"])
    parser.add_argument("--smoothing_param", type=float, default=0.0)
    parser.add_argument("--threshold_method", default="best_f1", choices=["quantile", "best_f1", "best_fc1", "dynamic"])
    parser.add_argument("--threshold_param", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = normalize_name(args.dataset)
    dataset_dir = PROCESSED_ROOT / dataset_name

    meta = json.loads((dataset_dir / "meta.json").read_text(encoding="utf-8"))
    labels = np.load(dataset_dir / "label.npy").astype(np.int64)
    window_end_indices_path = dataset_dir / "windows" / "test_window_end_indices.npy"
    if args.window_end_indices is not None:
        window_end_indices_path = Path(args.window_end_indices)
    window_end_indices = np.load(window_end_indices_path).astype(np.int64).reshape(-1)
    window_scores = np.load(args.window_scores).astype(np.float32)

    win_size = int(meta["windowing"]["win_size"])
    effective_length = min(window_scores.shape[0], window_end_indices.shape[0])
    window_scores = window_scores[:effective_length]
    window_end_indices = window_end_indices[:effective_length]

    if window_scores.ndim == 1:
        point_scores = aggregate_window_scores_to_points(
            window_scores=window_scores,
            window_end_indices=window_end_indices,
            total_length=len(labels),
            win_size=win_size,
            method=args.aggregation,
        )
    elif window_scores.ndim == 2:
        point_scores = aggregate_window_point_scores_to_points(
            window_point_scores=window_scores,
            window_end_indices=window_end_indices,
            total_length=len(labels),
            method=args.aggregation,
        )
    else:
        raise ValueError(f"Unsupported window_scores rank: {window_scores.ndim}")
    point_scores = apply_smoothing(point_scores, args.smoothing, args.smoothing_param)
    threshold = threshold_scores(point_scores, args.threshold_method, labels=labels, param=args.threshold_param)
    results = evaluate_point_scores(point_scores, labels, threshold)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
