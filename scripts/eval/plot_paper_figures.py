from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image

from run_counterfactual_v1 import (
    contribution_scores,
    load_feature_names,
    load_json as cf_load_json,
    load_model_space_arrays,
    topk_indices,
)
from run_event_aware_v2 import build_point_scores as event_build_point_scores


ROOT = Path(__file__).resolve().parents[2]
TABLE_ROOT = ROOT / "outputs" / "tables"
FIG_ROOT = ROOT / "outputs" / "figures" / "paper"
ANALYSIS_ROOT = ROOT / "analysis_figures"
CASE_ROOT = FIG_ROOT / "case_sources"

MAIN_DATASETS = ["SMAP", "MSL", "HAI21.03"]
ALL_DATASETS = ["SMAP", "MSL", "HAI21.03"]
ABLATIONS = ["abs_same_ref", "zscore_same_ref", "robust_same_ref"]
CASE_SEED = 20260322
DISPLAY_NAMES = {
    "SMAP": "SMAP",
    "MSL": "MSL",
    "HAI21.03": "HAI 21.03",
}

COLORS = {
    "point_adjust": "#439cc4",
    "strict": "#0868a6",
    "baseline": "#439cc4",
    "v1": "#7bc6be",
    "v2": "#0868a6",
    "cf": "#0868a6",
    "random": "#7bc6be",
    "stability": "#b4deb6",
    "abs_same_ref": "#b4deb6",
    "zscore_same_ref": "#439cc4",
    "robust_same_ref": "#0868a6",
    "top1": "#b4deb6",
    "top3": "#439cc4",
    "top5": "#0868a6",
    "random_ref": "#eaf3e2",
}


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_ROOT / name)


def display_labels(items: list[str]) -> list[str]:
    return [DISPLAY_NAMES.get(item, item) for item in items]


def display_feature_label(name: str) -> str:
    return name.replace("_", " ")


def load_exact_main3_summary() -> pd.DataFrame:
    seed_paths = [
        TABLE_ROOT / "results_counterfactual_exact_main3_seed20260322_combined.csv",
        TABLE_ROOT / "results_counterfactual_exact_main3_seed20260323.csv",
        TABLE_ROOT / "results_counterfactual_exact_main3_seed20260324.csv",
    ]
    frames = []
    for path in seed_paths:
        df = pd.read_csv(path)
        keep_cols = [
            "Dataset",
            "Top1 CF Gain",
            "Top3 CF Gain",
            "Top5 CF Gain",
            "Top1 Random Gain",
            "Top3 Random Gain",
            "Top5 Random Gain",
        ]
        frames.append(df[keep_cols].copy())
    merged = pd.concat(frames, ignore_index=True)
    summary = (
        merged.groupby("Dataset", as_index=False)
        .agg(
            {
                "Top1 CF Gain": "mean",
                "Top3 CF Gain": "mean",
                "Top5 CF Gain": "mean",
                "Top1 Random Gain": "mean",
                "Top3 Random Gain": "mean",
                "Top5 Random Gain": "mean",
            }
        )
    )
    return summary


def load_case_score_inputs(dataset: str, config: dict) -> dict:
    override = config["score_source_overrides"][dataset]
    score_root = ROOT / override / dataset
    labels = np.load(ROOT / "data_processed" / dataset / "label.npy", mmap_mode="r").astype(np.int64)
    window_point_scores = np.load(score_root / "test_window_point_scores.npy", mmap_mode="r")
    window_end_indices = np.load(score_root / "test_window_end_indices.npy", mmap_mode="r").reshape(-1)
    effective = min(window_point_scores.shape[0], window_end_indices.shape[0])
    window_point_scores = window_point_scores[:effective]
    window_end_indices = window_end_indices[:effective]
    covered = min(int(window_end_indices.max()) + 1 if window_end_indices.size else 0, len(labels))
    labels_aligned = labels[:covered] if config.get("trim_to_coverage", True) else labels
    return {
        "labels": np.asarray(labels_aligned, dtype=np.int64),
        "window_point_scores": window_point_scores,
        "window_end_indices": np.asarray(window_end_indices, dtype=np.int64),
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "Times New Roman",
            "font.size": 23,
            "axes.titlesize": 20,
            "axes.labelsize": 24,
            "xtick.labelsize": 21.5,
            "ytick.labelsize": 21.5,
            "legend.fontsize": 20,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle="-", linewidth=0.55, alpha=0.18, color="#b4deb6")
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(colors="black", width=1.2, length=5)
    add_axis_arrows(ax)


def add_axis_arrows(ax: plt.Axes) -> None:
    arrow_props = {
        "arrowstyle": "-|>",
        "color": "black",
        "lw": 1.1,
        "mutation_scale": 12,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    ax.annotate(
        "",
        xy=(1.015, 0.0),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=arrow_props,
        clip_on=False,
    )
    ax.annotate(
        "",
        xy=(0.0, 1.015),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=arrow_props,
        clip_on=False,
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=28,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="black",
    )


def save(fig: plt.Figure, filename: str) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = FIG_ROOT / filename
    fig.savefig(out_path, dpi=700, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(out_path)


def save_case_source(fig: plt.Figure, filename: str) -> None:
    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = CASE_ROOT / filename
    fig.savefig(out_path, dpi=700, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(out_path)


def plot_motivation_gap() -> None:
    df = load_csv("baseline_main_table.csv").set_index("Dataset").loc[MAIN_DATASETS].reset_index()
    x = np.arange(len(df))
    width = 0.34

    fig, ax = plt.subplots(figsize=(12.6, 7.8))
    style_axis(ax)
    add_panel_label(ax, "A")
    ax.bar(
        x - width / 2,
        df["F1"],
        width=width,
        color=COLORS["point_adjust"],
        label="Point-level adjusted F1",
    )
    ax.bar(
        x + width / 2,
        df["Fc1"],
        width=width,
        color=COLORS["strict"],
        label="Unified event-level Fc1",
    )

    ax.set_title("")
    ax.set_xticks(x, display_labels(df["Dataset"].tolist()))
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.53, 1.08))

    for idx, row in df.iterrows():
        ax.text(idx - width / 2, row["F1"] + 0.025, f'{row["F1"]:.3f}', ha="center", va="bottom", fontsize=24, color="black")
        ax.text(idx + width / 2, row["Fc1"] + 0.025, f'{row["Fc1"]:.3f}', ha="center", va="bottom", fontsize=24, color="black")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "figure_01_metric_gap.png")


def plot_event_aware_main() -> None:
    df = load_csv("event_aware_v2_comparison.csv").set_index("Dataset").loc[MAIN_DATASETS].reset_index()
    x = np.arange(len(df))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 7.8))

    def annotate_bars(ax, xs, values, *, fontsize=12, extra_levels=None):
        y_max = ax.get_ylim()[1]
        base_pad = max(y_max * 0.018, 0.002)
        if extra_levels is None:
            extra_levels = [0] * len(values)
        for x_pos, value, level in zip(xs, values, extra_levels):
            ax.text(
                float(x_pos),
                float(value) + base_pad * (1.0 + float(level)),
                f"{float(value):.3f}" if y_max <= 1.0 else f"{float(value):.1f}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.12),
                clip_on=False,
            )

    ax = axes[0]
    style_axis(ax)
    add_panel_label(ax, "A")
    ax.bar(x - width, df["baseline_unified_f1"], width=width, color=COLORS["baseline"], label="Raw")
    ax.bar(x, df["v1_unified_f1"], width=width, color=COLORS["v1"], label="Smoothed")
    ax.bar(x + width, df["v2_unified_f1"], width=width, color=COLORS["v2"], label="Event-aware")
    ax.set_title("Unified F1", pad=10)
    ax.set_xticks(x, display_labels(df["Dataset"].tolist()))
    ax.set_ylim(0, max(df["v2_unified_f1"].max() * 1.25, 0.08))
    ax.set_ylabel("Unified F1")
    for idx, row in df.iterrows():
        annotate_bars(
            ax,
            [idx - width, idx, idx + width],
            [row["baseline_unified_f1"], row["v1_unified_f1"], row["v2_unified_f1"]],
            fontsize=11,
        )

    ax = axes[1]
    style_axis(ax)
    add_panel_label(ax, "B")
    ax.bar(x - width, df["baseline_unified_delay"], width=width, color=COLORS["baseline"], label="Raw")
    ax.bar(x, df["v1_unified_delay"], width=width, color=COLORS["v1"], label="Smoothed")
    ax.bar(x + width, df["v2_unified_delay"], width=width, color=COLORS["v2"], label="Event-aware")
    ax.set_title("Unified Delay", pad=10)
    ax.set_xticks(x, display_labels(df["Dataset"].tolist()))
    ax.set_ylabel("Unified Delay")
    delay_max = max(
        float(df["baseline_unified_delay"].max()),
        float(df["v1_unified_delay"].max()),
        float(df["v2_unified_delay"].max()),
    )
    ax.set_ylim(0, delay_max * 1.08)
    for idx, row in df.iterrows():
        values = [row["baseline_unified_delay"], row["v1_unified_delay"], row["v2_unified_delay"]]
        max_value = max(values)
        levels = [1 if max_value > 150 and float(v) < max_value * 0.2 else 0 for v in values]
        annotate_bars(
            ax,
            [idx - width, idx, idx + width],
            values,
            fontsize=11,
            extra_levels=levels,
        )

    handles = [
        Patch(facecolor=COLORS["baseline"], label="Raw"),
        Patch(facecolor=COLORS["v1"], label="Smoothed"),
        Patch(facecolor=COLORS["v2"], label="Event-aware"),
    ]
    fig.legend(handles=handles, frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "figure_02_event_tradeoff.png")


def plot_cf_v1_main() -> None:
    df = load_exact_main3_summary().set_index("Dataset").loc[MAIN_DATASETS].reset_index()
    x = np.arange(len(df))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 7.8))

    def annotate_bars(ax, xs, values, x_offsets=None, level_offsets=None, fontsize=13):
        y_max = ax.get_ylim()[1]
        base_pad = max(0.00045, y_max * 0.018)
        if x_offsets is None:
            x_offsets = [0.0] * len(values)
        if level_offsets is None:
            level_offsets = [0] * len(values)
        for x_pos, value, x_offset, level in zip(xs, values, x_offsets, level_offsets):
            ax.text(
                float(x_pos) + float(x_offset),
                float(value) + base_pad * (1.0 + float(level)),
                f"{float(value):.3f}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.12),
                clip_on=False,
            )

    ax = axes[0]
    style_axis(ax)
    add_panel_label(ax, "A")
    ax.bar(
        x - width / 2,
        df["Top3 CF Gain"],
        width=width,
        color=COLORS["top3"],
        label="Top-3 exact repair",
    )
    ax.bar(
        x + width / 2,
        df["Top3 Random Gain"],
        width=width,
        color=COLORS["random_ref"],
        label="Top-3 random repair",
    )
    left_ymax = max(df["Top3 CF Gain"].max(), df["Top3 Random Gain"].max())
    ax.set_ylim(0, left_ymax + 0.006)
    ax.set_title("Top-3 CF Gain vs Random", pad=18)
    ax.set_xticks(x, display_labels(df["Dataset"].tolist()))
    ax.set_ylabel("Normalized score drop")
    for idx, row in df.iterrows():
        annotate_bars(
            ax,
            [idx - width / 2, idx + width / 2],
            [row["Top3 CF Gain"], row["Top3 Random Gain"]],
            x_offsets=[-0.01, 0.01],
            level_offsets=[0, 0 if abs(float(row["Top3 CF Gain"]) - float(row["Top3 Random Gain"])) > 0.004 else 1],
            fontsize=13,
        )

    ax = axes[1]
    style_axis(ax)
    add_panel_label(ax, "B")
    ax.bar(x - width, df["Top1 CF Gain"], width=width, color=COLORS["top1"])
    ax.bar(x, df["Top3 CF Gain"], width=width, color=COLORS["top3"])
    ax.bar(x + width, df["Top5 CF Gain"], width=width, color=COLORS["top5"])
    right_ymax = float(df[["Top1 CF Gain", "Top3 CF Gain", "Top5 CF Gain"]].to_numpy().max())
    ax.set_title("Top-k Exact CF Gain", pad=18)
    ax.set_xticks(x, display_labels(df["Dataset"].tolist()))
    ax.set_ylim(0, right_ymax + 0.008)
    ax.set_ylabel("Normalized score drop")
    for idx, row in df.iterrows():
        values = [row["Top1 CF Gain"], row["Top3 CF Gain"], row["Top5 CF Gain"]]
        if max(values) < 0.01:
            x_offsets = [-0.03, 0.0, 0.03]
            level_offsets = [0, 1, 2]
        else:
            x_offsets = [-0.01, 0.0, 0.01]
            level_offsets = [0, 0, 0]
        annotate_bars(
            ax,
            [idx - width, idx, idx + width],
            values,
            x_offsets=x_offsets,
            level_offsets=level_offsets,
            fontsize=12,
        )

    handles = [
        Patch(facecolor=COLORS["top1"], label="Top-1 exact repair"),
        Patch(facecolor=COLORS["top3"], label="Top-3 exact repair"),
        Patch(facecolor=COLORS["top5"], label="Top-5 exact repair"),
        Patch(facecolor=COLORS["random_ref"], label="Top-3 random repair"),
    ]
    fig.legend(handles=handles, frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.04), fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "figure_03_counterfactual_v1.png")


def plot_cf_v3_ablation() -> None:
    df = load_csv("results_counterfactual_v3_seed_summary.csv")
    df = df[df["Dataset"].isin(MAIN_DATASETS) & df["Ablation"].isin(ABLATIONS)].copy()

    fig, axes = plt.subplots(1, len(MAIN_DATASETS), figsize=(17.6, 7.6), sharey=False)
    y_max = float(df["Top3 CF Gain Mean Mean"].max())
    y_upper = max(0.035, np.ceil((y_max + 0.003) / 0.005) * 0.005)
    y_ticks = np.arange(0.0, y_upper + 1e-9, 0.005)

    for idx, (ax, dataset) in enumerate(zip(axes, MAIN_DATASETS)):
        style_axis(ax)
        add_panel_label(ax, chr(ord("A") + idx))
        subset = df[df["Dataset"] == dataset].set_index("Ablation").loc[ABLATIONS].reset_index()
        x = np.arange(len(subset))
        bars = ax.bar(
            x,
            subset["Top3 CF Gain Mean Mean"],
            color=[COLORS[name] for name in subset["Ablation"]],
            width=0.6,
        )
        ax.set_title(DISPLAY_NAMES.get(dataset, dataset), fontsize=20, weight="bold", color="black")
        ax.set_xticks(x, ["abs", "zscore", "robust"])
        ax.set_ylabel("Top-3 CF gain")
        ax.set_ylim(0.0, y_upper)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:.3f}" for tick in y_ticks])
        for bar, value in zip(bars, subset["Top3 CF Gain Mean Mean"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(value) + 0.0008,
                f"{float(value):.4f}",
                ha="center",
                va="bottom",
                fontsize=20,
                color="black",
            )

    legend_handles = [
        Patch(facecolor=COLORS["abs_same_ref"], label="abs"),
        Patch(facecolor=COLORS["zscore_same_ref"], label="zscore"),
        Patch(facecolor=COLORS["robust_same_ref"], label="robust"),
    ]
    fig.legend(handles=legend_handles, frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.06), fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "figure_04_contribution_ablation.png")


def load_case_seed_details() -> pd.DataFrame:
    path = TABLE_ROOT / f"counterfactual_v1_event_details_seed{CASE_SEED}.csv"
    return pd.read_csv(path)


def build_case_payload(dataset: str) -> dict:
    details = load_case_seed_details()
    subset = details[details["Dataset"] == dataset].copy()
    subset["gain_margin"] = subset["Top3 Gain"] - subset["Top3 Random Gain"]
    row = subset.sort_values(["gain_margin", "Top3 Gain"], ascending=[False, False]).iloc[0]

    cf_config = cf_load_json(ROOT / "configs" / "eval" / "counterfactual_v1.json")
    v2_config = cf_load_json(ROOT / "configs" / "eval" / "event_aware_v2.json")
    loaded = load_case_score_inputs(dataset, cf_config)
    v2_cfg = v2_config["event_aware_v2"]
    point_scores = event_build_point_scores(
        loaded["labels"],
        loaded["window_point_scores"],
        loaded["window_end_indices"],
        aggregation=v2_cfg["aggregation"],
        smoothing=v2_cfg["smoothing"],
        smoothing_param=v2_cfg["smoothing_param"],
    )

    train_data, test_data = load_model_space_arrays(dataset)
    test_data = test_data[: len(loaded["labels"])]
    train_mean = train_data.mean(axis=0).astype(np.float32)
    feature_names = load_feature_names(dataset)

    event = (int(row["Analyzed Event Start"]), int(row["Analyzed Event End"]))
    event_start, event_end = event
    event_segment = np.asarray(test_data[event_start:event_end + 1], dtype=np.float32)
    event_point_scores = point_scores[event_start:event_end + 1]
    contributions = contribution_scores(event_segment, event_point_scores, {"mean": train_mean})
    top5_idx = np.asarray(topk_indices(contributions, 5), dtype=np.int64)
    score_before = float(row["Score Before"])
    after_scores = {
        1: score_before * (1.0 - float(row["Top1 Gain"])),
        3: score_before * (1.0 - float(row["Top3 Gain"])),
        5: score_before * (1.0 - float(row["Top5 Gain"])),
    }
    random_after_scores = {
        1: score_before * (1.0 - float(row["Top1 Random Gain"])),
        3: score_before * (1.0 - float(row["Top3 Random Gain"])),
        5: score_before * (1.0 - float(row["Top5 Random Gain"])),
    }

    return {
        "dataset": dataset,
        "event": event,
        "point_scores": point_scores,
        "labels": loaded["labels"],
        "top5_names": [feature_names[int(idx)] for idx in top5_idx],
        "top5_contributions": [float(contributions[int(idx)]) for idx in top5_idx],
        "score_before": float(score_before),
        "score_after_top1": float(after_scores[1]),
        "score_after_top3": float(after_scores[3]),
        "score_after_top5": float(after_scores[5]),
        "score_after_rand1": float(random_after_scores[1]),
        "score_after_rand3": float(random_after_scores[3]),
        "score_after_rand5": float(random_after_scores[5]),
    }


def plot_case_source(payload: dict) -> Path:
    dataset = payload["dataset"]
    event_start, event_end = payload["event"]
    point_scores = payload["point_scores"]
    labels = payload["labels"]

    margin = max(50, event_end - event_start + 1)
    plot_start = max(0, event_start - margin)
    plot_end = min(len(point_scores) - 1, event_end + margin)
    x = np.arange(plot_start, plot_end + 1)
    local_scores = point_scores[plot_start:plot_end + 1]
    local_labels = labels[plot_start:plot_end + 1]

    fig, axes = plt.subplots(3, 1, figsize=(13.8, 11.6), gridspec_kw={"height_ratios": [2.2, 1.55, 1.6]})

    ax = axes[0]
    style_axis(ax)
    ax.plot(x, local_scores, color=COLORS["point_adjust"], linewidth=1.2)
    ax.axvspan(event_start, event_end, color=COLORS["strict"], alpha=0.12)
    in_true = False
    true_start = 0
    for idx, value in zip(x, local_labels):
        if value == 1 and not in_true:
            true_start = idx
            in_true = True
        elif value == 0 and in_true:
            ax.axvspan(true_start, idx - 1, color=COLORS["stability"], alpha=0.10)
            in_true = False
    if in_true:
        ax.axvspan(true_start, plot_end, color=COLORS["stability"], alpha=0.10)
    ax.set_title(f"{dataset} Case Study", pad=8, fontsize=20, color="black")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Score")

    ax = axes[1]
    style_axis(ax)
    y_pos = np.arange(len(payload["top5_names"]))
    ax.barh(y_pos, payload["top5_contributions"], color=COLORS["point_adjust"])
    ax.set_yticks(y_pos, labels=[display_feature_label(name) for name in payload["top5_names"]])
    ax.invert_yaxis()
    ax.set_title("Top Suspicious Variables", pad=6, fontsize=18, color="black")
    ax.set_xlabel("Contribution")
    ax.grid(axis="x", linestyle="-", linewidth=0.55, alpha=0.18, color="#b4deb6")

    ax = axes[2]
    style_axis(ax)
    bar_labels = ["before", "top1", "top3", "top5", "rand1", "rand3", "rand5"]
    bar_values = [
        payload["score_before"],
        payload["score_after_top1"],
        payload["score_after_top3"],
        payload["score_after_top5"],
        payload["score_after_rand1"],
        payload["score_after_rand3"],
        payload["score_after_rand5"],
    ]
    bar_colors = [COLORS["random_ref"], COLORS["strict"], COLORS["strict"], COLORS["strict"], COLORS["point_adjust"], COLORS["point_adjust"], COLORS["point_adjust"]]
    ax.bar(np.arange(len(bar_labels)), bar_values, color=bar_colors, width=0.72)
    ax.set_xticks(np.arange(len(bar_labels)), labels=bar_labels)
    ax.set_title("Repair Score Comparison", pad=6, fontsize=18, color="black")
    ax.set_ylabel("Event Score")

    fig.tight_layout()
    filename = f"{dataset}_case_source.png"
    save_case_source(fig, filename)
    plt.close(fig)
    return CASE_ROOT / filename


def trim_near_white_border(image: np.ndarray, threshold: float = 0.985, pad: int = 6) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] >= 3:
        rgb = image[..., :3]
        mask = np.any(rgb < threshold, axis=2)
    else:
        mask = image < threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)[:2]
    y1, x1 = coords.max(axis=0)[:2]
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(image.shape[0], y1 + pad + 1)
    x1 = min(image.shape[1], x1 + pad + 1)
    return image[y0:y1, x0:x1]


def soften_case_image(image: np.ndarray, saturation: float = 0.78, lift: float = 0.985) -> np.ndarray:
    # Very large rasterized case images can trigger unnecessary memory spikes here.
    # In that case, keep the image unchanged instead of failing the whole export.
    if image.size > 24_000_000:
        return image
    if image.dtype.kind in {"u", "i"}:
        image = image.astype(np.float32) / 255.0
    image = image.astype(np.float32)
    rgb = image[..., :3]
    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )[..., None]
    softened = gray * (1.0 - saturation) + rgb * saturation
    softened = np.clip(softened * lift + (1.0 - lift), 0.0, 1.0)
    if image.shape[-1] == 4:
        alpha = image[..., 3:4]
        return np.concatenate([softened, alpha], axis=2)
    return softened


def downsample_large_image(image: np.ndarray, max_side: int = 2600) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    step = int(np.ceil(longest / max_side))
    return image[::step, ::step]


def plot_case_montage() -> None:
    datasets = MAIN_DATASETS
    fig, axes = plt.subplots(1, len(datasets), figsize=(22.0, 8.0))

    for idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        path = plot_case_source(build_case_payload(dataset))
        with Image.open(path) as pil_image:
            pil_image = pil_image.convert("RGBA")
            pil_image.thumbnail((1800, 1800))
            image = np.asarray(pil_image, dtype=np.float32) / 255.0
        image = trim_near_white_border(image)
        image = downsample_large_image(image)
        image = soften_case_image(image)
        ax.imshow(image)
        add_panel_label(ax, chr(ord("A") + idx))
        ax.set_title(dataset, fontsize=16, weight="bold", pad=10)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#7bc6be")
        ax.axis("off")

    fig.tight_layout(pad=1.2, w_pad=1.4)
    save(fig, "figure_05_case_studies.png")


def main() -> None:
    configure_style()
    plot_motivation_gap()
    plot_event_aware_main()
    plot_cf_v1_main()
    plot_cf_v3_ablation()
    plot_case_montage()
    for dataset in MAIN_DATASETS:
        plot_case_source(build_case_payload(dataset))


if __name__ == "__main__":
    main()
