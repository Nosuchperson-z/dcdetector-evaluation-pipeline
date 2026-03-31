import argparse
import copy
from pathlib import Path

from run_counterfactual_v2 import (
    ROOT,
    load_cf_v1_reference,
    load_csv_rows,
    load_json,
)
from run_counterfactual_v2 import analyze_dataset as analyze_dataset_v2
from run_counterfactual_v1 import (
    build_model_bundle,
    parse_optional_bool,
    resolve_output_tag,
    tagged_output_dir,
    tagged_output_path,
)
from run_event_aware_v2 import format_value, write_csv, write_xlsx


TABLE_ROOT = ROOT / "outputs" / "tables"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Counterfactual v3 ablations on top of v2 pipeline.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval" / "counterfactual_v3.json"))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--num-random-trials", type=int, default=None)
    parser.add_argument("--event-sampling-seed", type=int, default=None)
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--save-case-figures", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def build_cf_config(base_config: dict, global_cfg: dict, ablation: dict, args) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["contribution_method"] = ablation["contribution_method"]
    cfg["random_baseline_modes"] = list(ablation["random_baseline_modes"])
    cfg["deletion_method"] = ablation["deletion_method"]
    cfg["save_case_figures"] = bool(global_cfg["save_case_figures"])
    if args.datasets:
        cfg["datasets"] = args.datasets
    else:
        cfg["datasets"] = list(global_cfg["datasets"])
    if args.max_events is not None:
        cfg["max_events_per_dataset"] = int(args.max_events)
    if args.num_random_trials is not None:
        cfg["num_random_trials"] = int(args.num_random_trials)
    if args.event_sampling_seed is not None:
        cfg["event_sampling_seed"] = int(args.event_sampling_seed)
    save_case_figures_override = parse_optional_bool(args.save_case_figures)
    if save_case_figures_override is not None:
        cfg["save_case_figures"] = save_case_figures_override
    output_tag = resolve_output_tag(args.output_tag, args.event_sampling_seed)
    cfg["analysis_dir"] = str(tagged_output_dir(Path(global_cfg["analysis_dir"]) / ablation["name"], output_tag))
    return cfg


def main():
    args = parse_args()
    global_cfg = load_json(args.config)
    base_cf_v2 = load_json(ROOT / global_cfg["base_counterfactual_v2_config"])
    v2_config = load_json(ROOT / base_cf_v2["event_config"])
    baseline_rows = load_csv_rows(TABLE_ROOT / "event_aware_v2_comparison.csv")
    cf_v1_rows = load_cf_v1_reference()

    output_tag = resolve_output_tag(args.output_tag, args.event_sampling_seed)
    results_csv = tagged_output_path(ROOT / global_cfg["results_csv"], output_tag)
    comparison_csv = tagged_output_path(ROOT / global_cfg["comparison_csv"], output_tag)
    results_xlsx = tagged_output_path(ROOT / global_cfg["results_xlsx"], output_tag)

    summary_rows = []
    comparison_rows = []

    datasets = args.datasets if args.datasets else global_cfg["datasets"]

    for ablation in global_cfg["ablations"]:
        cfg = build_cf_config(base_cf_v2, global_cfg, ablation, args)
        for dataset in datasets:
            print(f"processing {ablation['name']} / {dataset}")
            bundle = build_model_bundle(dataset, requested_device=args.device)
            summary_row, _, _ = analyze_dataset_v2(
                dataset=dataset,
                cf_config=cfg,
                v2_config=v2_config,
                bundle=bundle,
                baseline_rows=baseline_rows,
            )
            cf_v1_ref = cf_v1_rows[dataset]
            summary_rows.append(
                {
                    "Ablation": ablation["name"],
                    "Dataset": dataset,
                    "Contribution Method": cfg["contribution_method"],
                    "Random Baseline Modes": ",".join(cfg["random_baseline_modes"]),
                    "Deletion Method": cfg["deletion_method"],
                    "Total Predicted Event Count": summary_row["Total Predicted Event Count"],
                    "Analyzed Event Count": summary_row["Analyzed Event Count"],
                    "Selection Mode": summary_row["Selection Mode"],
                    "Event Sampling Seed": int(cfg.get("event_sampling_seed", 20260322)),
                    "Threshold Selection Length": summary_row["Threshold Selection Length"],
                    "Evaluation Offset": summary_row["Evaluation Offset"],
                    "Detector Unified F1 (from Event-aware v2)": format_value(summary_row["Detector Unified F1"]),
                    "Detector Unified Fc1 (from Event-aware v2)": format_value(summary_row["Detector Unified Fc1"]),
                    "Detector Unified Delay (from Event-aware v2)": format_value(summary_row["Detector Unified Delay"]),
                    "Top1 CF Gain Mean": format_value(summary_row["Top1 CF Gain Mean"]),
                    "Top3 CF Gain Mean": format_value(summary_row["Top3 CF Gain Mean"]),
                    "Top5 CF Gain Mean": format_value(summary_row["Top5 CF Gain Mean"]),
                    "Top3 CF Gain Median": format_value(summary_row["Top3 CF Gain Median"]),
                    "Top3 Positive Gain Ratio": format_value(summary_row["Top3 Positive Gain Ratio"]),
                    "Top3 Random A Gain Mean": format_value(summary_row["Top3 Random A Gain Mean"]),
                    "Top3 Random B Gain Mean": format_value(summary_row["Top3 Random B Gain Mean"]),
                    "Top3 Win Rate vs Random A": format_value(summary_row["Top3 Win Rate vs Random A"]),
                    "Top3 Win Rate vs Random B": format_value(summary_row["Top3 Win Rate vs Random B"]),
                    "Top3 Deletion Gain Mean": format_value(summary_row["Top3 Deletion Gain Mean"]),
                    "Top3 Stability": format_value(summary_row["Top3 Stability"]),
                }
            )
            comparison_rows.append(
                {
                    "Ablation": ablation["name"],
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
                    "event_sampling_seed": int(cfg.get("event_sampling_seed", 20260322)),
                    "top3_cf_gain_v1": format_value(cf_v1_ref["top3_cf_gain_v1"]),
                    "top3_cf_gain_v3": format_value(summary_row["Top3 CF Gain Mean"]),
                    "top3_random_gain_v1": format_value(cf_v1_ref["top3_random_gain_v1"]),
                    "top3_random_a_gain_v3": format_value(summary_row["Top3 Random A Gain Mean"]),
                    "top3_random_b_gain_v3": format_value(summary_row["Top3 Random B Gain Mean"]),
                    "top3_deletion_gain_v3": format_value(summary_row["Top3 Deletion Gain Mean"]),
                    "topk_stability_v1": format_value(cf_v1_ref["topk_stability_v1"]),
                    "topk_stability_v3": format_value(summary_row["Top3 Stability"]),
                }
            )
            del bundle

    if not summary_rows:
        raise RuntimeError("No ablation result was produced.")

    write_csv(results_csv, list(summary_rows[0].keys()), summary_rows)
    write_csv(comparison_csv, list(comparison_rows[0].keys()), comparison_rows)
    write_xlsx(
        results_xlsx,
        [
            ("counterfactual_v3", summary_rows),
            ("comparison", comparison_rows),
        ],
    )
    print(results_csv)
    print(comparison_csv)
    print(results_xlsx)


if __name__ == "__main__":
    main()
