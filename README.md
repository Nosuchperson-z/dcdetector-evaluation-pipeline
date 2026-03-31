# DCdetector Minimal Public Release

This folder is a smaller GitHub-ready release that keeps only the main DCdetector pipeline used in the paper.

Included:
- preprocessing scripts
- DCdetector training entrypoint
- official-metric aggregation
- event-aware evaluation (`v1`, `v2`)
- counterfactual evaluation (`v1`, `exact`, `v2`, `v3`)
- paper figure plotting
- multi-seed summary script
- mainline configs for SMAP, MSL, and HAI21.03 only

Excluded:
- raw and processed datasets
- checkpoints
- saved scores, tables, figures, and run artifacts
- auxiliary backbone bridges and third-party backbone-specific scripts
- manuscript drafts and private notes

## Expected layout

```text
repo/
  configs/
  scripts/
  docs/
  KDD2023-DCdetector/
```

The evaluation and training scripts expect the official DCdetector codebase at `KDD2023-DCdetector/`.

## Main evaluation settings kept in this release

- event-aware practical protocol: `configs/eval/event_aware_v2.json`
- exact counterfactual mainline: `configs/eval/counterfactual_exact_main3.json`
- score-weighted baseline: `configs/eval/counterfactual_v1_main3.json`
- robust ablations: `configs/eval/counterfactual_v2.json`, `configs/eval/counterfactual_v3.json`

The public configs in this folder keep the full random-baseline setting:

- `num_random_trials = 20`

The main counterfactual configs are also set to full predicted-event coverage:

- `event_selection_mode = all_predicted`
- no stratified event subsampling

## 50-seed runs

The multi-seed summary script is still included, but note the difference:

- for full-coverage configs, `event_sampling_seed` does not control a stratified event subsample
- seed sweeps are only meaningful if you intentionally switch back to a sampled event-selection mode

If you want a 50-seed sampled ablation, you can still run it by changing `event_selection_mode` back to a sampled setting and varying `event_sampling_seed`.

PowerShell example:

```powershell
0..49 | ForEach-Object {
  $seed = 20260322 + $_
  python scripts/eval/run_counterfactual_v2.py --event-sampling-seed $seed --output-tag seed$seed
}
python scripts/eval/summarize_counterfactual_seed_runs.py --mode v2
```

You can do the same for `run_counterfactual_v1.py` or `run_counterfactual_v3.py`.

## Data and outputs

This release does not ship:
- `data_raw/`
- `data_processed/`
- `outputs/`
- `analysis_figures/`
- `runs/`

You are expected to prepare these locally before running the pipeline.
