#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEVICE="${DEVICE:-cuda:0}"

CORE_PROTOCOL="experiment_specs/main_accuracy.json"
CORE_CONFIG="experiment_specs/main_selected_hyperparameters.json"
CORE_OUTPUT=".work/main_accuracy"
BASELINE_PROTOCOL="experiment_specs/structural_baselines.json"
BASELINE_OUTPUT=".work/structural_baselines"
SCALE_PROTOCOL="experiment_specs/large_dataset_accuracy.json"
SCALE_OUTPUT=".work/large_dataset_accuracy"

python main_accuracy.py \
    --stage evaluate \
    --protocol "$CORE_PROTOCOL" \
    --frozen-config "$CORE_CONFIG" \
    --outdir "$CORE_OUTPUT" \
    --datasets actor twitch pokec yelp \
    --device "$DEVICE"

python main_accuracy.py \
    --stage summarize \
    --protocol "$CORE_PROTOCOL" \
    --frozen-config "$CORE_CONFIG" \
    --outdir "$CORE_OUTPUT"

python structural_baselines.py \
    --stage prepare \
    --protocol "$BASELINE_PROTOCOL" \
    --outdir "$BASELINE_OUTPUT" \
    --device "$DEVICE"

python structural_baselines.py \
    --stage evaluate \
    --protocol "$BASELINE_PROTOCOL" \
    --outdir "$BASELINE_OUTPUT" \
    --include-full \
    --device "$DEVICE"

python structural_baselines.py \
    --stage summarize \
    --protocol "$BASELINE_PROTOCOL" \
    --outdir "$BASELINE_OUTPUT" \
    --parent-results "$CORE_OUTPUT/evaluation_runs"

python large_dataset_accuracy.py --stage prepare --protocol "$SCALE_PROTOCOL" --outdir "$SCALE_OUTPUT" --device "$DEVICE"
python large_dataset_accuracy.py --stage evaluate-core --protocol "$SCALE_PROTOCOL" --outdir "$SCALE_OUTPUT" --device "$DEVICE"
python large_dataset_accuracy.py --stage evaluate-structural --protocol "$SCALE_PROTOCOL" --outdir "$SCALE_OUTPUT" --device "$DEVICE"
python large_dataset_accuracy.py --stage summarize --protocol "$SCALE_PROTOCOL" --outdir "$SCALE_OUTPUT"
python summarize_main_accuracy.py

python run_experiment.py core-controls mlp-evaluate --device "$DEVICE"
python run_experiment.py core-controls mlp-summarize
python run_experiment.py core-controls random-resampled-evaluate --device "$DEVICE"
python run_experiment.py core-controls random-resampled-summarize

python collect_outputs.py --scope main --require-complete

echo "Core Table 2--4 summaries are under outputs/tables/."
