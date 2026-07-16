#!/bin/bash

# Runs Monte-Carlo trials of OpenVINS on a set of simulation datasets,
# then evaluates each dataset's results using evaluate_vins_dataset.py.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/../../devel/setup.bash"

MONTECARLO_SCRIPT="$SCRIPT_DIR/run_ov_montecarlo.py"
EVAL_SCRIPT="$SCRIPT_DIR/evaluation/evaluate_vins_dataset.py"

DATASET_BASEDIR="$REPO_ROOT/ov_data"
CONFIG_PATH="$REPO_ROOT/config/rpng_sim/estimator_config.yaml"

# Datasets to run Monte-Carlo trials on
DATASETS=("sim/udel_gore")

CONSISTENCY_METHODS=(
    "none"
    "fej"
    "dri_fej"
)
FEATURE_REPRESENTATIONS=(
    "GLOBAL_3D"
    "ANCHORED_MSCKF_INVERSE_DEPTH"
)

BASE_SAVE_DIR="/home/mitchell/experiments/ov_montecarlo"
NUM_TRIALS=25

# ---------------------------------------------------------------------------
# Create the save directory and run the Monte-Carlo trials
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${BASE_SAVE_DIR}_${TIMESTAMP}"

echo "Running Monte-Carlo trials..."
python3 "$MONTECARLO_SCRIPT" \
    --dataset_basedir "$DATASET_BASEDIR" \
    --datasets "${DATASETS[@]}" \
    --config_path "$CONFIG_PATH" \
    --save_basedir "$SAVE_DIR" \
    --consistency_methods "${CONSISTENCY_METHODS[@]}" \
    --feature_representations "${FEATURE_REPRESENTATIONS[@]}" \
    --num_trials "$NUM_TRIALS"

# ---------------------------------------------------------------------------
for dataset in "${DATASETS[@]}"; do
    results_dir="$SAVE_DIR/algorithms"
    gt_path="$SAVE_DIR/truths/${dataset//\//_}.txt"
    python3 "$EVAL_SCRIPT" \
        --results_dir "$results_dir" \
        --gt_path "$gt_path" \
        --save_dir "$SAVE_DIR/results" \
        --show_plots
done

echo "Monte-Carlo trials and evaluation complete. Results saved in $SAVE_DIR"