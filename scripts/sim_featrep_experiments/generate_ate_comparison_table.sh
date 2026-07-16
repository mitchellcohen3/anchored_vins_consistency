#!/bin/bash

# Runs the OpenVINS "error_comparison" executable to compute ATE/RPE CSVs
# (ate_results.csv, ate_2d_results.csv, rpe_results.csv, rpe_raw_samples.csv)
# from trajectory estimates from several algorithms on several datasets.
#
# Then generates the noise comparison LaTeX table from those CSVs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/../../devel/setup.bash"

ALIGN_MODE="none"
BASEDIR="/home/mitchell/workspaces/anchored_vins_analysis_ws/src/anchored_vins_analysis/experiments/sim_camnoise_experiment_merged"
FOLDER_GT="$BASEDIR/truths"
FOLDER_ALGORITHMS="$BASEDIR/algorithms"

NOISE_LEVELS=(1.0 4.0)

ALGORITHMS=(
    "none_global3d"
    "fej_global3d"
    "dri_fej_global3d"
    "none_anchoredmsckfinversedepth"
    "fej_anchoredmsckfinversedepth"
    "dri_fej_anchoredmsckfinversedepth"
)

echo "[BASH] Running error_comparison: align_mode=$ALIGN_MODE, gt=$FOLDER_GT, algorithms=$FOLDER_ALGORITHMS"

rosrun ov_eval error_comparison \
    "$ALIGN_MODE" \
    "$FOLDER_GT" \
    "$FOLDER_ALGORITHMS" \
    --no-plots

echo "[BASH] CSVs written to $FOLDER_ALGORITHMS"

python3 "$SCRIPT_DIR/generate_noise_comparison_table.py" \
    --folder_algorithms "$FOLDER_ALGORITHMS" \
    --algorithms "${ALGORITHMS[@]}" \
    --noise_levels "${NOISE_LEVELS[@]}" \
    "$@"

echo "[BASH] CSVs written to $FOLDER_ALGORITHMS"

python3 "$SCRIPT_DIR/generate_noise_comparison_table.py" \
    --folder_algorithms "$FOLDER_ALGORITHMS" \
    "$@"
