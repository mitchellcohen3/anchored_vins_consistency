#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

result_basedir="/home/mitchell/experiments/ov_montecarlo_20260710_123804"
results_dir="${result_basedir}/algorithms"
gt_path="${result_basedir}/truths/sim_udel_gore.txt"

EVAL_SCRIPT="$SCRIPT_DIR/evaluation/evaluate_vins_dataset.py"
python3 ${EVAL_SCRIPT} \
    --results_dir ${results_dir} \
    --gt_path ${gt_path} \
    --save_dir "${result_basedir}/results" \
    --show_plots \
    --labels_map_yaml "${SCRIPT_DIR}/labels_map.yaml"

# Now, call the openvins script
# rosrun ov_eval error_dataset se3 ${gt_path} ${results_dir}