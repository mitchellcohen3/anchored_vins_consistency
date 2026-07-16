#!/usr/bin/env bash

# Runs the error_comparison script on the results of the TUM-VI dataset, and then
# generates the ATE table for the paper
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="/home/mitchell/workspaces/visual_inertial_ws"
source "${WS_DIR}/devel/setup.bash"

## Specify the base experiment directory here!!
EXP_DIR="/home/mitchell/experiments/ov_tumvi_featrep_20260710_151734"
ALGORITHM_DIR="${EXP_DIR}/algorithms"
OUTPUT_DIR="${EXP_DIR}/results"

mkdir -p ${OUTPUT_DIR}
TRUTH_DIR="/media/mitchell/T7/data/tum_vi/truths"

# Call the error_comparison script from OpenVINS to generate
# ATE/RPE CSVs for all algorithms on all datasets
rosrun ov_eval error_comparison \
    "posyaw" \
    ${TRUTH_DIR} \
    ${ALGORITHM_DIR} \
    "--no-plots" \
    "--output-dir" ${OUTPUT_DIR}

# Now, generate ATE table and RPE boxplot
EVAL_SCRIPT="${SCRIPT_DIR}/compare_mono_vs_stereo.py"
LABELS_MAP_YAML="${SCRIPT_DIR}/../labels_map.yaml"

ALGORITHMS=(
    "none_GLOBAL_3D"
    "fej_GLOBAL_3D"
    "dri_fej_GLOBAL_3D"
    "none_ANCHORED_MSCKF_INVERSE_DEPTH"
    "fej_ANCHORED_MSCKF_INVERSE_DEPTH"
    "dri_fej_ANCHORED_MSCKF_INVERSE_DEPTH"
)

DATASETS=(
    "dataset-room1_512_16"
    "dataset-room2_512_16"
    "dataset-room3_512_16"
    "dataset-room4_512_16"
    "dataset-room5_512_16"
)

python3 ${EVAL_SCRIPT} \
    --results_folder ${OUTPUT_DIR} \
    --labels_map_yaml ${LABELS_MAP_YAML} \
    --algorithms ${ALGORITHMS[@]} \
    --datasets ${DATASETS[@]} \