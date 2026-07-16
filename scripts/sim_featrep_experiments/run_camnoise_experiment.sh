#!/bin/bash

# Runs the Monte-Carlo experiments for varying camera noise levels.

# For each combination of dataset, consistency method, feature representation,
# camera noise level, and trial, launches the ov_msckf simulation node and saves
# the resulting pose estimate. Results are saved in the format expected by the
# OpenVINS evaluation scripts:
#   base_save_dir/algorithms/<consistency_method>_<feat_rep>__sigma<sigma_pix>/<dataset>/run<N>.txt
#   base_save_dir/truths/<dataset>.txt
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/../../devel/setup.bash"

## Add path to the base save directory here 
BASE_SAVE_DIR="$REPO_ROOT/experiments/sim_camnoise_experiment"

CONFIG_PATH="$REPO_ROOT/config/rpng_sim/estimator_config.yaml"
DATA_BASEDIR="$REPO_ROOT/ov_data"


TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${BASE_SAVE_DIR}_${TIMESTAMP}"

DATASETS=(
    # "sim/udel_gore"
    "sim/tum_corridor1_512_16_okvis"
    # "sim/udel_arl"
)

CONSISTENCY_METHODS=(
    "none"
    "fej"
    "dri_fej"
)

FEATURE_REPRESENTATIONS=(
    "GLOBAL_3D"
    "ANCHORED_MSCKF_INVERSE_DEPTH"
)

CAM_NOISE_LEVELS=(
    1.0 
    # 2.0 
    # 3.0 
    4.0
)

NUM_TRIALS=25

# ---------------------------------------------------------------------------
# Run the Monte-Carlo trials for this configuration
ALGORITHM_DIR="$SAVE_DIR/algorithms"
TRUTHS_DIR="$SAVE_DIR/truths"
mkdir -p "$ALGORITHM_DIR" "$TRUTHS_DIR"

echo "[BASH] Results will be saved to: $SAVE_DIR"
echo "[BASH] Running Monte-Carlo trials with the following parameters:"
echo "[BASH] Datasets: ${DATASETS[*]}"
echo "[BASH] Consistency Methods: ${CONSISTENCY_METHODS[*]}"
echo "[BASH] Feature Representations: ${FEATURE_REPRESENTATIONS[*]}"
echo "[BASH] Camera Noise Levels (sigma_pix): ${CAM_NOISE_LEVELS[*]}"

BIG_START_TIME=$(date +%s)

# Loop through each dataset, consistency method, feature representation,
# camera noise level, and perform Monte-Carlo trials for each combination
for DATASET in "${DATASETS[@]}"; do
    for CONSISTENCY_METHOD in "${CONSISTENCY_METHODS[@]}"; do
        for FEAT_REP in "${FEATURE_REPRESENTATIONS[@]}"; do
            for SIGMA_PIX in "${CAM_NOISE_LEVELS[@]}"; do
                for ((TRIAL_NUM = 0; TRIAL_NUM < NUM_TRIALS; TRIAL_NUM++)); do
                    echo "[BASH] Starting trial $TRIAL_NUM"
                    echo "[BASH] Dataset: $DATASET, Consistency Method: $CONSISTENCY_METHOD, Feature Representation: $FEAT_REP, Cam noise: $SIGMA_PIX"

                    IMU_TRAJECTORY_PATH="$DATA_BASEDIR/${DATASET}.txt"

                    FEAT_REP_STR="${FEAT_REP,,}"
                    FEAT_REP_STR="${FEAT_REP_STR//_/}"
                    ESTIMATOR_STR="${CONSISTENCY_METHOD}_${FEAT_REP_STR}__sigma${SIGMA_PIX}"
                    DATASET_US="${DATASET//\//_}"

                    FILENAME_EST="$ALGORITHM_DIR/$ESTIMATOR_STR/$DATASET_US/run${TRIAL_NUM}.txt"
                    FILENAME_GT="$TRUTHS_DIR/${DATASET_US}.txt"

                    roslaunch ov_msckf simulation.launch \
                        verbosity:="ERROR" \
                        sim_traj_path:="$IMU_TRAJECTORY_PATH" \
                        seed:="$TRIAL_NUM" \
                        max_cameras:="2" \
                        feat_dist_min:="5.0" \
                        feat_dist_max:="7.0" \
                        use_stereo:="true" \
                        max_slam:="25" \
                        consistency_method:="$CONSISTENCY_METHOD" \
                        feat_rep:="$FEAT_REP" \
                        dosave_pose:="true" \
                        dosave_state:="false" \
                        path_state_est:="" \
                        path_state_gt:="" \
                        path_state_std:="" \
                        path_est:="$FILENAME_EST" \
                        path_gt:="$FILENAME_GT" \
                        config_path:="$CONFIG_PATH" \
                        sim_sigma_pix:="$SIGMA_PIX" \
                        sim_end_time:="-1" \
                        integration:="analytical" \
                        nav_state_representation:="decoupled_right" \
                        sim_do_imu_perturbation:="false" \
                        freq_cam:="10" \
                        freq_imu:="400" \
                        sim_calib_extrinsics:="false" \
                        sim_calib_cam_intrinsics:="false" \
                        sim_calib_timeoffset:="false" \
                        sim_do_calib_imu_intrinsics:="false" \
                        sim_do_calib_g_sensitivity:="false"
                done
            done
        done
    done
done

BIG_END_TIME=$(date +%s)
echo "[BASH] All trials completed in $((BIG_END_TIME - BIG_START_TIME)) seconds"

cp "${BASH_SOURCE[0]}" "$SAVE_DIR"

# Run our evaluation for each dataset
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_camnoise_experiment_dataset.py"

for DATASET in "${DATASETS[@]}"; do
    DATASET_US="${DATASET//\//_}"
    EVAL_SAVE_DIR="$SAVE_DIR/results/$DATASET_US"
    mkdir -p "$EVAL_SAVE_DIR"
    echo "[BASH] Running evaluation for dataset: $DATASET, saving results to $EVAL_SAVE_DIR"

    python3 "$EVAL_SCRIPT" \
        --results_dir "$SAVE_DIR" \
        --dataset "$DATASET_US" \
        --labels_map_yaml "$SCRIPT_DIR/../labels_map.yaml" \
        --load_from_pickle \
        --save_dir "$EVAL_SAVE_DIR"
done
