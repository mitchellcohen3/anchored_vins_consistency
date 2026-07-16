#!/bin/bash
# Runs a single simulation run of OpenVINS and evaluates the
# results using the evaluate_vins_singlerun_sim.py script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/../../devel/setup.bash"

EVAL_SCRIPT="$SCRIPT_DIR/evaluation/evaluate_vins_singlerun_sim.py"

CONFIG_PATH="$REPO_ROOT/config/rpng_sim/estimator_config.yaml"
DATASET_BASEDIR="$REPO_ROOT/ov_data"

# Select the trajectory to run, from within the ov_data directory
TRAJECTORY="sim/udel_gore"

BASE_SAVE_DIR="/home/mitchell/experiments/ov_sim"
SHOW_PLOTS=false

# Estimator configuration parameters
NAV_STATE_REPRESENTATION="decoupled_right" 
INTEGRATION="analytical"
CONSISTENCY_METHOD="dri_fej" # none | fej | dri_fej
FEAT_REP="GLOBAL_3D" # GLOBAL_3D | ANCHORED_MSCKF_INVERSE_DEPTHA

# Create a new output directory for this run with the current time
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${BASE_SAVE_DIR}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Define the output file paths
FILENAME_EST="$OUTPUT_DIR/poses_est.txt"
FILENAME_GT="$OUTPUT_DIR/poses_gt.txt"
FILENAME_STATE_EST="$OUTPUT_DIR/state_est.txt"
FILENAME_STATE_GT="$OUTPUT_DIR/state_gt.txt"
FILENAME_STATE_STD="$OUTPUT_DIR/state_std.txt"
FILENAME_TIMING="$OUTPUT_DIR/timing.txt"

IMU_TRAJECTORY_PATH="$DATASET_BASEDIR/${TRAJECTORY}.txt"

# ---------------------------------------------------------------------------
echo "Running OpenVINS simulation..."
roslaunch ov_msckf simulation.launch \
    verbosity:=WARNING \
    max_slam:=25 \
    feat_dist_min:=5.0 \
    feat_dist_max:=7.0 \
    sim_traj_path:="$IMU_TRAJECTORY_PATH" \
    seed:=10 \
    max_cameras:=2 \
    feat_rep:="$FEAT_REP" \
    dosave_pose:=true \
    path_state_est:="$FILENAME_STATE_EST" \
    path_state_gt:="$FILENAME_STATE_GT" \
    path_state_std:="$FILENAME_STATE_STD" \
    path_est:="$FILENAME_EST" \
    path_gt:="$FILENAME_GT" \
    config_path:="$CONFIG_PATH" \
    sim_end_time:=-1 \
    integration:="$INTEGRATION" \
    nav_state_representation:="$NAV_STATE_REPRESENTATION" \
    sim_calib_extrinsics:=false \
    sim_calib_cam_intrinsics:=false \
    sim_calib_timeoffset:=false \
    sim_do_calib_imu_intrinsics:=false \
    sim_do_calib_g_sensitivity:=false \
    sim_do_perturbation:=false \
    sim_sigma_pix:=1.0 \
    sim_do_imu_perturbation:=false \
    sim_seed_imu_perturb:=10 \
    sim_sigma_init_att:=1e-6 \
    sim_sigma_init_vel:=1e-6 \
    sim_sigma_init_bg:=0.03 \
    sim_sigma_init_ba:=0.03 \
    init_scaling:=1e-7 \
    record_timing_information:=true \
    record_timing_filepath:="$FILENAME_TIMING" \
    consistency_method:="$CONSISTENCY_METHOD" \
    freq_cam:=10 \
    freq_imu:=400 \
    use_stereo:=true

# For consistency evaluation, if we used the RI method,
# evaluate the results using SE_2(3) and a left perturbation
# otherwise, default OpenVINS is a decoupled/right perturbation for 
# the attitude
STATE_REPRESENTATION="decoupled"
LIE_DIRECTION="right"
if [[ "$CONSISTENCY_METHOD" == "dri_fej" ]]; then
    STATE_REPRESENTATION="SE23"
    LIE_DIRECTION="left"
fi

# Evaluate the results
echo "Evaluating results in $OUTPUT_DIR"

python3 "$EVAL_SCRIPT" \
    --results_dir "$OUTPUT_DIR" \
    --state_representation "$STATE_REPRESENTATION" \
    --lie_direction "$LIE_DIRECTION" \
    --show_plots

echo "Pipeline complete. Results in $OUTPUT_DIR"
