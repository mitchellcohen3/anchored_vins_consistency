#!/bin/bash
# Runs the feature representation experiment on all sequences of the TUM-VI dataset.
# Varies the following parameters:
#   1) Consistency method: none, fej, dri_fej
#   2) Feature representation: GLOBAL_3D, ANCHORED_MSCKF_INVERSE_DEPTH
#   3) Number of cameras: 1, 2 - to test mono vs stereo performance

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# source the catkin ws before running this script
source "$SCRIPT_DIR/../../../../devel/setup.bash"
REPO_ROOT="$SCRIPT_DIR/../.."
CONFIG_PATH="$REPO_ROOT/config/tum_vi/estimator_config.yaml"

# specify the dataset base directory here 
DATASET_BASEDIR="/media/mitchell/T7/data/tum_vi"
SAVE_BASEDIR="$REPO_ROOT/experiments/tumvi_featrep_experiment_$(date +%Y%m%d_%H%M%S)"
ALG_SAVE_DIR="$SAVE_BASEDIR/algorithms"
TIMING_SAVE_DIR="$SAVE_BASEDIR/timings"
mkdir -p "$ALG_SAVE_DIR"
mkdir -p "$TIMING_SAVE_DIR"

# Copy the truth folder to the save_basedir for reference
cp -r "$DATASET_BASEDIR/truths" "$SAVE_BASEDIR/truths"

# Copy the estimator config file to the save directory for reference
cp "$CONFIG_PATH" "$SAVE_BASEDIR/"

BAG_NAMES=(
    "dataset-room1_512_16"
    "dataset-room2_512_16"
    "dataset-room3_512_16"
    "dataset-room4_512_16"
    "dataset-room5_512_16"
)
START_TIMES=(0 0 0 0 0)

# Define the parameters to test here
NUM_TRIALS=1
CONSISTENCY_METHODS=(
    "none" 
    "fej" 
    "dri_fej"
)
FEAT_REPS=(
    "GLOBAL_3D"
    "ANCHORED_MSCKF_INVERSE_DEPTH"
)
NUM_CAMERAS=(
    1 
    2
)

big_time=$SECONDS

# Loop over the datasets
for i in "${!BAG_NAMES[@]}"; do
    bag_name="${BAG_NAMES[$i]}"
    start_time="${START_TIMES[$i]}"

    echo "[INFO] Starting experiments for dataset: $bag_name, starting at $start_time s"

    bag_path="$DATASET_BASEDIR/$bag_name.bag"
    path_gt="$DATASET_BASEDIR/truths/$bag_name.txt"

    # Loop over all parameter combinations for this dataset
    for consistency_method in "${CONSISTENCY_METHODS[@]}"; do
        for feat_rep in "${FEAT_REPS[@]}"; do
            for num_camera in "${NUM_CAMERAS[@]}"; do
                for trial_num in $(seq 0 $((NUM_TRIALS - 1))); do

                    estimator_str="${consistency_method}_${feat_rep}__${num_camera}cam"

                    filename_est="$ALG_SAVE_DIR/$estimator_str/$bag_name/${trial_num}_estimate.txt"
                    filename_time="$TIMING_SAVE_DIR/$estimator_str/$bag_name/${trial_num}_timing.txt"

                    small_time=$SECONDS
                    roslaunch ov_msckf serial.launch \
                        max_cameras:="$num_camera" \
                        use_stereo:=true \
                        consistency_method:="$consistency_method" \
                        feat_rep:="$feat_rep" \
                        config_path:="$CONFIG_PATH" \
                        bag:="$bag_path" \
                        bag_start:="$start_time" \
                        dobag:=true \
                        dosave:=false \
                        path_est:="$filename_est" \
                        verbosity:=ERROR \
                        dotime:=true \
                        path_time:="$filename_time" \
                        dolivetraj:=false \
                        dataset:="$bag_name" \
                        path_gt:="$path_gt"

                    if [ $? -ne 0 ]; then
                        echo "[ERROR] Error running openvins serial!"
                    fi
                    echo "[INFO] Trial finished in $((SECONDS - small_time)) s."
                done
            done
        done
    done
done

echo "[INFO] All trials completed in $((SECONDS - big_time)) s."
echo "[INFO] Results saved in $SAVE_BASEDIR"

# Copy the script to the save directory for reference
cp "$0" "$SAVE_BASEDIR/$(basename "$0")"
