"""Runs OpenVins on all sequences of the EuroC dataset."""

import matplotlib.pyplot as plt
import shutil
import subprocess
import os
import argparse
import json
import time

import logging
import itertools

from pyvins.file_utils import create_new_folder

from pyvins.trajectory import TrajectoryResult
from pyvins.plot_utils import set_plot_theme

logging.basicConfig(level=logging.INFO)

cur_dir = os.path.dirname(os.path.abspath(__file__))

set_plot_theme()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_basedir", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--dataset", type=str, default="sim/udel_gore")
    parser.add_argument("--run_all", action="store_true", help="Run all configurations")
    parser.add_argument(
        "--run_eval", action="store_true", help="Run evaluation after each trial"
    )
    args = parser.parse_args()

    # Specify dataset directory,
    args.dataset_basedir = "/home/mitchell/Documents/data/tum_vi"
    args.config_path = "/home/mitchell/Documents/catkin_ws_visual_inertial/src/open_vins/config/tum_vi/estimator_config.yaml"

    save_basedir = "/home/mitchell/experiments/openvins_tumvi_stereo_newbranch_withcalib"
    save_basedir = create_new_folder(save_basedir, add_timestamp=True)
    args.alg_save_dir = os.path.join(save_basedir, "algorithms")
    args.timing_save_dir = os.path.join(save_basedir, "timings")

    # Copy the estimator config file to the save directory for reference
    shutil.copy2(
        args.config_path,
        save_basedir,
    )

    os.makedirs(args.alg_save_dir, exist_ok=True)
    os.makedirs(args.timing_save_dir, exist_ok=True)

    args.run_all = True
    args.run_eval = False

    if args.run_all:
        bag_names = [
            "dataset-room1_512_16",
            "dataset-room2_512_16",
            "dataset-room3_512_16",
            "dataset-room4_512_16",
            "dataset-room5_512_16",
            # "dataset-room6_512_16",
        ]

        start_times = [0] * len(bag_names)
    else:
        bag_names = ["dataset-room1_512_16"]
        start_times = [0]

    if len(bag_names) != len(start_times):
        raise ValueError("bag_names and start_times must have the same length")

    verbosity = "INFO"

    # Define the parameters to test here
    num_trials = 1
    consistency_method_list = ["none", "fej"]
    feat_rep_list = [
        "GLOBAL_3D",
        "ANCHORED_MSCKF_INVERSE_DEPTH",
        "ANCHORED_3D",
    ]
    init_to_gt = False

    if init_to_gt:
        logging.error("Init to gt not yet implemented!")
        logging.error("Exiting...")
        exit(1)

    big_time = time.time()
    # Lopo over the datasets
    for start_time, bag_name in zip(start_times, bag_names):

        logging.info(
            f"Starting experiments for dataset: {bag_name}, starting at {start_time} s"
        )

        param_combinations = itertools.product(
            consistency_method_list,
            feat_rep_list,
            range(num_trials),
        )

        # Loop over all parameter combinations for this dataset
        for cur_combination in param_combinations:
            consistency_method, feat_rep, trial_num = cur_combination
            bag_path = os.path.join(args.dataset_basedir, bag_name + ".bag")
            path_gt = os.path.join(args.dataset_basedir, "truths", bag_name + ".txt")

            estimator_str = consistency_method
            estimator_str += "_" + feat_rep.lower().replace("_", "")

            # Create the save directory for this trial
            filename_est = os.path.join(
                args.alg_save_dir,
                estimator_str,
                bag_name,
                f"{trial_num}_estimate.txt",
            )
            filename_time = os.path.join(
                args.timing_save_dir,
                estimator_str,
                bag_name,
                f"{trial_num}_timing.txt",
            )

            cmd = ["roslaunch", "ov_msckf", "serial.launch"]
            cmd_args = [
                "max_cameras:=" + "2",
                "use_stereo:=" + "true",
                "consistency_method:=" + consistency_method,
                "config_path:=" + args.config_path,
                "bag:=" + bag_path,
                "bag_start:=" + str(start_time),
                "dobag:=" + "true",
                "dosave:=" + "true",
                "path_est:=" + filename_est,
                "verbosity:=" + "WARNING",
                "feat_rep:=" + feat_rep,
                "path_gt:=" + "",
                "dotime:=" + "true",
                "path_time:=" + filename_time,
            ]

            cmd.extend(cmd_args)

            small_time = time.time()
            try:
                result = subprocess.run(cmd, check=True)
                # logging.info(f"Would run command: {' '.join(cmd)}")
            except subprocess.CalledProcessError as e:
                logging.error("Error running openvins serial!")
            except FileNotFoundError as e:
                logging.error("File not found!")
            logging.info(f"Trial finished in {time.time() - small_time} s.")

    logging.info(f"All trials completed in {time.time() - big_time} s.")
    logging.info(f"Results saved in {save_basedir}")

    # Copy the script to the save directory for reference
    shutil.copy2(
        os.path.abspath(__file__),
        os.path.join(save_basedir, os.path.basename(__file__)),
    )
