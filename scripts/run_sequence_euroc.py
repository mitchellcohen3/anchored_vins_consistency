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
    args = parser.parse_args()

    # Specify dataset directory,
    args.dataset_basedir = "/home/mitchell/Documents/data/euroc_mav"
    args.config_path = "/home/mitchell/Documents/catkin_ws_visual_inertial/src/open_vins/config/euroc_mav/estimator_config.yaml"

    save_basedir = "/home/mitchell/experiments/ov_euroc_featrep_mono_newbranch"
    save_basedir = create_new_folder(save_basedir, add_timestamp=False)
    args.alg_save_dir = os.path.join(save_basedir, "algorithms")
    args.timing_save_dir = os.path.join(save_basedir, "timings")
    args.run_all = False
    args.run_eval = False

    # Create a new save directory with a timestamp
    # args.save_dir = create_new_folder(args.alg_save_dir, add_timestamp=False)
    # Create new directories for the timing and algorithm results
    os.makedirs(args.alg_save_dir, exist_ok=True)
    os.makedirs(args.timing_save_dir, exist_ok=True)

    if args.run_all:
        bag_names = [
            "V1_01_easy",
            "V1_02_medium",
            "V1_03_difficult",
            "V2_01_easy",
            "V2_02_medium",
            "V2_03_difficult",
            "MH_01_easy",
            "MH_02_easy",
            "MH_03_medium",
            "MH_04_difficult",
            "MH_05_difficult",
        ]

        start_times = [5, 0, 0, 0, 0, 0, 40, 35, 10, 15, 5]
        # start_times = [0] * len(bag_names)
    else:
        bag_names = ["MH_01_easy"]
        start_times = [40]

    if len(bag_names) != len(start_times):
        raise ValueError("bag_names and start_times must have the same length")

    # For each dataset, loop through all possible configurations
    big_time = time.time()
    for start_time, bag_name in zip(start_times, bag_names):

        logging.info(
            f"Starting experiments for dataset: {bag_name}, starting at {start_time} s"
        )
        estimator_str = "ov_msckf"

        bag_path = os.path.join(args.dataset_basedir, bag_name + ".bag")
        path_gt = os.path.join(args.dataset_basedir, "truths", bag_name + ".txt")

        # Create the save directory for this trial 
        filename_est = os.path.join(args.alg_save_dir, estimator_str, bag_name, f"0_estimate.txt")
        filename_time = os.path.join(args.timing_save_dir, estimator_str, bag_name, f"0_timing.txt")

        cmd = ["roslaunch", "ov_msckf", "serial.launch"]
        cmd_args = [
            "max_cameras:=" + "1",
            "use_stereo:=" + "false",
            "consistency_method:=" + "fej",
            "config_path:=" + args.config_path,
            "bag:=" + bag_path,
            "bag_start:=" + str(start_time),
            "dobag:=" + "true",
            "dosave:=" + "true",
            "path_est:=" + filename_est,
            "verbosity:=" + "WARNING",
            "feat_rep:=" + "GLOBAL_3D",
            "path_gt:=" + "",
            "dotime:=" + "true",
            "path_time:=" + filename_time,
        ]

        cmd.extend(cmd_args)

        small_time = time.time()
        try:
            result = subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error("Error running openvins serial!")
        except FileNotFoundError as e:
            logging.error("File not found!")
        logging.info(f"Trial finished in {time.time() - small_time} s.")

    logging.info(f"All trials completed in {time.time() - big_time} s.")
    logging.info(f"Results saved in {save_basedir}")

    # Copy the script to the save directory for reference 
    shutil.copy2(os.path.abspath(__file__), os.path.join(save_basedir, os.path.basename(__file__)))