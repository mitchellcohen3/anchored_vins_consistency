"""Runs Monte-Carlo trials for OpenVins on simulated trajectories."""

import subprocess
import time
import os
import argparse
from matplotlib import pyplot as plt
import typing
import numpy as np
import seaborn as sns
import shutil

from pyvins.plot_utils import set_plot_theme
from pyvins.file_utils import create_new_folder

import logging

import itertools

logging.basicConfig(level=logging.INFO)

cur_dir = os.path.dirname(os.path.abspath(__file__))
set_plot_theme(enable_latex=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_basedir",
        type=str,
        default=os.path.join(cur_dir, "../ov_data/"),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["sim/udel_gore"],
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(cur_dir, "../config/rpng_sim/estimator_config.yaml"),
    )
    parser.add_argument(
        "--save_basedir",
        type=str,
        default="/home/mitchell/experiments/ov_montecarlo",
    )
    parser.add_argument(
        "--consistency_methods",
        type=str,
        nargs="+",
        default=["none", "fej", "dri_fej"],
    )
    parser.add_argument(
        "--feature_representations",
        type=str,
        nargs="+",
        default=[
            "GLOBAL_3D",
            "GLOBAL_FULL_INVERSE_DEPTH",
            "ANCHORED_3D",
            "ANCHORED_FULL_INVERSE_DEPTH",
            "ANCHORED_MSCKF_INVERSE_DEPTH",
        ],
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="Number of Monte-Carlo trials to run for each parameter combination",
    )
    args = parser.parse_args()

    args.save_basedir = create_new_folder(args.save_basedir, add_timestamp=False)
    alg_save_dir = os.path.join(args.save_basedir, "algorithms")
    timing_save_dir = os.path.join(args.save_basedir, "timings")
    result_save_dir = os.path.join(args.save_basedir, "results")
    full_state_dir = os.path.join(args.save_basedir, "full_state")
    truths_dir = os.path.join(args.save_basedir, "truths")
    os.makedirs(alg_save_dir, exist_ok=True)
    os.makedirs(timing_save_dir, exist_ok=True)
    os.makedirs(result_save_dir, exist_ok=True)
    os.makedirs(full_state_dir, exist_ok=True)
    os.makedirs(truths_dir, exist_ok=True)

    # Copy the default config file to the experiment directory
    shutil.copy2(args.config_path, args.save_basedir)

    # Loop through all parameter combinations
    param_combinations = itertools.product(
        args.datasets,
        args.consistency_methods,
        args.feature_representations,
        range(args.num_trials),
    )
    big_time = time.time()
    for combo in param_combinations:
        dataset, consistency_method, feat_rep, trial_num = combo
        small_time = time.time()
        logging.info(f"Starting trial {trial_num}")
        logging.info(f"Dataset: {dataset}, Consistency Method: {consistency_method}")
        logging.info(f"Feature Representation: {feat_rep}, Trial Number: {trial_num}")

        imu_trajectory_path = os.path.join(args.dataset_basedir, dataset + ".txt")

        # Create the experiment directory
        estimator_str = consistency_method
        estimator_str += "_" + feat_rep

        dataset_str = dataset.replace("/", "_") 
        filename_est = os.path.join(
            alg_save_dir,
            estimator_str,
            dataset_str,
            f"run_{trial_num}.txt",
        )

        filename_gt = os.path.join(
            truths_dir,
            dataset_str + ".txt",
        )

        # Full state save information
        path_save_full_state = os.path.join(
            full_state_dir,
            estimator_str,
            dataset_str,
            f"run_{trial_num}.txt",
        )
        filename_state_est = os.path.join(path_save_full_state, "state_est.txt")
        filename_state_gt = os.path.join(path_save_full_state, "state_gt.txt")
        filename_state_std = os.path.join(path_save_full_state, "state_std.txt")

        # timing information save path
        filename_timing = os.path.join(
            timing_save_dir,
            estimator_str,
            dataset_str,
            f"run_{trial_num}.txt",
        )


        cmd = ["roslaunch", "ov_msckf", "simulation.launch"]

        cmd_args = [
            "verbosity:=" + "WARNING",
            "sim_traj_path:=" + imu_trajectory_path,
            "seed:=" + str(trial_num),
            "max_cameras:=" + "2",
            "consistency_method:=" + consistency_method,
            "max_slam:=" + "25",
            "feat_rep:=" + feat_rep,
            "dosave_pose:=" + "true",
            "path_state_est:=" + filename_state_est,
            "path_state_gt:=" + filename_state_gt,
            "path_state_std:=" + filename_state_std,
            "path_est:=" + filename_est,
            "path_gt:=" + filename_gt,
            "config_path:=" + args.config_path,
            "sim_sigma_pix:=" + "1.0",
            "sim_end_time:=" + "-1",
            "integration:=" + "analytical",
            "nav_state_representation:=" + "decoupled_right",
            "sim_do_imu_perturbation:=" + "false",
            "sim_seed_imu_perturb:=" + "10",
            "sim_sigma_init_att:=" + "0.1",
            "sim_sigma_init_vel:=" + "0.1",
            "sim_sigma_init_bg:=" + "0.03",
            "sim_sigma_init_ba:=" + "0.03",
            "init_scaling:=" + "1e-7",
            "record_timing_information:=" + "true",
            "record_timing_filepath:=" + filename_timing,
            "freq_cam:=" + "10",
            "freq_imu:=" + "400",
            "use_stereo:=" + "true",
            "feat_dist_min:=" + "5.0",
            "feat_dist_max:=" + "7.0",
        ]

        cmd.extend(cmd_args)
        try:
            result = subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("Error running command: ", e)
        except FileNotFoundError as e:
            print("Error : command not found: ", e)

    logging.info(f"All trials completed in {time.time() - big_time:.2f} seconds")
    logging.info(f"Results saved to {args.save_basedir} ")
