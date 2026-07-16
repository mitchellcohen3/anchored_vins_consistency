"""Evaluates the output of the estimator for a single run on real data."""

from ast import arg
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from pyvins.trajectory import TrajectoryResult
from pyvins.plot_utils import set_plot_theme

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

cur_dir = os.path.dirname(os.path.abspath(__file__))
set_plot_theme()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Quadric SLAM Simulation")
    parser.add_argument("--results_dir", type=str, help="Results directory")
    parser.add_argument("--gt_path", type=str, help="Ground truth file path")
    parser.add_argument("--save_dir", type=str, help="Output directory", default=None)
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--show_plots", action="store_true", help="Show plots")
    parser.add_argument("--align", action="store_true", help="Align trajectories")
    parser.add_argument("--lie_direction", type=str, default="right", help="Lie group direction")
    args = parser.parse_args()

    args.results_dir = "/home/mitchell/experiments/openvins_euroc_20260220_163225/MH_01_easy/fej/trial_0"
    args.gt_path = os.path.join(args.results_dir, "poses_gt.txt")

    if args.save_dir is None:
        args.save_dir = os.path.join(args.results_dir, "results")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.verbose = True
    args.show_plots = True
    args.align = True

    # File paths
    state_est_path = os.path.join(args.results_dir, "state_est.txt")
    pose_est_path = os.path.join(args.results_dir, "poses_est.txt")
    state_std_path = os.path.join(args.results_dir, "state_std.txt")
    gt_path = os.path.join(args.results_dir, "poses_gt.txt")
    timing_path = os.path.join(args.results_dir, "timing.txt")

    # If we don't have certain files, set to None
    if not os.path.exists(state_est_path):
        state_est_path = None
    if not os.path.exists(state_std_path):
        state_std_path = None
    if not os.path.exists(timing_path):
        timing_path = None

    result = TrajectoryResult(
        poses_gt_path=gt_path,
        poses_est_path=pose_est_path,
        state_est_path=state_est_path,
        state_std_path=state_std_path,
        timing_path=timing_path,
        lie_direction="right",
        state_representation="decoupled",
        save_dir=args.save_dir,
        align=args.align,
    )
    result.generate_all_plots(args.save_dir)

    att_ate = np.rad2deg(result.pose_error_metrics.attitude_ate)
    pos_ate = result.pose_error_metrics.position_ate
    logging.info(f"Attitude ATE: {att_ate:.8f} deg")
    logging.info(f"Position ATE: {pos_ate:.8f} m")

    # # Now, call openvins evaluation script
    # import subprocess
    # eval_command = f"rosrun ov_eval error_singlerun se3 {args.gt_path} {pose_est_path}"
    # logging.info(f"Running evaluation command: {eval_command}")
    # try:
    #     result = subprocess.run(eval_command, check=True, shell=True)
    # except subprocess.CalledProcessError as e:
    #     logging.error(f"Error running evaluation command: {e}")

    if args.show_plots:
        plt.show()

