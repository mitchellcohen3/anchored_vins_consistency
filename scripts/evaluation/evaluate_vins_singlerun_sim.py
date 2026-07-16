"""Evaluates the output of the estimator for a single simulation run.

The estimator is assumed to generate the following files:
    - imu_cov.txt
    - imu_data.txt
    - imu_est.txt
    - imu_gt.txt
    - new_params.yaml
    - timing_information.txt (optional)
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from pyvins.trajectory import VinsSimulationResult
from pyvins.plot_utils import set_plot_theme

logging.basicConfig(level=logging.INFO)

set_plot_theme()
plt.rc("grid", linestyle="--", alpha=0.8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates one run of the VINS simulation"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Results directory"
    )
    parser.add_argument("--save_dir", type=str, help="Directory to save plots")
    parser.add_argument("--show_plots", action="store_true", help="Show plots")
    parser.add_argument(
        "--state_representation",
        type=str,
        default="decoupled",
        choices=["decoupled", "SE23"],
        help="State representation used by the estimator",
    )
    parser.add_argument(
        "--lie_direction",
        type=str,
        default="right",
        choices=["left", "right"],
        help="Lie direction used by the estimator",
    )
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = os.path.join(args.results_dir, "plots")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    result = VinsSimulationResult.from_files(
        os.path.join(args.results_dir, "state_est.txt"),
        os.path.join(args.results_dir, "state_gt.txt"),
        os.path.join(args.results_dir, "state_std.txt"),
        poses_est_path=os.path.join(args.results_dir, "poses_est.txt"),
        timing_fpath=os.path.join(args.results_dir, "timing.txt"),
        state_representation=args.state_representation,
        lie_direction=args.lie_direction,
        plot_save_dir=args.save_dir,
    )
    result.generate_plots(args.save_dir)
    logging.info(f"Plots saved to {args.save_dir}")

    if args.show_plots:
        plt.show()

    logging.info("==============================")
    logging.info(f"Position ATE: {result.pose_error_metrics.position_ate:.4f} m") 
    logging.info(f"Orientation ATE: {np.rad2deg(result.pose_error_metrics.attitude_ate):.4f} deg")
    logging.info("==============================")