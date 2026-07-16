"""This script loads a trajectory from a TUM format file, plots
it, and does some analysis on the trajectory.

This script is helpful when trying to get a sense of the particular 
characteristics of a trajectory, such as the velocity profile, 
or to check for any issues with the trajectory file itself.A
"""

import os

import logging
import numpy as np
import matplotlib.pyplot as plt

import argparse

from pyvins.plot_utils import set_plot_theme
from pyvins.file_utils import load_poses_from_tum
from pyvins.trajectory import TrajectoryAnalyzer

from navlie.utils import plot_poses
from navlie.bspline import SE3Bspline

set_plot_theme(enable_latex=True)

# Configure logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot a trajectory from a results file.")
    parser.add_argument(
        "--traj_file", type=str, required=False, help="Path to the trajectory file. "
    )
    parser.add_argument(
        "--plot_save_dir",
        type=str,
        default=None,
        help="Directory to save plots to. If not set, plots are shown instead. ",
    )
    args = parser.parse_args()

    base_folder = (
        "/home/mitchell/Documents/catkin_ws_visual_inertial/src/open_vins/ov_data/"
    )
    # traj_file_name = "udel_gore.txt"
    # traj_file_name = "udel_arl.txt"
    # traj_file_name = "tum_corridor1_512_16_okvis.txt"
    # traj_file_name = "udel_gore.txt"
    # args.traj_file = os.path.join(base_folder, "kaist/urban28.txt")
    # args.traj_file = os.path.join(base_folder, "euroc_mav/V1_01_easy.txt")
    # args.traj_file = os.path.join(base_folder, "sim", traj_file_name)
    # args.traj_file = os.path.join(base_folder, "sim/udel_arl.txt")
    # args.traj_file = os.path.join(base_folder, "sim/udel_garage.txt")
    # args.traj_file = os.path.join(base_folder, "tum_vi", "dataset-room2_512_16.txt")
    args.traj_file = "/home/mitchell/Documents/data/d435i_mcgill/truths/lab2_tum.txt"

    # args.plot_save_dir = (
    #     "/home/mitchell/Documents/visual_inertial_navigation/figs/trajectories/"
    #     + traj_file_name.split(".")[0]
    # )
    args.plot_save_dir = None
    gt_poses = load_poses_from_tum(args.traj_file, delimiter=" ")

    analyzer = TrajectoryAnalyzer(gt_poses)
    analyzer.plot_trajectory(step=500)
    analyzer.plot_velocities()
    plt.show()