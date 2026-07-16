"""Evaluates different VINS algorithms on a single dataset.

Can be used to evaluate the pose results of Monte-Carlo trials, or
real world datasets with multiple runs.

The folder structure is assumed to be the same as OpenVINS evaluation scripts:
    - algorithms
        - alg_1
            - dataset_1
                - trial_1
                - trial_2
                - ..
        - alg_2
            - dataset_1
                - trial_1
                - trial_2
                - ...
    - truths
        - dataset_1.txt
        - dataset_2.txt

The user must specify the results directory for the dataset and the ground truth file path.
"""

import time
import yaml
import typing
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pyvins.file_utils as file_utils
from pyvins.trajectory import TrajectoryResult
from pyvins.aggregation import AveragedVinsResult, compare_averaged_vins_results
from pyvins.processing import evaluate_pose_stats_dataset
from pyvins.plot_utils import set_plot_theme
from pyvins.file_utils import load_poses_from_file

from navlie.utils import plot_poses
import logging

logging.basicConfig(level=logging.INFO)

import pickle
from tqdm import tqdm

colors = set_plot_theme(palette="colorblind", enable_latex=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Monte-Carlo results")
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Results directory containing algorithm subfolders",
    )
    parser.add_argument("--save_dir", type=str, help="Output directory")
    parser.add_argument(
        "--load_from_pickle",
        action="store_true",
        help="Whether to load results from pickle files if they exist",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        help="List of algorithms to evaluate (subfolder names in result_dir)",
    )
    parser.add_argument("--gt_path", type=str, help="Path to the ground truth file")
    parser.add_argument("--show_plots", action="store_true", help="Show plots")
    parser.add_argument("--align", action="store_true", help="Align trajectories")
    parser.add_argument(
        "--labels_map_yaml",
        type=str,
        help="Path to yaml file mapping algorithm names to labels for plotting",
    )
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.algorithms is None:
        args.algorithms = file_utils.get_all_subdirectory_names(args.results_dir)
        args.algorithms = [alg for alg in args.algorithms if "results" not in alg]
        args.labels = args.algorithms

    args.labels_map = {}
    if args.labels_map_yaml is not None:
        with open(args.labels_map_yaml, "r") as f:
            args.labels_map = yaml.safe_load(f)

    args.labels = [args.labels_map.get(alg, alg) for alg in args.algorithms]

    logging.info(f"Evaluating algorithms: {args.algorithms}")
    logging.info(f"Labels for plotting: {args.labels}")
    logging.info(f"Results directory: {args.results_dir}")
    logging.info(f"Found algorithms: {args.algorithms}")
    logging.info(f"Ground truth path: {args.gt_path}")

    gt_exists = os.path.exists(args.gt_path)
    if not gt_exists:
        logging.error(f"Ground truth file does not exist: {args.gt_path}")
        exit(1)

    # Load the groundtruth states to plot the trajectories
    gt_poses, _ = load_poses_from_file(args.gt_path)

    averaged_vins_results_list = []
    est_poses_by_alg = []
    dataset_name = os.path.basename(args.gt_path).split(".")[0]
    logging.info(f"Evaluating algorithms on dataset: {dataset_name}")

    big_time_start = time.time()
    for alg, label in zip(args.algorithms, args.labels):
        alg_folder = os.path.join(args.results_dir, alg)

        lie_direction = "right"
        state_representation = "decoupled"
        if "dri" in alg.lower():
            lie_direction = "left"
            state_representation = "SE23"

        print(f"Evaluating algorithm: {alg} with label: {label}")

        # Try to load a pickle file
        result = None
        pickle_file = os.path.join(args.save_dir, f"{alg}_result.pkl")
        if os.path.exists(pickle_file) and args.load_from_pickle:
            with open(pickle_file, "rb") as f:
                result = pickle.load(f)

        # Evaluate the results from a single algorithm
        if result is None:

            # Find all runs for this algorithm
            run_folder = os.path.join(alg_folder, dataset_name)

            # Get the results for all runs of this algorithm on this dataset
            print("Run folder")
            est_files = file_utils.get_files_with_extension(run_folder, "txt")
            traj_results = []
            for est_file in est_files:
                logging.info(f"Evaluating run: {est_file}")
                start_time = time.time()
                traj_result = TrajectoryResult(
                    poses_gt_path=args.gt_path,
                    poses_est_path=est_file,
                    align=args.align,
                    lie_direction=lie_direction,
                    state_representation=state_representation,
                    compute_rpe=False,
                )
                end_time = time.time()
                logging.info(f"Evaluated run in {end_time - start_time:.2f} seconds")
                traj_results.append(traj_result)

            result = AveragedVinsResult(traj_results)
            result.save_to_pickle(pickle_file)
            plt.close("all")
        averaged_vins_results_list.append(result)

    # # Plot the positions of all algorithms together
    # logging.info("Plotting groundtruth poses...")
    # fig, ax = plot_poses(
    #     gt_poses,
    #     label="Groundtruth",
    #     kwargs_line={"color": "k"},
    #     plot_2d=True,
    #     step=None,
    # )
    # logging.info("Plotting estimated poses...")
    # for i in range(len(est_poses_by_alg)):
    #     logging.info(f"Plotting poses for algorithm: {args.labels[i]}")
    #     poses = est_poses_by_alg[i]
    #     label = args.labels[i]
    #     color = colors[i]
    #     plot_poses(
    #         poses,
    #         ax=ax,
    #         label=label,
    #         plot_2d=True,
    #         step=None,
    #         kwargs_line={"color": color},
    #     )
    # fig.legend()
    # fig.tight_layout()
    # ax.set_xlabel("x (m)")
    # ax.set_ylabel("y (m)")
    # fig.savefig(os.path.join(args.save_dir, "trajectories.pdf"), dpi=300)

    # Compare the results of all algorithms
    compare_averaged_vins_results(
        averaged_vins_results_list,
        args.labels,
        colors,
        save_dir=args.save_dir,
    )

    for result, label in zip(averaged_vins_results_list, args.labels):
        logging.info(f"Results for algorithm: {label}")
        logging.info(f"Number of runs: {result.num_runs}")
        logging.info(f"ATE RMSE: {result.position_ate_stats.mean:.4f} m")
        logging.info(f"ATE std: {result.position_ate_stats.std:.4f} m")

    ## Save
    if args.show_plots:
        plt.show()

    big_time_end = time.time()
    logging.info(f"Total evaluation time: {big_time_end - big_time_start:.2f} seconds")
    logging.info(f"Results saved to: {args.save_dir}")
