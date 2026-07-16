"""This script loops through OpenVINS MonteCarlo results, extracts stats, and saves them in CSV files.

The expected folder structure is as follows:

results_dir 
    - algorithms
        - algorithm_1
            - dataset_name_1
                - run1.txt
                - run2.txt
            - dataset_name_2
                - run1.txt
                - run2.txt
    - truths
        - dataset_name_1.txt
        - dataset_name_2.txt

This script also saves pickled AveragedVinsResult objects for each algorithm/dataset combination,
so that we can easily load them for later analysis without having to re-parse the raw text files.

This essentially does the same as the error_comparison.cpp executable, but
also saves pickles to plot time-series information.

After running this script, a user can call plot_averaged_vins_results to 
"""

import pandas as pd
import numpy as np
import time
import logging
import argparse
import os

import pickle

from pyvins.file_utils import get_all_subdirectory_names
from pyvins.trajectory import TrajectoryResult
from pyvins.aggregation import AveragedVinsResult

logging.basicConfig(level=logging.INFO)

def evaluate_trials(path_gt: str, folder_est: str) -> AveragedVinsResult:
    """Evaluates all runs for a given algorithm/dataset combination and returns the averaged results."""
    run_files = [f for f in os.listdir(folder_est) if f.endswith(".txt")]
    logging.info(f"Found {len(run_files)} runs in {folder_est}")

    traj_results = []
    logging.getLogger().setLevel(logging.WARNING)
    for run_file in run_files:
        path_run = os.path.join(folder_est, run_file)

        # Extract the Lie direction from the filename
        if "dri"  in folder_est:
            lie_direction = "left"
            state_representation = "SE23"
        else: 
            lie_direction = "right"
            state_representation = "decoupled" 

        traj_result = TrajectoryResult(
            poses_gt_path=path_gt,
            poses_est_path=path_run,
            align=False,
            lie_direction=lie_direction,
            state_representation=state_representation
        )
        traj_results.append(traj_result)

    # Set logging level back to info 
    logging.getLogger().setLevel(logging.INFO)
    averaged_result = AveragedVinsResult(traj_results)
    return averaged_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SLAM results on real data, generates CSV files and outputs table for the paper."
    )
    parser.add_argument("--align_mode")
    parser.add_argument("--folder_gt")
    parser.add_argument("--folder_algorithms")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of dataset names to include. If omitted, all datasets are used.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Subset of algorithm names to include. If omitted, all algorithms are used.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="Display labels for each algorithm listed in --algorithms (same order).",
    )

    parser.add_argument("--output_dir")

    args = parser.parse_args()
    basedir = "/home/mitchell/experiments/ov_exp_featrep_sim_camnoise_20260303_103037"
    args.folder_gt = os.path.join(basedir, "truths")
    args.folder_algorithms = os.path.join(basedir, "algorithms")
    args.output_dir = os.path.join(basedir, "results")

    # Make the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    args.align = False
    args.algorithms = None

    if args.algorithms is None: 
        # Discover all algorithms by listing the subdirectories in folder_algorithms
        args.algorithms = get_all_subdirectory_names(args.folder_algorithms)
        print(f"Discovered algorithms: {args.algorithms}")

    if args.datasets is None:
        # Get the filenames in the truths folder and extract dataset names
        truth_files = os.listdir(args.folder_gt)
        args.datasets = [os.path.splitext(f)[0] for f in truth_files if f.endswith(".txt")]
        print(f"Discovered datasets: {args.datasets}")
 
    # Loop through algorithms and datasets, load results, and save CSVs
    ate_csv_rows = []
    nees_csv_rows = []

    start_time = time.time()
    for alg in args.algorithms:
        for dataset in args.datasets:
            logging.info(f"Processing algorithm {alg}, dataset {dataset}...")

            path_gt = os.path.join(args.folder_gt, f"{dataset}.txt")
            path_alg = os.path.join(args.folder_algorithms, alg, dataset)

            # Check if this algorithm/dataset combination exists
            if not os.path.exists(path_alg):
                logging.warning(f"Results for algorithm {alg}, dataset {dataset} not found!")
                continue
        
            # Try loading from pickle
            averaged_result = None
            pickle_path = os.path.join(args.output_dir, f"{alg}_{dataset}_averaged_result.pkl")
            if os.path.exists(pickle_path):
                logging.info(f"Found existing pickle for algorithm {alg}, dataset {dataset}. Loading...")
                with open(pickle_path, "rb") as f:
                    averaged_result = pickle.load(f)
            else:
                # Load from raw results and evaluate
                averaged_result = evaluate_trials(path_gt, path_alg)
                output_path = os.path.join(args.output_dir, f"{alg}_{dataset}_averaged_result.pkl")
                with open(output_path, "wb") as f:
                    pickle.dump(averaged_result, f)
                logging.info(f"Saved averaged result to {output_path}")

            
            # Print the ATE for this configuration
            logging.info(f"ATE for algorithm {alg}, dataset {dataset}:")
            logging.info(f"Attitude: {np.rad2deg(averaged_result.attitude_ate_stats.mean):.3f}")
            logging.info(f"Position: {averaged_result.position_ate_stats.mean:.3f}")

            ate_csv_rows.append({
                "algorithm": alg,
                "dataset": dataset,
                "mean_ori": np.rad2deg(averaged_result.attitude_ate_stats.mean),
                "std_ori": np.rad2deg(averaged_result.attitude_ate_stats.std),
                "mean_pos": averaged_result.position_ate_stats.mean,
                "std_pos": averaged_result.position_ate_stats.std,
                "num_runs": averaged_result.num_runs,
            })

            nees_csv_rows.append({
                "algorithm": alg,
                "dataset": dataset,
                "mean_ori_nees": averaged_result.mean_att_nees,
                "mean_pos_nees": averaged_result.mean_pos_nees,
                "num_runs": averaged_result.num_runs,
            })


    end_time = time.time()
    logging.info(f"Finished processing all algorithms and datasets in {end_time - start_time:.2f} seconds.")

    # Now, generate summary CSV files for each dataset, containing the ATE stats for all algorithms
    df_ate = pd.DataFrame(ate_csv_rows)
    ate_csv_path = os.path.join(args.output_dir, "ate_results.csv")
    df_ate.to_csv(ate_csv_path, index=False)
    logging.info(f"Saved ATE results to {ate_csv_path}")

    # Save NEES results to CSV
    df_nees = pd.DataFrame(nees_csv_rows)
    nees_csv_path = os.path.join(args.output_dir, "nees_results.csv")
    df_nees.to_csv(nees_csv_path, index=False)
    logging.info(f"Saved NEES results to {nees_csv_path}")