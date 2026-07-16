"""
Evaluates Monte-Carlo results for multiple algorithms at multiple noise levels, on
a single dataset.

Each algorithm is identified by a directory name of the form:
    <algorithm_name>__sigma<noise_level>

The assumed directory structure (as produced by run_camnoise_experiment.sh) is:
- algorithms/
    - algorithm1__param1.0/
        - dataset_name/
            - run0.txt
            - run1.txt
            - ...
    - algorithm2__param1.0/
        - dataset_name/
            - run0.txt
            - ...
    - algorithm1__param2.0/
        - dataset_name/
            - run0.txt
            - ...
- truths/
    - dataset_name.txt,

where each run*.txt file contains estimated poses/pose covariances for a single Monte-Carlo trial.

This script can also be used to aggregate results across multiple runs, at different values
of any parameter that can take on a continuous range of values. Some examples of experiments
that can be run include:
    - Varying noise levels
    - Varying initial perturbation scaling
    - Varying the number of SLAM features

The only requirement is that the parameter value is encoded in the directory name in a consistent way,
and that the directory structure follows the pattern described above.
"""

import time
import typing
import os
import argparse
import re
import glob
import logging
import yaml

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

import pyvins.file_utils as file_utils
from pyvins.aggregation import AveragedVinsResult, compare_averaged_vins_results
from pyvins.processing import get_label_for_alg
from pyvins.trajectory import TrajectoryResult
from pyvins.plot_utils import set_plot_theme
from pyvins.noise_experiment_plots import (
    generate_sensitivty_comparison_plots,
    generate_paper_plot,
)

logging.basicConfig(level=logging.INFO)

# colors = set_plot_theme("deep", enable_latex=True)
colors = set_plot_theme(palette="colorblind", enable_latex=True)

def parse_directory_name(dir_name: str, split_str: str = "__"):
    """Gets the algorithm parameters from the folder name.

    Expected format: <alg_name>__sigma<noise_level>
    """
    parts = dir_name.rsplit(split_str, 1)
    if len(parts) == 2:
        alg_name = parts[0]
        noise_level_str = parts[1]

        try:
            noise_level = re.search(r"-?[\d.]+$", noise_level_str)
            if noise_level:
                noise_level = noise_level.group()
            else:
                raise ValueError("No numeric value found in noise level string")
            # noise_level = float(noise_level_str.replace("sigma", ""))
            return alg_name, noise_level
        except ValueError:
            return alg_name, None
    else:
        return None, None


def get_params_from_folder_name(folder_name: str):
    if "dri" in folder_name:
        return "left", "SE23"
    else:
        return "right", "decoupled"

def evaluate_pose_only_trials(
    alg_dataset_dir: str,
    gt_path: str,
    lie_direction: str = "right",
    state_representation: str = "decoupled",
) -> typing.List[TrajectoryResult]:
    """Evaluates pose-only Monte-Carlo trials stored as flat run*.txt files.

    Expects `alg_dataset_dir` to directly contain run0.txt, run1.txt, ... files
    (no per-trial subfolders, no state/covariance files).
    """
    run_files = sorted(glob.glob(os.path.join(alg_dataset_dir, "run*.txt")))
    results_list = []
    for run_file in run_files:
        try:
            result = TrajectoryResult(
                poses_gt_path=gt_path,
                poses_est_path=run_file,
                state_representation=state_representation,
                lie_direction=lie_direction,
                align=False,
            )
            results_list.append(result)
        except Exception as e:
            logging.warning(f"Error processing run file {run_file}: {e}")
    return results_list


def evaluate_noise_experiment(
    results_dir: str,
    dataset: str,
    algorithms: typing.List[str],
    labels: typing.List[str],
    colors: typing.List[str],
    save_dir: str,
    load_from_pickle: bool = True,
):
    if len(algorithms) != len(labels):
        logging.warning("Length of algorithms and labels do not match!")
        logging.warning(f"Algorithms: {algorithms}")
        logging.warning(f"Labels: {labels}")
        raise ValueError("Length of algorithms and labels must match!")

    logging.info(f"Evaluating noise experiment for {len(algorithms)} algorithms")

    algorithms_dir = os.path.join(results_dir, "algorithms")
    gt_path = os.path.join(results_dir, "truths", f"{dataset}.txt")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    # Find how many noise levels there are by inspecting the directories
    all_subdirs = file_utils.get_all_subdirectory_names(algorithms_dir)
    alg_name_to_label_dict = dict(zip(algorithms, labels))

    # For each subdirectory, extract the algorithm name and noise level
    eval_subdirs = []
    for subdir in all_subdirs:
        alg_name, noise_level = parse_directory_name(subdir)
        if alg_name not in algorithms:
            continue
        eval_subdirs.append(subdir)

    logging.info(
        f"Found {len(eval_subdirs)} algorithm/noise level combinations to evaluate"
    )
    mc_results_dict: typing.Dict[typing.Tuple[str, float], AveragedVinsResult] = {}
    for alg_folder in tqdm(eval_subdirs):
        logging.info(f"Processing directory: {alg_folder}")

        # Get the algorithm name and sigma from the folder name
        alg_name, noise_level = parse_directory_name(alg_folder)
        logging.info(f"Algorithm: {alg_name}, Noise Level: {noise_level}")

        alg_path = os.path.join(algorithms_dir, alg_folder, dataset)
        # Evaluate the Monte-Carlo trials for this algorithm and noise level
        mc_results = None
        results_save_dir = os.path.join(alg_path, "results")
        if load_from_pickle:
            # Search for the pickle file
            summary_data_path = os.path.join(
                results_save_dir,
                "summary_data.pickle",
            )
            if os.path.exists(summary_data_path):
                with open(summary_data_path, "rb") as f:
                    mc_results = pickle.load(f)
        if mc_results is None:
            lie_direction, state_representation = get_params_from_folder_name(
                alg_folder
            )

            logging.info(f"Lie direction: {lie_direction}")
            logging.info(f"State representation: {state_representation}")

            results_list = evaluate_pose_only_trials(
                alg_path,
                gt_path=gt_path,
                lie_direction=lie_direction,
                state_representation=state_representation,
            )
            mc_results = AveragedVinsResult(results_list)
            logging.getLogger().setLevel(logging.INFO)
            os.makedirs(results_save_dir, exist_ok=True)
            mc_results.save_to_pickle(
                os.path.join(results_save_dir, "summary_data.pickle")
            )
            plt.close("all")

        # Get the label for this algorithm
        label = alg_name_to_label_dict[alg_name]
        key = (label, float(noise_level))
        mc_results_dict[key] = mc_results

    comparison_results_dir = os.path.join(save_dir, "comparison_results")
    os.makedirs(comparison_results_dir, exist_ok=True)

    # Generate comparison plots across the continuous range of noise levels, 
    # comparing all algorithms on the same plot
    generate_sensitivty_comparison_plots(
        mc_results_dict,
        algorithm_order=labels,
        save_dir=comparison_results_dir,
        xlabel=r"$\sigma_p$ (px)",
        log_scale=False,
    )

    # Generate the time-series plot from the paper
    generate_paper_plot(
        mc_results_dict,
        algorithm_order=labels,
        colors=colors,
        save_dir=comparison_results_dir,
        noise_levels_plot=[1.0, 4.0]
    )

    # Pickle the mc_results
    pickle_fname = os.path.join(comparison_results_dir, "results.pkl")
    with open(pickle_fname, "wb") as f:
        pickle.dump(mc_results_dict, f)

    # Generate comparison plots at each noise level, comparing all algorithms at that noise level
    noise_levels = sorted(set(key[1] for key in mc_results_dict.keys()))
    for noise_level in noise_levels:
        results_list: typing.List[AveragedVinsResult] = []
        labels_list: typing.List[str] = []

        for label in labels:
            key = (label, noise_level)
            if key not in mc_results_dict:
                logging.warning(
                    f"Missing results for algorithm {label} at noise level {noise_level}, skipping this algorithm for this noise level."
                )
                continue
            results_list.append(mc_results_dict[key])
            labels_list.append(label)
        cur_save_dir = os.path.join(comparison_results_dir, f"noise_{noise_level}")
        os.makedirs(cur_save_dir, exist_ok=True)
        compare_averaged_vins_results(results_list, labels_list, colors, cur_save_dir)
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Monte-Carlo results")
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Experiment root directory (contains algorithms/ and truths/)",
    )
    parser.add_argument("--dataset", type=str, help="Dataset name to evaluate")
    parser.add_argument("--save_dir", type=str, help="Output directory")
    parser.add_argument("--load_from_pickle", action="store_true")
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        help="Algorithms to evaluate",
        default=None,
    )
    parser.add_argument(
        "--labels_map_yaml",
        type=str,
        default=None,
        help="Path to yaml file mapping algorithm names to labels for plotting",
    )
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = os.path.join(args.results_dir, args.dataset)

    if args.algorithms is None:
        logging.info(
            "No algorithms specified, attempting to auto-detect from directory names..."
        )
        # Automwatically detect algorithms from results directory
        algorithms_dir = os.path.join(args.results_dir, "algorithms")
        all_subdirs = file_utils.get_all_subdirectory_names(algorithms_dir)
        detected_algorithms = set()
        for subdir in all_subdirs:
            alg_name, noise_level = parse_directory_name(subdir)
            if alg_name is not None:
                detected_algorithms.add(alg_name)
        args.algorithms = sorted(list(detected_algorithms))
        logging.info(f"Auto-detected algorithms: {args.algorithms}")

    labels = args.algorithms

    if args.labels_map_yaml is not None:
        logging.info(f"Loading algorithm labels from {args.labels_map_yaml}")
        with open(args.labels_map_yaml, "r") as f:
            labels_map = yaml.safe_load(f)
        labels = [
            labels_map.get(alg, get_label_for_alg(alg)) for alg in args.algorithms
        ]

    alg_colors = colors

    # Start timing
    start_time = time.time()
    evaluate_noise_experiment(
        args.results_dir,
        args.dataset,
        args.algorithms,
        labels,
        alg_colors,
        args.save_dir,
        args.load_from_pickle,
    )
    end_time = time.time()
    logging.info(f"Total evaluation time: {end_time - start_time:.2f} seconds")
    plt.show()
    logging.info(f"Results saved to {args.save_dir}")
