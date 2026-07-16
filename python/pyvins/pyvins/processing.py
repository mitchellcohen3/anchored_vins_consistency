"""Batch and Monte-Carlo trial processing utilities for VINS evaluation."""

import logging
import multiprocessing
import os
import typing

import matplotlib
if multiprocessing.current_process().name != "MainProcess":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyvins import file_utils
from pyvins.containers import Statistics
from pyvins.trajectory import TrajectoryResult, VinsSimulationResult
from pyvins.aggregation import AveragedVinsResult

def get_label_for_alg(alg_name: str) -> str:
    """Helper function to get display label for an algorithm name."""
    label_map = {
        "none_global3d": "Std-G3D",
        "none_anchored3d": "Std-AEP",
        "none_anchoredmsckfinversedepth": "Std-AID",
        "fej_global3d": "FEJ-G3D",
        "fej_anchored3d": "FEJ-AEP",
        "fej_anchoredmsckfinversedepth": "FEJ-AID",
        "dri_fej_global3d": "RI-G3D",
        "dri_fej_anchored3d": "RI-AEP",
        "dri_fej_anchoredmsckfinversedepth": "RI-AID",
    }
    return label_map.get(alg_name, alg_name)

def load_reprojection_errors_file(outlier_info_file: str) -> typing.List[Statistics]:
    """Load reprojection errors from a file.

    Parameters
    ----------
    outlier_info_file : str
        Path to the outlier info file containing reprojection errors.

    Returns
    -------
    List[Statistics]
        List of Statistics objects containing reprojection error data.
    """
    stats_list: typing.List[Statistics] = []
    with open(outlier_info_file, "r") as f:
        for line in f:
            row = line.strip().split(",")
            row = np.array([float(x) for x in row])
            stamp = row[0]
            errors = row[1:]
            if len(errors) != 0:
                stats = Statistics(errors, stamp)
                stats_list.append(stats)

    return stats_list


def process_timing_file(
    timing_fpath: str,
    save_dir: str = None,
    print_info: bool = False,
) -> typing.Dict[str, typing.Any]:
    """Processes the timing file and generates timing related plots.

    The timing file is a CSV where the first column is the algorithm timestamp,
    the subsequent columns are the times for each component of the algorithm
    and the final column is the total time for that iteration.

    Parameters
    ----------
    timing_fpath : str
        Path to the timing CSV file.
    save_dir : str, optional
        Directory to save plots. If None, plots are not saved.
    print_info : bool, optional
        Whether to print runtime statistics.

    Returns
    -------
    dict
        Dictionary containing timing statistics and data.
    """
    df = pd.read_csv(timing_fpath, sep=",")
    df = df.copy()

    # Convert computation times to ms
    df.iloc[:, 1:] = df.iloc[:, 1:] * 1000.0
    names = df.columns.to_list()

    # Get the total time
    total_times = df[names[-1]]
    total_runtime = total_times.sum() / 1000.0

    # Times for each part
    df_parts = df.iloc[:, :-1]

    times = df[names[0]]
    parts = [df_parts[col] for col in df_parts.columns if col != names[0]]

    # Generate stackplot
    fig, ax = plt.subplots(1, 1)
    ax.stackplot(times, *parts, labels=[col for col in df.columns if col != names[0]])
    ax.legend(loc="upper right")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Execution Time (ms)")
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "timing_stackplot.png"))

    # Generate lineplot of total times
    fig, ax = plt.subplots(1, 1)
    ax.plot(times, total_times)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Execution Time (ms)")
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "timing_lineplot.png"))

    # Generate histogram of total times
    fig, ax = plt.subplots(1, 1)
    ax.hist(total_times, bins=10, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Total Time (ms)")
    ax.set_ylabel("Frequency")
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "timing_histogram.png"))

    # Compute statistics
    column_means = df_parts.mean()
    column_stds = df_parts.std()

    # Compute the total data time
    if print_info:
        dataset_length = df[names[0]].iloc[-1] - df[names[0]].iloc[0]
        print(f"Total runtime: {round(total_runtime, 3)}s")
        print(f"Dataset length: {round(dataset_length, 3)}s")
        realtime_factor = dataset_length / total_runtime
        print(f"Realtime factor: {round(realtime_factor, 3)}")

    output_dict = {
        "total_times": total_times,
        "average_times": column_means,
        "std_times": column_stds,
        "timing_dataframe": df,
    }
    return output_dict


def discover_algorithms_and_datasets(
    result_dir: str,
) -> typing.Tuple[typing.List[str], typing.List[str]]:
    """Automatically discovers algorithms and datasets in the result directory.

    The assumed folder structure is:
    result_dir/
        ├── dataset1/
        │   ├── algorithm1/
        │   ├── algorithm2/
        ├── dataset2/
        │   ├── algorithm1/
        │   ├── algorithm2/

    Parameters
    ----------
    result_dir : str
        Results directory for MonteCarlo runs.

    Returns
    -------
    Tuple[List[str], List[str]]
        Tuple of (algorithms, datasets) found in the directory.
    """
    algorithms = []
    datasets = []

    if not os.path.exists(result_dir):
        print(f"Results directory {result_dir} does not exist!")
        return algorithms, datasets

    # Get all subdirectories in the result directory,
    # corresponding to different datasets
    datasets = file_utils.get_all_subdirectory_names(result_dir)

    # For each dataset, look for algorithms
    for dataset in datasets:
        if "results" in dataset:
            continue

        dataset_path = os.path.join(result_dir, dataset)

        # Get the subdirectories in the dataset directory
        algs = file_utils.get_all_subdirectory_names(dataset_path)
        for alg in algs:
            if alg not in algorithms:
                algorithms.append(alg)

    return algorithms, datasets


def _process_single_trial(
    trial_folder_path: str,
    file_name_dict: typing.Dict[str, str],
    state_representation: str,
    lie_direction: str,
    generate_all_plots: bool,
    verbose: bool,
) -> typing.Optional[VinsSimulationResult]:
    """Helper function to process a single trial folder. Used for parallel processing.

    Parameters
    ----------
    trial_folder_path : str
        Path to the trial folder.
    file_name_dict : dict
        Dictionary mapping file types to filenames.
    state_representation : str
        State representation ('SE23' or 'decoupled').
    lie_direction : str
        Lie group direction ('left' or 'right').
    generate_all_plots : bool
        Whether to generate plots for this trial.
    verbose : bool
        Whether to print verbose output on errors.

    Returns
    -------
    VinsSimulationResult or None
        The processed result, or None if an error occurred.
    """
    # Set logging level in worker process
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        state_est_path = os.path.join(trial_folder_path, file_name_dict["state_est"])
        state_gt_path = os.path.join(trial_folder_path, file_name_dict["state_gt"])
        state_std_path = os.path.join(trial_folder_path, file_name_dict["state_std"])
        poses_est_path = os.path.join(trial_folder_path, file_name_dict["poses_est"])

        save_dir = (
            os.path.join(trial_folder_path, "plots") if generate_all_plots else None
        )
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        result = VinsSimulationResult.from_files(
            state_est_path=state_est_path,
            state_gt_path=state_gt_path,
            state_std_path=state_std_path,
            poses_est_path=poses_est_path,
            timing_fpath=None,
            state_representation=state_representation,
            lie_direction=lie_direction,
            plot_save_dir=save_dir,
        )
        return result
    except Exception as e:
        print(f"Error processing trial folder {trial_folder_path}: {e}")
        if verbose:
            import traceback

            print(f"Full traceback for trial {trial_folder_path}:")
            traceback.print_exc()
        return None


def _process_single_pose_trial(
    path_est: str,
    path_gt: str,
    state_representation: str,
    lie_direction: str,
    align: bool = True,
) -> TrajectoryResult:
    """Helper function to process a single pose trial folder. Used for parallel processing."""
    logging.getLogger().setLevel(logging.WARNING)
    result = TrajectoryResult(
        poses_gt_path=path_gt,
        poses_est_path=path_est,
        state_representation=state_representation,
        lie_direction=lie_direction,
        align=align,
    )

    return result


def _process_single_trial_wrapper(
    args: typing.Tuple,
) -> typing.Optional[VinsSimulationResult]:
    """Wrapper to unpack arguments for ProcessPoolExecutor."""
    return _process_single_trial(*args)


def _process_single_pose_trial_wrapper(
    args: typing.Tuple,
) -> typing.Optional[TrajectoryResult]:
    """Wrapper to unpack arguments for ProcessPoolExecutor."""
    return _process_single_pose_trial(*args)


def evaluate_pose_stats_dataset(
    base_folder: str,
    path_gt: str,
    filename_est: str,
    state_representation: str = "decoupled",
    lie_direction: str = "right",
    num_workers: int = None,
) -> AveragedVinsResult:
    """Evaluates just the pose results for a given dataset folder.

    Parameters
    ----------
    base_folder : str
        Path to the base folder containing trial subfolders.
    path_gt : str
        Path to the ground truth file.
    filename_est : str
        Filename of the estimated poses file in each trial folder.
    state_representation : str, optional
        State representation ('SE23' or 'decoupled'). Default is 'decoupled'.
    lie_direction : str, optional
        Lie group direction ('left' or 'right'). Default is 'right'.
    num_workers : int, optional
        Number of parallel workers. Default is None (sequential processing).

    Returns
    -------
    AveragedVinsResult
        Aggregated results over all trials.
    """
    folders = [
        name
        for name in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, name)) and name != "results"
    ]
    logging.info(f"Found {len(folders)} trial folders in {base_folder}")

    trial_args = [
        (
            os.path.join(base_folder, folder, filename_est),
            path_gt,
            state_representation,
            lie_direction,
        )
        for folder in folders
    ]

    trial_results: typing.List[TrajectoryResult] = []
    if num_workers is not None and num_workers > 1 and len(folders) > 1:
        # Use parallel processing with spawn context to avoid issues with forked processes
        logging.info(f"Processing {len(folders)} trials with {num_workers} workers")
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results_iter = pool.imap_unordered(
                _process_single_pose_trial_wrapper, trial_args
            )

            for result in results_iter:
                if result is not None:
                    trial_results.append(result)
    else:
        # Fall back to sequential processing for single trial or single worker
        for args in trial_args:
            result = _process_single_pose_trial_wrapper(args)
            if result is not None:
                trial_results.append(result)

    # Aggregate all results into a single Monte-Carlo result
    return trial_results


def evaluate_mc_trials(
    base_mc_folder: str,
    generate_all_plots: bool = False,
    verbose: bool = False,
    show_progress: bool = False,
    file_name_dict: typing.Dict[str, str] = None,
    align: bool = True,
    lie_direction: str = "right",
    state_representation: str = "decoupled",
    num_workers: int = None,
) -> typing.List[VinsSimulationResult]:
    """Evaluates all trials in a Monte-Carlo folder.

    Parameters
    ----------
    base_mc_folder : str
        Path to the base Monte-Carlo folder containing trial subfolders.
    generate_all_plots : bool, optional
        Whether to generate plots for each trial. Default is False.
    verbose : bool, optional
        Whether to print verbose output. Default is False.
    show_progress : bool, optional
        Whether to show a progress bar. Default is False.
    file_name_dict : dict, optional
        Dictionary mapping file types to filenames.
    align : bool, optional
        Whether to align trajectories. Default is True.
    lie_direction : str, optional
        Lie group direction ('left' or 'right'). Default is 'right'.
    state_representation : str, optional
        State representation ('SE23' or 'decoupled'). Default is 'decoupled'.
    num_workers : int, optional
        Number of parallel workers. Default is None (uses CPU count).

    Returns
    -------
    AveragedVinsResult
        Aggregated results over all trials.
    """
    if file_name_dict is None:
        file_name_dict = {
            "params": "new_params.yaml",
            "state_est": "state_est.txt",
            "state_gt": "state_gt.txt",
            "state_std": "state_std.txt",
            "poses_est": "poses_est.txt",
            "timing": "timing.txt",
        }

    folders = [
        name
        for name in os.listdir(base_mc_folder)
        if os.path.isdir(os.path.join(base_mc_folder, name)) and name != "results"
    ]
    logging.info(f"Found {len(folders)} trial folders in {base_mc_folder}")

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    # Prepare arguments for parallel processing
    trial_args = [
        (
            os.path.join(base_mc_folder, folder),
            file_name_dict,
            state_representation,
            lie_direction,
            generate_all_plots,
            verbose,
        )
        for folder in folders
    ]

    # Determine number of workers
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), len(folders))

    trial_results: typing.List[VinsSimulationResult] = []

    if num_workers > 1 and len(folders) > 1:
        # Use parallel processing with spawn context to avoid issues with forked processes
        logging.info(f"Processing {len(folders)} trials with {num_workers} workers")
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            if show_progress:
                results_iter = tqdm(
                    pool.imap_unordered(_process_single_trial_wrapper, trial_args),
                    total=len(trial_args),
                    desc="Processing trials",
                )
            else:
                results_iter = pool.imap_unordered(
                    _process_single_trial_wrapper, trial_args
                )

            for result in results_iter:
                if result is not None:
                    trial_results.append(result)
    else:
        # Fall back to sequential processing for single trial or single worker
        for args in tqdm(trial_args, disable=not show_progress):
            result = _process_single_trial_wrapper(args)
            if result is not None:
                trial_results.append(result)

    # Aggregate all results into a single Monte-Carlo result
    # Create directory for saving plots if needed
    save_dir = os.path.join(base_mc_folder, "results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logging.info(
        f"Processed {len(trial_results)} successful trials out of {len(folders)}"
    )
    trial_results = [result for result in trial_results if result is not None]
    return trial_results


def evaluate_vins_algorithm(
    results_dir: str,
    gt_path: str,
    verbose: bool = False,
    generate_all_plots: bool = False,
    lie_direction: str = "left",
    state_representation: str = "decoupled",
    align: bool = True,
) -> AveragedVinsResult:
    """Evaluates multiple runs of a VINS algorithm on a single dataset.

    Parameters
    ----------
    results_dir : str
        Directory containing trial subfolders.
    params_fname : str
        Filename of parameters file (currently unused).
    gt_path : str
        Path to ground truth file.
    verbose : bool, optional
        Whether to print verbose output. Default is False.
    generate_all_plots : bool, optional
        Whether to generate plots. Default is False.
    lie_direction : str, optional
        Lie group direction ('left' or 'right'). Default is 'left'.
    state_representation : str, optional
        State representation. Default is 'decoupled'.
    align : bool, optional
        Whether to align trajectories. Default is True.

    Returns
    -------
    AveragedVinsResult
        Averaged results over all trials.
    """
    # Get all trial folders in the results directory
    trial_folders = file_utils.get_all_subdirectory_names(results_dir)
    results_list = []
    for trial in trial_folders:
        if "results" in trial:
            continue

        trial_path = os.path.join(results_dir, trial)
        logging.info(f"Processing trial folder: {trial_path}")

        try:
            # Load the parameters for this trial
            poses_est_path = os.path.join(trial_path, "poses_est.txt")

            if not verbose:
                logging.getLogger().setLevel(logging.WARNING)

            # Save the results
            save_dir = os.path.join(trial_path, "results")
            file_utils.create_new_folder(save_dir, overwrite=True)
            result = TrajectoryResult(
                poses_gt_path=gt_path,
                poses_est_path=poses_est_path,
                align=align,
                lie_direction=lie_direction,
                state_representation=state_representation,
                save_dir=None,
            )

        except Exception as e:
            print(f"Error processing trial {trial_path}: {e}")
            continue

        # Collect results for averaging
        results_list.append(result)

    # Average the results across all trials
    averaged_result = AveragedVinsResult(results_list)
    return averaged_result
