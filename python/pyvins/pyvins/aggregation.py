"""Multi-run averaging and comparison classes for VINS evaluation.

One of the main classes here is AveragedVinsResult, which computes
averaged error metrics over multiple runs of the same VINS algorithm.

The function compare_averaged_results generates comparison plots
between multiple AveragedVinsResult instances.
"""

import logging
import os
import pickle
import typing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyvins.trajectory import VinsSimulationResult
from pyvins.containers import Statistics, CalibrationErrorMetrics
from pyvins.metrics import (
    compute_rmses_over_time_dataset,
    PoseConsistencyMetrics,
    VinsConsistencyMetricsSim,
    CalibrationErrors,
)
from pyvins.tables import create_alg_comparison_table, generate_ate_comparison_table, generate_rmse_nees_comparison_table

import pyvins.plot_utils as plot_utils
from pyvins.plot_utils import plot_nees as plot_nees_customA
from pyvins.plot_utils import plot_three_sigma


def process_montecarlo_results(
    sim_result_list: typing.List[VinsSimulationResult],
    save_dir: str = None,
):
    """Plots the IMU consistency metrics for multiple simulations results."""
    metrics = []
    for sim_result in sim_result_list:
        if sim_result.consistency_metrics is not None:
            metrics.append(sim_result.consistency_metrics)

    # Generate 3-sigma plots for each of the consistency metrics
    plot_imu_consistency(metrics, save_dir)


def plot_calibration_errors(
    errors: typing.List[CalibrationErrors],
    save_dir: str = None,
):
    """Plots the calibration errors over time."""

    # Create plots for each camera ID
    cam_ids = errors[0].extrinsic_errors.keys()
    for cam_id in cam_ids:
        fig, ax = plt.subplots(3, 2, sharex=True, figsize=(10, 8))
        for cur_errors in errors:
            stamps = cur_errors.stamps
            errors_extrinsics = cur_errors.extrinsic_errors[cam_id]
            three_sigma_extrinsics = cur_errors.extrinsics_3_sigma[cam_id]

            plot_utils.plot_extrinsics_errors(
                stamps,
                errors_extrinsics,
                cam_id=cam_id,
                ax=ax,
                three_sigma=three_sigma_extrinsics,
            )
        fig.suptitle(f"Extrinsics Calibration Errors for Camera {cam_id}")
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(
                os.path.join(
                    save_dir, f"calibration_errors_extrinsics_cam_{cam_id}.pdf"
                )
            )


def plot_imu_consistency(
    metrics: typing.Union[
        VinsConsistencyMetricsSim, typing.List[VinsConsistencyMetricsSim]
    ],
    save_dir: str = None,
    combined_plot: bool = False,
):
    """Plots the IMU consistency metrics over time."""
    if isinstance(metrics, VinsConsistencyMetricsSim):
        metrics = [metrics]

    # Ensure that the state dimension is always 15
    for metric in metrics:
        dofs = metric.delta_xi.shape[1]
        if dofs != 15:
            raise ValueError("The dimension of the state must be 15")

    slices = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)]
    titles = [
        "Attitude Errors",
        "Velocity Errors",
        "Position Errors",
        "Gyro Bias Errors",
        "Accel Bias Errors",
    ]

    labels_list = [
        [
            r"$\delta \xi^{\phi_1}$ (rad)",
            r"$\delta \xi^{\phi_2}$ (rad)",
            r"$\delta \xi^{\phi_3}$ (rad)",
        ],
        [
            r"$\delta \xi^{v_1}$ (m/s)",
            r"$\delta \xi^{v_2}$ (m/s)",
            r"$\delta \xi^{v_3}$ (rad)",
        ],
        [
            r"$\delta \xi^{r_1}$ (m)",
            r"$\delta \xi^{r_2}$ (m)",
            r"$\delta \xi^{r_3}$ (m)",
        ],
        [
            r"$\delta \xi^{b^g_1}$ (rad)",
            r"$\delta \xi^{b^g_2}$ (rad)",
            r"$\delta \xi^{b^a_3}$ (rad/s)",
        ],
        [
            r"$\delta \xi^{b^a_1}$ (m/$s^2$)",
            r"$\delta \xi^{b^a_2}$ (m/$s^2$)",
            r"$\delta \xi^{b^a_3}$ (m/s$^2$)",
        ],
    ]

    if combined_plot:
        fig, axes = plt.subplots(3, 5, sharex=True, figsize=(15, 10))

        for col_idx, (s, title, labels) in enumerate(zip(slices, titles, labels_list)):
            ax = axes[:, col_idx]

            for i, result in enumerate(metrics):
                enable_sigma_bounds = i == 0
                error = result.delta_xi[:, s]
                sigma = result.three_sigma[:, s]
                stamps = result.stamps
                plot_three_sigma(
                    stamps,
                    error,
                    sigma,
                    ax=ax,
                    enable_sigma_bounds=enable_sigma_bounds,
                )
                ax[2].set_xlabel("Time (s)")
                for j, label in enumerate(labels):
                    ax[j].set_ylabel(label)

            for row_idx, label in enumerate(labels):
                axes[row_idx, col_idx].set_ylabel(label)
                if col_idx == 0:
                    axes[row_idx, col_idx].text(
                        -0.15,
                        0.5,
                        ["X", "Y", "Z"][row_idx],
                        transform=axes[row_idx, col_idx].transAxes,
                        rotation=90,
                        verticalalignment="center",
                        fontweight="bold",
                    )

            axes[0, col_idx].set_title(title, fontsize=10)

        for col_idx in range(5):
            axes[-1, col_idx].set_xlabel("Time (s)")

        fig.suptitle("IMU State Consistency Metrics", fontsize=16)
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "imu_consistency_combined.pdf"))
    else:
        for s, title, labels in zip(slices, titles, labels_list):
            fig, ax = plt.subplots(3, 1, sharex=True)

            for i, result in enumerate(metrics):
                enable_sigma_bounds = i == 0
                error = result.delta_xi[:, s]
                sigma = result.three_sigma[:, s]
                stamps = result.stamps
                plot_three_sigma(
                    stamps,
                    error,
                    sigma,
                    ax=ax,
                    enable_sigma_bounds=enable_sigma_bounds,
                )
                ax[2].set_xlabel("Time (s)")
                for j, label in enumerate(labels):
                    ax[j].set_ylabel(label)
            fig.suptitle(title)
            fig.tight_layout()

            if save_dir is not None:
                fig.savefig(
                    os.path.join(save_dir, f"{title.lower().replace(' ', '_')}.pdf")
                )


class AveragedVinsResult:
    """Averaged result over multiple runs of the same algorithm on the same dataset.

    This can either be used for multiple runs on the same dataset,
    or for Monte-Carlo runs on simulated data.
    """

    # Data about each of the runs
    num_runs: int
    alg_name: str
    dataset_name: str

    # Aggregated error metrics
    # These are statistics computed over the scalar ATEs for each run
    attitude_ate_stats: Statistics
    position_ate_stats: Statistics

    # Mean RMSEs at each timestamp, averaged over all trials
    # (K, ) arrays, where K is the number of timestamps in one run
    mean_att_rmses: np.ndarray
    mean_pos_rmses: np.ndarray

    # Mean NEES values over all runs
    # (K, ) arrays, where K is the number of timestamps in one run
    mean_att_nees_per_timestamp: np.ndarray
    mean_pos_nees_per_timestamp: np.ndarray

    # Scalar mean NEES values, averaged over all timestamps and runs
    mean_att_nees: float
    mean_pos_nees: float

    # Statistics of the NEES distributions over many runs
    mean_att_nees_distribution: Statistics = None
    mean_pos_nees_distribution: Statistics = None

    # Timestamps for the errors
    stamps_sync: np.ndarray

    # Calibration errors (if we have them)
    mean_att_rmses_extrinsics: typing.Dict[int, np.ndarray] = None
    mean_pos_rmses_extrinsics: typing.Dict[int, np.ndarray] = None
    mean_time_offset_rmse: float = None
    mean_intrinsics_rmse: float = None
    timestamps_calib: np.ndarray = None

    failed_runs: int = 0
    failed_run_folders: typing.List[str] = None

    def __init__(
        self,
        results: typing.List[typing.Any],  # TrajectoryResult or VinsSimulationResult
        alg_name: str = None,
        dataset_name: str = None,
        plot_dir: str = None,
    ):
        """Computes the averaged results over multiple runs."""
        self.num_runs = len(results)
        if self.num_runs == 0:
            raise ValueError("No results provided for averaging!")

        self.alg_name = alg_name
        self.dataset_name = dataset_name

        # Get the timestamps for the state estimates
        self.stamps_sync = np.array([x.stamp for x in results[0].gt_poses])

        # Compute statistics over the trials
        self._compute_mean_rmses(results)
        self._compute_ate_statistics(results)
        self._compute_mean_nees(results)
        # self._compute_calib_rmses(results)

        if self.failed_runs > 0:
            logging.warning("Some runs were ommited from averaging due to failure.")
            logging.warning(f"Number of failed runs: {self.failed_runs} out of {self.num_runs}")

    def _compute_mean_rmses(self, results: typing.List[VinsSimulationResult]):
        """Computes the mean RMSEs at each timestamp over all runs."""
        # for result in results:
        #     size_error = result.pose_error_metrics.attitude_errors.shape
        #     if size_error[0] != len(self.stamps_sync):
        #         logging.error("Error: Inconsistent number of timestamps in results, cannot compute mean RMSEs!")
        #         logging.error(f"Expected {len(self.stamps_sync)} timestamps, but got {size_error[0]} in one of the results.")
        #         logging.error("Omitting this result from mean RMSE computation...")
        #         self.mean_att_rmses = None
        #         self.mean_pos_rmses = None
        #         return

        # Build errors arrays for each run
        att_errors = []
        pos_errors = []
        for x in results:
            if x.pose_error_metrics.attitude_errors.shape[0] != len(self.stamps_sync):
                logging.error("Warning: inconsistent number of timestamps in results...")
                logging.error("Not including this result in mean RMSE computation.")
                self.failed_runs += 1
                if self.failed_run_folders is None:
                    self.failed_run_folders = []
                self.failed_run_folders.append(x.evaluation_folder)
                continue
            att_errors.append(x.pose_error_metrics.attitude_errors)
            pos_errors.append(x.pose_error_metrics.position_errors)

        att_errors = np.array(att_errors)
        pos_errors = np.array(pos_errors)

        self.mean_att_rmses = compute_rmses_over_time_dataset(att_errors)
        self.mean_pos_rmses = compute_rmses_over_time_dataset(pos_errors)

    def _compute_mean_nees(self, results: typing.List[VinsSimulationResult]):
        """Computes the mean NEES at each timestamp, over all runs."""
        # Get the NEES values for each run and compute the mean over all runs
        att_nees_list = []
        pos_nees_list = []
        for trial in results:
            if trial.consistency_metrics is None:
                logging.error("Cannot compute mean NEES, no consistency metrics found!")
                continue
            if trial.consistency_metrics.att_nees.shape[0] != len(self.stamps_sync):
                logging.error("Warning: inconsistent number of timestamps in consistency metrics...")
                logging.error("Not including this trial in mean NEES computation.")
                continue

            att_nees_list.append(trial.consistency_metrics.att_nees)
            pos_nees_list.append(trial.consistency_metrics.pos_nees)

        # Size of these lists: [num_trials x num_timestamps]
        att_nees_list = np.array(att_nees_list)
        pos_nees_list = np.array(pos_nees_list)

        # Average NEES at each timestamp, averaged over the runs
        self.mean_att_nees_per_timestamp = np.mean(att_nees_list, axis=0)
        self.mean_pos_nees_per_timestamp = np.mean(pos_nees_list, axis=0)

        # Scalar mean NEES value, averaged over all timestamps
        self.mean_att_nees = np.mean(self.mean_att_nees_per_timestamp)
        self.mean_pos_nees = np.mean(self.mean_pos_nees_per_timestamp)

        # Compute statistics of the NEES distributions
        att_nees_flat = []
        pos_nees_flat = []
        for trial in results:
            if trial.consistency_metrics is None:
                print(
                    "Cannot compute mean NEES distribution, no consistency metrics found!"
                )
                return

            for i in range(len(trial.consistency_metrics.stamps)):
                att_nees_flat.append(trial.consistency_metrics.att_nees[i])
                pos_nees_flat.append(trial.consistency_metrics.pos_nees[i])

        self.mean_att_nees_distribution = Statistics(np.array(att_nees_flat))
        self.mean_pos_nees_distribution = Statistics(np.array(pos_nees_flat))

    def _compute_ate_statistics(self, results):
        """Compute the mean ATE statistics over all runs."""
        self.attitude_ate_stats = Statistics(
            np.array([r.pose_error_metrics.attitude_ate for r in results])
        )
        self.position_ate_stats = Statistics(
            np.array([r.pose_error_metrics.position_ate for r in results])
        )

    def _compute_calib_rmses(self, results):
        """Computes mean RMSEs of calibration parameters over all runs, if available."""
        for result in results:
            if not hasattr(result, "calibration_errors"):
                logging.info("No calibration errors found, skipping...")
                return
            if result.calibration_errors is None:
                logging.info("No calibration errors found, skipping...")
                return

        # Evaluate extrinsics errors
        self.mean_att_rmses_extrinsics = {}
        self.mean_pos_rmses_extrinsics = {}
        for cam_idx in results[0].calibration_errors.extrinsic_errors.keys():
            att_errors = []
            pos_errors = []
            for result in results:
                att_errors.append(
                    result.calibration_errors.extrinsic_errors[cam_idx][:, 0:3]
                )
                pos_errors.append(
                    result.calibration_errors.extrinsic_errors[cam_idx][:, 3:6]
                )
            att_errors = np.array(att_errors)
            pos_errors = np.array(pos_errors)

            self.mean_att_rmses_extrinsics[cam_idx] = compute_rmses_over_time_dataset(
                att_errors
            )
            self.mean_pos_rmses_extrinsics[cam_idx] = compute_rmses_over_time_dataset(
                pos_errors
            )

        # Evaluate time offset errors
        time_offset_errors = []
        for result in results:
            time_offset_errors.append(result.calibration_errors.offset_dt_errors)
        time_offset_errors = np.array(time_offset_errors)

        self.mean_time_offset_rmse = np.sqrt(np.mean(time_offset_errors**2, axis=0))
        self.timestamps_calib = results[0].calibration_errors.stamps

    def _plot_nees_density(self, save_dir: str):
        """Plots the density of the NEES values over all runs."""
        if (
            self.mean_att_nees_distribution is None
            or self.mean_pos_nees_distribution is None
        ):
            print("No NEES distribution data to plot!")
            return

        fig, ax = plt.subplots(1, 2)
        plot_utils.plot_nees_density(
            self.mean_att_nees_distribution.values, ax=ax[0], n_dof=3
        )
        ax[0].set_title("Attitude NEES Density")
        plot_utils.plot_nees_density(
            self.mean_pos_nees_distribution.values,
            ax=ax[1],
            n_dof=3,
        )
        ax[1].set_title("Position NEES Density")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "nees_density.pdf"))
        return fig, ax

    def plot_mean_pose_rmses(self, save_dir: str = None):
        """Plots the mean position and orientation RMSEs over all runs."""
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(
            self.stamps_sync, np.rad2deg(self.mean_att_rmses), label=self.alg_name
        )
        ax[1].plot(self.stamps_sync, self.mean_pos_rmses, label=self.alg_name)
        ax[0].set_ylabel("Orientation RMSE (deg)")
        ax[1].set_ylabel("Position RMSE (m)")
        ax[1].set_xlabel("Time (s)")
        fig.suptitle(
            f"Mean Pose RMSEs for {self.alg_name} on {self.dataset_name} ({self.num_runs} runs)"
        )
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "mean_pose_rmses.pdf"))
        return fig, ax

    def plot_mean_pose_nees(self, save_dir: str = None):
        """Plots the mean NEES for the attitude and position over all runs."""
        if self.mean_att_nees_per_timestamp is None or self.mean_pos_nees_per_timestamp is None:
            return
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(self.stamps_sync, self.mean_att_nees_per_timestamp, label="Attitude NEES")
        ax[1].plot(self.stamps_sync, self.mean_pos_nees_per_timestamp, label="Position NEES")
        ax[0].axhline(y=3.0, color="tab:red", linestyle="--", label="Expected NEES")
        ax[1].axhline(y=3.0, color="tab:red", linestyle="--", label="Expected NEES")
        ax[0].set_ylabel("Attitude NEES")
        ax[1].set_ylabel("Position NEES")
        ax[1].set_xlabel("Time (s)")
        fig.suptitle(
            f"Mean NEES for {self.alg_name} on {self.dataset_name} ({self.num_runs} runs)"
        )
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "mean_pose_nees.pdf"))
        return fig, ax

    def save_to_pickle(self, pickle_fname: str):
        with open(pickle_fname, "wb") as f:
            pickle.dump(self, f)

    def _plot_consistency(self, results, save_dir: str = None):
        """Plots the consistency metrics for all trials on the same plots if they exist."""
        imu_consistency = []
        for result in results:
            if result.consistency_metrics is None:
                continue
            if type(result.consistency_metrics) is not VinsConsistencyMetricsSim:
                continue
            imu_consistency.append(result.consistency_metrics)

        if len(imu_consistency) > 0:
            plot_imu_consistency(imu_consistency, save_dir)

        # Get the calibration errors if we have them
        calib_errors = []
        for result in results:
            if not hasattr(result, "calibration_errors"):
                logging.info(
                    "No calibration errors found, skipping calibration consistency plot..."
                )
                return
            if result.calibration_errors is None:
                continue
            calib_errors.append(result.calibration_errors)

        if len(calib_errors) > 0:
            plot_calibration_errors(calib_errors, save_dir)


def plot_mean_pose_stats(
    results: typing.List[AveragedVinsResult],
    labels: typing.List[str],
    colors: typing.List[str] = None,
    save_dir: str = None,
    shift_stamps: bool = True,
):
    """Plots the mean position and orientation RMSEs over all trials for multiple results."""
    if colors is None:
        colors = sns.color_palette("deep", n_colors=len(labels))

    if len(colors) < len(results):
        raise ValueError("Number of colors must match number of results")
    if len(labels) != len(results):
        raise ValueError("Number of labels must match number of results")

    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 6))

    for i, (result, label) in enumerate(zip(results, labels)):
        if shift_stamps:
            stamps = result.stamps_sync - result.stamps_sync[0]
        else:
            stamps = result.stamps_sync

        if result.mean_att_rmses is None:
            continue
        mean_att_rmses = np.rad2deg(result.mean_att_rmses)
        color = colors[i]
        ax[0, 0].plot(
            stamps,
            mean_att_rmses,
            label=label,
            color=color,
        )

        ax[1, 0].plot(
            stamps,
            result.mean_pos_rmses,
            label=label,
            color=color,
        )

        if np.any(result.mean_att_nees_per_timestamp > 1e6) or np.any(result.mean_pos_nees_per_timestamp > 1e6):
            logging.info(f"Skipping NEES plot for result {label} due to large values.")
            continue
        ax[0, 1].plot(
            stamps,
            result.mean_att_nees_per_timestamp,
            label=label,
            color=color,
        )

        ax[1, 1].plot(
            stamps,
            result.mean_pos_nees_per_timestamp,
            label=label,
            color=color,
        )

    ax[0, 1].axhline(y=3.0, color="tab:red", linestyle="--", label="Expected NEES")
    ax[1, 1].axhline(y=3.0, color="tab:red", linestyle="--", label="Expected NEES")
    ax[0, 0].set_ylabel("Orientation RMSE (degrees)")
    ax[1, 0].set_ylabel("Position RMSE (m)")
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 1].set_xlabel("Time (s)")
    ax[0, 0].set_title("RMSE")
    ax[0, 1].set_title("NEES")
    ax[0, 1].set_ylabel("Orientation NEES")
    ax[1, 1].set_ylabel("Position NEES")
    ax[0, 0].legend()
    fig.tight_layout()

    if save_dir is not None:
        fig.savefig(
            os.path.join(save_dir, "mean_rmses_comparison.pdf"),
            dpi=300,
            bbox_inches="tight",
        )

    return fig, ax


def plot_ate_distribution(
    results: typing.List[AveragedVinsResult],
    labels: typing.List[str],
    colors: typing.List[str],
    save_dir: str = None,
    **kwargs,
):
    title = kwargs.pop("title", None)

    boxplot_defaults = {
        "linewidth": 1.5,
        "showfliers": False,
        "boxprops": {"alpha": 1.0},
        "width": 0.8,
    }
    boxplot_defaults.update(kwargs)

    data_for_plotting = []
    for i, (result, label) in enumerate(zip(results, labels)):
        att_values = np.rad2deg(result.attitude_ate_stats.values)
        pos_values = result.position_ate_stats.values

        for val in att_values:
            data_for_plotting.append(
                {
                    "Algorithm": label,
                    "Error Type": "Attitude",
                    "ATE": val,
                    "Units": "deg",
                }
            )
        for val in pos_values:
            data_for_plotting.append(
                {
                    "Algorithm": label,
                    "Error Type": "Position",
                    "ATE": val,
                    "Units": "m",
                }
            )

    # Create DataFrame for plotting
    df = pd.DataFrame(data_for_plotting)
    fig, axes = plt.subplots(1, 2)
    att_data = df[df["Error Type"] == "Attitude"]
    sns.boxplot(
        data=att_data,
        x="Algorithm",
        y="ATE",
        hue="Algorithm",
        ax=axes[0],
        order=labels,
        fill=False,
        gap=0.1,
        **boxplot_defaults,
    )
    axes[0].set_ylabel("Attitude ATE (deg)")
    axes[0].set_xlabel("Algorithm")
    axes[0].tick_params(axis="x", rotation=45)

    pos_data = df[df["Error Type"] == "Position"]
    sns.boxplot(
        data=pos_data,
        x="Algorithm",
        y="ATE",
        ax=axes[1],
        order=labels,
        hue="Algorithm",
        fill=False,
        gap=0.1,
        **boxplot_defaults,
    )
    axes[1].set_ylabel("Position ATE (m)")
    axes[1].set_xlabel("Algorithm")
    axes[1].tick_params(axis="x", rotation=45)
    
    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(
            os.path.join(save_dir, "ate_distribution.pdf"), dpi=300, bbox_inches="tight"
        )
    return fig, axes


def plot_nees_distribution(
    results: typing.List[AveragedVinsResult],
    labels: typing.List[str],
    colors: typing.List[str],
    save_dir: str = None,
    **kwargs,
):
    title = kwargs.pop("title", None)
    boxplot_defaults = {
        "linewidth": 1.5,
        "showfliers": False,
        "boxprops": {"alpha": 1.0},
        "width": 0.8,
    }

    boxplot_defaults.update(kwargs)

    data_for_plotting = []
    for i, (result, label) in enumerate(zip(results, labels)):
        # att_values = result.mean_att_nees_distribution.values
        # pos_values = result.mean_pos_nees_distribution.values

        # Single number means for each timestamp
        att_mean = np.mean(result.mean_att_nees_per_timestamp)
        pos_mean = np.mean(result.mean_pos_nees_per_timestamp)

        # print(f"Number of attitude NEES samples for {label}: {len(att_values)}")
        # print(f"Number of position NEES samples for {label}: {len(pos_values)}")A

        data_for_plotting.append(
            {
                "Algorithm": label,
                "NEES Type": "Attitude",
                "NEES": att_mean,
            }
        )
        data_for_plotting.append(
            {
                "Algorithm": label,
                "NEES Type": "Position",
                "NEES": pos_mean,
            }
        )

    # Create DataFrame for plotting
    df = pd.DataFrame(data_for_plotting)
    fig, axes = plt.subplots(1, 2)
    att_data = df[df["NEES Type"] == "Attitude"]
    sns.barplot(
        data=att_data,
        x="Algorithm",
        y="NEES",
        hue="Algorithm",
        ax=axes[0],
        order=labels,
        fill=False,
        gap=0.1,
    )
    axes[0].set_ylabel("Attitude NEES")
    axes[0].set_xlabel("Algorithm")
    axes[0].tick_params(axis="x", rotation=45)

    pos_data = df[df["NEES Type"] == "Position"]
    sns.boxplot(
        data=pos_data,
        x="Algorithm",
        y="NEES",
        ax=axes[1],
        order=labels,
        hue="Algorithm",
        fill=False,
        gap=0.1,
        # **boxplot_defaults,
    )
    axes[1].set_ylabel("Position NEES")
    axes[1].set_xlabel("Algorithm")
    axes[1].tick_params(axis="x", rotation=45)

    # Plot horizontal line for expected NEES
    axes[0].axhline(y=3.0, color="tab:red", linestyle="--", label="Expected NEES")
    axes[1].axhline(y=3.0, color="tab:red", linestyle="--", label="Expected NEES")
    axes[1].legend()

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(
            os.path.join(save_dir, "nees_distribution.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
    return fig, axes


def plot_nees_density(
    results: typing.List[AveragedVinsResult],
    labels: typing.List[str],
    colors: typing.List[str],
    save_dir: str = None,
):
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    for i, (result, label) in enumerate(zip(results, labels)):
        if result.mean_att_nees_per_timestamp is None or result.mean_pos_nees_per_timestamp is None:
            logging.warning(
                f"Skipping NEES density plot for result {label} as NEES data is missing."
            )
            return

        show_theoretical = i == 0
        plot_utils.plot_nees_density(
            result.mean_att_nees_distribution.values,
            n_dof=3,
            ax=ax[0],
            label=label,
            normalize=True,
            show_theoretical=show_theoretical,
        )
        plot_utils.plot_nees_density(
            result.mean_pos_nees_distribution.values,
            n_dof=3,
            ax=ax[1],
            label=label,
            normalize=True,
            show_theoretical=show_theoretical,
        )
    ax[0].set_title("Attitude NEES Density")
    ax[0].set_xlabel("NEES")
    ax[0].set_ylabel("Density")
    ax[1].set_title("Position NEES Density")
    ax[1].set_xlabel("NEES")
    ax[1].set_ylabel("Density")
    ax[0].legend()
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(
            os.path.join(save_dir, "nees_density.pdf"), dpi=300, bbox_inches="tight"
        )
    return fig, ax


def compare_sim_consistency_metrics(
    results: typing.List[typing.List[VinsConsistencyMetricsSim]],
    labels: typing.List[str],
    save_dir: typing.List[str],
    color: typing.List[str] = None,
):
    """Generates a combined plot comparing the pose
    consistency metrics across multiple algorithms
    """

    if color is None:
        color = "tab:blue"

    fig, ax = plt.subplots(6, len(results), sharex=True, figsize=(8, 6))
    for col_idx, (result, label) in enumerate(zip(results, labels)):
        for j, metrics in enumerate(result):
            stamps = metrics.stamps - metrics.stamps[0]
            for k in range(6):
                ax[k, col_idx].plot(stamps, metrics.delta_xi[:, k], alpha=0.5, color=color,)

                if j == 0:
                    ax[k, col_idx].plot(
                        stamps,
                        metrics.three_sigma[:, k],
                        color="k",
                        linestyle="--",
                        label=r"$3\sigma$"
                    )
                    ax[k, col_idx].plot(
                        stamps,
                        -metrics.three_sigma[:, k],
                        color="k",
                        linestyle="--",
                    )

        ax[0, col_idx].set_title(label)
        ax[0, 0].set_ylabel("x (rad)", fontsize=10)
        ax[1, 0].set_ylabel("y (rad)", fontsize=10)
        ax[2, 0].set_ylabel("z (rad)", fontsize=10)
        ax[3, 0].set_ylabel("x (m)", fontsize=10)
        ax[4, 0].set_ylabel("y (m)", fontsize=10)
        ax[5, 0].set_ylabel("z (m)", fontsize=10)
        ax[5, col_idx].set_xlabel("Time (s)")
    # fig.suptitle("Pose Consistency Metrics Comparison", fontsize=16)
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "consistency_comparison.pdf"), dpi=300)
    return fig, ax

def compare_averaged_vins_results(
    results: typing.List[AveragedVinsResult],
    labels: typing.List[str],
    colors: typing.List[str] = None,
    save_dir: str = None,
    print_table: bool = True,
):
    """Generates comparison plots for averaged trajectory results,
    on a single dataset or for Monte-Carlo runs.

    This function generates the following plots:
    - Mean pose RMSEs over time for each algorithm
    - Distribution of ATEs across trials for each algorithm
    - Distribution of NEES values across trials for each algorithm
    - NEES density plots for each algorithm.

    It alsop generates
    """

    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if colors is None:
        colors = sns.color_palette("deep", n_colors=len(labels))

    plot_mean_pose_stats(results, labels, colors, save_dir)
    plot_ate_distribution(results, labels, colors, save_dir)
    plot_nees_distribution(results, labels, colors, save_dir)
    plot_nees_density(results, labels, colors, save_dir)

    # Generate ATE comparison table
    rows = []
    for label, result in zip(labels, results):
        att_ate = result.attitude_ate_stats.mean
        att_std = result.attitude_ate_stats.std

        att_ate = np.rad2deg(att_ate)
        att_std = np.rad2deg(att_std)

        rows.append(
            {
                "algorithm": label,
                "position_ate": result.position_ate_stats.mean,
                "attitude_ate": att_ate,
                "position_std": result.position_ate_stats.std,
                "attitude_std": att_std,
                "num_trials": result.num_runs,
            }
        )

    df = pd.DataFrame(rows)
    table_str = generate_ate_comparison_table(
        df,
        position_col="position_ate",
        attitude_col="attitude_ate",
        algorithm_col="algorithm",
        output_format="latex",
        precision=3,
        position_unit="m",
        attitude_unit="deg",
        highlight_best=True,
        # save_path=os.path.join(save_dir, "ate_comparison_table.md")
    )

    # Generate a NEES comparison table
    if print_table:
        print(table_str)

    # Generate RMSE/NEES comparison table
    rmse_nees_data: typing.Dict[str, typing.Dict[str, float]] = {}
    for label, result in zip(labels, results):
        avg_att_ate = np.rad2deg(result.attitude_ate_stats.mean)
        avg_pos_ate = result.position_ate_stats.mean

        if result.mean_att_nees_per_timestamp is None: 
            continue

        avg_att_nees = np.mean(result.mean_att_nees_per_timestamp)
        avg_pos_nees = np.mean(result.mean_pos_nees_per_timestamp)
        rmse_nees_data[label] = {
            "att_ate": avg_att_ate,
            "pos_ate": avg_pos_ate,
            "att_nees": avg_att_nees,
            "pos_nees": avg_pos_nees,
        }

    rmse_nees_save_path = (
        os.path.join(save_dir, "rmse_nees_comparison_table.txt")
        if save_dir is not None
        else None
    )
    rmse_nees_table_str = generate_rmse_nees_comparison_table(
        rmse_nees_data,
        precision=3,
        save_path=rmse_nees_save_path,
    )

    if print_table:
        print(rmse_nees_table_str)