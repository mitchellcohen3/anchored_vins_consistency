"""Single-run trajectory evaluation results for VINS."""

import json
import logging
import os
import time
import typing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from navlie.bspline import SE3Bspline
from navlie.lib.states import SE3State
from navlie.types import StateWithCovariance
from navlie.utils import GaussianResultList, plot_poses
from navlie.utils.alignment import associate_and_align_trajectories

from pymlg import SO3, SE3

import pyvins.file_utils as file_utils
import pyvins.plot_utils as plot_utils
from pyvins.plot_utils import plot_three_sigma, plot_nees as plot_nees_custom

from pyvins.containers import (
    VinsStateInfo,
    VinsCovarianceInfo,
    VinsSimulationData,
    TimingData,
)
from pyvins.metrics import (
    PoseErrorMetrics,
    PoseConsistencyMetrics,
    VinsConsistencyMetricsSim,
    CalibrationErrors,
)
from pyvins.association import (
    associate_states,
    associate_states_and_covariances,
)


def load_timing_file(timing_fpath: str) -> pd.DataFrame:
    """Loads the timing file and converts computation times to ms."""
    try:
        df = pd.read_csv(timing_fpath, sep=",")
        df = df.copy()

        # Convert computation times to ms (from seconds)
        df.iloc[:, 1:] = df.iloc[:, 1:] * 1000.0
        return df
    except Exception as e:
        print(f"Error loading timing file: {e}")
        return None


def plot_imu_consistency(
    metrics: typing.Union[
        "VinsConsistencyMetricsSim", typing.List["VinsConsistencyMetricsSim"]
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

    # For each slice, plot the results
    if combined_plot:
        # Create a 3x5 subplot
        fig, axes = plt.subplots(3, 5, sharex=True, figsize=(15, 10))

        for col_idx, (s, title, labels) in enumerate(zip(slices, titles, labels_list)):
            ax = axes[:, col_idx]

            # For each result, plot the three sigma bounds
            for i, result in enumerate(metrics):
                if i == 0:
                    enable_sigma_bounds = True
                else:
                    enable_sigma_bounds = False
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

            # Set labels for each subplot in the column
            for row_idx, label in enumerate(labels):
                axes[row_idx, col_idx].set_ylabel(label)
                if col_idx == 0:  # Add row titles only to the leftmost column
                    axes[row_idx, col_idx].text(
                        -0.15,
                        0.5,
                        ["X", "Y", "Z"][row_idx],
                        transform=axes[row_idx, col_idx].transAxes,
                        rotation=90,
                        verticalalignment="center",
                        fontweight="bold",
                    )

            # Add column title at the top
            axes[0, col_idx].set_title(title, fontsize=10)

        # Set x-label only for bottom row
        for col_idx in range(5):
            axes[-1, col_idx].set_xlabel("Time (s)")

        fig.suptitle("IMU State Consistency Metrics", fontsize=16)
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "imu_consistency_combined.pdf"))
    else:
        for s, title, labels in zip(slices, titles, labels_list):
            fig, ax = plt.subplots(3, 1, sharex=True)

            # For each result, plot the three sigma bounds
            for i, result in enumerate(metrics):
                if i == 0:
                    enable_sigma_bounds = True
                else:
                    enable_sigma_bounds = False
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


class VinsSimulationResult:
    """Container for the result of a single VINS simulation result.

    Stores the loaded data, consistency metrics, and error metrics.
    """

    data: VinsSimulationData

    consistency_metrics: VinsConsistencyMetricsSim
    pose_error_metrics: PoseErrorMetrics
    calibration_errors: CalibrationErrors

    timing_data: TimingData = None

    # Summary statistics about the trajectory lengths
    est_trajectory_length: float
    gt_trajectory_length: float

    evaluation_folder: str = None

    def __init__(
        self,
        data: VinsSimulationData,
        state_representation: str,
        lie_direction: str,
        timing_path: str = None,
        evaluation_folder: str = None,
        plot_save_dir: str = None,
    ):
        # Check if state
        """Evaluates the results from simulation data."""
        self.data = data
        self.evaluation_folder = evaluation_folder

        # Evaluate consistency
        self.consistency_metrics = VinsConsistencyMetricsSim(
            self.data.state_est.imu_states,
            self.data.state_gt.imu_states,
            self.data.cov_info.imu_std,
            self.data.cov_info.att_cov,
            self.data.cov_info.pos_cov,
            state_representation=state_representation,
            lie_direction=lie_direction,
        )

        # Evaluate pose error metrics
        self.pose_error_metrics = PoseErrorMetrics(
            self.data.state_gt.imu_states,
            self.data.state_est.imu_states,
            lie_direction="right",
        )

        self.calibration_errors = None
        if (
            self.data.state_est.has_calibration_parameters()
            and self.data.state_gt.has_calibration_parameters()
        ):
            # Convert the covariances to 3 sigma bounds if we have them

            self.calibration_errors = CalibrationErrors(
                self.data.state_gt,
                self.data.state_est,
                self.data.cov_info.intrinsic_std,
                self.data.cov_info.extrinsic_std,
                self.data.cov_info.offset_dt_std,
            )

        # Load timing data if we have it
        if timing_path is not None:
            df = pd.read_csv(timing_path, sep=",")
            df = df.copy()

            # Convert computating times to ms (from seconds)
            df.iloc[:, 1:] = df.iloc[:, 1:] * 1000.0

            names = df.columns.tolist()
            timestamps = df[names[0]]
            frame_processing_times_ms = df[names[-1]].to_numpy()

            self.timing_data = TimingData(
                timestamps=timestamps,
                frame_processing_times_ms=frame_processing_times_ms,
                raw_timing_df=df,
            )

        # Comptue the trajectory lengths
        est_traj_length = 0.0
        gt_traj_length = 0.0
        for i in range(1, len(self.data.state_est.imu_states)):
            pos_gt_i = self.data.state_gt.imu_states[i].position
            pos_gt_prev = self.data.state_gt.imu_states[i - 1].position
            gt_traj_length += np.linalg.norm(pos_gt_i - pos_gt_prev)

            pos_est_i = self.data.state_est.imu_states[i].position
            pos_est_prev = self.data.state_est.imu_states[i - 1].position
            est_traj_length += np.linalg.norm(pos_est_i - pos_est_prev)

        self.est_trajectory_length = est_traj_length
        self.gt_trajectory_length = gt_traj_length

        logging.info(f"Estimated trajectory length: {self.est_trajectory_length:.2f} m")
        logging.info(
            f"Groundtruth trajectory length: {self.gt_trajectory_length:.2f} m"
        )

        total_time = (
            self.data.state_est.imu_states[-1].stamp
            - self.data.state_est.imu_states[0].stamp
        )
        logging.info(f"Trajectory time: {total_time/60.0:.3f} minutes")

    @classmethod
    def from_files(
        cls,
        state_est_path: str,
        state_gt_path: str,
        state_std_path: str,
        poses_est_path: str = None,
        timing_fpath: str = None,
        **kwargs,
    ):
        data = VinsSimulationData.from_files(
            state_est_path,
            state_gt_path,
            state_std_path,
            poses_est_path,
        )
        evaluation_folder = os.path.dirname(state_est_path)
        return cls(data, timing_path=timing_fpath, evaluation_folder=evaluation_folder, **kwargs,)

    def generate_plots(self, save_dir: str):
        """Generates and saves all plots to the specified directory."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self._plot_trajectories(save_dir)
        self._plot_consistency(save_dir)
        self._plot_rmses_over_time(save_dir)
        self._plot_calibration_errors(save_dir, show_titles=True)
        self._plot_timing_data(save_dir)

    def _plot_calibration_errors(self, save_dir: str = None, show_titles: bool = True):
        """Plots the calibration errors over time."""
        if self.calibration_errors is None:
            return

        # Time offset errors
        stamps = self.calibration_errors.stamps - self.calibration_errors.stamps[0]
        fig, ax = plt.subplots(1, 1)
        ax.plot(stamps, self.calibration_errors.offset_dt_errors)

        # If we have 3-sigma bounds, additionally plot them
        if self.data.cov_info.offset_dt_std is not None:
            three_sigma = 3.0 * np.array(self.data.cov_info.offset_dt_std)
            ax.plot(stamps, three_sigma, color="k", linestyle="--")
            ax.plot(stamps, -three_sigma, color="k", linestyle="--")

        if show_titles:
            ax.set_title("Time Offset Errors")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (s)")
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "time_offset_errors.pdf"))

        # Plot the intrinsic errors
        for (
            cam_id,
            intrinsic_errors,
        ) in self.calibration_errors.intrinsic_errors.items():

            # Get the three sigma bounds if we have them
            three_sigma = None
            if self.data.cov_info.intrinsic_std is not None:
                if cam_id in self.data.cov_info.intrinsic_std:
                    three_sigma = 3.0 * np.array(
                        self.data.cov_info.intrinsic_std[cam_id]
                    )

            fig, ax = plot_utils.plot_intrinsics_errors(
                stamps,
                intrinsic_errors,
                cam_id,
                three_sigma=three_sigma,
                figsize=(5, 6),
            )
            fig.suptitle(f"Intrinsic Errors for Camera {cam_id}")
            fig.tight_layout()
            if save_dir is not None:
                fig.savefig(
                    os.path.join(save_dir, f"intrinsic_errors_cam_{cam_id}.pdf")
                )

        # Plot the extrinsic errors
        for (
            cam_id,
            extrinsic_errors,
        ) in self.calibration_errors.extrinsic_errors.items():

            if self.data.cov_info.extrinsic_std is not None:
                if cam_id in self.data.cov_info.extrinsic_std:
                    three_sigma = 3.0 * np.array(
                        self.data.cov_info.extrinsic_std[cam_id]
                    )
                else:
                    three_sigma = None

            fig, ax = plot_utils.plot_extrinsics_errors(
                stamps,
                extrinsic_errors,
                cam_id,
                three_sigma=three_sigma,
                figsize=(5, 6),
                show_titles=show_titles,
            )
            if save_dir is not None:
                fig.savefig(
                    os.path.join(save_dir, f"extrinsic_errors_cam_{cam_id}.pdf")
                )

    def _plot_trajectories(self, save_dir: str = None, plot_landmarks: bool = True):
        traj_data = {
            "Groundtruth": self.data.state_gt.imu_states,
            "Estimate": self.data.state_est.imu_states,
        }

        fig, ax = plot_utils.plot_trajectories(traj_data)

        # If we have landmarks, plot them
        if self.data.landmarks_gt is not None and plot_landmarks:
            ax.scatter(
                self.data.landmarks_gt[:, 0],
                self.data.landmarks_gt[:, 1],
                self.data.landmarks_gt[:, 2],
                c="tab:green",
                marker="o",
                label="Landmarks",
                alpha=0.3,
                s=1.0,
            )

        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "trajectories.pdf"))

    def _plot_consistency(self, save_dir: str = None):
        if self.consistency_metrics is None:
            return

        plot_imu_consistency(self.consistency_metrics, save_dir, combined_plot=True)

        # Plot the NEES over time
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        stamps = self.consistency_metrics.stamps
        att_nees = self.consistency_metrics.att_nees
        pos_nees = self.consistency_metrics.pos_nees

        plot_nees_custom(stamps, att_nees, ax=ax[0], expected_nees=3)
        ax[0].set_title("Attitude NEES")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("NEES")
        plot_nees_custom(stamps, pos_nees, ax=ax[1], expected_nees=3)
        ax[1].set_title("Position NEES")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("NEES")

    def _plot_rmses_over_time(self, save_dir: str = None):
        stamps = [x.stamp for x in self.data.state_gt.imu_states]
        fig, ax = plot_utils.plot_pose_rmses_over_time(
            stamps,
            self.pose_error_metrics.attitude_rmses_over_time,
            self.pose_error_metrics.position_rmses_over_time,
        )
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "rmses_over_time.pdf"))

    def _plot_timing_data(self, save_dir: str = None):
        if self.timing_data is None:
            return

        fig, ax = plot_utils.plot_timing_dataframe(self.timing_data.timing_df)
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "timing_data.pdf"))

    @property
    def gt_poses(self) -> typing.List[SE3State]:

        gt_poses = []
        for imu_state in self.data.state_gt.imu_states:
            pose = SE3State(
                value=SE3.from_components(imu_state.attitude, imu_state.position),
                stamp=imu_state.stamp,
            )
            gt_poses.append(pose)
        return gt_poses

    @property
    def est_poses(self) -> typing.List[SE3State]:
        est_poses = []
        for imu_state in self.data.state_est.imu_states:
            pose = SE3State(
                value=SE3.from_components(imu_state.attitude, imu_state.position),
                stamp=imu_state.stamp,
            )
            est_poses.append(pose)
        return est_poses


class TrajectoryResult:
    """Container for the result of a trajectory evaluation.

    A state estimation algorithm must output estimated poses, as well as optionally
    covariances.

    If covariances are provided, consistency metrics will be computed.
    """

    gt_poses: typing.List[SE3State]
    est_poses: typing.List[SE3State]
    pose_covariances: typing.List[np.ndarray]

    # The full estimated states
    est_states: VinsStateInfo
    cov_info: VinsCovarianceInfo

    consistency_metrics: PoseConsistencyMetrics
    pose_error_metrics: PoseErrorMetrics

    timing_df: pd.DataFrame
    results_dir: str

    def __init__(
        self,
        poses_gt_path: str,
        poses_est_path: str,
        state_est_path: str = None,
        state_std_path: str = None,
        timing_path: str = None,
        align: bool = False,
        verbose: bool = False,
        lie_direction: str = "left",
        state_representation: str = "decoupled",
        save_dir: str = None,
        n_to_align: int = -1,
    ):
        """Loads the groundtruth, estimates, and covariance files for a run on real data."""
        self.verbose = verbose
        self.lie_direction = lie_direction
        self.aligned = align

        gt_poses, _ = file_utils.load_poses_from_file(poses_gt_path)
        est_poses, pose_covariances = file_utils.load_poses_from_file(poses_est_path)

        logging.info(f"Loaded {len(gt_poses)} groundtruth poses from {poses_gt_path}")
        logging.info(f"Loaded {len(est_poses)} estimated poses from {poses_est_path}")

        if pose_covariances:
            # Find associate timestamps
            gt_poses, est_poses, pose_covariances = associate_states_and_covariances(
                gt_poses,
                est_poses,
                pose_covariances,
            )
        else:
            gt_poses, est_poses = associate_states(gt_poses, est_poses)

        logging.info(f"Associated {len(gt_poses)} poses after timestamp matching")

        # Align the trajectories if desired
        if align:
            gt_poses, est_poses, transformation_dict = associate_and_align_trajectories(
                gt_poses,
                est_poses,
                align=True,
                verbose=verbose,
                n_to_align=n_to_align,
            )

        # Compute error metrics
        self.est_states = None
        self.cov_info = None
        if state_est_path is not None:
            logging.info(f"Loading state estimates from {state_est_path}")
            self.est_states = VinsStateInfo.from_file(state_est_path)

        if state_std_path is not None and os.path.exists(state_std_path):
            logging.info(f"Loading covariances from {state_est_path}")
            self.cov_info = VinsCovarianceInfo.from_files(state_std_path, poses_est_path)

        # Evaluate the pose consistency
        self.consistency_metrics = None
        if pose_covariances:
            self.consistency_metrics = PoseConsistencyMetrics.compute_from_poses(
                gt_poses,
                est_poses,
                pose_covariances,
                lie_direction=lie_direction,
                state_representation=state_representation,
            )

        # Compute the pose errors
        self.pose_error_metrics = PoseErrorMetrics(
            gt_poses,
            est_poses,
            lie_direction="right",
        )

        # Load the timing information if we have it
        if timing_path is not None and os.path.exists(timing_path):
            self.timing_df = load_timing_file(timing_path)
        else:
            self.timing_df = None

        self.gt_poses = gt_poses
        self.est_poses = est_poses
        self.pose_covariances = pose_covariances

        # Compute some stats about the trajectory
        gt_positions = np.array([x.position for x in self.gt_poses])
        est_positions = np.array([x.position for x in self.est_poses])
        gt_traj_length = compute_trajectory_length(gt_positions)
        est_traj_length = compute_trajectory_length(est_positions)
        total_traj_time = self.gt_poses[-1].stamp - self.gt_poses[0].stamp

        logging.info(f"Total groundtruth trajectory length: {gt_traj_length:.2f} m")
        logging.info(f"Total estimated trajectory length: {est_traj_length:.2f} m")
        logging.info(f"Total trajectory time: {total_traj_time:.2f} s")

    def generate_all_plots(self, save_dir: str = None):
        """Generates and saves all plots to the specified directory,"""
        self._plot_nees(save_dir)
        self._plot_pose_three_sigma(save_dir)
        self._plot_rmses_over_time(save_dir)
        self._plot_trajectories(save_dir)

        self._plot_extrinsics(save_dir)
        self._plot_timeoffset(save_dir)
        # self.est_states.plot_velocities(save_dir)
        self.plot_timing(save_dir)

    def _plot_nees(self, save_dir: str = None):
        if self.consistency_metrics is None:
            return

        stamps = self.consistency_metrics.stamps
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(stamps, self.consistency_metrics.att_nees)
        ax[1].plot(stamps, self.consistency_metrics.pos_nees)
        ax[0].set_ylabel("Attitude NEES")
        ax[1].set_ylabel("Position NEES")
        ax[1].set_xlabel("Time (s)")
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "nees.pdf"))
        return fig, ax

    def _plot_pose_three_sigma(self, save_dir: str = None):
        if self.consistency_metrics is None:
            return

        stamps = self.consistency_metrics.stamps
        delta_xi = self.consistency_metrics.delta_xi
        three_sigma = self.consistency_metrics.three_sigma
        fig, ax = plot_three_sigma(
            stamps,
            delta_xi,
            three_sigma,
        )
        ax[0, 0].set_title("Attitude Errors")
        ax[0, 1].set_title("Position Errors")
        ax[0, 0].set_ylabel("x (rad)")
        ax[1, 0].set_ylabel("y (rad)")
        ax[2, 0].set_ylabel("z (rad)")
        ax[0, 1].set_ylabel("x (m)")
        ax[1, 1].set_ylabel("y (m)")
        ax[2, 1].set_ylabel("z (m)")
        ax[2, 0].set_xlabel("Time (s)")
        ax[2, 1].set_xlabel("Time (s)")

        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "pose_consistency.pdf"))
        return fig, ax

    def _plot_rmses_over_time(self, save_dir):
        """Plots position and attitude RMSEs over time."""
        stamps = [x.stamp for x in self.gt_poses]
        fig, ax = plot_utils.plot_pose_rmses_over_time(
            stamps,
            self.pose_error_metrics.attitude_rmses_over_time,
            self.pose_error_metrics.position_rmses_over_time,
        )
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "pose_rmses.pdf"))
        return fig, ax

    def plot_timing(self, save_dir: str = None):
        """Plots the timing dataframe if it exists."""
        if self.timing_df is None:
            return

        # Plot the timing information
        plot_utils.plot_timing_dataframe(self.timing_df, save_dir)

    def _plot_extrinsics(self, save_dir: str = None):
        if self.est_states is None:
            return
        if self.est_states.camimu_extrinsics is None:
            return

        # Plot the extrinsics
        for cam_id, extrinsics in self.est_states.camimu_extrinsics.items():
            phi_list = []
            pos_list = []
            for T_bc in extrinsics:
                phi = SO3.Log(T_bc[0:3, 0:3])
                position = T_bc[0:3, 3]
                phi = phi.ravel()
                position = position.ravel()
                phi_list.append(phi)
                pos_list.append(position)
            phi_list = np.array(phi_list)
            pos_list = np.array(pos_list)
            stamps = self.est_states.stamps
            fig, ax = plot_utils.plot_pose_timeseries_2(stamps, phi_list, pos_list)
            fig.suptitle(f"Extrinsics for {cam_id}")
            if save_dir is not None:
                fig.savefig(os.path.join(save_dir, f"extrinsics_{cam_id}.pdf"))

    def _plot_timeoffset(self, save_dir: str = None):
        """Plots the time offset estimates if they exist."""
        if self.est_states is None:
            return
        if self.est_states.camimu_time_offset is None:
            return

        time_offsets = np.array(self.est_states.camimu_time_offset)
        stamps = self.est_states.stamps
        fig, ax = plt.subplots(1, 1)
        ax.plot(stamps, time_offsets)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Camera-IMU Time Offset (s)")
        ax.set_title("Camera-IMU Time Offset Estimates Over Time")
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "time_offset.pdf"))
        return fig, ax

    def _plot_trajectories(self, save_path: str = None, shift_origin: bool = True):
        """Plots the estimated and groundtruth trajectories."""
        gt_poses = []
        est_poses = []
        if shift_origin:
            origin_gt = self.gt_poses[0].position
            for gt_pose in self.gt_poses:
                shifted_position = gt_pose.position - origin_gt
                shifted_pose = SE3State(
                    value=SE3.from_components(gt_pose.attitude, shifted_position),
                    stamp=gt_pose.stamp,
                )
                gt_poses.append(shifted_pose)
            for est_pose in self.est_poses:
                shifted_position = est_pose.position - origin_gt
                shifted_pose = SE3State(
                    value=SE3.from_components(est_pose.attitude, shifted_position),
                    stamp=est_pose.stamp,
                )
                est_poses.append(shifted_pose)
        else:
            gt_poses = self.gt_poses
            est_poses = self.est_poses

        plot_2d_list = [False]
        for plot_2d in plot_2d_list:
            fig, ax = plot_poses(
                gt_poses,
                label="Groundtruth",
                step=None,
                plot_2d=plot_2d,
            )
            fig, ax = plot_poses(
                est_poses,
                label="Estimate",
                ax=ax,
                step=None,
                plot_2d=plot_2d,
            )
            ax.legend()
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            if not plot_2d:
                ax.set_zlabel("z (m)")

            if self.aligned:
                ax.set_title("Aligned Trajectories")
            else:
                ax.set_title("Trajectories")

            if save_path is not None:
                fig.savefig(os.path.join(save_path, "trajectories.pdf"))

        return fig, ax


class TrajectoryAnalyzer:
    """A class to analyze a trajectory, computing velocities and plotting results."""

    # Translational velocities
    linear_velocities: np.ndarray
    # Angular velocities
    angular_velocities: np.ndarray
    # Body frame translational velocities
    body_frame_velocities: np.ndarray
    # Timestamps corresponding to the velocities
    timestamps: np.ndarray
    # Poses evaluated at the timestamps
    eval_poses: typing.List[SE3State]
    # Original input poses
    poses: typing.List[SE3State]

    # Trajectory statistics
    trajectory_length: float
    average_speed: float
    max_speed: float

    def __init__(
        self,
        poses: typing.List[SE3State],
        eval_dt: float = 0.10,
    ):
        self.poses = poses
        spline = SE3Bspline(poses)
        start_time = spline.start_time
        end_time = spline.end_time

        logging.info("Analyzing trajectory...")
        logging.info(f"Number of input poses: {len(poses)}")
        eval_stamps = np.arange(start_time, end_time, eval_dt)
        omegas = []
        velocities = []
        stamps = []
        eval_poses = []
        body_frame_velocities = []
        for t in eval_stamps:
            pose = spline.get_pose(t)
            omega_b_ba, v_a_ba = spline.get_velocity(t)
            if pose is not None and omega_b_ba is not None:
                omegas.append(omega_b_ba)
                velocities.append(v_a_ba)

                # Compute body frame velocities
                v_b_ba = pose.attitude.T @ v_a_ba
                body_frame_velocities.append(v_b_ba)
                stamps.append(t)
                eval_poses.append(pose)

        end_time = stamps[-1]
        start_time = stamps[0]
        print(f"Dataset total duration: {end_time - start_time:.2f} seconds")
        self.linear_velocities = np.array(velocities)
        self.angular_velocities = np.array(omegas)
        self.body_frame_velocities = np.array(body_frame_velocities)
        self.timestamps = np.array(stamps)
        self.eval_poses = eval_poses

        # Compute trajectory length
        traj_length = 0.0
        for i in range(1, len(self.eval_poses)):
            p1 = self.eval_poses[i - 1].position
            p2 = self.eval_poses[i].position
            traj_length += np.linalg.norm(p2 - p1)

        self.trajectory_length = traj_length
        logging.info(f"Trajectory length: {self.trajectory_length:.2f} m")

        # Compute average speed
        total_time = self.timestamps[-1] - self.timestamps[0]
        avg_speed = traj_length / total_time
        self.average_speed = avg_speed
        logging.info(f"Average speed: {self.average_speed:.2f} m/s")
        # Compute max speed
        speeds = np.linalg.norm(self.linear_velocities, axis=1)
        max_speed = np.max(speeds)
        self.max_speed = max_speed
        logging.info(f"Max speed: {self.max_speed:.2f} m/s")

        # Print total trajectory time
        logging.info(f"Total trajectory time: {total_time:.2f} seconds")

    def plot_trajectory(self, step=500):
        fig, ax = plot_poses(
            self.eval_poses,
            label="Trajectory",
            step=step,
            plot_2d=False,
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title("Trajectory")
        return fig, ax

    def plot_velocities(self):
        fig, ax = plt.subplots(3, 1, figsize=(10, 8))
        ax[0].set_title("Angular Velocities")
        ax[0].plot(self.timestamps, self.angular_velocities[:, 0], label="Omega X")
        ax[0].plot(self.timestamps, self.angular_velocities[:, 1], label="Omega Y")
        ax[0].plot(self.timestamps, self.angular_velocities[:, 2], label="Omega Z")
        ax[0].set_ylabel("Angular Velocity [rad/s]")
        ax[0].legend()

        ax[1].set_title("Inertial Velocities")
        ax[1].plot(self.timestamps, self.linear_velocities[:, 0], label="Velocity X")
        ax[1].plot(self.timestamps, self.linear_velocities[:, 1], label="Velocity Y")
        ax[1].plot(self.timestamps, self.linear_velocities[:, 2], label="Velocity Z")
        ax[1].set_ylabel("Linear Velocity [m/s]")
        ax[1].set_xlabel("Time [s]")
        ax[1].legend()

        ax[2].set_title("Body Frame Velocities")
        ax[2].plot(
            self.timestamps, self.body_frame_velocities[:, 0], label="Velocity X"
        )
        ax[2].plot(
            self.timestamps, self.body_frame_velocities[:, 1], label="Velocity Y"
        )
        ax[2].plot(
            self.timestamps, self.body_frame_velocities[:, 2], label="Velocity Z"
        )
        ax[2].set_ylabel("Linear Velocity [m/s]")
        ax[2].set_xlabel("Time [s]")
        ax[2].legend()
        fig.tight_layout()


def compute_trajectory_length(positions: np.ndarray) -> float:
    """Computes the total length of a trajectory.

    Positions should be an N x 3, where N is the number of stamps.
    """
    if positions.shape[1] != 3:
        logging.info("Positions should be N x 3!")

    traj_length = 0.0
    for i in range(1, positions.shape[0]):
        delta_pos = positions[i, :] - positions[i - 1, :]
        traj_length += np.linalg.norm(delta_pos)
    return traj_length
