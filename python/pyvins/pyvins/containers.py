"""Data containers and dataclasses for VINS evaluation."""

import os
import typing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass, field

from navlie.lib.imu import IMUState
from navlie.lib.states import SE3State
from navlie.utils import GaussianResultList

from pymlg import SO3, SE3

import pyvins.file_utils as file_utils
from pyvins.association import associate_states_and_covariances, associate_states
from pyvins.utils import shift_stamps

import logging


class Statistics:
    """A class designed to compute and store statistics about a set of values."""

    def __init__(self, values: np.ndarray):
        self.values = values

        # Compute stats
        self.mean = np.mean(values)
        self.std = np.std(values)
        self.median = np.median(values)
        self.min = np.min(values)
        self.max = np.max(values)

        self.rmse = np.sqrt(np.mean(values**2))

    def __str__(self):
        return f"Num values: {len(self.values)}, Mean: {self.mean:.3f}, Std: {self.std:.3f}, Median: {self.median}, Min: {self.min}, Max: {self.max}"


@dataclass
class ErrorMetrics:
    """Container for some computed error metrics."""

    phi_errors_rad: np.ndarray
    pos_errors: np.ndarray
    attitude_rmses: np.ndarray
    position_rmses: np.ndarray


@dataclass
class ConsistencyMetrics:
    """Container for consistency metrics."""

    att_nees: np.ndarray
    pos_nees: np.ndarray
    pose_grl: GaussianResultList
    att_result_list: GaussianResultList
    pos_result_list: GaussianResultList


@dataclass
class CalibrationErrorMetrics:
    """Container for calibration error metrics."""

    # [cam_idx -> errors], where errors are an [N, 3] array where N is the
    # number of states
    att_errors: typing.Dict[int, np.ndarray] = field(default_factory=dict)
    pos_errors: typing.Dict[int, np.ndarray] = field(default_factory=dict)
    stamps: typing.Dict[int, np.ndarray] = field(default_factory=dict)

    att_rmse: typing.Dict[int, np.ndarray] = field(default_factory=dict)
    pos_rmse: typing.Dict[int, np.ndarray] = field(default_factory=dict)


class CalibrationParameters:
    """Container for calibration parameters and their associated covariances."""

    extrinsics: typing.Dict[str, typing.List[np.ndarray]]
    intrinsics: typing.Dict[str, typing.List[np.ndarray]]
    offset_dt: typing.List[float]
    stamps: np.ndarray

    def __init__(self, extrinsics, intrinsics, offset_dt, stamps):
        self.extrinsics = extrinsics
        self.intrinsics = intrinsics
        self.offset_dt = offset_dt
        self.stamps = stamps


class VinsStateInfo:
    """Container for the full VINS state information with self calibration."""

    # IMU states over the trajectory
    imu_states: typing.List[IMUState]

    # Camera intrinsics and extrinsics
    camimu_extrinsics: typing.Dict[str, typing.List[np.ndarray]]
    cam_intrinsics: typing.Dict[str, np.ndarray]
    camimu_time_offset: np.ndarray

    # Timestamps of all states (N, )
    stamps: np.ndarray

    def __init__(
        self,
        imu_states,
        stamps,
        camimu_extrinsics=None,
        cam_intrinsics=None,
        camimu_time_offset=None,
    ):
        self.imu_states = imu_states
        self.camimu_extrinsics = camimu_extrinsics
        self.cam_intrinsics = cam_intrinsics
        self.camimu_time_offset = camimu_time_offset
        self.stamps = stamps

    @classmethod
    def from_file(cls, state_file_path: str):
        """Loads in the VINS state information from the specified file."""
        state_dict = file_utils.load_ov_state_est_file(state_file_path)
        return cls(
            imu_states=state_dict["imu_states"],
            stamps=state_dict["stamps"],
            camimu_time_offset=state_dict["offset_dt"],
            camimu_extrinsics=state_dict["extrinsics"],
            cam_intrinsics=state_dict["intrinsics"],
        )

    def has_calibration_parameters(self) -> bool:
        """Returns true if the state contains calibration parameters."""
        return (
            self.camimu_extrinsics is not None
            and self.cam_intrinsics is not None
            and self.camimu_time_offset is not None
        )

    def plot_biases(self, save_dir: str = None):
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        stamps = np.array([state.stamp for state in self.imu_states])
        gyro_biases = np.array([state.bias_gyro for state in self.imu_states])
        accel_biases = np.array([state.bias_accel for state in self.imu_states])
        ax[0].plot(stamps, gyro_biases[:, 0], color="tab:blue", label="x")
        ax[0].plot(stamps, gyro_biases[:, 1], color="tab:red", label="y")
        ax[0].plot(stamps, gyro_biases[:, 2], color="tab:green", label="z")
        ax[0].set_ylabel(r"Gyro Bias (rad/s)")
        ax[1].plot(stamps, accel_biases[:, 0], color="tab:blue", label="x")
        ax[1].plot(stamps, accel_biases[:, 1], color="tab:red", label="y")
        ax[1].plot(stamps, accel_biases[:, 2], color="tab:green", label="z")
        ax[1].set_ylabel(r"Accel Bias (m/$s^2$)")
        fig.suptitle(f"IMU Biases")
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, f"biases.pdf"), bbox_inches="tight", dpi=300)
    
    def plot_intrinsics(self, save_dir: str = None):
        if self.cam_intrinsics is None:
            return
    
        for cam_id, intrinsics in self.cam_intrinsics.items():
            fig, ax = plt.subplots(2, 1, figsize=(8, 6))
            intrinsics = np.array(intrinsics)
            ax[0].plot(self.stamps, intrinsics[:, 0], color="tab:blue", label="fx")
            ax[0].plot(self.stamps, intrinsics[:, 1], color="tab:red", label="fy")
            ax[0].set_ylabel("Focal Length (px)")
            ax[1].plot(self.stamps, intrinsics[:, 2], color="tab:blue", label="cx")
            ax[1].plot(self.stamps, intrinsics[:, 3], color="tab:red", label="cy")
            ax[1].set_ylabel("Principal Point (px)")
            fig.suptitle(f"Intrinsics for {cam_id}")
            fig.tight_layout()
            if save_dir is not None:
                fig.savefig(os.path.join(save_dir, f"intrinsics_{cam_id}.pdf"))

    def plot_time_offset(self, save_dir: str = None):
        if self.camimu_time_offset is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(self.stamps, self.camimu_time_offset, color="tab:blue")
        ax.set_ylabel("Camera-IMU Time Offset (s)")
        fig.suptitle(f"Camera-IMU Time Offset")
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, f"time_offset.pdf"), bbox_inches="tight", dpi=300)

    def plot_extrinsics(self, save_dir: str = None):
        if self.camimu_extrinsics is None:
            return

        for cam_id, extrinsics in self.camimu_extrinsics.items():
            fig, ax = plt.subplots(2, 1, figsize=(8, 6))
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
            ax[0].plot(self.stamps, phi_list[:, 0], color="tab:blue", label="x")
            ax[0].plot(self.stamps, phi_list[:, 1], color="tab:red", label="y")
            ax[0].plot(self.stamps, phi_list[:, 2], color="tab:green", label="z")
            ax[0].set_ylabel("Calib Ori. (rad)")

            ax[1].plot(self.stamps, pos_list[:, 0], color="tab:blue", label="x")
            ax[1].plot(self.stamps, pos_list[:, 1], color="tab:red", label="y")
            ax[1].plot(self.stamps, pos_list[:, 2], color="tab:green", label="z")
            ax[1].set_ylabel("Calib Pos. (m)")
            fig.suptitle(f"Extrinsics for {cam_id}")
            fig.tight_layout()
            if save_dir is not None:
                fig.savefig(os.path.join(save_dir, f"extrinsics_{cam_id}.pdf"))

    def plot_velocities(self, save_dir: str = None):
        if self.imu_states is None:
            return

        stamps = np.array([state.stamp for state in self.imu_states])
        velocities = np.array([state.velocity for state in self.imu_states])
        fig, ax = plt.subplots(3, 1, figsize=(8, 6))
        ax[0].plot(stamps, velocities[:, 0], color="tab:blue")
        ax[1].plot(stamps, velocities[:, 1], color="tab:blue")
        ax[2].plot(stamps, velocities[:, 2], color="tab:blue")
        ax[0].set_ylabel(r"$v_x$ (m/s)")
        ax[1].set_ylabel(r"$v_y$ (m/s)")
        ax[2].set_ylabel(r"$v_z$ (m/s)")
        fig.suptitle(f"IMU Velocities")
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, f"velocities.pdf"), bbox_inches="tight", dpi=300)


class VinsCovarianceInfo:
    """Covariance information for the full VINS state."""

    # List of
    imu_std: typing.List[np.ndarray]

    imu_cov: typing.List[np.ndarray] = None
    extrinsic_std: typing.Dict[str, typing.List[np.ndarray]]
    intrinsic_std: typing.Dict[str, typing.List[np.ndarray]]
    offset_dt_std: typing.List[float]

    # pose_cov: typing.List[np.ndarray]
    att_cov: typing.List[np.ndarray]
    pos_cov: typing.List[np.ndarray]

    # Covariance timestamps
    stamps: np.ndarray

    def __init__(
        self,
        imu_std: typing.List[np.ndarray],
        imu_cov: typing.List[np.ndarray] = None,
        extrinsic_std=None,
        intrinsic_std=None,
        offset_dt_std=None,
        att_cov: typing.List[np.ndarray] = None,
        pos_cov: typing.List[np.ndarray] = None,
    ):
        self.imu_std = imu_std
        self.imu_cov = imu_cov
        self.extrinsic_std = extrinsic_std
        self.intrinsic_std = intrinsic_std
        self.offset_dt_std = offset_dt_std
        self.att_cov = att_cov
        self.pos_cov = pos_cov

    @classmethod
    def from_files(cls, std_file_path: str, pose_fpath: str):
        """Loads the covariance information from the specified file."""
        cov_dict = file_utils.load_std_from_ov_file(std_file_path)
        imu_std = cov_dict["imu_std"]
        extrinsic_std = cov_dict["sigma_extrinsics"]
        intrinsic_std = cov_dict["sigma_intrinsics"]
        offset_dt_std = cov_dict["sigma_camimu_dt"]

        pose_cov = file_utils.load_poses_and_covariances_from_ov_file(pose_fpath)[1]

        # Get the attitude and position covariances from the pose covariance
        att_cov = []
        pos_cov = []
        for cov in pose_cov:
            att_cov.append(cov[0:3, 0:3])
            pos_cov.append(cov[3:6, 3:6])

        return cls(
            imu_std=imu_std,
            extrinsic_std=extrinsic_std,
            intrinsic_std=intrinsic_std,
            offset_dt_std=offset_dt_std,
            att_cov=att_cov,
            pos_cov=pos_cov,
        )

    # @classmethod
    # def from_file(cls, std_file_path: str):
    #     """Loads the covariance information from the specified file."""
    #     cov_dict = file_utils.load_std_from_ov_file(std_file_path)
    #     return cls(
    #         imu_cov=cov_dict["imu"],
    #         extrinsic_std=cov_dict["sigma_extrinsics"],
    #         intrinsic_std=cov_dict["sigma_intrinsics"],
    #         offset_dt_std=cov_dict["sigma_camimu_dt"],
    #     )


class TimingData:
    """Container for timing data from a VINS run."""

    timestamps: np.ndarray
    frame_processing_times_ms: np.ndarray
    timing_df: pd.DataFrame

    def __init__(
        self,
        timestamps: np.ndarray,
        frame_processing_times_ms: np.ndarray,
        raw_timing_df: pd.DataFrame,
    ):
        self.timestamps = timestamps
        self.frame_processing_times_ms = frame_processing_times_ms
        self.timing_df = raw_timing_df
        self.frame_processing_stats = Statistics(frame_processing_times_ms)


class PoseData:
    """Container for pose data."""

    poses: typing.List[SE3State]
    pose_covariances: typing.List[np.ndarray]

    def __init__(
        self,
        poses: typing.List[SE3State] = None,
        pose_covariances: typing.List[np.ndarray] = None,
    ):
        self.poses = poses
        self.pose_covariances = pose_covariances


class VinsSimulationData:
    """A container to store the data from a VINS simulation run."""

    state_gt: VinsStateInfo
    state_est: VinsStateInfo
    cov_info: VinsCovarianceInfo

    landmarks_gt: np.ndarray = None

    def __init__(
        self,
        state_gt: VinsStateInfo,
        state_est: VinsStateInfo,
        cov_info: VinsCovarianceInfo,
        landmarks_gt: np.ndarray = None,
    ):
        if (len(state_gt.imu_states) != len(state_est.imu_states)) or (
            len(state_gt.imu_states) != len(cov_info.imu_std)
        ):
            raise ValueError(
                "Lengths of groundtruth, estimated, and covariance do not match!"
            )
        self.state_gt = state_gt
        self.state_est = state_est
        self.cov_info = cov_info
        self.landmarks_gt = landmarks_gt

    @classmethod
    def from_files(
        cls,
        state_est_path: str,
        state_gt_path: str,
        state_std_path: str,
        poses_est_path: str,
        landmarks_gt_path: str = None,
        do_shift_stamps: bool = False,
    ):
        """Loads in the data from the specified output files."""
        gt_states = VinsStateInfo.from_file(state_gt_path)
        est_states = VinsStateInfo.from_file(state_est_path)
        cov_info = VinsCovarianceInfo.from_files(state_std_path, poses_est_path)

        logging.info(f"Loaded {len(gt_states.imu_states)} groundtruth IMU states")
        logging.info(f"Loaded {len(est_states.imu_states)} estimated IMU states")
        logging.info(f"Loaded {len(cov_info.imu_std)} IMU covariance matrices")
        logging.info(f"Loaded {len(cov_info.pos_cov)} pose covariance matrices")

        # Associate the IMU states and covariances
        imu_gt, imu_est, matching_indices = associate_states(
            gt_states.imu_states,
            est_states.imu_states,
            return_matching_idx=True,
        )
        matching_idx_short = matching_indices["matching_indices_short"]
        matching_idx_long = matching_indices["matching_indices_long"]

        if len(cov_info.imu_std) > len(gt_states.imu_states):
            matching_idx = matching_idx_long
        else:
            matching_idx = matching_idx_short

        # Get the covariance matches
        imu_std = [cov_info.imu_std[i] for i in matching_idx]
        att_cov = [cov_info.att_cov[i] for i in matching_idx]
        pos_cov = [cov_info.pos_cov[i] for i in matching_idx]
        cov_info.imu_std = imu_std
        cov_info.att_cov = att_cov
        cov_info.pos_cov = pos_cov
        gt_states.imu_states = imu_gt
        est_states.imu_states = imu_est

        if do_shift_stamps:
            shift_stamps(est_states.imu_states)
            shift_stamps(gt_states.imu_states)
            cov_info.stamps = cov_info.stamps - cov_info.stamps[0]
            est_stamps = np.array([state.stamp for state in est_states.imu_states])
            gt_stamps = np.array([state.stamp for state in gt_states.imu_states])
            cov_stamps = cov_info.stamps
            assert np.allclose(est_stamps, gt_stamps)
            assert np.allclose(est_stamps, cov_stamps)

        landmarks = None
        if landmarks_gt_path is not None:
            landmarks = np.loadtxt(landmarks_gt_path, delimiter=" ")
            logging.info("Loaded groundtruth landmarks from: %s", landmarks_gt_path)

        return cls(
            state_gt=gt_states,
            state_est=est_states,
            cov_info=cov_info,
            landmarks_gt=landmarks,
        )

    def plot_trajectories(self, save_dir: str = None):
        traj_data = {"Groundtruth": self.imu_gt, "Estimate": self.imu_est}
        fig, ax = plot_utils.plot_trajectories(traj_data)
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "trajectories.pdf"))
