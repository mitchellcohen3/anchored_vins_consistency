"""Error and consistency metric computation for VINS evaluation."""

import typing

import numpy as np

from navlie.lib.imu import IMUState
from navlie.lib.states import SE3State
from navlie.types import StateWithCovariance, State
from navlie.utils import GaussianResult, GaussianResultList

from pymlg import SO3, SE3, SE23

from pyvins.containers import VinsStateInfo


def minus_SO3(Y: np.ndarray, X: np.ndarray, lie_direction: str) -> np.ndarray:
    """Computes the minus operation for SO3, given a lie direction."""
    if lie_direction == "left":
        return SO3.Log(Y @ X.T)
    elif lie_direction == "right":
        return SO3.Log(X.T @ Y)


def minus_SE3(Y: np.ndarray, X: np.ndarray, lie_direction: str) -> np.ndarray:
    """Computes the minus operation for SE3, given a lie direction."""
    if lie_direction == "left":
        return SE3.Log(Y @ SE3.inverse(X))
    elif lie_direction == "right":
        return SE3.Log(SE3.inverse(X) @ Y)


def minus_SE23(Y: np.ndarray, X: np.ndarray, lie_direction: str) -> np.ndarray:
    """Computes the minus operation for SE23, given a lie direction."""
    if lie_direction == "left":
        return SE23.Log(Y @ SE23.inverse(X))
    elif lie_direction == "right":
        return SE23.Log(SE23.inverse(X) @ Y)


def compute_rmses_over_time(errors: np.ndarray) -> np.ndarray:
    """Computes a scalar RMSE value at each timestep.

    errors is a (N, dof) array of errors at each timestep,
    where N is the number of timestamps and dof is the dimension of the error vector.

    This computes
        RMSE(t) = sqrt( ||error(t)||^2 / dof )
    """
    dof = errors.shape[1]
    return np.sqrt(((np.linalg.norm(errors, axis=1)) ** 2) / dof)


def compute_ate(errors: np.ndarray) -> float:
    """Computes a scalar RMSE value over the entire trajectory.

    This computes ATE = sqrt(mean(||error(t)||^2)) over all timesteps t
    """
    return np.sqrt(np.mean(np.linalg.norm(errors, axis=1) ** 2))


def compute_rmses_over_time_dataset(errors: np.ndarray) -> float:
    """Computes RMSE at each timestep of the trajectory, for a given dataset with N runs.

    errors is a (N, T, dof) array of errors at each timestep, where N is the number of runs,
    T is the number of timestamps, and dof is the dimension of the error vector.
    """
    return np.sqrt(np.mean(np.sum(errors**2, axis=2), axis=0))


def compute_pose_errors(
    gt_states: typing.List[SE3State],
    est_states: typing.List[SE3State],
    lie_direction: str,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Computes the pose errors between two lists of SE3State objects."""

    if len(gt_states) != len(est_states):
        raise ValueError("Lengths of groundtruth and estimated states do not match!")

    phi_errors = []
    pos_errors = []
    for gt_state, est_state in zip(gt_states, est_states):
        pos_error = gt_state.position - est_state.position
        phi_error = minus_SO3(
            gt_state.attitude,
            est_state.attitude,
            lie_direction,
        )
        phi_error = phi_error.ravel()
        pos_error = pos_error.ravel()
        phi_errors.append(phi_error)
        pos_errors.append(pos_error)

    phi_errors = np.array(phi_errors)
    pos_errors = np.array(pos_errors)
    return phi_errors, pos_errors


def compute_gaussian_result_list(
    states_gt: typing.List[State],
    states_est: typing.List[State],
    cov_mats: typing.List[np.ndarray],
) -> GaussianResultList:
    """Computes a GaussianResultList from the groundtruth states, estimated states, and covariance matrices.

    Should work with any state types.
    """

    if len(states_gt) != len(states_est) or len(states_gt) != len(cov_mats):
        raise ValueError("Lengths of states and covariances do not match!")

    # Create GaussianResultList
    gaussian_result_list: typing.List[GaussianResult] = []
    for i, state_est in enumerate(states_est):
        state_with_cov = StateWithCovariance(state_est, cov_mats[i])
        state_gt = states_gt[i]
        gaussian_result = GaussianResult(state_with_cov, state_gt)
        gaussian_result_list.append(gaussian_result)

    return GaussianResultList(gaussian_result_list)


class PoseErrorMetrics:
    """Container for pose error metrics."""

    # [N x 3] arrays, containing the attitude and position errors at each timestep
    attitude_errors: np.ndarray
    position_errors: np.ndarray

    # Computed RMSEs from the errors
    # These are (N, ) arrays containing the RMSE at each timestep
    orientation_rmses_over_time: np.ndarray
    position_rmses_over_time: np.ndarray

    # The absolute trajectory errors over the entire trajectory
    # These are scalar quantities representing the RMSE over the entire trajectory
    attitude_ate: float
    position_ate: float

    def __init__(
        self,
        gt_poses: typing.List[SE3State],
        est_poses: typing.List[SE3State],
        lie_direction: str = "left",
    ):
        """Computes the errors between the groundtruth and the estimate."""
        if len(gt_poses) != len(est_poses):
            raise ValueError("Lengths of groundtruth and estimate do not match!")

        # Compute actual errors on physical quantities (not Lie algebra)
        att_errors, pos_errors = compute_pose_errors(
            gt_poses,
            est_poses,
            lie_direction,
        )

        # Compute RMSEs for each timestamp
        self.attitude_rmses_over_time = compute_rmses_over_time(att_errors)
        self.position_rmses_over_time = compute_rmses_over_time(pos_errors)

        # Compute ATEs
        self.attitude_ate = compute_ate(att_errors)
        self.position_ate = compute_ate(pos_errors)

        # Store the actual errors
        self.attitude_errors = att_errors
        self.position_errors = pos_errors


class PoseConsistencyMetrics:
    """Container for consistency metrics for a VINS run."""

    # Attitude NEES throughout the trajectory
    # (N, ) array
    att_nees: np.ndarray
    pos_nees: np.ndarray

    # (N, 6) array of pose errors in the Lie algebra
    delta_xi: np.ndarray
    # (N, 6) array of three-sigma values
    three_sigma: np.ndarray
    # (N, ) array of timestamps
    stamps: np.ndarray

    def __init__(
        self,
        att_nees: np.ndarray,
        pos_nees: np.ndarray,
        delta_xi: np.ndarray,
        three_sigma: np.ndarray,
        stamps: np.ndarray,
    ):
        self.att_nees = att_nees
        self.pos_nees = pos_nees
        self.delta_xi = delta_xi
        self.three_sigma = three_sigma
        self.stamps = stamps

    @classmethod 
    def compute_from_poses(cls,
        gt_poses: typing.List[SE3State],
        est_poses: typing.List[SE3State],
        pose_covariances: typing.List[np.ndarray],
        state_representation: str = "decoupled",
        lie_direction: str = "left",
    ):
        """Computes the consistency metrics for the VINS poses."""
        if len(gt_poses) != len(est_poses) or len(gt_poses) != len(pose_covariances):
            raise ValueError(
                "Lengths of groundtruth, estimate, and covariance do not match!"
            )

        N = len(gt_poses)
        att_nees = np.zeros(N)
        pos_nees = np.zeros(N)
        delta_xi_pose = []
        three_sigma_pose = []
        stamps = []

        for i, (gt_pose, est_pose, cov) in enumerate(
            zip(
                gt_poses,
                est_poses,
                pose_covariances,
            )
        ):
            error = np.zeros(6)

            if state_representation == "SE3" or state_representation == "SE23":
                error_se3 = minus_SE3(
                    gt_pose.value,
                    est_pose.value,
                    lie_direction,
                )
                error[0:3] = error_se3[0:3].ravel()
                error[3:6] = error_se3[3:6].ravel()
            elif state_representation == "decoupled":
                att_error = minus_SO3(
                    gt_pose.attitude,
                    est_pose.attitude,
                    lie_direction,
                )
                error[0:3] = att_error.ravel()
                error[3:6] = (gt_pose.position - est_pose.position).ravel()
            else:
                raise ValueError(
                    f"Unknown state representation: {state_representation}"
                )

            delta_xi_pose.append(error)

            # Compute the NEES values
            att_nees[i] = error[0:3] @ np.linalg.solve(cov[0:3, 0:3], error[0:3])
            pos_nees[i] = error[3:6] @ np.linalg.solve(cov[3:6, 3:6], error[3:6])

            # Compute the 3-sigma bounds
            three_sigma_pose.append(3.0 * np.sqrt(np.diag(cov)))
            stamps.append(gt_pose.stamp)

        delta_xi = np.array(delta_xi_pose)
        three_sigma = np.array(three_sigma_pose)
        stamps = np.array(stamps)
        return cls(att_nees, pos_nees, delta_xi, three_sigma, stamps)


class VinsConsistencyMetricsSim:
    """Container for consistency metrics in a VINS simulation run.

    In a simulation, we have access to the full groundtruth state,
    and therefore we can plot the consistency of the entire state
    """

    # (N, ) array that store the NEES through the trajectory
    att_nees: np.ndarray
    pos_nees: np.ndarray

    # Errors in the Lie algebra
    # Needed to plot the 3-sigma bounds
    # (N, 15) array where N is the number of states
    delta_xi: np.ndarray

    # Precompute the 3-sigma bounds
    # (N, 15) array where N is the number of states
    three_sigma: np.ndarray

    # (N, ) array of timestamps
    stamps: np.ndarray

    def __init__(
        self,
        imu_est: typing.List[IMUState],
        imu_gt: typing.List[IMUState],
        imu_std: typing.List[np.ndarray],
        att_cov: typing.List[np.ndarray],
        pos_cov: typing.List[np.ndarray],
        state_representation: str = "SE23",
        lie_direction: str = "left",
    ):
        """Computes the consistency metrics for the full IMU state."""
        # Validate input state representation and lie direction
        valid_state_representations = ["SE23", "decoupled"]
        valid_lie_directions = ["left", "right"]
        if state_representation not in valid_state_representations:
            raise ValueError(f"Invalid state representation!")
        if lie_direction not in valid_lie_directions:
            raise ValueError(f"Invalid lie direction!")

        # Ensure that all the input lists are the same length
        n = len(imu_gt)
        if not all(len(lst) == n for lst in [imu_est, imu_std, att_cov, pos_cov]):
            raise ValueError("Input lists must all be the same length!")

        # Compute the attitude and position NEES manually
        att_nees: np.ndarray = np.zeros(len(imu_gt))
        pos_nees: np.ndarray = np.zeros(len(imu_gt))

        # Compute the errors in the Lie algebra
        delta_xi = []
        three_sigma = []
        stamps = []

        idx = 0
        for i in range(len(imu_gt)):
            cur_error = np.zeros(15)
            cur_3_sigma = np.zeros(15)
            x_gt = imu_gt[i]
            x_est = imu_est[i]
            cov_att = att_cov[i]
            cov_pos = pos_cov[i]

            # Compute the error in the Lie algebra differently depending on
            # the state representation
            if state_representation == "SE23":
                X_nav_gt = SE23.from_components(
                    x_gt.attitude, x_gt.velocity, x_gt.position
                )
                X_nav_est = SE23.from_components(
                    x_est.attitude, x_est.velocity, x_est.position
                )
                error_nav = minus_SE23(X_nav_gt, X_nav_est, lie_direction)
                cur_error[0:9] = error_nav.ravel()
            elif state_representation == "decoupled":
                att_error = minus_SO3(x_gt.attitude, x_est.attitude, lie_direction)
                cur_error[0:3] = att_error.ravel()
                cur_error[3:6] = x_gt.velocity - x_est.velocity
                cur_error[6:9] = x_gt.position - x_est.position
            else:
                raise ValueError(
                    f"Unknown state representation: {state_representation}"
                )

            # Fill in the bias errors
            cur_error[9:12] = x_gt.bias_gyro - x_est.bias_gyro
            cur_error[12:15] = x_gt.bias_accel - x_est.bias_accel

            delta_xi.append(cur_error)

            cur_3_sigma[0:3] = 3.0 * imu_std[i][0:3]
            cur_3_sigma[3:6] = 3.0 * imu_std[i][6:9]
            cur_3_sigma[6:9] = 3.0 * imu_std[i][3:6]
            cur_3_sigma[9:12] = 3.0 * imu_std[i][9:12]
            cur_3_sigma[12:15] = 3.0 * imu_std[i][12:15]
            three_sigma.append(cur_3_sigma)

            att_nees[idx] = cur_error[0:3] @ np.linalg.solve(cov_att, cur_error[0:3])
            pos_nees[idx] = cur_error[6:9] @ np.linalg.solve(cov_pos, cur_error[6:9])

            # Compute the 3-sigma bounds
            stamps.append(x_gt.stamp)
            idx += 1

        self.att_nees = att_nees
        self.pos_nees = pos_nees
        self.delta_xi = np.array(delta_xi)
        self.three_sigma = np.array(three_sigma)
        self.stamps = np.array(stamps)


class CalibrationErrors:
    """Container for calibration errors."""

    # cam_id -> errors_array
    # All of the errors_array are (N, dof) arrays,
    # where N is the number of timestamps.
    extrinsic_errors: typing.Dict[str, np.ndarray]
    intrinsic_errors: typing.Dict[str, np.ndarray]
    offset_dt_errors: np.ndarray
    stamps: np.ndarray

    intrinsic_3_sigma: typing.Dict[str, np.ndarray] = None
    extrinsics_3_sigma: typing.Dict[str, np.ndarray] = None
    time_offset_3_sigma: np.ndarray = None

    def __init__(
        self,
        gt_state: VinsStateInfo,
        est_state: VinsStateInfo,
        intrinsics_3_sigma_dict: typing.Dict[str, np.ndarray] = None,
        extrinsics_3_sigma_dict: typing.Dict[str, np.ndarray] = None,
        time_offset_3_sigma: np.ndarray = None,
    ):
        """Computes the calibration errors between groundtruth and estimated calib parameters."""
        self.extrinsic_errors = {}
        self.intrinsic_errors = {}

        # Get the extrinsics and intrinsics
        gt_extrinsics_dict = gt_state.camimu_extrinsics
        est_extrinsics_dict = est_state.camimu_extrinsics
        gt_intrinsics_dict = gt_state.cam_intrinsics
        est_intrinsics_dict = est_state.cam_intrinsics

        for cam_id in gt_extrinsics_dict.keys():
            if cam_id in est_extrinsics_dict:
                gt_extrinsics = gt_extrinsics_dict[cam_id][0]
                est_extrinsics = est_extrinsics_dict[cam_id]

                # Compute errors
                # All the groundtruth calibration parameters are assumed to be time-invariant.
                extrinsic_errors = []
                for i in range(len(est_extrinsics)):
                    T_bc_gt: np.ndarray = gt_extrinsics
                    T_bc_est: np.ndarray = est_extrinsics[i]

                    C_bc_gt = T_bc_gt[0:3, 0:3]
                    C_bc_est = T_bc_est[0:3, 0:3]
                    r_bc_gt = T_bc_gt[0:3, 3]
                    r_bc_est = T_bc_est[0:3, 3]

                    # Compute the attitude and position errors
                    att_error = minus_SO3(C_bc_gt, C_bc_est, "right")
                    pos_error = r_bc_gt - r_bc_est
                    extrinsic_error = np.hstack((att_error.ravel(), pos_error.ravel()))
                    extrinsic_errors.append(extrinsic_error)

                self.extrinsic_errors[cam_id] = np.array(extrinsic_errors)

            # Compute intrinsic errors
            if cam_id in est_intrinsics_dict:
                gt_intrinsics = gt_intrinsics_dict[cam_id][0]
                est_intrinsics = est_intrinsics_dict[cam_id]

                intrinsic_errors = []
                for i in range(len(est_intrinsics)):
                    intrinsic_error = est_intrinsics[i] - gt_intrinsics
                    intrinsic_errors.append(intrinsic_error)

                self.intrinsic_errors[cam_id] = np.array(intrinsic_errors)

        # Time offset errors
        self.offset_dt_errors = (
            np.array(est_state.camimu_time_offset) - gt_state.camimu_time_offset[0]
        )

        # Store timestamps
        self.stamps = est_state.stamps
        self.intrinsic_3_sigma = intrinsics_3_sigma_dict
        self.extrinsics_3_sigma = extrinsics_3_sigma_dict
        self.time_offset_3_sigma = time_offset_3_sigma
