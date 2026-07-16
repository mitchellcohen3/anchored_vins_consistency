import yaml
import os
from datetime import datetime
import typing
import numpy as np

from navlie.lib.imu import IMUState
from navlie.lib.states import SE3State
from pymlg import SO3, SE3, SE23


def load_config(config_file: str):
    """Loads the configuration file."""
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def print_config(config: dict):
    """Prints the configuration dictionary."""
    for key, value in config.items():
        print(f"{key}: {value}")


def create_save_dir(basedir: str):
    # Create directory tosave the results
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(basedir, "results_" + current_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def parse_yaml(config_path: str) -> typing.Dict[typing.Any, typing.Any]:
    """Parses a yaml file generated for C++.
    Here, we need to remove the header since the header is a YAML1.0
    specification.
    """
    remove_yaml_header(config_path)
    if config_path is None:
        return
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def parse_multi_yaml(config_path: str) -> typing.Dict[typing.Any, typing.Any]:
    if config_path is None:
        return
    remove_yaml_header(config_path)
    with open(config_path, "r") as f:
        config = list(yaml.safe_load_all(f))

    # Merge all list entries
    config = {k: v for d in config for k, v in d.items()}
    return config


def remove_yaml_header(fpath: str):
    """Removes the first line of a yaml file and saves it back to the same file.

    Only removes the header if the first line is a comment.
    """

    with open(fpath, "r") as f:
        content = f.read()

    # Split after the first newline
    split_content = content.split("\n", 1)

    # Check if the first line is a comment that needs to be removed
    if split_content[0].startswith("%"):
        with open(fpath, "w") as f:
            f.write(split_content[1])


def get_files_with_extension(folder_path: str, extension: str) -> typing.List[str]:
    """Returns a list of files in the given folder with the specified extension."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist.")

    files = [f for f in os.listdir(folder_path) if f.endswith(extension)]
    file_paths = [os.path.join(folder_path, f) for f in files]
    return file_paths


def get_all_subdirectory_paths(root_path: str) -> typing.List[str]:
    """Returns a list of all directories in the given root path."""
    folders = [
        name
        for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]

    # Get the full paths of the directories
    full_paths = [os.path.join(root_path, folder) for folder in folders]
    return full_paths


def get_all_subdirectory_names(root_path: str) -> typing.List[str]:
    """Returns a list of all directory names in the given root path."""
    folders = [
        name
        for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]
    return folders


def create_new_folder(
    folder_path: str, overwrite: bool = False, add_timestamp: bool = True
):
    """Creates a new folder at the specified path."""
    # Add current timestamp to the folder name if specified
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_path = f"{folder_path}_{timestamp}"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return folder_path

    if overwrite:
        # If the folder exists and overwrite is True, remove it
        import shutil

        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        return folder_path

    return folder_path


def load_poses_from_tum(file: str, delimiter=",") -> typing.List[SE3State]:
    """Loads the IMU states from tum format into SE3 state containers.

    Note, TUM format is in the form
        t, pos, qx, qy, qz, qw.
    """

    txt_file = np.loadtxt(file, delimiter=delimiter, usecols=range(0, 8))

    if len(txt_file.shape) == 1:
        txt_file = txt_file.reshape((1, -1))

    pose_list: typing.List[SE3State] = []
    for i in range(txt_file.shape[0]):
        data_row = txt_file[i, :]
        stamp = data_row[0]
        position = data_row[1:4]
        qx = data_row[4]
        qy = data_row[5]
        qz = data_row[6]
        qw = data_row[7]
        quat = np.array([qx, qy, qz, qw])
        C = SO3.from_quat(quat, order="xyzw")
        cur_pose = SE3State(value=SE3.from_components(C, position), stamp=stamp)
        pose_list.append(cur_pose)
    return pose_list


def load_imu_states_from_asl(file: str) -> typing.List[IMUState]:
    """Loads IMU states from a text file in ASL format.

    Note, ASL format is in the form
    t, pos, qw, qx, qy, qz, vel, bg, ba.

    Optimized version that extracts data in bulk using numpy slicing.
    """
    results = np.loadtxt(file, delimiter=",")

    # Ensure that the dimensions of the input file match what is expected
    if results.shape[1] != 17:
        raise ValueError(f"Expected 17 columns in ASL file, but got {results.shape[1]}")

    n_rows = results.shape[0]

    # Extract all columns at once (vectorized)
    stamps = results[:, 0]
    positions = results[:, 1:4]
    quats = results[:, 4:8]  # qw, qx, qy, qz
    velocities = results[:, 8:11]
    bias_gyros = results[:, 11:14]
    bias_accels = results[:, 14:17]

    # Pre-allocate list
    imu_states: typing.List[IMUState] = [None] * n_rows

    # Build IMU states - loop needed for object creation
    for i in range(n_rows):
        C = SO3.from_quat(quats[i], order="wxyz")
        nav_state = SE23.from_components(C, velocities[i], positions[i])
        imu_states[i] = IMUState(
            nav_state,
            bias_gyro=bias_gyros[i],
            bias_accel=bias_accels[i],
            stamp=stamps[i],
        )

    return imu_states


def load_poses_from_file(
    file: str,
) -> typing.Tuple[typing.List[SE3State], typing.List[np.ndarray]]:
    """Loads poses and possibily covariances from a file."""

    txt_file = np.loadtxt(file, delimiter=" ")

    if len(txt_file.shape) == 1:
        txt_file = txt_file.reshape((1, -1))

    pose_list: typing.List[SE3State] = []
    cov_list: typing.List[np.ndarray] = []

    for i in range(txt_file.shape[0]):
        data_row = txt_file[i, :]
        stamp = data_row[0]
        position = data_row[1:4]
        qx = data_row[4]
        qy = data_row[5]
        qz = data_row[6]
        qw = data_row[7]
        quat = np.array([qx, qy, qz, qw])
        C = SO3.from_quat(quat, order="xyzw")
        cur_pose = SE3State(value=SE3.from_components(C, position), stamp=stamp)
        pose_list.append(cur_pose)

        if data_row.shape[0] > 8:
            # Assemble the covariance matrix
            cov_mat = np.zeros((6, 6))
            att_cov = np.zeros((3, 3))
            upper_indices = np.triu_indices(3)
            att_cov[upper_indices] = data_row[8:14]
            att_cov = att_cov + att_cov.T - np.diag(np.diag(att_cov))

            pos_cov = np.zeros((3, 3))
            pos_cov[upper_indices] = data_row[14:20]
            pos_cov = pos_cov + pos_cov.T - np.diag(np.diag(pos_cov))

            cov_mat[0:3, 0:3] = att_cov
            cov_mat[3:6, 3:6] = pos_cov
            cov_list.append(cov_mat)

    return pose_list, cov_list


def load_poses_and_covariances_from_ov_file(
    file: str,
) -> typing.Tuple[typing.List[SE3State], typing.List[np.ndarray]]:
    """Loads poses and covariances from an OpenVins format file.

    The OpenVins format is:
        t, px, py, pz, qx, qy, qz, qw, Pr11 Pr12 Pr13 Pr22 Pr23 Pr33 Pt11 Pt12 Pt13 Pt13 Pt22 Pt23 Pt33
    """
    txt_file = np.loadtxt(file, delimiter=" ")

    if len(txt_file.shape) == 1:
        txt_file = txt_file.reshape((1, -1))

    # Ensure that the dimensions of the input file match what is expected
    if txt_file.shape[1] != 20:
        raise ValueError(
            f"Expected 20 columns in the OpenVins file, but got {txt_file.shape[1]}"
        )

    pose_list: typing.List[SE3State] = []
    cov_list: typing.List[np.ndarray] = []
    for i in range(txt_file.shape[0]):
        data_row = txt_file[i, :]
        stamp = data_row[0]
        position = data_row[1:4]
        qx = data_row[4]
        qy = data_row[5]
        qz = data_row[6]
        qw = data_row[7]
        quat = np.array([qx, qy, qz, qw])
        C = SO3.from_quat(quat, order="xyzw")
        cur_pose = SE3State(value=SE3.from_components(C, position), stamp=stamp)
        pose_list.append(cur_pose)

        # # Assemble the covariance matrix
        cov_mat = np.zeros((6, 6))
        att_cov = np.zeros((3, 3))
        upper_indices = np.triu_indices(3)
        att_cov[upper_indices] = data_row[8:14]
        att_cov = att_cov + att_cov.T - np.diag(np.diag(att_cov))

        pos_cov = np.zeros((3, 3))
        pos_cov[upper_indices] = data_row[14:20]
        pos_cov = pos_cov + pos_cov.T - np.diag(np.diag(pos_cov))

        cov_mat[0:3, 0:3] = att_cov
        cov_mat[3:6, 3:6] = pos_cov
        cov_list.append(cov_mat)

    return pose_list, cov_list


def load_ov_state_est_file(file: str) -> typing.Dict[str, typing.Any]:
    """Loads the IMU states from an OpenVins state estimate file.

    Optimized version that extracts data in bulk using numpy slicing
    and minimizes per-row Python operations.
    """
    data = np.genfromtxt(file, delimiter=" ")
    n_rows = data.shape[0]

    # Extract all columns at once (vectorized)
    stamps = data[:, 0]
    quats = data[:, 1:5]
    positions = data[:, 5:8]
    velocities = data[:, 8:11]
    bias_gyros = data[:, 11:14]
    bias_accels = data[:, 14:17]

    # Pre-allocate list for IMU states
    imu_states: typing.List[IMUState] = [None] * n_rows

    # Build IMU states - loop is unavoidable due to object creation,
    # but we've minimized indexing operations
    for i in range(n_rows):
        C = SO3.from_quat(quats[i], order="xyzw")
        nav_state = SE23.from_components(C, velocities[i], positions[i])
        imu_states[i] = IMUState(
            nav_state,
            bias_gyro=bias_gyros[i],
            bias_accel=bias_accels[i],
            stamp=stamps[i],
        )

    # Handle calibration parameters if present
    extrinsics_dict = None
    intrinsics_dict = None
    offset_dt_list = None

    if data.shape[1] > 17:
        num_cams = int(data[0, 18])

        # Extract time offsets (vectorized)
        offset_dt_list = data[:, 17].tolist()

        # Pre-allocate camera dictionaries with numpy arrays
        extrinsics_dict = {}
        intrinsics_dict = {}

        for cam_idx in range(num_cams):
            cam_start_idx = 19 + (cam_idx * 15)

            # Extract all intrinsics for this camera at once (N, 8)
            intrinsics_all = data[:, cam_start_idx : cam_start_idx + 8]
            intrinsics_dict[f"cam_{cam_idx}"] = [
                intrinsics_all[i] for i in range(n_rows)
            ]

            # Extract all extrinsic quaternions and translations at once
            q_bc_all = data[:, cam_start_idx + 8 : cam_start_idx + 12]  # (N, 4)
            t_bc_all = data[:, cam_start_idx + 12 : cam_start_idx + 15]  # (N, 3)

            # Build extrinsics list - loop needed for SE3 object creation
            extrinsics_list = [None] * n_rows
            for i in range(n_rows):
                C_bc = SO3.from_quat(q_bc_all[i], order="xyzw")
                extrinsics_list[i] = SE3.from_components(C_bc, t_bc_all[i])
            extrinsics_dict[f"cam_{cam_idx}"] = extrinsics_list

    output_dict = {
        "imu_states": imu_states,
        "extrinsics": extrinsics_dict,
        "intrinsics": intrinsics_dict,
        "offset_dt": offset_dt_list,
        "stamps": stamps,  # Already a numpy array, no need to convert
    }
    return output_dict


def load_std_from_ov_file(file: str) -> typing.List[np.ndarray]:
    """Loads the covariances from an OpenVins state std file.

    Optimized version that builds covariance matrices using vectorized
    operations where possible.
    """
    data = np.genfromtxt(file, delimiter=" ")
    n_rows = data.shape[0]

    # Extract IMU state standard deviations
    # Column layout: [stamp, att(3), pos(3), vel(3), bg(3), ba(3), ...]
    sigma_imu_state = data[:, 1:16]

    imu_state_std: typing.List[np.ndarray] = []
    for i in range(n_rows):
        imu_state_std.append(sigma_imu_state[i, :])

    # Extract standard deviation columns (vectorized)
    # Column layout: [stamp, att(3), pos(3), vel(3), bg(3), ba(3), ...]
    # sigma_att = data[:, 1:4]  # (N, 3)
    # sigma_pos = data[:, 4:7]  # (N, 3)
    # sigma_vel = data[:, 7:10]  # (N, 3)
    # sigma_bg = data[:, 10:13]  # (N, 3)
    # sigma_ba = data[:, 13:16]  # (N, 3)

    # # Build all covariance matrices at once
    # # Pre-allocate 3D array: (N, 15, 15)
    # imu_covs = np.zeros((n_rows, 15, 15))

    # # Fill diagonal blocks using vectorized operations
    # # Attitude: indices 0:3
    # imu_covs[:, 0, 0] = sigma_att[:, 0] ** 2
    # imu_covs[:, 1, 1] = sigma_att[:, 1] ** 2
    # imu_covs[:, 2, 2] = sigma_att[:, 2] ** 2

    # # Velocity: indices 3:6
    # imu_covs[:, 3, 3] = sigma_vel[:, 0] ** 2
    # imu_covs[:, 4, 4] = sigma_vel[:, 1] ** 2
    # imu_covs[:, 5, 5] = sigma_vel[:, 2] ** 2

    # # Position: indices 6:9
    # imu_covs[:, 6, 6] = sigma_pos[:, 0] ** 2
    # imu_covs[:, 7, 7] = sigma_pos[:, 1] ** 2
    # imu_covs[:, 8, 8] = sigma_pos[:, 2] ** 2

    # # Gyro bias: indices 9:12
    # imu_covs[:, 9, 9] = sigma_bg[:, 0] ** 2
    # imu_covs[:, 10, 10] = sigma_bg[:, 1] ** 2
    # imu_covs[:, 11, 11] = sigma_bg[:, 2] ** 2

    # # Accel bias: indices 12:15
    # imu_covs[:, 12, 12] = sigma_ba[:, 0] ** 2
    # imu_covs[:, 13, 13] = sigma_ba[:, 1] ** 2
    # imu_covs[:, 14, 14] = sigma_ba[:, 2] ** 2

    # Convert to list of 2D arrays (required by downstream code)
    # imu_cov_list = [imu_covs[i] for i in range(n_rows)]

    # Handle calibration parameters if present
    extrinsics_dict = None
    intrinsics_dict = None
    offset_dt_list = None

    if data.shape[1] > 16:
        num_cams = int(data[0, 17])

        # Extract time offset sigmas (vectorized)
        offset_dt_list = data[:, 16].tolist()

        # Extract camera calibration sigmas
        extrinsics_dict = {}
        intrinsics_dict = {}

        for cam_idx in range(num_cams):
            cam_start_idx = 18 + (cam_idx * 14)

            # Extract all at once using numpy slicing
            intrinsics_dict[f"cam_{cam_idx}"] = data[
                :, cam_start_idx : cam_start_idx + 8
            ].copy()
            extrinsics_dict[f"cam_{cam_idx}"] = data[
                :, cam_start_idx + 8 : cam_start_idx + 14
            ].copy()

    cov_dict = {
        "imu_std": imu_state_std,
        "sigma_camimu_dt": offset_dt_list,
        "sigma_extrinsics": extrinsics_dict,
        "sigma_intrinsics": intrinsics_dict,
    }

    return cov_dict


def write_poses_to_tum(
    pose_state_list: typing.List[IMUState],
    outfile: str,
    delimiter=" ",
):
    """Writes a list of pose states to a text file in TUM format.

    Note, tum format is in the form
    t, x, y, z, qx, qy, qz, qw
    """

    n_timesteps = len(pose_state_list)
    output_mat = np.ndarray((n_timesteps, 8))

    for i in range(n_timesteps):
        cur_state = pose_state_list[i]

        quat = SO3.to_quat(cur_state.attitude, order="xyzw").ravel()
        output_mat[i, 0] = cur_state.stamp
        output_mat[i, 1:4] = cur_state.position
        output_mat[i, 4:8] = quat

    np.savetxt(outfile, output_mat, delimiter=delimiter)


def load_covariances_from_file(file: str, dof: int) -> typing.List[np.ndarray]:
    """Loads the full covariances from a file, where each row of the file is
    assumed to have a timestmap and the covariance entries in column major order."""
    cov_mats_file_np = np.loadtxt(file, delimiter=",")
    cov_mats: typing.List[np.ndarray] = []
    stamps: typing.List[float] = []
    for row in cov_mats_file_np:
        cov_mats.append(row[1:].reshape((dof, dof), order="F"))
        stamps.append(row[0])

    return cov_mats, stamps


from dataclasses import dataclass


@dataclass
class OvEvalResult:
    # Summary metrics
    nees_ori_mean: float
    nees_pos_mean: float
    error_ori_rmse: float
    error_pos_rmse: float

    # Time series data
    nees_timestamps: np.ndarray
    nees_ori_values: np.ndarray
    nees_pos_values: np.ndarray
    error_timestamps: np.ndarray
    error_ori_values: np.ndarray
    error_pos_values: np.ndarray


def process_ov_eval_output(output_file: str) -> OvEvalResult:
    """
    Docstring for process_ov_eval_output

    :param output_file: Description
    :type output_file: str
    """
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            lines = f.readlines()

            num_nees, num_errors = lines[0].split()
            num_nees, num_errors = int(num_nees), int(num_errors)

            print(f"Number of NEES entries: {num_nees}")
            # Line 2: Average NEES
            nees_ori_mean, nees_pos_mean = lines[1].split()
            nees_ori_mean, nees_pos_mean = float(nees_ori_mean), float(nees_pos_mean)

            # Line 3: Average errors
            error_ori_rmse, error_pos_rmse = lines[2].split()
            error_ori_rmse, error_pos_rmse = float(error_ori_rmse), float(
                error_pos_rmse
            )

            # Lines 4 to 4 + num_nees: Per-timestep NEES
            nees_timestamps = []
            nees_ori_values = []
            nees_pos_values = []
            for i in range(3, 3 + num_nees):
                timestamp, nees_ori, nees_pos = lines[i].split()
                nees_timestamps.append(float(timestamp))
                nees_ori_values.append(float(nees_ori))
                nees_pos_values.append(float(nees_pos))

            nees_ori_values = np.array(nees_ori_values)
            nees_pos_values = np.array(nees_pos_values)
            nees_timestamps = np.array(nees_timestamps)

            # Lines 4 + num_nees to 4 + num_nees + num_errors: Per-timestep errors
            error_timestamps = []
            error_ori_values = []
            error_pos_values = []
            for i in range(3 + num_nees, 3 + num_nees + num_errors):
                timestamp, error_ori, error_pos = lines[i].split()
                error_timestamps.append(float(timestamp))
                error_ori_values.append(float(error_ori))
                error_pos_values.append(float(error_pos))

            error_ori_values = np.array(error_ori_values)
            error_pos_values = np.array(error_pos_values)
            error_timestamps = np.array(error_timestamps)

            return OvEvalResult(
                nees_ori_mean=nees_ori_mean,
                nees_pos_mean=nees_pos_mean,
                error_ori_rmse=error_ori_rmse,
                error_pos_rmse=error_pos_rmse,
                nees_timestamps=nees_timestamps,
                nees_ori_values=nees_ori_values,
                nees_pos_values=nees_pos_values,
                error_timestamps=error_timestamps,
                error_ori_values=error_ori_values,
                error_pos_values=error_pos_values,
            )
