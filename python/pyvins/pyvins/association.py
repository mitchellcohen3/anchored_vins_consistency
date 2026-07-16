"""State association and alignment utilities for VINS evaluation."""

import copy
import logging
import typing

import numpy as np

from evo.core.sync import matching_time_indices

from navlie.lib.imu import IMUState
from navlie.lib.states import SE3State
from navlie.types import State

from pymlg import SE3


def associate_states(
    states_1: typing.List[State],
    states_2: typing.List[State],
    max_diff: float = 0.02,
    offset_2: float = 0.0,
    return_matching_idx: bool = False,
):
    """Matches the stamps of two lists of states.

    Very similar to the function associate_trajectories from evo, but uses navlie's
    state types rather than the PoseTrajectory3d from evo.
    """
    snd_longer = len(states_2) > len(states_1)
    traj_long = copy.deepcopy(states_2) if snd_longer else copy.deepcopy(states_1)
    traj_short = copy.deepcopy(states_1) if snd_longer else copy.deepcopy(states_2)

    stamps_long = np.array([x.stamp for x in traj_long])
    stamps_short = np.array([x.stamp for x in traj_short])

    # Call evos function to actually match the stamps
    matching_indices_short, matching_indices_long = matching_time_indices(
        stamps_short,
        stamps_long,
        max_diff,
        offset_2 if snd_longer else -offset_2,
    )

    if len(matching_indices_short) != len(matching_indices_long):
        print("Matching time indices returned unequal number of indices")

    traj_short_reduced = [traj_short[idx] for idx in matching_indices_short]
    traj_long_reduced = [traj_long[idx] for idx in matching_indices_long]

    traj_1 = traj_short_reduced if snd_longer else traj_long_reduced
    traj_2 = traj_long_reduced if snd_longer else traj_short_reduced

    if return_matching_idx:
        matching_idx_dict = {
            "matching_indices_short": matching_indices_short,
            "matching_indices_long": matching_indices_long,
        }
        return traj_1, traj_2, matching_idx_dict
    else:
        return traj_1, traj_2


def associate_states_and_covariances(
    state_list_gt_raw: typing.List[State],
    state_list_est_raw: typing.List[State],
    cov_list_est_raw: typing.List[np.ndarray],
    verbose: bool = False,
):
    """A function to match the timestamps of states and covariances."""

    # Ensure we have equal number of estimates and covariances
    assert len(state_list_est_raw) == len(cov_list_est_raw)
    gt_list, est_list, matching_indices = associate_states(
        state_list_gt_raw,
        state_list_est_raw,
        return_matching_idx=True,
    )
    # Make sure we now have the same number of estimates and groundtruths
    # after matching
    assert len(gt_list) == len(est_list)

    if verbose:
        print(f"Number of matching states found: {len(gt_list)}")

    # Get the covariances at the correct stamps
    matching_idx_short = matching_indices["matching_indices_short"]
    matching_idx_long = matching_indices["matching_indices_long"]

    if len(cov_list_est_raw) > len(state_list_gt_raw):
        matching_idx = matching_idx_long
    else:
        matching_idx = matching_idx_short

    cov_est = [cov_list_est_raw[i] for i in matching_idx]

    return gt_list, est_list, cov_est


def imu_list_to_pose_list(imu_states: typing.List[IMUState]) -> typing.List[SE3State]:
    """Converts a list of IMU states to a list of SE3States."""
    pose_list: typing.List[SE3State] = []
    for imu_state in imu_states:
        pose = SE3State(
            value=SE3.from_components(imu_state.attitude, imu_state.position),
            stamp=imu_state.stamp,
            direction=imu_state.direction,
        )
        pose_list.append(pose)
    return pose_list

def shift_stamps(state_list: typing.List[State]) -> None:
    """Shifts all timestamps so the first state has stamp 0."""
    init_stamp = state_list[0].stamp
    for state in state_list:
        state.stamp -= init_stamp
