"""A collection of tools for interating with evo."""

import typing
import matplotlib.pyplot as plt
import numpy as np
import argparse

from navlie.lib.states import SE3State
from navlie.utils.alignment import state_list_to_evo_traj

# To interface with evo
from evo.main_ape import ape
import evo.common_ape_rpe as common
from evo.core.metrics import PoseRelation, APE, RPE
from evo.core import sync, trajectory, metrics
from evo.main_ape_parser import parser as ape_parser
from evo.tools.settings import SETTINGS

# from evo.core.metrics import RPE
from evo.main_rpe import rpe


def compute_ate(
    gt_states: typing.List[SE3State],
    est_states: typing.List[SE3State],
    max_diff: float = 0.02,
    offset: float = 0.0,
    pose_relation: metrics.PoseRelation = PoseRelation.translation_part,
    verbose: bool = False,
    align: bool = False,
):
    """Computes the ATE between two lists of SE3 states.

    See the following notebook for an example of the metrics.py API:
        https://github.com/MichaelGrupp/evo/blob/2ecb88dac02968589602e4485af75fc9c88bf5c1/notebooks/metrics.py_API_Documentation.ipynb
    """

    # First, convert the trajectories to evo type
    traj_ref = state_list_to_evo_traj(gt_states)
    traj_est = state_list_to_evo_traj(est_states)

    # Associate the stamps
    traj_ref_sync, traj_est_sync = sync.associate_trajectories(
        traj_ref,
        traj_est,
        max_diff=max_diff,
        offset_2=offset,
    )

    if verbose:
        print("Reference trajectory: ")
        print(traj_ref_sync)
        print("Estimated trajectory: ")
        print(traj_est_sync)

    ape_result = ape(
        traj_ref_sync,
        traj_est_sync,
        pose_relation,
        align=align,
        correct_scale=False,
    )
    return ape_result


def plot_evo_ape(
    ape_result,
    ref_traj,
    est_traj,
    plot_mode: str = "xy",
    save_plot: str = None,
    show_plots: bool = True,
):
    args = argparse.Namespace(
        plot_mode=plot_mode,  # Replace with an appropriate value for plot.PlotMode
        plot_x_dimension="distances",  # or "seconds", depending on your use case
        plot_colormap_min=None,
        plot_colormap_max=None,
        plot_colormap_max_percentile=None,  # Replace if needed
        ros_map_yaml=None,  # Replace if using a ROS map
        plot=show_plots,  # Set to True if you want to display plots
        save_plot=save_plot,  # Replace with a filename if saving the plot
        no_warnings=True,  # Set to True to suppress warnings
        serialize_plot=None,  # Replace with a filename if serializing the plot
        map_tile=False,
    )
    common.plot_result(args, ape_result, ref_traj, est_traj)


def run_evo_ape(
    result_file: str,
    gt_file: str,
    file_type: str = "tum",
    show_plot: bool = True,
    plot_mode: str = "xy",
    align: bool = True,
    save_result: str = None,
    verbose: bool = False,
):
    # Get the default evo parser
    evo_ape_parser = ape_parser()
    # Create a namespace to overwrite defaults from the evo_ape_parser
    namespace = argparse.Namespace()
    args = [file_type, gt_file, result_file]

    args_ape = evo_ape_parser.parse_args(args, namespace=namespace)
    args_ape.plot = show_plot
    args_ape.align = align
    args_ape.plot_mode = plot_mode
    args_ape.verbose = verbose

    if save_result is not None:
        args_ape.save_results = save_result

    main_ape.run(args_ape)


def evaluate_ape_complete(
    gt_states: typing.List[SE3State],
    est_states: typing.List[SE3State],
    align: bool = True,
):
    # First, convert the trajectories to evo type
    traj_ref = state_list_to_evo_traj(gt_states)
    traj_est = state_list_to_evo_traj(est_states)

    # Associate the stamps
    traj_ref_sync, traj_est_sync = sync.associate_trajectories(
        traj_ref,
        traj_est,
    )

    position_ape_result = ape(
        traj_ref_sync,
        traj_est_sync,
        PoseRelation.translation_part,
        align=align,
    )

    att_ape_result = ape(
        traj_ref_sync,
        traj_est_sync,
        PoseRelation.rotation_angle_deg,
        align=align,
    )

    return position_ape_result, att_ape_result


def compute_rpe(
    gt_states: typing.List[SE3State],
    est_states: typing.List[SE3State],
    max_diff: float = 0.02,
    align: bool = True,
    pose_relation: metrics.PoseRelation = PoseRelation.translation_part,
):
    """Computes the RPE between two lists of SE3 states."""
    traj_ref = state_list_to_evo_traj(gt_states)
    traj_est = state_list_to_evo_traj(est_states)

    traj_ref_sync, traj_est_sync = sync.associate_trajectories(
        traj_ref,
        traj_est,
        max_diff=max_diff,
    )

    rpe_result = rpe(
        traj_ref_sync,
        traj_est_sync,
        pose_relation=pose_relation,
        align=align,
    )

    return rpe_result

def plot_evo_rpe(rpe_result, ref_traj, est_traj, plot_mode="xy", save_plot=None, show_plots=True):
    args = argparse.Namespace(
        plot_mode=plot_mode,  # Replace with an appropriate value for plot.PlotMode
        plot_x_dimension="distances",  # or "seconds", depending on your use case
        plot_colormap_min=None,
        plot_colormap_max=None,
        plot_colormap_max_percentile=None,  # Replace if needed
        ros_map_yaml=None,  # Replace if using a ROS map
        plot=show_plots,  # Set to True if you want to display plots
        save_plot=save_plot,  # Replace with a filename if saving the plot
        no_warnings=True,  # Set to True to suppress warnings
        serialize_plot=None,  # Replace with a filename if serializing the plot
        map_tile=False,
    )
    common.plot_result(args, rpe_result, ref_traj, est_traj)


