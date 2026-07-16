"""A collection of plotting utilities for visualizing visual-inertial navigation results."""

import numpy as np
import matplotlib.pyplot as plt
import typing

from navlie.lib.states import SE3State
from navlie.lib.imu import IMUState, IMU
from navlie.utils import GaussianResultList, MonteCarloResult, plot_nees, plot_poses
import os

import seaborn as sns
import pandas as pd
from scipy import stats


def plot_intrinsics_errors(
    stamps: np.ndarray,
    errors: np.ndarray,
    cam_id: int,
    ax: plt.Axes = None,
    three_sigma: np.ndarray = None,
    figsize: typing.Tuple[int, int] = (10, 8),
):
    """Plots the camera intrinsics errors over time.

    Parameters
    ----------
    errors : np.ndarray
        Array of shape (N, 8) containing the intrinsics errors over time.
    cam_id : int
        Camera ID for labeling the plot.
    ax : plt.Axes, optional
        Axes to plot on, by default None. If None, new axes will be created.
    """

    # Check that errors has the correct shape
    if errors.shape[1] != 8:
        raise ValueError("errors must have shape (N, 8)")

    if three_sigma is not None:
        if three_sigma.shape != errors.shape:
            raise ValueError("three_sigma must have the same shape as errors")

    if ax is None:
        fig, ax = plt.subplots(4, 2, figsize=figsize)
    else:
        # Ensure that the axes are in the correct shape
        fig = ax.get_figure()
        if ax.shape != (4, 2):
            raise ValueError("ax must be of shape (4, 2)")
    ax[0, 0].plot(stamps, errors[:, 0])
    ax[1, 0].plot(stamps, errors[:, 1])
    ax[2, 0].plot(stamps, errors[:, 2])
    ax[3, 0].plot(stamps, errors[:, 3])
    ax[0, 1].plot(stamps, errors[:, 4])
    ax[1, 1].plot(stamps, errors[:, 5])
    ax[2, 1].plot(stamps, errors[:, 6])
    ax[3, 1].plot(stamps, errors[:, 7])

    # Plot the three sigmas
    if three_sigma is not None:
        ax[0, 0].plot(stamps, three_sigma[:, 0], color="k", linestyle="--")
        ax[0, 0].plot(stamps, -three_sigma[:, 0], color="k", linestyle="--")
        ax[1, 0].plot(stamps, three_sigma[:, 1], color="k", linestyle="--")
        ax[1, 0].plot(stamps, -three_sigma[:, 1], color="k", linestyle="--")
        ax[2, 0].plot(stamps, three_sigma[:, 2], color="k", linestyle="--")
        ax[2, 0].plot(stamps, -three_sigma[:, 2], color="k", linestyle="--")
        ax[3, 0].plot(stamps, three_sigma[:, 3], color="k", linestyle="--")
        ax[3, 0].plot(stamps, -three_sigma[:, 3], color="k", linestyle="--")
        ax[0, 1].plot(stamps, three_sigma[:, 4], color="k", linestyle="--")
        ax[0, 1].plot(stamps, -three_sigma[:, 4], color="k", linestyle="--")
        ax[1, 1].plot(stamps, three_sigma[:, 5], color="k", linestyle="--")
        ax[1, 1].plot(stamps, -three_sigma[:, 5], color="k", linestyle="--")
        ax[2, 1].plot(stamps, three_sigma[:, 6], color="k", linestyle="--")
        ax[2, 1].plot(stamps, -three_sigma[:, 6], color="k", linestyle="--")
        ax[3, 1].plot(stamps, three_sigma[:, 7], color="k", linestyle="--")
        ax[3, 1].plot(stamps, -three_sigma[:, 7], color="k", linestyle="--")

    ax[0, 0].set_ylabel(r"$f_x$")
    ax[1, 0].set_ylabel(r"$f_y$")
    ax[2, 0].set_ylabel(r"$c_x$")
    ax[3, 0].set_ylabel(r"$c_y$")
    ax[0, 1].set_ylabel(r"$k_1$")
    ax[1, 1].set_ylabel(r"$k_2$")
    ax[2, 1].set_ylabel(r"$p_1$")
    ax[3, 1].set_ylabel(r"$p_2$")
    ax[3, 0].set_xlabel("Time (s)")
    ax[3, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig, ax


def plot_extrinsics_errors(
    stamps: np.ndarray,
    errors: np.ndarray,
    cam_id: int,
    ax: plt.Axes = None,
    three_sigma: np.ndarray = None,
    figsize: typing.Tuple[int, int] = (10, 8),
    show_titles: bool = True,
    error_color: str = "tab:blue",
    error_alpha: float = 0.5,
):
    """Plots the camera extrinsics errors over time

    Errors are assumed to be in the order:
    phi_1, phi_2, phi_3, r_1, r_2, r_3,

    where the phi errors are assumed to be in radians
    ."""
    # Check that errors has the correct shape
    if errors.shape[1] != 6:
        raise ValueError("errors must have shape (N, 6)")

    if three_sigma is not None:
        if three_sigma.shape != errors.shape:
            raise ValueError("three_sigma must have the same shape as errors")

    if ax is None:
        fig, ax = plt.subplots(3, 2, sharex=True, figsize=figsize)
    else:
        # Ensure that the axes are in the correct shape
        if ax.shape != (3, 2):
            raise ValueError("ax must be of shape (3, 2)")
        fig = ax[0, 0].get_figure()

    # Conver the relevant errors to degrees
    errors[:, 0:3] = np.rad2deg(errors[:, 0:3])
    ax[0, 0].plot(stamps, errors[:, 0], color=error_color, alpha=error_alpha)
    ax[1, 0].plot(stamps, errors[:, 1], color=error_color, alpha=error_alpha)
    ax[2, 0].plot(stamps, errors[:, 2], color=error_color, alpha=error_alpha)
    ax[0, 1].plot(stamps, errors[:, 3], color=error_color, alpha=error_alpha)
    ax[1, 1].plot(stamps, errors[:, 4], color=error_color, alpha=error_alpha)
    ax[2, 1].plot(stamps, errors[:, 5], color=error_color, alpha=error_alpha)

    # Plot the three sigmas
    if three_sigma is not None:
        three_sigma[:, 0:3] = np.rad2deg(three_sigma[:, 0:3])
        ax[0, 0].plot(stamps, three_sigma[:, 0], color="k", linestyle="--")
        ax[0, 0].plot(stamps, -three_sigma[:, 0], color="k", linestyle="--")
        ax[1, 0].plot(stamps, three_sigma[:, 1], color="k", linestyle="--")
        ax[1, 0].plot(stamps, -three_sigma[:, 1], color="k", linestyle="--")
        ax[2, 0].plot(stamps, three_sigma[:, 2], color="k", linestyle="--")
        ax[2, 0].plot(stamps, -three_sigma[:, 2], color="k", linestyle="--")
        ax[0, 1].plot(stamps, three_sigma[:, 3], color="k", linestyle="--")
        ax[0, 1].plot(stamps, -three_sigma[:, 3], color="k", linestyle="--")
        ax[1, 1].plot(stamps, three_sigma[:, 4], color="k", linestyle="--")
        ax[1, 1].plot(stamps, -three_sigma[:, 4], color="k", linestyle="--")
        ax[2, 1].plot(stamps, three_sigma[:, 5], color="k", linestyle="--")
        ax[2, 1].plot(stamps, -three_sigma[:, 5], color="k", linestyle="--")

    ax[0, 0].set_ylabel(r"$\delta \xi_1^\phi$ (deg)")
    ax[1, 0].set_ylabel(r"$\delta \xi_2^\phi$ (deg)")
    ax[2, 0].set_ylabel(r"$\delta \xi_3^\phi$ (deg)")
    ax[0, 1].set_ylabel(r"$\delta \xi_1^r$ (m)")
    ax[1, 1].set_ylabel(r"$\delta \xi_2^r$ (m)")
    ax[2, 1].set_ylabel(r"$\delta \xi_3^r$ (m)")
    ax[2, 1].set_xlabel("Time (s)")
    ax[2, 0].set_xlabel("Time (s)")
    if show_titles:
        fig.suptitle(f"Extrinsic Errors for Camera {cam_id}")
    fig.tight_layout()
    return fig, ax


def set_plot_theme(
    palette: str = "deep", enable_latex: bool = True
) -> typing.List[str]:
    """Sets some plotting defaults."""
    sns.set_theme(style="whitegrid")
    plt.rc("lines", linewidth=1.5)
    plt.rc("axes", grid=True)
    plt.rc("grid", linestyle="--")
    plt.rc("grid", alpha=0.9)

    # plt.rcParams.update({"font.size": 14})
    # plt.rc("text", usetex=True)
    # plt.rcParams.update(
    #     {
    #         "text.usetex": enable_latex,
    #         "font.family": "serif",
    #         "font.size": 20,
    #         "axes.labelsize": 20,
    #         "axes.titlesize": 20,
    #         "legend.fontsize": 14,
    #     }
    # )

    plt.rcParams.update(
        {
            "text.usetex": enable_latex,
            "font.family": "serif",
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
        }
    )

    if palette == "colorblind":
        # colors = [
        #     "#E69F00",  # orange
        #     "#56B4E9",  # sky blue
        #     "#009E73",  # bluish green
        #     "#F0E442",  # yellow
        #     "#0072B2",  # blue
        #     "#D55E00",  # vermillion
        #     "#CC79A7",  # reddish purple
        #     "#999999",  # grey
        # ]
        # sns.set_palette(colors)
        colors = sns.color_palette("colorblind")
        sns.set_palette(colors)
    elif palette == "deep":
        colors = sns.color_palette("deep")
    elif palette == "tab10":
        colors = sns.color_palette("tab10")
    return colors


def plot_trajectories(
    traj_data: typing.Dict[str, typing.List[SE3State]],
    ax=None,
    colors: typing.List[str] = None,
):
    """Plots multiple trajectories on the same 3D plot."""

    if colors is None:
        colors = sns.color_palette("deep", len(traj_data))

    for i, (traj_name, poses) in enumerate(traj_data.items()):
        fig, ax = plot_poses(
            poses,
            step=None,
            label=traj_name,
            linewidth=1.5,
            ax=ax,
        )

    ax.set_title("Trajectories Comparison")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_nav_rmses_over_time(
    stamps: np.ndarray,
    attitude_rmses: np.ndarray,
    velocity_rmses: np.ndarray,
    position_rmses: np.ndarray,
    label=None,
    ax=None,
    color=None,
    alpha=1.0,
):
    """Plots the navigation RMSEs over time for attitude, velocity, and position."""

    if ax is None:
        fig, ax = plt.subplots(3, 1, sharex=True)
    else:
        fig = ax[0].get_figure()

    plot_kwargs = {}
    if color is not None:
        plot_kwargs["color"] = color
    if alpha != 1.0:
        plot_kwargs["alpha"] = alpha
    if label is not None:
        plot_kwargs["label"] = label

    attitude_rmses_deg = np.rad2deg(attitude_rmses)

    # Convert attitude RMSEs from rad to degrees
    ax[0].plot(stamps, attitude_rmses_deg, **plot_kwargs)
    ax[0].set_ylabel("Attitude RMSE (deg)")
    ax[0].grid(True)

    # Plot velocity RMSE
    ax[1].plot(stamps, velocity_rmses, **plot_kwargs)
    ax[1].set_ylabel("Velocity RMSE (m/s)")
    ax[1].grid(True)

    # Plot position RMSE
    ax[2].plot(stamps, position_rmses, **plot_kwargs)
    ax[2].set_ylabel("Position RMSE (m)")
    ax[2].set_xlabel("Time (s)")
    ax[2].grid(True)

    # Add legend if label was provided
    if label is not None:
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

    fig.tight_layout()
    return fig, ax


def plot_pose_rmses_over_time(
    stamps: np.ndarray,
    attitude_rmses: np.ndarray,
    position_rmses: np.ndarray,
    label=None,
    ax=None,
    color=None,
    alpha=1.0,
):
    """Plots the RMSEs over time for attitude and position.

    Parameters
    -----------
    stamps: Stamps to plot over
    """
    if ax is None:
        fig, ax = plt.subplots(2, 1, sharex=True)
    else:
        fig = ax[0].get_figure()

    plot_kwargs = {}
    if color is not None:
        plot_kwargs["color"] = color
    if alpha != 1.0:
        plot_kwargs["alpha"] = alpha
    if label is not None:
        plot_kwargs["label"] = label

    attitude_rmses_deg = np.rad2deg(attitude_rmses)

    ax[0].plot(stamps, attitude_rmses_deg, **plot_kwargs)
    ax[1].plot(stamps, position_rmses, **plot_kwargs)
    ax[0].set_ylabel("Orientation RMSE (deg)")
    ax[0].set_xlabel("Time (s)")
    ax[1].set_ylabel("Position RMSE (m)")
    ax[1].set_xlabel("Time (s)")

    if label is not None:
        ax[0].legend()
        ax[1].legend()

    fig.tight_layout()

    return fig, ax


def compare_nav_rmses_over_time(
    rmse_data: typing.Dict[str, typing.Dict[str, np.ndarray]],
    save_dir: str = None,
    colors: typing.List[str] = None,
    alpha: float = 0.8,
    figsize: typing.Tuple[int, int] = (10, 8),
    ax=None,
) -> typing.Tuple[plt.Figure, plt.Axes]:

    algorithms = list(rmse_data.keys())
    n_algorithms = len(algorithms)

    # Set up colors
    if colors is None:
        colors = sns.color_palette("deep", n_algorithms)
    elif len(colors) < n_algorithms:
        colors = (colors * ((n_algorithms // len(colors)) + 1))[:n_algorithms]

    # Validate required keys
    required_keys = ["stamps", "attitude_rmses", "position_rmses"]
    for alg_name, data in rmse_data.items():
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing key '{key}' for algorithm '{alg_name}'")

    fig = None
    for i, (alg_name, data) in enumerate(rmse_data.items()):
        plot_pose_rmses_over_time(
            stamps=data["stamps"],
            attitude_rmses=data["attitude_rmses"],
            position_rmses=data["position_rmses"],
            label=alg_name,
            ax=ax,
            color=colors[i],
            alpha=alpha,
        )

    # Update figure size if different from default
    if figsize != (10, 8):
        fig.set_size_inches(figsize)

    # Add a title
    fig.suptitle("Navigation RMSEs Comparison", fontsize=16)
    fig.tight_layout()

    # Save if directory provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "nav_rmses_comparison.pdf"))
        fig.savefig(os.path.join(save_dir, "nav_rmses_comparison.png"), dpi=300)

    return fig, ax


def plot_nees(
    stamps: np.ndarray,
    nees: np.ndarray,
    ax: plt.Axes = None,
    label: str = None,
    color=None,
    expected_nees_color="tab:red",
    alpha: float = 1.0,
    expected_nees: float = 1.0,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Makes a plot of the NEES, showing the actual NEES values, the expected NEES,
    and the bounds of the specified confidence interval.

    Parameters
    ----------
    results : GaussianResultList or MonteCarloResult
        Results to plot
    ax : plt.Axes, optional
        Axes on which to draw, by default None. If None, new axes will be
        created.
    label : str, optional
        Label to assign to the NEES line, by default None
    color : optional
        Fed directly to the ``plot(..., color=color)`` function, by default None
    confidence_interval : float or None, optional
        Desired probability confidence region, by default 0.95. Must lie between
        0 and 1. If None, no confidence interval will be plotted.
    normalize : bool, optional
        Whether to normalize the NEES by the degrees of freedom, by default False
    alpha: float, optional
        Alpha value for the plot, by default 1.0
    Returns
    -------
    plt.Figure
        Figure on which the plot was drawn
    plt.Axes
        Axes on which the plot was drawn
    """

    if ax is None:
        fig, ax = plt.subplots(
            1,
            1,
            sharex=True,
        )
    else:
        fig = ax.get_figure()

    axs_og = ax
    kwargs = {}
    if color is not None:
        kwargs["color"] = color
        kwargs["alpha"] = alpha

    expected_nees_label = "Expected NEES"
    _, exisiting_labels = ax.get_legend_handles_labels()

    if expected_nees_label in exisiting_labels:
        expected_nees_label = None

    # fmt:off
    ax.plot(stamps, nees, label=label, **kwargs)

    if expected_nees is not None:
        expected_nees_color = expected_nees_color or "tab:red"
        ax.axhline(expected_nees, color=expected_nees_color, label=expected_nees_label, linestyle="--")

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()

    return fig, axs_og


def format_imu_consistency_plot(
    fig: plt.Figure,
    axs: plt.Axes,
):
    axs[0, 0].set_title("Attitude")
    axs[0, 1].set_title("Velocity")
    axs[0, 2].set_title("Position")
    axs[0, 3].set_title("Gyro Bias")
    axs[0, 4].set_title("Accel Bias")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 2].set_xlabel("Time (s)")
    axs[2, 3].set_xlabel("Time (s)")
    axs[2, 4].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("x")
    axs[1, 0].set_ylabel("y")
    axs[2, 0].set_ylabel("z")
    fig.tight_layout()

    return fig, axs


def plot_imu_errors_with_covariance(
    result_list: typing.List[GaussianResultList],
    save_dir: str = None,
    file_extension: str = ".png",
    plot_all_sigma_bounds: bool = False,
):
    """Generates 3 Sigma plots for the IMU states.

    This can be used to plot Monte-Carlo results by passing in a list
    of GaussianResultList objects.

    Parameters
    ----------
    results : typing.List[GaussianResultList]
        a GaussianResultList object, or a list of GaussianResultList objects
    save_dir : str, optional
        The directory to save to plots in, by default None
    """
    # check if result_list is a single GaussianResultList
    if isinstance(result_list, GaussianResultList):
        result_list = [result_list]

    # Ensure that the dimension of the state is always 15
    # TODO: Add this
    for result in result_list:
        dofs = result.dof
        if dofs[0] != 15:
            raise ValueError("The dimension of the state must be 15")

    slices = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)]
    titles = [
        "Attitude Errors",
        "Velocity Errors",
        "Position Errors",
        "Gyro Bias Errors",
        "Accel Bias Errors",
    ]

    labels_list = get_list_of_imu_labels()
    # labels = ["x", "y", "z"]

    # For each slice, plot the results
    # This creates a separate figure for each slice
    for s, title, labels in zip(slices, titles, labels_list):
        fig, ax = plt.subplots(3, 1, sharex=True)

        # For each result, plot the three sigma bounds
        for i, result in enumerate(result_list):
            if i == 0:
                enable_sigma_bounds = True
            else:
                enable_sigma_bounds = False
            error = result.error[:, s]
            sigma = result.three_sigma[:, s]
            stamps = result.stamp
            plot_three_sigma(
                stamps,
                error,
                sigma,
                ax=ax,
                enable_sigma_bounds=enable_sigma_bounds,
            )
            ax[2].set_xlabel("Time (s)")
            for i, label in enumerate(labels):
                ax[i].set_ylabel(label)
        fig.suptitle(title)
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(
                os.path.join(save_dir, f"{title.lower().replace(' ', '_')}.pdf")
            )


def plot_pose_timeseries(
    stamps,
    phi: np.ndarray,
    pos: np.ndarray,
    ax: plt.Axes = None,
    label: str = None,
):
    """Plots the pose timeseries for attitude and position."""
    if ax is None:
        fig, ax = plt.subplots(2, 1, sharex=True)
    else:
        fig = ax[0].figure

    ax[0].plot(stamps, phi[:, 0], label=label)
    ax[0].plot(stamps, phi[:, 1])
    ax[0].plot(stamps, phi[:, 2])
    ax[1].plot(stamps, pos[:, 0])
    ax[1].plot(stamps, pos[:, 1])
    ax[1].plot(stamps, pos[:, 2])

    ax[0].set_ylabel("Attitude (rad)")
    ax[1].set_ylabel("Position (m)")
    ax[1].set_xlabel("Time (s)")
    ax[0].legend()
    fig.tight_layout()
    return fig, ax


def plot_pose_timeseries_2(
    stamps,
    phi: np.ndarray,
    pos: np.ndarray,
    ax: plt.Axes = None,
):
    if ax is None:
        fig, ax = plt.subplots(3, 2, sharex=True)
    else:
        fig = ax[0].figure

    for i in range(3):
        ax[i, 0].plot(stamps, phi[:, i])
        ax[i, 1].plot(stamps, pos[:, i])

    ax[0, 0].set_title("Calib. Ori.")
    ax[0, 1].set_title("Calib. Pos.")
    fig.tight_layout()
    return fig, ax


def plot_pose_error_timeseries(
    stamps: np.ndarray,
    phi_errors: np.ndarray,
    pos_errors: np.ndarray,
    ax: plt.Axes = None,
):
    """Plots the pose errors over time"""
    # Plot the timeseries for each of these
    if ax is None:
        fig, ax = plt.subplots(2, 1, sharex=True)
    else:
        fig = ax[0].figure
    ax[0].plot(stamps, phi_errors[:, 0], label="x")
    ax[0].plot(stamps, phi_errors[:, 1], label="y")
    ax[0].plot(stamps, phi_errors[:, 2], label="z")
    ax[1].plot(stamps, pos_errors)
    ax[0].set_ylabel("Attitude Error (deg)")
    ax[1].set_ylabel("Position Error (m)")
    ax[1].set_xlabel("Time (s)")
    ax[0].legend()
    fig.tight_layout()
    return fig, ax


# def plot_rmses_over_time(
#     stamps: np.ndarray,
#     phi_rmses: np.ndarray,
#     pos_rmses: np.ndarray,
#     ax: plt.Axes = None,
# ):
#     """Plots the RMSEs over time"""
#     # Plot the timeseries for each of these
#     if ax is None:
#         fig, ax = plt.subplots(2, 1, sharex=True)
#     else:
#         fig = ax[0].figure
#     ax[0].plot(stamps, phi_rmses)
#     ax[1].plot(stamps, pos_rmses)
#     ax[0].set_ylabel("Attitude RMSE (deg)")
#     ax[1].set_ylabel("Position RMSE (m)")
#     ax[1].set_xlabel("Time (s)")
#     fig.tight_layout()
#     return fig, ax


def plot_imu_monte_carlo_result(
    mc_results: MonteCarloResult,
    save_dir: str = None,
):
    """Generates plots related to the Monte-Carlo results for the IMU states.

    Parameters
    ----------
    mc_results : MonteCarloResult
        The Monte-Carlo results to plot.
    save_dir: str, optional
        Directory to save the plots in. If None, plots will not be saved.

    """
    # Ensure that the dimension of the state is always 15
    if not np.all(mc_results.dof == 15):
        raise ValueError("The dimension of the state must be 15")

    # Compute individual Monte-Carlo results
    att_result_list: typing.List[GaussianResultList] = []
    pos_result_list: typing.List[GaussianResultList] = []
    nav_mc_list: typing.List[GaussianResultList] = []
    for grl in mc_results.trial_results:
        att_result_list.append(grl[:, 0:3])
        pos_result_list.append(grl[:, 6:9])
        nav_mc_list.append(grl[:, 0:9])

    att_mc_results = MonteCarloResult(att_result_list)
    pos_mc_results = MonteCarloResult(pos_result_list)

    fig, ax = plot_nees(mc_results.stamp, mc_results.nees, color="tab:blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES")
    fig.suptitle(f"State NEES over {mc_results.num_trials} trials")
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "average_nees.pdf"))

    # Plot attitude nees
    fig, ax = plot_nees(att_mc_results.stamp, att_mc_results.nees, color="tab:blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES")
    fig.suptitle(f"Orientation NEES over {mc_results.num_trials} trials")
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "average_orientation_nees.pdf"))

    # Plot position NEES
    fig, ax = plot_nees(pos_mc_results.stamp, pos_mc_results.nees, color="tab:blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES")
    fig.suptitle(f"Position NEES over {mc_results.num_trials} trials")
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "average_position_nees.pdf"))

    stamps = mc_results.stamp
    # Generate a plot of all the NEES values plotted on the same plot
    fig, ax = plt.subplots(1, 1)
    for result in mc_results.trial_results:
        plot_nees(result.stamp, result.nees, ax=ax, color="tab:blue", alpha=0.5)
        fig.suptitle("NEES for all trials")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("NEES")
        # fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "nees_all_trials.pdf"))


def plot_three_sigma(
    stamps: np.ndarray,
    error: np.ndarray,
    three_sigma: np.ndarray,
    ax: plt.Axes = None,
    error_color=None,
    enable_sigma_bounds: bool = True,
    error_alpha: float = 0.5,
    three_sigma_color: str = "black",
    error_label: str = None,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """Plots the error and three sigma bounds directly.

    This function does not depend on GaussianResultList, so it can be used more generally.
    """

    dim = error.shape[1]

    if dim < 3:
        n_rows = dim
    else:
        n_rows = 3

    n_cols = int(np.ceil(dim / 3))

    if ax is None:
        fig, ax = plt.subplots(n_rows, n_cols, sharex=True)
    else:
        fig: plt.Figure = ax.ravel("F")[0].get_figure()

    if error_color is None:
        error_color = "tab:blue"

    ax_og = ax
    # Plot the error and three sigma bounds
    ax = ax.ravel("F")
    for i in range(three_sigma.shape[1]):
        ax[i].plot(
            stamps,
            error[:, i],
            label=error_label,
            color=error_color,
            alpha=error_alpha,
        )
        if enable_sigma_bounds:
            ax[i].plot(
                stamps,
                three_sigma[:, i],
                color=three_sigma_color,
                linewidth=1.5,
                linestyle="--",
            )
            ax[i].plot(
                stamps,
                -three_sigma[:, i],
                color=three_sigma_color,
                linewidth=1.5,
                linestyle="--",
            )

        if i == 0 and error_label is not None:
            ax[i].legend()

    return fig, ax_og


def plot_timeseries_R3(
    stamps: np.ndarray,
    states: np.ndarray,
    ax: plt.Axes = None,
    color: str = "tab:blue",
    label: str = None,
):
    """Plots states in R^3 over time.

    Parameters
    ----------
    stamps : np.ndarray
        N x 1 array of timestamps
    states : np.ndarray
        N x 3 array of states
    ax: plt.Axes
        Axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(3, 1, sharex=True)
    else:
        fig: plt.Figure = ax.ravel("F")[0].get_figure()

    # Plot the states
    for i in range(3):
        if i == 0:
            ax[i].plot(stamps, states[:, i], color=color, label=label)
        else:
            ax[i].plot(stamps, states[:, i], color=color)

    ax[0].set_ylabel("x")
    ax[1].set_ylabel("y")
    ax[2].set_ylabel("z")
    ax[2].set_xlabel("Time (s)")
    return fig, ax


def plot_imu_data(
    imu_data_list: typing.List[IMU],
    imu_states: typing.List[IMUState] = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Plots IMU data, and optionally, IMU biases."""
    gyro_np = np.array([data.gyro for data in imu_data_list])
    accel_np = np.array([data.accel for data in imu_data_list])
    stamps = np.array([data.stamp for data in imu_data_list])

    # Only extract biases if we passed in IMU states
    if imu_states is not None:
        gyro_bias_np = np.array([state.bias_gyro for state in imu_states])
        accel_bias_np = np.array([state.bias_accel for state in imu_states])

    # Create plots
    if ax is None:
        if imu_states is None:
            n_plots = 2
        else:
            n_plots = 4

        fig, ax = plt.subplots(n_plots, 1, sharex=True)
    # Plot gyro and accel
    ax[0].plot(stamps, gyro_np)
    ax[1].plot(stamps, accel_np)

    if imu_states is not None:
        ax[2].plot(stamps, gyro_bias_np[0 : len(stamps), :])
        ax[3].plot(stamps, accel_bias_np[0 : len(stamps), :])
        ax[2].set_ylabel("Gyro Bias")
        ax[3].set_ylabel("Accel Bias")

    ax[0].set_ylabel("Gyro (rad/s)")
    ax[1].set_ylabel("Accel (m/s$^2$)")
    ax[-1].set_xlabel("Time (s)")
    return fig, ax


def get_list_of_imu_labels() -> typing.List[typing.List[str]]:
    labels = [
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
    return labels


def plot_timing_dataframe(
    timing_df: pd.DataFrame,
    save_dir: str = None,
    columns_boxplot: typing.List[str] = None,
    figsize: typing.Tuple[int, int] = (10, 6),
):
    """Generates timing related plots from a timing dataframe."""
    df = timing_df.copy()
    print(df)
    names = df.columns.to_list()
    print(names)

    total_times = df[names[-1]]

    # Get the times for each part
    df_parts = df.iloc[:, :-1]

    times = df[names[0]]
    parts = [df_parts[col] for col in df_parts.columns if col != names[0]]

    if len(parts) == 0:
        # Only plot the total times
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        # Generate lineplot of total times
        ax[0].plot(times.to_numpy(), total_times.to_numpy())
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Execution Time (ms)")

        # Generate histogram of total times
        ax[1].hist(total_times, bins=10, alpha=0.7, edgecolor="black")
        ax[1].set_xlabel("Total Time (ms)")
        ax[1].set_ylabel("Frequency")
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "timing_info.pdf"))
        return fig, ax

    # Create a figure with four subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Generate stackplot
    ax1.stackplot(times, *parts, labels=[col for col in df.columns if col != names[0]])
    ax1.legend(loc="upper right")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Execution Time (ms)")

    # Generate lineplot of total times
    ax2.plot(times, total_times)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Execution Time (ms)")

    # Generate histogram of total times
    ax3.hist(total_times, bins=10, alpha=0.7, edgecolor="black")
    ax3.set_xlabel("Total Time (ms)")
    ax3.set_ylabel("Frequency")
    fig.tight_layout()

    # Compute statistics
    column_means = df_parts.mean()
    column_stds = df_parts.std()

    if columns_boxplot is None:
        columns_boxplot = [
            " preintegration",
            " triangulation",
            " assembly",
            " optimization",
            " covariance",
            " marginalization",
        ]
    ax4 = sns.boxplot(
        data=df[df.columns[1:]],
        ax=ax4,
        showfliers=False,
    )
    ax4.set_ylabel("Execution Time (ms)")
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    # ax.set_yscale("log")
    ax4.set_xlabel("Estimator Component")

    fig.suptitle(f"Timing Information Summary")
    fig.tight_layout()

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "timing_info.pdf"))

    return fig, axes


def plot_ate_distribution(
    attitude_ates: typing.Dict[str, np.ndarray],
    position_ates: typing.Dict[str, np.ndarray],
    colors: typing.List[str] = None,
    save_dir: str = None,
    **kwargs,
):
    """Generates ATE distribution plots for attitude and position.

    Parameters
    ----------
    attitude_ates : typing.Dict[str, np.ndarray]
        Dictionary of attitude ATEs for each algorithm
    position_ates : typing.Dict[str, np.ndarray]
        Dictionary of position ATEs for each algorithm
    colors: typing.Dict[str, str], optional
        Dictionary mapping algorithm names to colors, by default None
    save_dir : str, optional
        Directory to save the plots in, by default None
    """

    title = kwargs.pop("title", None)

    # Extract plotting parameters with defaults
    boxplot_defaults = {
        "linewidth": 1.5,
        "showfliers": False,
        "boxprops": {"alpha": 1.0},
        "width": 0.5,
    }

    boxplot_defaults.update(kwargs)

    # Prepare data for plotting
    data_for_plotting = []
    for key in attitude_ates.keys():
        att_vals = np.rad2deg(attitude_ates[key])
        for val in att_vals:
            data_for_plotting.append(
                {
                    "Algorithm": key,
                    "Error Type": "Attitude",
                    "ATE": val,
                    "Unit": "degrees",
                }
            )

        if key in position_ates:
            pos_vals = position_ates[key]
            for val in pos_vals:
                data_for_plotting.append(
                    {
                        "Algorithm": key,
                        "Error Type": "Position",
                        "ATE": val,
                        "Unit": "m",
                    }
                )

    # Convert to DataFrame for plotting
    df = pd.DataFrame(data_for_plotting)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Create color palette if colors are provided
    palette = sns.color_palette("deep")
    # if colors is not None:
    #     # Get unique algorithms in order they appear
    #     unique_algorithms = df["Algorithm"].unique()
    #     palette = [colors.get(alg, "#1f77b4") for alg in unique_algorithms]

    # Attitude subplot
    att_data = df[df["Error Type"] == "Attitude"]
    sns.boxplot(
        data=att_data,
        x="Algorithm",
        y="ATE",
        ax=axes[0],
        palette=palette,
        **boxplot_defaults,
    )
    axes[0].set_ylabel("Attitude ATE (deg)")
    axes[0].set_xlabel("Algorithm")
    axes[0].tick_params(axis="x", rotation=45)

    # Position subplot
    pos_data = df[df["Error Type"] == "Position"]
    sns.boxplot(
        data=pos_data,
        x="Algorithm",
        y="ATE",
        ax=axes[1],
        palette=palette,
        **boxplot_defaults,
    )
    axes[1].set_ylabel("Position ATE (m)")
    axes[1].set_xlabel("Algorithm")
    axes[1].tick_params(axis="x", rotation=45)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, axes


def plot_ate_barplot(
    attitude_ates: typing.Dict[str, float],
    position_ates: typing.Dict[str, float],
    colors: typing.Dict[str, str] = None,
    figsize: typing.Tuple[int, int] = (8, 5),
    save_dir: str = None,
    **kwargs,
):
    """Generates ATE distribution plots for attitude and position.

    Parameters
    ----------
    attitude_ates : typing.Dict[str, np.ndarray]
        Dictionary of attitude ATEs for each algorithm
    position_ates : typing.Dict[str, np.ndarray]
        Dictionary of position ATEs for each algorithm
    save_dir : str, optional
        Directory to save the plots in, by default None
    """

    title = kwargs.pop("title", None)

    # Extract plotting parameters with defaults
    barplot_defaults = {
        "alpha": 0.8,
        "edgecolor": "black",
        "linewidth": 0.8,
        "width": 0.9,
    }

    barplot_defaults.update(kwargs)

    data_for_plotting = []

    # Process data
    for alg, rmse_val in attitude_ates.items():
        att_val = np.rad2deg(rmse_val)
        data_for_plotting.append(
            {
                "Algorithm": alg,
                "Error Type": "Attitude",
                "ATE": att_val,
                "Unit": "degrees",
            }
        )

    # Process position data
    for algorithm, rmse_value in position_ates.items():
        pos_val = rmse_value  # Position values should already be in meters
        data_for_plotting.append(
            {
                "Algorithm": algorithm,
                "Error Type": "Position",
                "ATE": pos_val,
                "Unit": "m",
            }
        )

    # Convert to DataFrame for plotting
    df = pd.DataFrame(data_for_plotting)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Create color palette if colors provided
    palette = colors
    # if colors is not None:
    #     # Get unique algorithms in order they appear
    #     unique_algorithms = df["Algorithm"].unique()
    #     palette = [colors.get(alg, "#1f77b4") for alg in unique_algorithms]

    # Attitude subplot
    att_data = df[df["Error Type"] == "Attitude"]
    sns.barplot(
        data=att_data,
        x="Algorithm",
        y="ATE",
        ax=axes[0],
        palette=palette,
        **barplot_defaults,
    )
    axes[0].set_ylabel("Attitude ATE (deg)")
    axes[0].set_xlabel("Algorithm")
    axes[0].tick_params(axis="x", rotation=45)

    # Position subplot
    pos_data = df[df["Error Type"] == "Position"]
    sns.barplot(
        data=pos_data,
        x="Algorithm",
        y="ATE",
        ax=axes[1],
        palette=palette,
        **barplot_defaults,
    )
    axes[1].set_ylabel("Position ATE (m)")
    axes[1].set_xlabel("Algorithm")
    axes[1].tick_params(axis="x", rotation=45)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "ate_barplot.pdf"))

    return fig, axes


def plot_nees_density(
    nees_values: np.ndarray,
    n_dof: int,
    ax=None,
    label=None,
    color=None,
    show_theoretical=True,
    alpha=0.8,
    num_bins=20,
    normalize=True,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """Plots a histogram of the NEES values with theoretical chi-square distrbution overlay."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    if normalize:
        plot_data = nees_values / n_dof
        x_max = 4
    else:
        plot_data = nees_values
        x_max = max(20, 3 * n_dof)

    counts, bin_edges = np.histogram(plot_data, bins=num_bins, density=True)

    # ax.plot(bin_edges[:-1], counts, color=color, alpha=alpha, label=label)
    ax.step(bin_edges[:-1], counts, where="post", color=color, alpha=alpha, label=label)

    # Plot theoretical chi-square distribution (only once)
    if show_theoretical:
        # x = np.linspace(0, max(ax.get_xlim()[1], np.max(nees_values)), 1000)
        x = np.linspace(0, max(x_max, bin_edges[-1]), 1000)

        if normalize:
            chi2_pdf = n_dof * stats.chi2.pdf(n_dof * x, df=n_dof)
        else:
            chi2_pdf = stats.chi2.pdf(x, df=n_dof)
        ax.plot(x, chi2_pdf, "k--", linewidth=2, label="Theoretical")

    # Labels and formatting
    ax.set_xlabel("NEES", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_pose_stats(
    axes_row,
    algorithms_data: typing.Dict[str, typing.Dict[str, str]],
    is_first_row: typing.List = True,
    is_last_row: bool = True,
):
    pass
