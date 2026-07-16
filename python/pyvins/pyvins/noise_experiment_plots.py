"""This module contains code for plotting results of noise experiments for VINS algorithms."""

import logging
import typing
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyvins.aggregation import AveragedVinsResult


def generate_paper_plot(
    results_dict: typing.Dict[typing.Tuple[str, float], AveragedVinsResult],
    algorithm_order: typing.List[str],
    colors: typing.List[str],
    noise_levels_plot: typing.List[float] = None,
    save_dir: str = None,
):
    """Plot showcasing RMSEs for different noise values"""

    # Ensure that the list of colors matches the number of algorithms
    if len(colors) < len(algorithm_order):
        raise ValueError(
            f"Not enough colors provided for the number of algorithms. "
            f"Expected at least {len(algorithm_order)}, but got {len(colors)}."
        )

    # Create a grid of subplots for each noise level
    fig, ax = plt.subplots(2 * len(noise_levels_plot), 2, figsize=(12, 14), sharex=True)
    for alg_name, color in zip(algorithm_order, colors):
        for i, noise_level in enumerate(noise_levels_plot):
            cur_results = results_dict[(alg_name, noise_level)]
            stamps = cur_results.stamps_sync - cur_results.stamps_sync[0]
            att_rmse = np.rad2deg(cur_results.mean_att_rmses)
            pos_rmse = cur_results.mean_pos_rmses
            att_nees_over_time = cur_results.mean_att_nees_per_timestamp
            pos_nees_over_time = cur_results.mean_pos_nees_per_timestamp

            # Plot on correct subplot
            row_idx = 2 * i
            ax[row_idx, 0].plot(stamps, att_rmse, label=alg_name, color=color)
            ax[row_idx+1, 0].plot(stamps, pos_rmse, label=alg_name, color=color)
            ax[row_idx, 1].plot(stamps, att_nees_over_time, label=alg_name, color=color)
            ax[row_idx+1, 1].plot(stamps, pos_nees_over_time, label=alg_name, color=color)


    # Plot a horizontal line for the expected values of the NEES
    for i, noise_level in enumerate(noise_levels_plot):
        row_idx = 2 * i
        ax[row_idx, 1].axhline(3.0, color="k", linestyle="--", label="Expected NEES")
        ax[row_idx+1, 1].axhline(3.0, color="k", linestyle="--", label="Expected NEES")
    # Add titles and labels
    for i, noise_level in enumerate(noise_levels_plot):
        row_idx = 2 * i
        ax[row_idx, 0].set_title(rf"$\sigma_p$={int(noise_level)} px")
        ax[row_idx, 1].set_title(rf"$\sigma_p$={int(noise_level)} px")
        ax[row_idx, 0].set_ylabel("Attitude RMSE (deg)")
        ax[row_idx+1, 0].set_ylabel("Position RMSE (m)")
        ax[row_idx+1, 0].set_xlabel("Time (s)")
        ax[row_idx, 1].set_ylabel("Attitude NEES")
        ax[row_idx+1, 1].set_ylabel("Position NEES")
        ax[row_idx+1, 1].set_xlabel("Time (s)")
        # ax[row_idx, 0].legend()

    # ax[0, 1].legend(loc="upper left", bbox_to_anchor=(1.01, 1),
    # borderaxespad=0)
    ax[0, 0].legend()
    ax[2, 1].set_ylim(-5, 10)
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "paper_plot.pdf"), bbox_inches="tight", dpi=300)
    plt.show()

        


def generate_sensitivty_comparison_plots(
    results_dict: typing.Dict[typing.Tuple[str, float], AveragedVinsResult],
    algorithm_order: typing.List[str],
    save_dir: str = None,
    xlabel: str = "Noise Level",
    log_scale: bool = True,
):
    """The main function used to generate all plots for comparing results from
    different algorithms over a range of noise levels.

    This function should be used to analyze the sensitivty of different algorithms to a continuous
    parameter, such as noise level.

    results_dict is a dictionary with keys of the form (algorithm_name, noise_level),
    and values of form AveragedVinsResults, which contains statistics about the trajectory errors for that
    algorithm and noise level.
    """

    # Generate dataframe for plotting
    data = []
    for (alg_name, noise_level), results in results_dict.items():
        att_ate_list = results.attitude_ate_stats.values
        pos_ate_list = results.position_ate_stats.values
        for att_ate, pos_ate in zip(att_ate_list, pos_ate_list):
            data.append(
                {
                    "Algorithm": alg_name,
                    xlabel: float(noise_level),
                    "Attitude ATE": np.rad2deg(att_ate),
                    "Position ATE": pos_ate,
                }
            )

    df = pd.DataFrame(data)
    y_cols = ["Attitude ATE", "Position ATE"]
    fig, axes = generate_rmse_boxplot(
        df,
        x_col=xlabel,
        y_cols=y_cols,
        hue_col="Algorithm",
        hue_order=algorithm_order,
        figsize=(10, 5.5),
        y_labels=["Attitude ATE (deg)", "Position ATE (m)"],
        save_dir=save_dir,
        showfliers=False,
    )

    axes[0].set_title("")
    axes[1].set_title("")
    if log_scale:
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
    axes[1].set_xlabel(xlabel)
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "rmse_boxplot.pdf"))

    ### Generate RMSE lineplot, showing the mean RMSE for each algorithm at each noise level
    # Get the unique algorithm names and noise levels
    alg_names = set(key[0] for key in results_dict.keys())
    noise_levels_list = set(key[1] for key in results_dict.keys())
    # Convert to numerical values and sort
    noise_levels = sorted([float(n) for n in noise_levels_list])
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    # Plot each algorithm in the specified order
    for alg_name in algorithm_order:
        if alg_name not in alg_names:
            logging.warning(
                f"Algorithm {alg_name} not found in results, skipping line plot."
            )
            continue
        # Extract RMSE values for this algorithm across noise levels
        att_rmses = []
        pos_rmses = []
        for noise_level in noise_levels:
            key = (alg_name, noise_level)
            if key in results_dict:
                result = results_dict[key]
                att_rmses.append(np.rad2deg(result.attitude_ate_stats.mean))
                pos_rmses.append(result.position_ate_stats.mean)
            else:
                logging.warning(
                    f"Missing results for algorithm {alg_name} at noise level {noise_level}, skipping this point."
                )
                att_rmses.append(np.nan)
                pos_rmses.append(np.nan)
        ax[0].plot(noise_levels, att_rmses, marker="o", label=alg_name)
        ax[1].plot(noise_levels, pos_rmses, marker="o", label=alg_name)
    ax[0].set_title("Attitude RMSE")
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel("RMSE (deg)")
    ax[0].legend()
    ax[1].set_title("Position RMSE")
    ax[1].set_xlabel("Noise Level")
    ax[1].set_ylabel("RMSE (m)")
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "rmse_lineplot.pdf"))


def generate_rmse_lineplot(
    rmse_data: typing.Dict[str, typing.Dict[str, np.ndarray]],
):
    """Generates a line plot of RMSE values for different algorithms and noise levels.

    Format of rmse_data:
    {
        "algorithm_name": {
            "noise_levels": [1.0, 2.0, 3.0, ...],
            "att_rmses": [val1, val2, val3, ...],
            "pos_rmses": [val1, val2, val3, ...],
            "att_neess": [val1, val2, val3, ...],
            "pos_neess": [val1, val2, val3, ...],
        },
    }
        ...
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for alg_name, data in rmse_data.items():
        noise_levels = data["noise_levels"]
        att_rmses = data["att_rmses"]
        pos_rmses = data["pos_rmses"]
        att_neess = data["att_neess"]
        pos_neess = data["pos_neess"]

        ax[0, 0].plot(noise_levels, att_rmses, marker="o", label=alg_name)
        ax[0, 1].plot(noise_levels, pos_rmses, marker="o", label=alg_name)
        ax[1, 0].plot(noise_levels, att_neess, marker="o", label=alg_name)
        ax[1, 1].plot(noise_levels, pos_neess, marker="o", label=alg_name)
    ax[0, 0].set_title("Attitude RMSE")
    ax[0, 0].set_xlabel("Noise Level")
    ax[0, 0].set_ylabel("RMSE (deg)")
    ax[0, 0].legend()

    return fig, ax


def generate_rmse_boxplot(
    df: pd.DataFrame,
    x_col: str = "Noise Level",
    y_cols: typing.List[str] = None,
    hue_col: str = "Algorithm",
    hue_order: typing.List[str] = None,
    y_labels: typing.List[str] = None,
    figsize: tuple = (14, 6),
    save_dir: str = None,
    showfliers: bool = False,
):

    if y_labels is None:
        y_labels = y_cols

    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col])
    df[x_col] = df[x_col].astype(int)
    fig, axes = plt.subplots(
        len(y_cols),
        1,
        figsize=figsize,
        sharex=True,
    )

    # Plot the attitude RMSEs
    sns.boxplot(
        data=df,
        x=x_col,
        y=y_cols[0],
        hue=hue_col,
        hue_order=hue_order,
        ax=axes[0],
        fill=False,
        width=0.8,
        gap=0.1,
        showfliers=showfliers,
    )
    # for patch in axes[0].patches:
    #     r, g, b, a = patch.get_facecolor()
    #     patch.set_facecolor((r, g, b, 0.4))

    axes[0].set_title(f"{y_cols[0]} by {x_col}")
    axes[0].set_ylabel(y_labels[0])

    sns.boxplot(
        data=df,
        x=x_col,
        y=y_cols[1],
        hue=hue_col,
        hue_order=hue_order,
        ax=axes[1],
        fill=False,
        width=0.8,
        gap=0.1,
        showfliers=showfliers,
    )
    axes[1].set_title(f"{y_cols[1]} by {x_col}")
    axes[1].set_ylabel(y_labels[1])
    axes[0].legend()
    axes[1].get_legend().remove()
    fig.tight_layout()

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "rmse_boxplot.pdf"))
    return fig, axes


def generate_nees_latex_table(
    data: typing.Dict[str, typing.Dict[str, typing.Tuple[float, float]]],
    noise_levels: typing.List[str],
    algorithms: typing.List[str],
    caption: str = None,
    label: str = None,
    metrics: typing.Tuple[str, str] = ("Attitude", "Position"),
    precision: int = 3,
) -> str:
    """
    Generate a LaTeX table for NEES results with multiple algorithms and noise levels.

    Parameters:
    -----------
    data : dict
        Nested dictionary with structure:
        {algorithm: {noise_level: (orientation_value, position_value)}}
        Example: {"dli_fej": {"1-Pixel": (2.317, 2.200), "3-Pixel": (2.736, 3.092)}}

    noise_levels : list
        List of noise level names (e.g., ["1-Pixel Noise", "3-Pixel Noise"])

    algorithms : list
        List of algorithm names in desired order

    caption : str, optional
        Table caption

    label : str, optional
        Table label for referencing

    metrics : tuple, optional
        Names of the two metrics (default: ("Orientation", "Position"))

    precision : int, optional
        Number of decimal places (default: 3)

    Returns:
    --------
    str : LaTeX table code
    """

    # Start building the LaTeX code
    latex = []

    # Begin table environment
    if caption or label:
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"  \centering")

    # Calculate number of columns: 1 (algorithm) + len(noise_levels)
    n_cols = 1 + len(noise_levels)
    col_spec = "l" + "|" + "c" * len(noise_levels)

    # Begin tabular
    latex.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"    \hline")

    # Header row 1: Algorithm and Noise levels
    header1 = ["Algorithm"]
    for noise in noise_levels:
        header1.append(f"\\textbf{{NEES with {noise}}}")
    latex.append("    " + " & ".join(header1) + r" \\")

    # Header row 2: Metrics (spanning under each noise level)
    header2 = [""]
    for _ in noise_levels:
        header2.append(f"({metrics[0]} / {metrics[1]})")
    latex.append("    " + " & ".join(header2) + r" \\")
    latex.append(r"    \hline")

    # Data rows
    for algo in algorithms:
        row = [algo]
        for noise in noise_levels:
            if algo in data and noise in data[algo]:
                orient, pos = data[algo][noise]
                row.append(f"{orient:.{precision}f} / {pos:.{precision}f}")
            else:
                row.append("-- / --")  # Missing data
        latex.append("    " + " & ".join(row) + r" \\")

    # End tabular
    latex.append(r"    \hline")
    latex.append(r"  \end{tabular}")

    # Add caption and label if provided
    if caption:
        latex.append(f"  \\caption{{{caption}}}")
    if label:
        latex.append(f"  \\label{{{label}}}")

    # End table environment
    if caption or label:
        latex.append(r"\end{table}")

    return "\n".join(latex)


def generate_rmse_barplot(
    mc_results_by_alg: typing.Dict[typing.Tuple[str, float], AveragedVinsResult],
    figsize: tuple = (14, 6),
    save_dir: str = None,
    algorithm_colors: typing.Dict[str, str] = None,
    algorithm_order: typing.List[str] = None,
    algorithm_labels: typing.Dict[str, str] = None,
):
    """Generates a barplot of mean RMSE values for different algorithms and noise levels."""

    if algorithm_colors is None:
        colors = sns.color_palette("deep")

    data = []
    for (alg_name, noise_level), results in mc_results_by_alg.items():
        # Get the position RMSE from the results
        att_rmse = np.rad2deg(results.attitude_ate_stats.mean)
        pos_rmse = results.position_ate_stats.mean
        print(pos_rmse)
        data.append(
            {
                "Algorithm": alg_name,
                "Noise Level": noise_level,
                "Position RMSE": pos_rmse,
                "Attitude RMSE": att_rmse,
            }
        )

    df = pd.DataFrame(data)

    # Map algorithm names to labels if provided
    if algorithm_labels is not None:
        df["Algorithm"] = df["Algorithm"].map(algorithm_labels)

        if algorithm_order is not None:
            algorithm_order = [
                algorithm_labels.get(alg, alg) for alg in algorithm_order
            ]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # Plot the attitude RMSEs
    sns.barplot(
        data=df,
        x="Noise Level",
        y="Attitude RMSE",
        hue="Algorithm",
        hue_order=algorithm_order,
        palette=algorithm_colors,
        ax=axes[0],
    )
    axes[0].set_title("Attitude RMSE by Noise Level")
    axes[0].set_xlabel("Noise Level")
    axes[0].set_ylabel("RMSE (deg)")

    sns.barplot(
        data=df,
        x="Noise Level",
        y="Position RMSE",
        hue="Algorithm",
        hue_order=algorithm_order,
        palette=algorithm_colors,
        ax=axes[1],
    )
    axes[1].set_title("Position RMSE by Noise Level")
    axes[1].set_xlabel("Noise Level")
    axes[1].set_ylabel("RMSE (m)")
    axes[1].set_ylim(0, 0.5)
    fig.tight_layout()

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "rmse_barplot.pdf"))
    return fig, axes
