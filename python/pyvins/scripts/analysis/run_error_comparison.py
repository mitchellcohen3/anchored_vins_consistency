"""Runs the OpenVINS evaluation script error_comparison to compute
ATE/RPE metrics for all algorithms across multiple datasets.

Generates CSVs, and then loads in those CSVs to create LaTeX tables/figures that
can be directly included in a paper.
"""

import os
import argparse
import subprocess
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from pyvins.plot_utils import set_plot_theme
from pyvins.tables import create_ate_table, create_rpe_table
from pyvins.processing import get_label_for_alg

import pandas as pd
import typing

colors = set_plot_theme(palette="colorblind", enable_latex=True)

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SLAM results on real data, generates CSV files and outputs table for the paper."
    )
    parser.add_argument("--align_mode")
    parser.add_argument("--folder_gt")
    parser.add_argument("--folder_algorithms")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of dataset names to include. If omitted, all datasets are used.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Subset of algorithm names to include. If omitted, all algorithms are used.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="Display labels for each algorithm listed in --algorithms (same order).",
    )
    parser.add_argument("--output", default=None, help="Path to save the LaTeX table.")

    args = parser.parse_args()
    args.align_mode = "posyaw"

    basedir = "/home/mitchell/experiments/ov_tumvi_featrep_withcalib_20260224_171633"
    args.folder_gt = os.path.join(basedir, "truths")
    args.folder_algorithms = os.path.join(basedir, "algorithms")
    args.no_plots = True
    args.run_evaluation = True

    # This orders the algorithms by consistency method, but maybe we should order
    # by parameterization instead?
    # args.algorithms = [
    #     "none_global3d",
    #     # "none_global3d_stereo",
    #     "fej_global3d",
    #     # "fej_global3d_stereo",
    #     "dri_fej_global3d",
    #     # "dri_fej_global3d_stereo",
    #     # "dri_fej_global3d_stereo",
    #     # "dri_fej_global3d_covprop",
    #     # "dri_fej_global3d_nocovprop",
    #     # "none_global3d",
    #     # "fej_global3d",
    #     # "dri_fej_global3d",
    #     # "none_anchoredmsckfinversedepth",
    #     # "fej_anchoredmsckfinversedepth",
    #     # "dri_fej_anchoredmsckfinversedepth",
    #     # "none_anchored3d",
    #     # "fej_anchored3d",
    #     # "dri_fej_anchored3d",
    # ]

    # args.labels = [get_label_for_alg(a) for a in args.algorithms]

    args.algorithms = None
    args.labels = None
    # If we only want to report a subset of the datasets (useful if )

    if args.labels is not None and args.algorithms is None:
        parser.error("--algorithm_labels requires --algorithms to be specified.")
    if args.labels is not None and len(args.labels) != len(args.algorithms):
        parser.error(
            "--algorithm_labels must have the same number of entries as --algorithms."
        )

    # Run the evaluation script to compute the CSVs
    if args.run_evaluation:
        cmd = ["rosrun", "ov_eval", "error_comparison"]
        cmd_args = [
            args.align_mode,
            args.folder_gt,
            args.folder_algorithms,
            "--no-plots" if args.no_plots else "",
        ]
        cmd.extend(cmd_args)

        # Run with subprocess
        logging.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)