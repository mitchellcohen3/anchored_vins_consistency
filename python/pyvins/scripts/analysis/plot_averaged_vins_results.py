"""Loads averaged VINS results from pickled files and generates comparison plots.

These should all be on the same dataset.
""" 

import numpy as np
import re
import argparse
import os
import logging

import matplotlib.pyplot as plt

from pyvins.plot_utils import set_plot_theme
from pyvins.file_utils import get_files_with_extension
from pyvins.aggregation import AveragedVinsResult, compare_averaged_vins_results
from pyvins.processing import   get_label_for_alg

import pickle

logging.basicConfig(level=logging.INFO)

colors = set_plot_theme("colorblind", enable_latex=True)

if __name__ == "__main__":
    results_dir = "/home/mitchell/experiments/ov_exp_featrep_sim_camnoise_20260303_103037/results"
    result_files = get_files_with_extension(results_dir, ".pkl")
    logging.info(f"Found {len(result_files)} result files in {results_dir}")

    dataset_name = "sim_udel_gore"

    algorithms = [
        "none_global3d",
        "fej_global3d",
        "dri_fej_global3d",
        "none_anchoredmsckfinversedepth",
        "fej_anchoredmsckfinversedepth",
        "dri_fej_anchoredmsckfinversedepth",
    ]

    labels = [get_label_for_alg(alg) for alg in algorithms]


    traj_results = []

    for alg in algorithms:
        result_file = os.path.join(results_dir, f"{alg}__sigma1.0_{dataset_name}_averaged_result.pkl")

        # Check if the result file exists
        if not os.path.isfile(result_file):
            logging.warning(f"Result file {result_file} not found, skipping algorithm {alg}")
            continue
        
        with open(result_file, "rb") as f:
            traj_result = pickle.load(f)
        traj_results.append(traj_result)

    # Compare the results and generate plots
    compare_averaged_vins_results(
        traj_results,
        labels,
        colors,
    )
    plt.show()
