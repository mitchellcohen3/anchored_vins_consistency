"""This script generates the plots for the noise experiments in the paper.

It takes in a results directory that contains pickled AveragedVinsResult objects
for each algorithm/dataset combinations, and generates the plots for the paper.

NOTE: This script generates identical plots to evaluat_vins_noise_experiments.py!
"""

import argparse
import numpy as np
import logging

import time
import argparse
import os

import re

import pickle
from pyvins.file_utils import get_files_with_extension
from pyvins.processing import get_label_for_alg
from pyvins.plot_utils import set_plot_theme

from pyvins.noise_experiment_plots import (
    generate_sensitivty_comparison_plots,
    generate_paper_plot,
)

import matplotlib.pyplot as plt

colors = set_plot_theme("colorblind", enable_latex=True)

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots for the noise experiments."
    )
    parser.add_argument("--results_dir", type=str)

    args = parser.parse_args()

    args.results_dir = "/home/mitchell/Documents/paper_results/camnoise_experiment/ov_featrep_sim_camnoise_final/results"

    # Get all pickle files in this directory
    result_files = get_files_with_extension(args.results_dir, ".pkl")

    logging.info(f"Found {len(result_files)} result files in {args.results_dir}")

    algorithms = [
        "none_global3d",
        "fej_global3d",
        "dri_fej_global3d",
        "none_anchoredmsckfinversedepth",
        "fej_anchoredmsckfinversedepth",
        "dri_fej_anchoredmsckfinversedepth",
    ]

    labels = [get_label_for_alg(alg) for alg in algorithms]

    dataset = "sim_udel_gore"
    results_dict = {}
    for result_file in result_files:
        with open(result_file, "rb") as f:
            result = pickle.load(f)

        # Get the algorithm and dataset from the filename
        if dataset not in result_file:
            continue

        # Get the algorithm name from the filename
        fname = os.path.basename(result_file)
        alg_and_noise = re.match(r".*?sigma[\d.]+", fname).group()
        alg, noise = alg_and_noise.split("__sigma")
        logging.info(f"Algorithm: {alg}, Noise: {noise}")

        label = get_label_for_alg(alg)
        key = (label, float(noise))
        results_dict[key] = result

    # Generate the plots for the paper
    generate_sensitivty_comparison_plots(
        results_dict,
        algorithm_order=labels,
        save_dir=args.results_dir,
        xlabel=r"$\sigma$ (pixels)",
        log_scale=False,
    )

    generate_paper_plot(
        results_dict, 
        algorithm_order=labels,
        colors=colors,
        noise_levels_plot=[1.0, 4.0],
        save_dir=args.results_dir,
    )
    plt.show()