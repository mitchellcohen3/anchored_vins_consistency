"""This script loads state information from an OpenVINS file and plots it."""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pyvins.containers import VinsStateInfo
from pyvins.plot_utils import set_plot_theme

import logging

logging.basicConfig(level=logging.INFO)

set_plot_theme(enable_latex=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot VINS state information.")
    parser.add_argument(
        "--state_est_file",
        type=str,
        help="Path to the state estimate file.",
    )

    args = parser.parse_args()

    args.state_est_file = "/home/mitchell/experiments/ov_debug/state_est.txt"

    state_info = VinsStateInfo.from_file(args.state_est_file)
    state_info.plot_extrinsics()
    state_info.plot_velocities()
    state_info.plot_biases()
    state_info.plot_intrinsics()
    state_info.plot_time_offset()
    plt.show()
