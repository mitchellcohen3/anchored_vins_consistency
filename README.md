# Anchored VINS Consistency Analysis #
This repository contains the companion code to reproduce the figures and tables of our paper: "[Observability and Consistency Analysis for Visual-Inertial Navigation with Anchored Feature Parameterizations](https://arxiv.org/pdf/2606.19307)".

This repo is a fork of [OpenVINS](https://github.com/rpng/open_vins) and its main purpose is to empirically validate the observability and consistency results derived in the paper. It additionally provides an implementation of the Decoupled Right-Invariant EKF (DRI), based on **Yang et al. (2022) - "Decoupled Right-Invariant Error States for Consistent Visual-Inertial Navigation"**. To use the DRI-EKF, set the parameter `consistency_method: "dri_fej"` in the config file, which will enable the right-invariant error for the inertial state and pose clones, and utilize the forms of the right-invariant state transition matrix and measurement model Jacobians from Yang et al. (2022). 

## Installation
Follow the [OpenVINS installation guide](https://docs.openvins.com/gs-installing.html) to install the dependencies. This code was tested with **Ubuntu 20.04** and **ROS Noetic**, using Eigen 3.4.0, OpenCV 4.2.0, and Ceres 2.0.0, and can be built by running
```bash
mkdir -p ~/catkin_ws_anchored_vins/src  
cd ~/catkin_ws_anchored_vins/src 
git clone git@github.com:mitchellcohen3/anchored_vins_consistency.git
cd ~/catkin_ws_anchored_vins
catkin build -j4
```

## Reproducing paper results
The scripts to reproduce the figures and tables in the paper are located in the `scripts` folder, which additionally contains a few scripts that may be useful for analyzing the output of VINS estimators. The Python evaluation scripts use the Python package `pyvins`, located in the folder `python/pyvins`. Install the package in a virtual environment using
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e python/pyvins
```
which will additionally install all dependencies required to run the evaluation scripts.

To reproduce the results from the simulation experiment, run the script `scripts/sim_featrep_experiments/run_camnoise_experiment.sh`, which runs Monte-Carlo trials across different camera noise levels for each estimator, on all three datasets. The script will also generate plots of the results in the folder `results/plots`, used to compare the performance of the different estimators across camera noise levels (Figures I and II from the paper). See also the script `scripts/generate_ate_comparison_table.py` to generate Table II from the paper.

The script `scripts/tumvi_exp/run_tumvi_experiment.sh` is used to run the TUM-VI dataset experiments. Download the `room` sequences ROS bags (in 512x512 resolution) along with the groundtruth files [here](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset), and place them in folder with the structure:
```
dataset_path/
├── dataset-room1_512_16.bag
├── dataset-room2_512_16.bag
...
├── truths
    ├── dataset-room1_512_16.txt
    ├── dataset-room2_512_16.txt
    ....
```
See the script `scripts/tumvi_exp/evaluate_tum_vi_experiment.sh` to generate the evaluation plots.

## Citation
If you found this repository useful for your research, please cite it as below:
```bibtex
@article{cohen2026observability,
  title={Observability and Consistency Analysis for Visual-Inertial Navigation with Anchored Feature Parameterizations},
  author={Cohen, Mitchell and Korotkine, Vassili and Forbes, James Richard},
  journal={arXiv preprint arXiv:2606.19307},
  year={2026}
}

```
## Acknowledgments and License
This project is built on [OpenVINS](https://github.com/rpng/open_vins), released under GPLv3.
