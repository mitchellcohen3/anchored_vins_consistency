#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# source the caktin ws before runnign this script
source $SCRIPT_DIR/../../../../../devel/setup.bash

echo "Running OpenVINS Monte-Carlo simulations for feature representation experiments..."

RESULTS_BASEDIR="/home/mitchell/experiments/ov_tumvi_featrep_20260302_163647"
TRUTH_FOLDER=${RESULTS_BASEDIR}/truths
ALGORITHMS_FOLDER=${RESULTS_BASEDIR}/algorithms

rosrun ov_eval error_comparison \
    "posyaw" \
    ${TRUTH_FOLDER} \
    ${ALGORITHMS_FOLDER}