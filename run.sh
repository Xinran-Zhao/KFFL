#!/bin/bash
# Run this script from a GPU compute node (after: interactive -G 1)
# Usage: bash run.sh

module load mamba cuda-12.8.1-gcc-12.1.0
export CONDA_ENVS_PATH=/scratch/xzhao181/.conda/envs
source activate dl

# Fix: use conda env's libstdc++ instead of outdated system /lib64 version
# (needed for pandas and other compiled extensions requiring GLIBCXX_3.4.29)
export LD_LIBRARY_PATH=/scratch/xzhao181/.conda/envs/dl/lib:$LD_LIBRARY_PATH

cd /data/amciilab/xinran/KFFL
python test_functions.py
