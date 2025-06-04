#!/bin/bash

module load spack/default

module load gcc/12.4.0

module load python/3.12.1

# Activate the virtual environment
source /home/tfellner/RobustnessGeneralizationQML/.venv/bin/activate

python3 ./utils/submit_multiple_jobs.py -slurm "yes"