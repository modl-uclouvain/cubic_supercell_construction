#!/bin/bash
#SBATCH --no-requeue
#SBATCH --get-user-env
#SBATCH --output=_scheduler-stdout.txt
#SBATCH --error=_scheduler-stderr.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1920
#SBATCH --partition=debug

set -e
export OMP_NUM_THREADS=1

conda run -n jobflow --no-capture-output python build_cubic_supercells.py

 
