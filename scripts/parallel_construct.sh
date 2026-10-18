#!/bin/bash
#SBATCH --no-requeue
#SBATCH --get-user-env
#SBATCH --output=_scheduler-stdout.txt
#SBATCH --error=_scheduler-stderr.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1920
#SBATCH --partition=batch

set -e

export NUMBA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

conda run -n jobflow --no-capture-output \
    python build_cubic_supercells_parallel.py
