#!/bin/bash

#SBATCH --job-name=str_models7            # Name of the job
#SBATCH --partition=icelake              # Name of the partition you want to submit to
#SBATCH --account=BENDER-SL3-CPU         # Account/Project name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks per node
#SBATCH --cpus-per-task=76               # Number of CPUs per task
#SBATCH --time=11:59:00                  # Time limit hrs:min:sec (set the "wall time")
#SBATCH --output=str_models7_%j.out
#SBATCH --error=str_models7_%j.err

# Load necessary modules (if any, e.g., Python, GCC, etc.)
# module load python/3.8

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate my-rdkit-env

# Run your Python script
python ../06_LINCSL1000_models/LINCSL1000_models_v2.py

