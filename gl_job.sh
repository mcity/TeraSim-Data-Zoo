#!/bin/bash

#SBATCH --job-name=traffic_lights
#SBATCH --mail-user=erdao@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5g
#SBATCH --array=1000-1149  # How many workers to run in parallel
#SBATCH --time=00-03:00:00
#SBATCH --account=henryliu98
#SBATCH --partition=standard
#SBATCH --output=/home/erdao/WOMDScenParse/slurm-log/traffic_lights/%x-%j.log

cd /home/erdao/WOMDScenParse

# module purge
# module load python3.10-anaconda/2023.03

srun python3 scripts/hpc.py $SLURM_ARRAY_TASK_ID