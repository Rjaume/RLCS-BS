#!/bin/bash
#SBATCH --job-name=example_training
#SBATCH --output=output_training
#SBATCH --nodes=1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=general

parameters=$( cat parameters.txt)

exec=$( cat executable_path.txt)

$exec $parameters

