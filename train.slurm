#!/bin/bash
#SBATCH --job-name=rl_base
#SBATCH --output=outputs/slurm/%x.%j.out       # Save output to logs with job ID
#SBATCH --error=outputs/slurm/%x.%j.err       # Save errors to separate file
#SBATCH --time=01:00:00                  # Max runtime (hh:mm:ss)
#SBATCH --partition=gpu                  # Partition name (adjust if needed)
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=20                # Adjust CPU count if needed
#SBATCH --mem=32G                        # RAM requested
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --account=master

USER=$1

cd /home/$USER/rl-gen-of-kinetic-models/
apptainer shell --nv --bind "$(pwd)":/home/renaissance/work --bind "/scratch/izar/$USER/rl-for-kinetics/output:/home/renaissance/output" /scratch/izar/$USER/images/renaissance_with_ml.sif
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

nvidia-smi

# Run your training script
python train.py
# python train.py 'logger.tags=[dev]'