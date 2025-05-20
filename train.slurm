#!/bin/bash
#SBATCH --job-name=rl_base
#SBATCH --output=outputs/slurm/%x.%j.out
#SBATCH --error=outputs/slurm/%x.%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account=master

USER=$1
cd /home/$USER/rl-gen-of-kinetic-models/

apptainer shell \
  --nv \
  --bind "$(pwd)":/home/renaissance/work \
  --bind "/scratch/izar/$USER/rl-for-kinetics/output":/home/renaissance/output \
  /scratch/izar/$USER/images/renaissance_with_ml.sif << 'EOF'
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
nvidia-smi
python train.py training.num_episodes=10 'logger.tags=[dev]'
EOF
