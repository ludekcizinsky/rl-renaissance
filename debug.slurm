#!/bin/bash
#SBATCH --job-name=debug1
#SBATCH --output=outputs/slurm/%x.%j.out
#SBATCH --error=outputs/slurm/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account=master

USER=$1
cd /home/$USER/rl-gen-of-kinetic-models/

# make sure the output directory exists
mkdir -p /scratch/izar/$USER/rl-for-kinetics/outputs
mkdir -p /scratch/izar/$USER/rl-for-kinetics/outputs/slurm

apptainer shell \
  --nv \
  --bind "$(pwd)":/home/renaissance/work \
  --bind "/scratch/izar/$USER/rl-for-kinetics/output":/home/renaissance/output \
  /scratch/izar/$USER/images/renaissance_with_ml.sif << 'EOF'
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
nvidia-smi

# Debugs
# python train.py method.critic_lr=1e-3 training.num_episodes=40 'logger.tags=[debug, critic_lr]'
# python train.py method.clip_eps=0.2 training.num_episodes=40 'logger.tags=[debug, clip_eps]'
# python train.py method.max_log_std=3 training.num_episodes=40 'logger.tags=[debug, max_log_std]'

# python train.py method.actor_lr=2e-4 method.critic_lr=1e-3 method.clip_eps=0.2 method.min_log_std=-6 method.max_log_std=2 method.entropy_loss_weight=0.01 training.num_episodes=100 'logger.tags=[debug, best_setup, actor_lr]'
# python train.py method.entropy_loss_weight=0.02 training.num_episodes=40 'logger.tags=[debug, entropy_loss_weight]'
# python train.py method.value_loss_weight=1.0 training.num_episodes=40 'logger.tags=[debug, value_loss_weight]'
# python train.py 'logger.tags=[debug, best_setup, clip_eps_decay]'
EOF
