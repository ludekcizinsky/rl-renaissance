#!/bin/bash
#SBATCH --job-name=final2_rl_exp
#SBATCH --output=outputs/slurm/%x.%j.out
#SBATCH --error=outputs/slurm/%x.%j.err
#SBATCH --time=09:00:00
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


# EPS
python train.py method.clip_eps_kind=cosine 'logger.tags=[final, clip_eps_cosine]'
python train.py method.t_eps=50 'logger.tags=[final, clip_decay_fast]'
python train.py method.t_eps=200 'logger.tags=[final, clip_decay_slow]'
python train.py method.clip_eps_end=0.05 'logger.tags=[final, clip_eps_end_low]'
python train.py method.clip_eps_start=0.4 'logger.tags=[final, clip_eps_start_high]'

# Actor LR
python train.py method.actor_lr=5e-5 'logger.tags=[final, actor_lr_down]'
python train.py method.actor_lr=1e-5 'logger.tags=[final, actor_lr_down]'

# GAE lambda
python train.py method.gae_lambda=0.95 'logger.tags=[final, gae_lambda_down]'
python train.py method.gae_lambda=0.99 'logger.tags=[final, gae_lambda_up]'

# Value loss weight
python train.py method.value_loss_weight=0.3 'logger.tags=[final, value_loss_weight_down]'
EOF
