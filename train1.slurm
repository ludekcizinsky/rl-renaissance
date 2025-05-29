#!/bin/bash
#SBATCH --job-name=final1_rl_exp
#SBATCH --output=outputs/slurm/%x.%j.out
#SBATCH --error=outputs/slurm/%x.%j.err
#SBATCH --time=09:30:00
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

# Batch of ablations
# 1. lr scheduler (default constant)
python train.py lr_scheduler=cosine 'logger.tags=[final, lr_scheduler, cosine]'
python train.py lr_scheduler=linear_decay 'logger.tags=[final, lr_scheduler, linear_decay]'

# 2. PPO epochs per update (default 10)
python train.py training.num_epochs=3 'logger.tags=[final, num_epochs]'
python train.py training.num_epochs=6 'logger.tags=[final, num_epochs]'

# 3. Gradient norm clipping (default 0.5)
python train.py training.max_grad_norm=0.25 'logger.tags=[final, max_grad_norm]'
python train.py training.max_grad_norm=1.0 'logger.tags=[final, max_grad_norm]'

# 4. Entropy loss weight (default 0.01)
python train.py method.entropy_loss_weight=0.005 'logger.tags=[final, entropy_loss_weight]'
python train.py method.entropy_loss_weight=0.02 'logger.tags=[final, entropy_loss_weight]'

# 5. Action scale (default 1.0)
python train.py env.action_scale=0.25 'logger.tags=[final, action_scale]'
python train.py env.action_scale=0.5 'logger.tags=[final, action_scale]'
EOF
