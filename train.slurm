#!/bin/bash
#SBATCH --job-name=rl_exp
#SBATCH --output=outputs/slurm/%x.%j.out
#SBATCH --error=outputs/slurm/%x.%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
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

# Batch of experiments
# 1. lr scheduler (default constant)
#python train.py lr_scheduler=cosine 'logger.tags=[lr_scheduler, cosine]'
#python train.py lr_scheduler=linear_decay 'logger.tags=[lr_scheduler, linear_decay]'

# 2. action scale (default 1.0)
python train.py env.action_scale=0.25 'logger.tags=[action_scale]'
python train.py env.action_scale=0.5 'logger.tags=[action_scale]'
python train.py env.action_scale=0.75 'logger.tags=[action_scale]'

# 3. number of epochs (default 10)
python train.py training.num_epochs=5 'logger.tags=[num_epochs]'

# 4. max grad norm (default 0.5)
python train.py training.max_grad_norm=0.25 'logger.tags=[max_grad_norm]'
python train.py training.max_grad_norm=0.75 'logger.tags=[max_grad_norm]'
python train.py training.max_grad_norm=1.0 'logger.tags=[max_grad_norm]'

# 5. lr actor (default 3e-4)
python train.py method.actor_lr=1e-4 'logger.tags=[lr_actor]'
python train.py method.actor_lr=1e-5 'logger.tags=[lr_actor]'

# 6. lr critic (default 1e-3)
python train.py method.critic_lr=5e-4 'logger.tags=[lr_critic]'
python train.py method.critic_lr=1e-4 'logger.tags=[lr_critic]'

# 7. clip eps (default 0.2)
python train.py method.clip_eps=0.1 'logger.tags=[clip_eps]'

# 8. Value loss weight (default 0.5)
python train.py method.value_loss_weight=0.25 'logger.tags=[value_loss_weight]'
python train.py method.value_loss_weight=0.75 'logger.tags=[value_loss_weight]'
EOF
