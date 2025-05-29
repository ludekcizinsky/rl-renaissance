#!/bin/bash
#SBATCH --job-name=b2_rl_exp
#SBATCH --output=outputs/slurm/%x.%j.out
#SBATCH --error=outputs/slurm/%x.%j.err
#SBATCH --time=09:00:00
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

# Baseline
# python train.py 'logger.tags=[part2, baseline]'

# Batch of ablations
# 1. lr scheduler (default constant)
# python train.py lr_scheduler=cosine 'logger.tags=[part2, lr_scheduler, cosine]'
# python train.py lr_scheduler=linear_decay 'logger.tags=[part2, lr_scheduler, linear_decay]'

# 2. action scale (default 1.0)
# python train.py env.action_scale=0.5 'logger.tags=[part2, action_scale]'
# python train.py env.action_scale=1.5 'logger.tags=[part2, action_scale]'

# 3. PPO epochs per update (default 10)
# python train.py training.num_epochs=4 'logger.tags=[part2, num_epochs]'
# python train.py training.num_epochs=6 'logger.tags=[part2, num_epochs]'

# 4. Gradient norm clipping (default 0.5)
# python train.py training.max_grad_norm=0.25 'logger.tags=[part2, max_grad_norm]'
# python train.py training.max_grad_norm=1.0 'logger.tags=[part2, max_grad_norm]'

# 5. Actor learning rate (default 1e-4)
# python train.py method.actor_lr=5e-5 'logger.tags=[part2, lr_actor]'
# python train.py method.actor_lr=2e-4 'logger.tags=[part2, lr_actor]'

# 6. Critic learning rate (default 1e-3)
python train.py method.critic_lr=5e-4 'logger.tags=[part2, lr_critic]'
python train.py method.critic_lr=2e-3 'logger.tags=[part2, lr_critic]'

# 7. Clip epsilon (default 0.2)
python train.py method.clip_eps=0.1 'logger.tags=[part2, clip_eps]'
python train.py method.clip_eps=0.25 'logger.tags=[part2, clip_eps]'

# 8. Entropy loss weight (default 0.01)
#python train.py method.entropy_loss_weight=0.005 'logger.tags=[part2, entropy_loss_weight]'
#python train.py method.entropy_loss_weight=0.02 'logger.tags=[part2, entropy_loss_weight]'

# 9. Log std bounds (default [-6, 2])
python train.py method.min_log_std=-8 'logger.tags=[part2, log_std_bounds]'
python train.py method.min_log_std=-4 'logger.tags=[part2, log_std_bounds]'
python train.py method.max_log_std=3 'logger.tags=[part2, log_std_bounds]'

# 10. Number of episodes (default 100)
python train.py training.num_episodes=30 'logger.tags=[part2, num_episodes]'
python train.py training.num_episodes=60 'logger.tags=[part2, num_episodes]'
EOF
