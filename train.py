from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf

from renaissance.kinetics.jacobian_solver import check_jacobian

from helpers.ppo_agent import PPOAgent
from helpers.env import BatchKineticEnv
from helpers.utils import reward_func, load_pkl, batch_reward_func

import numpy as np

import logging
import wandb

@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def train(cfg: DictConfig):

    print("-" * 50)
    print(OmegaConf.to_yaml(cfg))  # print config to verify
    print("-" * 50)

    # Call solvers from SKimPy
    chk_jcbn = check_jacobian()
    logging.disable(logging.CRITICAL)

    # Integrate data
    print("FYI: Loading kinetic and thermodynamic data.")
    chk_jcbn._load_ktmodels(cfg.paths.met_model_name, 'fdp1') # Load kinetic and thermodynamic data
    chk_jcbn._load_ssprofile(cfg.paths.met_model_name, 'fdp1', cfg.constraints.ss_idx) # Integrate steady state information

    # Logger setup, todo: for now disabled, else we would get w&b run object
    logger = None # get_logger(cfg)

    # Initialize environment
    names_km = load_pkl(cfg.paths.names_km)
    reward_fn = partial(batch_reward_func, chk_jcbn, names_km, cfg.reward.eig_partition)
    env = BatchKineticEnv(cfg, reward_fn, batch_size=cfg.training.batch_size)
    env.seed(cfg.seed)

    # Initialize PPO agent (actor and critic)
    ppo_agent = PPOAgent(cfg, logger)

    # Optionally initialize wandb here if you want logging
    run = wandb.init(project="rl-gen-kinetic-models", config=OmegaConf.to_container(cfg, resolve=True))
   
    # Training loop
    for episode in range(cfg.training.num_episodes):
        # Collect a batch of trajectories in parallel (vectorized)
        trajectories = ppo_agent.collect_trajectories(env, 1)  # 1 batch env, batch_size inside env
        # Get the last step rewards for all trajectories
        rewards = []
        for trajectory in trajectories:
            # Get the last step reward for each trajectory
            rewards.append(trajectory["rewards"][-1].cpu().numpy())
        # Convert rewards to numpy array
        rewards = np.array(rewards)
        # Calculate min, max, mean rewards for the last step for all trajectories
        min_rew, max_rew, mean_rew = rewards.min(), rewards.max(), rewards.mean()
        print(f"Episode {episode+1}/{cfg.training.num_episodes} - Min reward: {min_rew:.4f}, Max reward: {max_rew:.4f}, Mean reward: {mean_rew:.4f}")
        policy_loss, value_loss, entropy = ppo_agent.update(trajectories)
        print(f"Episode {episode+1}/{cfg.training.num_episodes} - Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}, Entropy: {entropy:.4f}")
        # Log to wandb
        wandb.log({
            "episode": episode + 1,
            "min_reward": float(min_rew),
            "max_reward": float(max_rew),
            "mean_reward": float(mean_rew),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy": float(entropy),
        })
    wandb.finish()


if __name__ == "__main__":
    train()
