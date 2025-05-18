from functools import partial
import os
import sys
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from renaissance.kinetics.jacobian_solver import check_jacobian
from helpers.ppo_agent import PPOAgent
from helpers.env import KineticEnv
from helpers.utils import reward_func, load_pkl

import logging

@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def train(cfg: DictConfig):
    print("-" * 50)
    print("CONFIG AT START OF TRAIN:")
    print(OmegaConf.to_yaml(cfg))
    print("-" * 50)

    # Optionally initialize wandb here if you want logging
    run = wandb.init(project="rl-gen-kinetic-models", config=OmegaConf.to_container(cfg, resolve=True))

    # Call solvers from SKimPy
    chk_jcbn = check_jacobian()
    logging.disable(logging.CRITICAL)

    # Integrate data
    print("FYI: Loading kinetic and thermodynamic data.")
    chk_jcbn._load_ktmodels(cfg.paths.met_model_name, 'fdp1')
    chk_jcbn._load_ssprofile(cfg.paths.met_model_name, 'fdp1', cfg.constraints.ss_idx)

    logger = None  # or get_logger(cfg) if you have a logger

    # Initialize environment
    names_km = load_pkl(cfg.paths.names_km)
    reward_fn = partial(reward_func, chk_jcbn, names_km, cfg.reward.eig_partition)
    env = KineticEnv(cfg, reward_fn)
    env.seed(cfg.seed)

    # Initialize PPO agent
    ppo_agent = PPOAgent(cfg, logger)

    # Training loop
    for episode in range(cfg.training.num_episodes):
        trajectory = ppo_agent.collect_trajectory(env)
        rewards = trajectory["rewards"]
        min_rew, max_rew, mean_rew = rewards.min(), rewards.max(), rewards.mean()
        print(f"Episode {episode+1}/{cfg.training.num_episodes} - Min reward: {min_rew:.4f}, Max reward: {max_rew:.4f}, Mean reward: {mean_rew:.4f}")

        policy_loss, value_loss, entropy = ppo_agent.update(trajectory)
        print(f"Episode {episode+1}/{cfg.training.num_episodes} - Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}, Entropy: {entropy:.4f}")

        # Optionally log to wandb
        wandb.log({
            "episode": episode + 1,
            "min_reward": min_rew,
            "max_reward": max_rew,
            "mean_reward": mean_rew,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        })

    wandb.finish()

def agent_sweep():
    """
    This function is called by wandb.agent for each sweep run.
    It injects sweep parameters as Hydra overrides before Hydra builds the config.
    """
    if os.environ.get('WANDB_SWEEP_ID') is not None:
        # Remove any previous overrides to avoid duplicates
        sys.argv = sys.argv[:1]
        # Add sweep parameters as command-line overrides
        for param_path, param_value in wandb.config.items():
            if param_path.startswith('_'):
                continue
            sys.argv.append(f"{param_path}={param_value}")
            print(f"Hydra override: {param_path}={param_value}")
    train()

def create_and_run_sweep(num_agents=5):
    """
    Create a sweep and run the specified number of agents.
    """
    from register_sweep_params import SweepRegistry
    sweep_config = SweepRegistry.get_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="rl-gen-kinetic-models")
    wandb.agent(sweep_id, function=agent_sweep, count=num_agents)

if __name__ == "__main__":
    train()
