import sys
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf

from helpers.jacobian_solver import check_jacobian

from helpers.ppo_agent import PPOAgent
from helpers.env import BatchKineticEnv, KineticEnv
from helpers.utils import reward_func, load_pkl, batch_reward_func, log_rl_models, log_reward_distribution
from helpers.logger import get_wandb_run
import numpy as np

import logging

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

    # Logger setup
    cfg.launch_cmd = " ".join(sys.argv)
    run = get_wandb_run(cfg)

    # Initialize environment
    names_km = load_pkl(cfg.paths.names_km)
    reward_fn = partial(batch_reward_func, chk_jcbn, names_km, cfg.reward.eig_partition)
    env = BatchKineticEnv(cfg, reward_fn, batch_size=cfg.training.batch_size)
    env.seed(cfg.seed)

    # Initialize PPO agent (actor and critic)
    ppo_agent = PPOAgent(cfg, run)
   
    # Training loop
    try:
        for episode in range(cfg.training.num_episodes):
            # Collect trajectory
            trajectories = ppo_agent.collect_trajectories(env, episode)
            rewards = trajectories["rewards"]
            rewards = rewards.cpu().numpy().mean(axis=1)

        print("Mean episode rewards over steps: ", rewards)
        log_reward_distribution(rewards, episode)

        # Update PPO agent
        ppo_agent.update(trajectories)


        # Log models
        if cfg.training.save_trained_models:
            log_rl_models(ppo_agent.policy_net, ppo_agent.value_net, save_dir=cfg.paths.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Traceback: {e.__traceback__}")
        print("-" * 50)

    
    # Finish wandb run
    run.finish()

    run.finish()

if __name__ == "__main__":
    train()
