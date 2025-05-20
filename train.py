import sys
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf

from helpers.jacobian_solver import check_jacobian

from helpers.ppo_agent import PPOAgent
from helpers.env import KineticEnv
from helpers.utils import reward_func, load_pkl, log_rl_models, log_reward_distribution
from helpers.logger import get_wandb_run

import logging

import traceback

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
    reward_fn = partial(reward_func, chk_jcbn, names_km, cfg.reward.eig_partition)
    env = KineticEnv(cfg, reward_fn)
    env.seed(cfg.seed)

    # Initialize PPO agent (actor and critic)
    ppo_agent = PPOAgent(cfg, run)
   
    # Training loop
    try:
        for episode in range(cfg.training.num_episodes):
            # Collect trajectory
            trajectory = ppo_agent.collect_trajectory(env, episode)
            rewards = trajectory["rewards"]
            log_reward_distribution(rewards, episode)

            # Update PPO agent
            ppo_agent.update(trajectory)

        # Log models
        if cfg.training.save_trained_models:
            policy_net_dict, value_net_dict = ppo_agent.global_best_model
            log_rl_models(policy_net_dict, value_net_dict, save_dir=cfg.paths.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print("-" * 50)

    
    # Finish wandb run
    run.finish()

if __name__ == "__main__":
    train()
