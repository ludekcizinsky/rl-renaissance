from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf

from renaissance.kinetics.jacobian_solver import check_jacobian

from helpers.ppo_agent import PPOAgent
from helpers.env import KineticEnv
from helpers.utils import reward_func, load_pkl
from helpers.logger import get_wandb_run

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
    run = get_wandb_run(cfg)

    # Initialize environment
    names_km = load_pkl(cfg.paths.names_km)
    reward_fn = partial(reward_func, chk_jcbn, names_km, cfg.reward.eig_partition)
    env = KineticEnv(cfg, reward_fn)
    env.seed(cfg.seed)

    # Initialize PPO agent (actor and critic)
    ppo_agent = PPOAgent(cfg, run)
   
    # Training loop
    for episode in range(cfg.training.num_episodes):
        trajectory = ppo_agent.collect_trajectory(env)
        rewards = trajectory["rewards"]
        min_rew, max_rew, mean_rew = rewards.min(), rewards.max(), rewards.mean()
        run.log({"reward/min_rew": min_rew, "reward/max_rew": max_rew, "reward/mean_rew": mean_rew, "episode": episode})

        policy_loss, value_loss = ppo_agent.update(trajectory)
        run.log({"ppo/policy_loss": policy_loss, "ppo/value_loss": value_loss, "episode": episode})


if __name__ == "__main__":
    train()
