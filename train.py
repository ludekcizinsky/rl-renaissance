from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf

from renaissance.kinetics.jacobian_solver import check_jacobian

from helpers.ppo_agent import PPOAgent
from helpers.env import KineticEnv
from helpers.utils import reward_func, load_pkl

import logging
import numpy as np
import os

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
    reward_fn = partial(reward_func, chk_jcbn, names_km, cfg.reward.eig_partition)
    env = KineticEnv(cfg, reward_fn)
    env.seed(cfg.seed)

    # Initialize PPO agent (actor and critic)
    ppo_agent = PPOAgent(cfg, logger)

    best_reward = float('-inf')
    best_actor_path = f"{cfg.paths.output_dir}/run_0/best_actor.pth"
    best_critic_path = f"{cfg.paths.output_dir}/run_0/best_critic.pth"
    import os
    save_dir = os.path.join(cfg.paths.output_dir, "run_0")
    os.makedirs(save_dir, exist_ok=True)
    collect_rewards = []
    collect_policy_loss = []
    collect_value_loss = []
    # Training loop
    for episode in range(cfg.training.num_episodes):
        trajectory = ppo_agent.collect_trajectory(env)
        rewards = trajectory["rewards"]
        min_rew, max_rew, mean_rew = rewards.min(), rewards.max(), rewards.mean()
        last_reward = rewards[-1]
        print(f"Episode {episode+1}/{cfg.training.num_episodes} - Min reward: {min_rew:.4f}, Max reward: {max_rew:.4f}, Mean reward: {mean_rew:.4f}, Last reward: {last_reward:.4f}")

        policy_loss, value_loss, entropy = ppo_agent.update(trajectory)
        print(f"Episode {episode+1}/{cfg.training.num_episodes} - Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}, Entropy: {entropy:.4f}")

        if last_reward > best_reward:
            best_reward = last_reward
            import os, torch
            os.makedirs(cfg.paths.output_dir, exist_ok=True)
            torch.save(ppo_agent.policy_net.state_dict(), best_actor_path)
            torch.save(ppo_agent.value_net.state_dict(), best_critic_path)
            print(f"Best model saved at episode {episode+1} with mean reward {best_reward:.4f}")
            # Save the training data
            np.save(os.path.join(save_dir, "rewards.npy"), np.array(collect_rewards))
            np.save(os.path.join(save_dir, "policy_loss.npy"), np.array(collect_policy_loss))
            np.save(os.path.join(save_dir, "value_loss.npy"), np.array(collect_value_loss))
        collect_rewards.append(rewards.numpy())
        collect_policy_loss.append(policy_loss)
        collect_value_loss.append(value_loss)

    # Save the training data
    np.save(os.path.join(save_dir, "rewards.npy"), np.array(collect_rewards))
    np.save(os.path.join(save_dir, "policy_loss.npy"), np.array(collect_policy_loss))
    np.save(os.path.join(save_dir, "value_loss.npy"), np.array(collect_value_loss))
    # save the config file
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))



if __name__ == "__main__":
    train()


