import os
import pickle
import sys
import logging
from functools import partial
from typing import Any

import torch
import math
import numpy as np
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import wandb
from tqdm import tqdm

from helpers.jacobian_solver import check_jacobian
from helpers.env import KineticEnv

def get_timestep_embedding(timestep: int, embedding_dim: int, max_period: float = 10000) -> torch.Tensor:
    """
    Maps an integer timestep to a torch tensor using sinusoidal positional embeddings.

    Args:
        timestep (int): The current timestep.
        embedding_dim (int): The dimension of the embedding.
        max_period (float): The maximum period for the sinusoidal functions.
            Controls the frequency range of the sinusoidal functions. A larger value allows for more gradual 
            changes in the embeddings over time.

    Returns:
        torch.Tensor: The timestep embedding as a tensor.
    """
    half_dim = embedding_dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half_dim) / half_dim)
    args = timestep * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    if embedding_dim % 2 == 1:  # If embedding_dim is odd, pad with a zero
        embedding = torch.cat([embedding, torch.zeros(1)], dim=-1)
    
    return embedding


def get_initial_state(cfg: DictConfig) -> torch.Tensor:
    """
    Generate the initial state for the PPO algorithm.

    Args:
        cfg (DictConfig): The configuration for the PPO algorithm.

    Returns:
        torch.Tensor: The initial state for the PPO algorithm.
    """
    # Initial parameters, todo: make this deterministic
    p_curr = torch.normal(cfg.env.p0_init_mean, cfg.env.p0_init_std, size=(cfg.env.p_size,))

    # Timestep embedding
    p_curr += get_timestep_embedding(0, cfg.env.p_size)

    # Bound to [min_km, max_km]
    p_curr = torch.clamp(p_curr, cfg.constraints.min_km, cfg.constraints.max_km)

    return p_curr


def reward_func(chk_jcbn, names_km, eig_partition: float, gen_kinetic_params: torch.Tensor):
    """
    Calculate the reward for a 1D tensor of kinetic parameters.
    """

    # Ensure that the kinetic parameters are in the correct format
    gen_kinetic_params = gen_kinetic_params.detach().cpu().numpy()

    # For some reason, we need to convert the kinetic parameters to a pandas dataframe
    chk_jcbn._prepare_parameters([gen_kinetic_params], names_km)

    # Calculate the maximum eigenvalue of the Jacobian
    all_eigenvalues = chk_jcbn.calc_eigenvalues_recal_vmax()[0]
    max_eig = np.max(all_eigenvalues)

    # Calculate the reward
    # TODO: this is somewhat adapted from the original Renaissance code
    # but needs further investigation
    # reward = 0.01 / (1 + np.exp(max_eig - eig_partition))
    z = np.clip(max_eig - eig_partition, -20, +20)
    reward = 1.0 / (1.0 + np.exp(z)) + 1e-3  # now ∈ (0,1)

    return reward, all_eigenvalues


def load_pkl(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pkl(path: str, obj: Any):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def compute_grad_norm(model, norm_type: float = 2.0) -> float:
    """
    Compute the total gradient norm over all model parameters.
    
    Args:
        model (torch.nn.Module): your model
        norm_type (float): the p‐norm to use (default: 2)
    
    Returns:
        float: the total norm (as a Python float)
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def log_max_eig_dist_and_incidence_rate(max_eig_values, was_valid_solution, episode: int):


    data = np.asarray(max_eig_values)

    # 1) Set up figure & axis
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)

    # 2) Fit KDE
    kde = gaussian_kde(data, bw_method=0.2)

    # 3) Build evaluation grid
    x_min, x_max = data.min() - 1, data.max() + 1
    x = np.linspace(x_min, x_max, 500)
    y = kde(x)

    # 4) Plot
    ax.plot(x, y, lw=2)
    ax.fill_between(x, y, alpha=0.3)
    ax.set_xlabel("max eigenvalue")
    ax.set_ylabel("density")
    ax.set_title("Smoothed density of max eigenvalue")
    fig.tight_layout()


    incidence_rate = sum(was_valid_solution) / len(was_valid_solution)

    wandb.log({
        "reward/max_eig_dist": wandb.Image(fig), 
        "reward/incidence_rate": incidence_rate,
        "episode": episode
    })
    plt.close(fig)


def log_reward_distribution(rewards, episode: int):

    data = np.asarray(rewards)

    try:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=100)

        kde = gaussian_kde(data, bw_method=0.2)

        x_min, x_max = data.min() - 1, data.max() + 1
        x = np.linspace(x_min, x_max, 500)
        y = kde(x)

        ax.plot(x, y, lw=2)
        ax.fill_between(x, y, alpha=0.3)
        ax.set_xlabel("reward")
        ax.set_ylabel("density")
        ax.set_title("Smoothed density of reward")
        fig.tight_layout()
    except Exception as e:
        print(f"FYI: Error plotting reward distribution: {e}, using empty plot instead.")
        fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
        ax.set_title("Empty plot due to error computing KDE.")
        fig.tight_layout()


    # Calculate reward statistics
    reward_mean = np.mean(data)
    reward_std = np.std(data)
    reward_max = np.max(data)
    wandb.log({
        "reward/distribution": wandb.Image(fig), 
        "reward/mean": reward_mean,
        "reward/std": reward_std,
        "reward/max": reward_max,
        "episode": episode
    })
    plt.close(fig)


def log_rl_models(
    policy_net_dict: dict,
    value_net_dict: dict,
    best_setup, 
    first_valid_setup,
    save_dir:      str = ".",
):
    """
    Logs policy and value networks to W&B as a versioned Artifact.

    Args:
        policy_net:     Trained policy network (torch.nn.Module).
        value_net:      Trained value network (torch.nn.Module).
        description:    Artifact description.
        save_dir:       Directory where to save temporary .pt files.
    """

    # Unpack best setups
    best_mean, best_std, best_state, best_step, best_episode = best_setup
    if first_valid_setup is not None:
        first_valid_mean, first_valid_std, first_valid_state, first_valid_step, first_valid_episode = first_valid_setup

    # Prepare file paths
    run_name = wandb.run.name
    save_dir = os.path.join(save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    policy_path = os.path.join(save_dir, "policy.pt")
    value_path  = os.path.join(save_dir, "value.pt")
    best_path   = os.path.join(save_dir, f"best_setup_e{best_episode}_s{best_step}.pt")
    if first_valid_setup is not None:
        first_valid_path = os.path.join(save_dir, f"first_valid_setup_e{first_valid_episode}_s{first_valid_step}.pt")

    # Save state_dicts
    torch.save(policy_net_dict, policy_path)
    torch.save(value_net_dict,  value_path)

    # Save best setup
    best_bundle = {
        "best_state": best_state,
        "best_mean": best_mean,
        "best_std": best_std,
    }
    torch.save(best_bundle, best_path)

    # Save first valid setup
    if first_valid_setup is not None:
        first_valid_bundle = {
            "first_valid_state": first_valid_state,
            "first_valid_mean": first_valid_mean,
            "first_valid_std": first_valid_std,
        }
        torch.save(first_valid_bundle, first_valid_path)

    # Build and log the Artifact
    artifact = wandb.Artifact(
        name=run_name,
        type="model",
        description="Trained policy and value networks + best setup"
    )
    artifact.add_file(policy_path)
    artifact.add_file(value_path)
    artifact.add_file(best_path)
    if first_valid_setup is not None:
        artifact.add_file(first_valid_path)
    wandb.log_artifact(artifact)
    wandb.log_artifact(artifact, aliases=["best"])

    print(f"FYI: Logged models and best setup to W&B as {run_name}.")


def evaluate_best_setup(env, state, dist, n_samples):

    all_max_eigs = []
    is_valid_solution = []

    state = state.to("cpu")
    for _ in tqdm(range(n_samples), desc="Evaluating best setup"):
        # sample action and get next state accordingly
        action = dist.rsample()
        action = action.to("cpu")
        action = env.action_scale * action
        next_state = (state + action).clamp(min=env.min_val, max=env.max_val)

        # compute max eigenvalue
        _, all_eigenvalues = env.reward_fn(next_state)
        max_eig = np.max(all_eigenvalues)
        all_max_eigs.append(max_eig)
        is_valid_solution.append(max_eig < env.eig_cutoff)

    return all_max_eigs, is_valid_solution


def evaluate_and_log_best_setup(env, state, dist, n_samples, episode):

    all_max_eigs, is_valid_solution = evaluate_best_setup(env, state, dist, n_samples)
    log_max_eig_dist_and_incidence_rate(all_max_eigs, is_valid_solution, episode)


def get_incidence_rate(env, policy_net, max_steps_per_episode=50, num_samples=100, device="cuda"):

    # Do one episode and find the best state and action distribution
    best_setup = None
    best_reward = -math.inf
    state = env.reset().to(device)
    for _ in tqdm(range(max_steps_per_episode), desc="Getting best setup"):
        mean, std = policy_net(state)
        dist = torch.distributions.Normal(mean, std)

        action = dist.rsample()
        next_state, reward, done = env.step(action)
        next_state = next_state.to(device)

        if best_setup is None or reward > best_reward:
            best_setup = (dist, state)
            best_reward = reward

        state = next_state
        if done:
            break
    
    # Evaluate the best setup
    best_dist, best_state = best_setup
    all_max_eigs, is_valid_solution = evaluate_best_setup(env, best_state, best_dist, num_samples)
    incidence_rate = sum(is_valid_solution) / len(is_valid_solution)

    return incidence_rate, all_max_eigs


def setup_kinetic_env(cfg):

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

    # Initialize environment
    names_km = load_pkl(cfg.paths.names_km)
    reward_fn = partial(reward_func, chk_jcbn, names_km, cfg.reward.eig_partition)
    env = KineticEnv(cfg, reward_fn)
    env.seed(cfg.seed)

    return env