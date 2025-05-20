import os
import pickle
from typing import Any

import torch
import math
import numpy as np
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import wandb

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


def batch_reward_func(chk_jcbn, names_km, eig_partition: float, gen_kinetic_params: torch.Tensor, steps_ratio) -> torch.Tensor:
    """
    Calculate rewards for a batch of kinetic parameter tensors.
    Args:
        gen_kinetic_params: Tensor of shape (batch_size, param_dim)
        steps_ratio: Ratio of the number of steps taken to the maximum number of steps
        chk_jcbn: Jacobian solver object
        names_km: List of parameter names
        eig_partition: Eigenvalue partition for reward calculation
    Returns:
        rewards: Tensor of shape (batch_size,)
    """
    params_np = gen_kinetic_params.detach().cpu().numpy()
    rewards = []
    for param in params_np:
        chk_jcbn._prepare_parameters([param], names_km)
        all_eigenvalues = chk_jcbn.calc_eigenvalues_recal_vmax()[0]
        max_eig = np.max(all_eigenvalues)
        z = np.clip(max_eig - eig_partition, -100, +100)
        reward = (1.0 / (1.0 + np.exp(z)) + 1e-3)

        rewards.append(reward)
    return torch.tensor(np.array(rewards), dtype=torch.float32, device=gen_kinetic_params.device)


def load_pkl(name: str) -> Any:
    """load a pickle object"""
    name = name.replace('.pkl', '')
    with open(name + '.pkl', 'rb') as f:
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

    wandb.log({
        "reward/distribution": wandb.Image(fig), 
        "episode": episode
    })
    plt.close(fig)


def log_rl_models(
    policy_net: torch.nn.Module,
    value_net:  torch.nn.Module,
    description:   str = "Trained policy and value networks",
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


    # Prepare file paths
    run_name = wandb.run.name
    save_dir = os.path.join(save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    policy_path = os.path.join(save_dir, "policy.pt")
    value_path  = os.path.join(save_dir, "value.pt")

    # Save state_dicts
    torch.save(policy_net.state_dict(), policy_path)
    torch.save(value_net.state_dict(),  value_path)

    # Build and log the Artifact
    artifact = wandb.Artifact(
        name=run_name,
        type="model",
        description=description
    )
    artifact.add_file(policy_path)
    artifact.add_file(value_path)
    wandb.log_artifact(artifact)
    wandb.log_artifact(artifact, aliases=["latest"])

    print(f"FYI: Logged model to W&B as {run_name}.")