import pickle
from typing import Any

import torch
import math
import numpy as np
from omegaconf import DictConfig

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


def reward_func(chk_jcbn, names_km, eig_partition: float, include_penalty:bool, gen_kinetic_params: torch.Tensor) -> float:
    """
    Calculate the reward for a 1D tensor of kinetic parameters.
    """

    # Ensure that the kinetic parameters are in the correct format
    gen_kinetic_params = gen_kinetic_params.detach().cpu().numpy()

    # For some reason, we need to convert the kinetic parameters to a pandas dataframe
    chk_jcbn._prepare_parameters([gen_kinetic_params], names_km)

    # Calculate the maximum eigenvalue of the Jacobian
    max_eig = chk_jcbn.calc_eigenvalues_recal_vmax()[0]

    # Calculate the reward
    # TODO: this is somewhat adapted from the original Renaissance code
    # but needs further investigation
    # reward = 0.01 / (1 + np.exp(max_eig - eig_partition))
    #eig_values = np.clip(eig_values, -20, 100)
    #penalty = np.max([0, max_eig])
    #considered_avg = sum(eig_values[:10]) / 10
    #reward = np.exp(-0.1 * considered_avg) / 2
    z = np.clip(max_eig - eig_partition, -20, +20)
    penalty = np.clip(max_eig, 0, 100)
    reward = 1.0 / (1.0 + np.exp(z)) + 1e-3  # now ∈ (0,1)
    if include_penalty:
        reward -= penalty

    return reward


def load_pkl(name: str) -> Any:
    """load a pickle object"""
    name = name.replace('.pkl', '')
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)