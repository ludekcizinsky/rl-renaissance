"""
The general objective is that given a set of kinetic parameters, we want to make sure that the maximum eigenvalue of the Jacobian < -2.5 
(this is a constant derived from our biological system we are trying to model). The more negative the eigenvalue is the faster the system
returns to the steady state.

In general, if the system is stable, the eigenvalues should be only negative. 

In Renaissance, the quality of the generated network was assesed by the incidence rate, i.e., the fraction of generated kinetic parameters who max
eigenvalue is less than -2.5.
"""

import torch
import numpy as np
from omegaconf import DictConfig


def get_reward(lam_max: float, rew_cfg: DictConfig):
    """
    Get the reward for the given lambda max.
    """
    if rew_cfg.name == "sigmoid":
        return sigmoid_reward(lam_max, rew_cfg)
    elif rew_cfg.name == "tanh":
        return tanh_reward(lam_max, rew_cfg)
    else:
        raise ValueError(f"Reward function {rew_cfg.name} not found")

def _preprocess_lam_max(lam_max: float, rew_cfg: DictConfig):
    """
    Preprocess the lambda max to be used in the reward function.
    Currently, only supports center and clip.
    """
    # float to tensor
    lam_max = torch.tensor(lam_max)
    if rew_cfg.center:
        lam_max = lam_max - rew_cfg.eig_partition
    if rew_cfg.clip:
        lam_max = torch.clamp(lam_max, rew_cfg.clip_range[0], rew_cfg.clip_range[1])
    return lam_max

def _post_process_reward(reward: float, rew_cfg: DictConfig):
    # tensor to float
    reward = reward.item()
    return reward

def sigmoid_reward(lam_max: float, rew_cfg: DictConfig):
    """
    lam_max : tensor or float, largest real-part eigenvalue  
    eig_partition : float, the eigenvalue at which the reward is 0.5
    center : bool, whether to center the eigenvalue around 0
    clip : bool, whether to clip the eigenvalue between -20 and 20
    """
    lam_max = _preprocess_lam_max(lam_max, rew_cfg)
    reward = 1.0 / (1.0 + np.exp(lam_max)) + 1e-3  # now ∈ (0,1)
    reward = _post_process_reward(reward, rew_cfg)
    return reward

def tanh_reward(lam_max: float, rew_cfg: DictConfig):
    """
    lam_max : tensor or float, largest real-part eigenvalue  
    beta    : slope parameter; larger -> steeper transition around -2.5
    Returns a reward in (-1, 1), zero when lam_max == -2.5
    """
    lam_max = _preprocess_lam_max(lam_max, rew_cfg)
    reward = torch.tanh(-rew_cfg.beta * (lam_max))
    reward = _post_process_reward(reward, rew_cfg)
    return reward