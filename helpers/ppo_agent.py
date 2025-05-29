import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

from helpers.buffers import TrajectoryBuffer
from helpers.env import KineticEnv
from helpers.utils import compute_grad_norm
from helpers.lr_schedulers import get_lr_scheduler
from typing import Dict
import wandb

class PolicyNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        hidden_dim = 4 * cfg.env.p_size
        # Shared base network
        self.base = nn.Sequential(
            nn.Linear(cfg.env.p_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        # Separate heads for mean and log_std
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, cfg.env.p_size),
            nn.LayerNorm(cfg.env.p_size)
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(hidden_dim, cfg.env.p_size),
            nn.LayerNorm(cfg.env.p_size)
        )

        self.min_log_std = getattr(cfg.method, "min_log_std", -20)
        self.max_log_std = getattr(cfg.method, "max_log_std", 2)

    def forward(self, x):
        base_out = self.base(x)
        mean = self.mean_head(base_out)
        log_std = self.log_std_head(base_out)
        log_std = torch.clamp(log_std, min=self.min_log_std, max=self.max_log_std)
        std = torch.exp(log_std)
        return mean, std

    def load_pretrained_policy_net(self, policy_net_path):
        self.load_state_dict(torch.load(policy_net_path, map_location="cpu"))
        print(f"FYI: Loaded pretrained policy network from {policy_net_path}.")

class ValueNetwork(nn.Module):
    def __init__(self, cfg):
        super(ValueNetwork, self).__init__()

        hidden_dim = 4 * cfg.env.p_size
        self.net = nn.Sequential(
            nn.Linear(cfg.env.p_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a single value
        )

    def forward(self, x):
        value = self.net(x)
        return value.squeeze()

class PPOAgent:
    def __init__(self, cfg, run):

        self.cfg = cfg
        self.run = run
        self.n_eval_samples_in_episode = cfg.training.n_eval_samples_in_episode
        self.device = torch.device(cfg.device)

        self.policy_net = PolicyNetwork(cfg).to(self.device)
        self.value_net = ValueNetwork(cfg).to(self.device)
        self.policy_optimizer = optim.AdamW(self.policy_net.parameters(), lr=cfg.method.actor_lr)
        self.value_optimizer = optim.AdamW(self.value_net.parameters(), lr=cfg.method.critic_lr)

        self.policy_scheduler = get_lr_scheduler(cfg, self.policy_optimizer)
        self.value_scheduler = get_lr_scheduler(cfg, self.value_optimizer)

        self.global_step = 0
        self.global_best_reward = -math.inf
        self.global_best_model = None
        self.global_best_setup = None
        self.first_valid_setup = None

        # running mean/std buffers for obs
        self.obs_mean = torch.zeros(cfg.env.p_size, device=self.device)
        self.obs_var  = torch.ones(cfg.env.p_size, device=self.device)
        self.obs_count= 1e-4

    def normalize_obs(self, x):
        # update running stats
        batch_mean = x.mean(0)
        batch_var  = x.var(0, unbiased=False)
        batch_n    = x.shape[0]

        # Welford online update
        delta      = batch_mean - self.obs_mean
        tot_count  = self.obs_count + batch_n
        self.obs_mean += delta * batch_n / tot_count
        self.obs_var  = (self.obs_var * self.obs_count + batch_var * batch_n
                        + delta**2 * self.obs_count * batch_n / tot_count) / tot_count
        self.obs_count = tot_count

        # normalize
        return (x - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)

    def collect_trajectory(self, env: KineticEnv, episode: int):
        buf = TrajectoryBuffer()
        best_setup = None
        best_reward = -math.inf
        best_step = None

        state = env.reset().to(self.device)
        for step in range(self.cfg.training.max_steps_per_episode):

            mean, std = self.policy_net(state)
            mean = torch.nan_to_num(mean, nan=0.0, posinf=1e3, neginf=-1e3)
            std  = torch.nan_to_num(std, nan=1e-3, posinf=1e3, neginf=1e-3)
            std = std.clamp(min=1e-3, max=1e3)
            dist = torch.distributions.Normal(mean, std)

            action = dist.rsample()
            log_prob = dist.log_prob(action).sum()
            value = self.value_net(state)

            next_state, reward, done = env.step(action)
            next_state = next_state.to(self.device)

            if best_setup is None or reward > best_reward:
                best_setup = (dist, state, mean, std)
                best_reward = reward
                best_step = step

            buf.add(state, action, log_prob, value, reward, done)
            state = next_state
            if done:
                break
        
        # Parse best setup
        best_dist, best_state, best_mean, best_std = best_setup
        # if episode % 10 == 0:
        # evaluate_and_log_best_setup(env, best_state, best_dist, self.n_eval_samples_in_episode, episode)

        # Global best setup
        trajectory = buf.to_tensors()
        max_episode_reward = trajectory["rewards"].max().item()
        if max_episode_reward > self.global_best_reward:
            self.global_best_reward = max_episode_reward
            self.global_best_model = (self.policy_net.state_dict(), self.value_net.state_dict())
            self.global_best_setup = (best_mean, best_std, best_state, best_step, episode)
            print(f"FYI: New global best reward: {self.global_best_reward} in episode {episode} at step {best_step}.")

        # First valid setup
        if self.first_valid_setup is None and best_reward > 0.5:
            self.first_valid_setup = (best_mean, best_std, best_state, best_step, episode)
            print(f"FYI: First valid setup found in episode {episode} at step {best_step}.")

        # Log episode metrics
        self.run.log({
            "episode": episode,
            "episode/best_reward": best_reward,
            "episode/best_step": best_step,
        })

        buf.clear()
        return trajectory

    def compute_advantages(self, rewards, values, dones, last_value):
        """
        Compute Generalized Advantage Estimation (GAE) advantages and discounted returns.

        Args:
            rewards: Tensor of shape (T,)
            values: Tensor of shape (T,)
            dones: Tensor of shape (T,)
            last_value: float

        Returns:
            advantages: Tensor of shape (T,)
            returns: Tensor of shape (T,)

        Notes:
            - Advantage: how much better is the action taken by the policy compared to the average action
            - Returns: sum of discounted rewards
        """

        gamma = self.cfg.method.discount_factor
        gae_lambda = self.cfg.method.gae_lambda

        T = rewards.size(0)
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]   # 0 if episode ended
            next_val = values[t+1] if t+1 < T else last_value

            # d_t = r_t + \gamma*V(s_t+1) - V(s_t)
            delta = rewards[t] + gamma * next_val * mask - values[t]
            # A_t = d_t + \gamma*\lambda*A_t+1
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae

        # A_t = R_t - V(s_t) -> R_t = A_t + V(s_t)
        returns = advantages + values
        return advantages, returns

    def _optimize_policy(self, loss, max_grad_norm):
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        loss.backward()

        log_info = {}
        # --- gradient norms ---
        log_info["optim/pnet_pre_clip_grad_norm"] = compute_grad_norm(self.policy_net)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_grad_norm)
        log_info["optim/pnet_post_clip_grad_norm"] = compute_grad_norm(self.policy_net)

        log_info["optim/vnet_pre_clip_grad_norm"] = compute_grad_norm(self.value_net)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_grad_norm)
        log_info["optim/vnet_post_clip_grad_norm"] = compute_grad_norm(self.value_net)

        # --- step policy optimizer & log its LR ---
        self.policy_optimizer.step()
        self.policy_scheduler.step()
        log_info["optim/policy_lr"] = self.policy_optimizer.param_groups[0]['lr']

        # --- step value optimizer & log its LR ---
        self.value_optimizer.step()
        self.value_scheduler.step()
        log_info["optim/value_lr"] = self.value_optimizer.param_groups[0]['lr']

        return log_info

    def update(self, trajectory: Dict[str, torch.Tensor]):
        """
        Runs multiple PPO epochs over the data.

        Args:
            trajectory: dict from buf.to_tensors()
        """

        # unpack trajectory
        states = trajectory["states"] # (T, p_size)
        actions = trajectory["actions"] # (T, p_size)
        old_logp = trajectory["log_probs"] # (T,)
        values = trajectory["values"].squeeze(-1) # (T,)
        rewards = trajectory["rewards"] # (T,)
        dones = trajectory["dones"] # (T,)

        # normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)

        # compute advantages & returns
        with torch.no_grad():
            last_value = self.value_net(states[-1]).item()
        advs, returns = self.compute_advantages(rewards, values, dones, last_value)
        advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)

        # train policy and value network from the collected trajectory
        T = rewards.size(0)
        for _ in range(self.cfg.training.num_epochs):  
            idxs = torch.randperm(T) # shuffle indices
            for start in range(0, T, self.cfg.training.batch_size):
                step_log_info = {"global_step": self.global_step}

                # get batch of data
                batch_idxs = idxs[start : start + self.cfg.training.batch_size]
                b_states  = states[batch_idxs] # (batch_size, p_size)
                b_actions = actions[batch_idxs] # (batch_size, p_size)
                b_oldlp   = old_logp[batch_idxs] # (batch_size,)
                b_advs    = advs[batch_idxs] # (batch_size,)
                b_returns = returns[batch_idxs] # (batch_size,)

                # normalize states
                b_states = self.normalize_obs(b_states)
                b_advs = (b_advs - b_advs.mean()) / (b_advs.std(unbiased=False) + 1e-8)
                b_advs = torch.clamp(b_advs, -3.0, +3.0)

                # policy forward
                mean, std = self.policy_net(b_states)
                mean = torch.nan_to_num(mean, nan=0.0, posinf=1e3, neginf=-1e3)
                std  = torch.nan_to_num(std, nan=1e-3, posinf=1e3, neginf=1e-3)

                dist = torch.distributions.Normal(mean, std)
                new_logp = dist.log_prob(b_actions).sum(dim=-1)

                # ratio for clipped surrogate
                ratio = torch.exp(new_logp - b_oldlp)
                clip_frac = ((ratio - 1.0).abs() > self.cfg.method.clip_eps).float().mean()
                step_log_info["ppo/clip_fraction"] = clip_frac.item() # should be between 0.1 to 0.3

                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.method.clip_eps, 1.0 + self.cfg.method.clip_eps) * b_advs
                step_log_info["ppo/advs"] = b_advs.mean().item()
                step_log_info["ppo/per_dim_ratio"] = torch.exp((new_logp - b_oldlp) / self.cfg.env.p_size).mean().item()
                step_log_info["ppo/surr1"] = surr1.mean().item()
                step_log_info["ppo/surr2"] = surr2.mean().item()

                # policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                step_log_info["ppo/policy_loss"] = policy_loss.item()

                # value loss
                new_values = self.value_net(b_states).squeeze(-1)
                value_loss = (b_returns - new_values).pow(2).mean()
                step_log_info["ppo/value_loss"] = value_loss.item()

                # entropy
                entropy = dist.entropy().sum(dim=-1).mean()
                step_log_info["ppo/entropy"] = entropy.item()

                # combined loss
                loss = policy_loss \
                    + self.cfg.method.value_loss_weight * value_loss \
                    - self.cfg.method.entropy_loss_weight * entropy
                step_log_info["ppo/loss"] = loss.item()

                if torch.isnan(loss):
                    continue

                # optimize
                optim_log_info = self._optimize_policy(loss, self.cfg.training.max_grad_norm)
                step_log_info.update(optim_log_info)

                # log
                self.run.log(step_log_info)
                self.global_step += 1

        return policy_loss.item(), value_loss.item()