import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

from helpers.buffers import TrajectoryBuffer
from helpers.env import KineticEnv, BatchKineticEnv

from typing import Dict, List
from helpers.env import KineticEnv
from helpers.utils import compute_grad_norm, evaluate_and_log_best_setup, log_rl_models
from helpers.lr_schedulers import get_lr_scheduler
from typing import Dict


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

    def forward(self, x):
        base_out = self.base(x)
        mean = self.mean_head(base_out)
        log_std = self.log_std_head(base_out)
        std = torch.exp(log_std)
        return mean, std

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

    def collect_trajectory(self, env: KineticEnv, episode: int):
        buf = TrajectoryBuffer()
        best_setup = None
        best_reward = -math.inf

        state = env.reset().to(self.device)
        for step in range(self.cfg.training.max_steps_per_episode):
            mean, std = self.policy_net(state)
            dist = torch.distributions.Normal(mean, std)

            action = dist.rsample()
            log_prob = dist.log_prob(action).sum()
            value = self.value_net(state)
            next_state, reward, done = env.step(action)
            next_state = next_state.to(self.device)

            if best_setup is None or reward > best_reward:
                best_setup = (dist, state)
                best_reward = reward

            buf.add(state, action, log_prob, value, reward, done)
            state = next_state
            if done:
                break
        
        best_dist, best_state = best_setup
        if episode % 10 == 0:
            evaluate_and_log_best_setup(env, best_state, best_dist, self.n_eval_samples_in_episode, episode)

        trajectory = buf.to_tensors()
        mean_episode_reward = trajectory["rewards"].mean().item()
        if mean_episode_reward > self.global_best_reward:
            self.global_best_reward = mean_episode_reward
            self.global_best_model = (self.policy_net.state_dict(), self.value_net.state_dict())
            print(f"FYI: New global best reward: {self.global_best_reward} in episode {episode}.")

        buf.clear()
        return trajectory
    
    def collect_batch_trajectory(self, env: BatchKineticEnv):
        print("[PPOAgent] Starting collect_batch_trajectory (batch env)")
        buf = TrajectoryBuffer()
        state = env.reset()  # (batch_size, param_dim)
        print(f"[PPOAgent] Initial batch state: {state.shape}")
        batch_size = state.shape[0]
        for step in range(self.cfg.training.max_steps_per_episode):
            mean, std = self.policy_net(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=1)  # sum over param_dim
            value = self.value_net(state)
            next_state, reward, done = env.step(action)
            buf.add(state, action, log_prob, value, reward, done)
            print(f"[PPOAgent] Step {step}: reward (batch) mean={reward.mean().item():.4f}, done sum={done.sum().item()}")
            state = next_state
        trajectory = buf.to_tensors()
        print(f"[PPOAgent] Batch trajectory collected: states={trajectory['states'].shape}, rewards={trajectory['rewards'].shape}")
        buf.clear()
        return trajectory

    def collect_trajectories(self, env, num_trajectories: int):
        print(f"[PPOAgent] collect_trajectories: env={type(env).__name__}, num_trajectories={num_trajectories}")
        if isinstance(env, BatchKineticEnv):
            print("[PPOAgent] Using batch environment (BatchKineticEnv)")
            return self.collect_batch_trajectory(env)
        else:
            trajectories = []
            for i in range(num_trajectories):
                print(f"[PPOAgent] Collecting trajectory {i+1}/{num_trajectories}")
                trajectory = self.collect_trajectory(env)
                trajectories.append(trajectory)
            return trajectories

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
            mask = 1.0 - dones[t].float()   # 0 if episode ended
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

    def update(self, trajectories: List[Dict[str, torch.Tensor]]):
        """
        Runs multiple PPO epochs over the data.
        Args:
            trajectory: dict from buf.to_tensors()
        """
        print(f"[PPOAgent] update: {len(trajectories['rewards'])} trajectories")

        # unpack trajectory
        states = trajectories["states"] # (T, p_size)
        actions = trajectories["actions"] # (T, p_size)
        old_logp = trajectories["log_probs"] # (T,)
        values = trajectories["values"].squeeze(-1) # (T,)
        rewards = trajectories["rewards"] # (T,)
        dones = trajectories["dones"] # (T,)
        print(f"[PPOAgent] update: states={states.shape}, actions={actions.shape}, rewards={rewards.shape}")
        # --- flatten for batch envs (BatchKineticEnv) ---
        # If states/actions are 3D (steps, batch, param_dim), flatten to (steps*batch, param_dim)
        if len(states.shape) == 3:
            steps, batch, param_dim = states.shape
            states = states.reshape(-1, param_dim)
            actions = actions.reshape(-1, param_dim)
            old_logp = old_logp.reshape(-1)
            values = values.reshape(-1)
            rewards = rewards.reshape(-1)
            dones = dones.reshape(-1)
        # normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
        # Compute advantages & returns for the whole batch
        with torch.no_grad():
            last_value = self.value_net(states[-1])
        advs, returns = self.compute_advantages(
            rewards,
            values,
            dones,
            last_value
        )
        advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)

        # train policy and value network from the collected trajectory
        T = rewards.size(0)
        for _ in range(self.cfg.training.num_epochs):  
            idxs = torch.randperm(T) # shuffle indices
            for start in range(0, T, self.cfg.training.batch_size):
                step_log_info = {"global_step": self.global_step}
                # get batch of data
                batch_idxs = idxs[start : start + self.cfg.training.batch_size]
                b_states  = states[batch_idxs]
                b_actions = actions[batch_idxs]
                b_oldlp   = old_logp[batch_idxs]
                b_advs    = advs[batch_idxs]
                b_returns = returns[batch_idxs]
                mean, std = self.policy_net(b_states)
                dist = torch.distributions.Normal(mean, std)
                new_logp = dist.log_prob(b_actions).sum(dim=-1)

                # ratio for clipped surrogate
                ratio = torch.exp(new_logp - b_oldlp)
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

                # combined loss
                loss = policy_loss \
                     + self.cfg.method.value_loss_weight * value_loss
                step_log_info["ppo/loss"] = loss.item()

                # optimize
                optim_log_info = self._optimize_policy(loss, self.cfg.training.max_grad_norm)
                step_log_info.update(optim_log_info)

                # log
                self.run.log(step_log_info)
                self.global_step += 1

        return policy_loss.item(), value_loss.item()