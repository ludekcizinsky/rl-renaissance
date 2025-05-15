import torch
import torch.nn as nn
import torch.optim as optim

from helpers.buffers import TrajectoryBuffer
from helpers.env import KineticEnv

from typing import Dict

class PolicyNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        hidden_dim = 4*cfg.env.p_size
        self.net = nn.Sequential(
            nn.Linear(cfg.env.p_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cfg.env.p_size),
            nn.LayerNorm(cfg.env.p_size)
        )

        self.log_std = nn.Parameter(torch.full((cfg.env.p_size,), -0.5))

    def bound_action(self, action):
        action = torch.clamp(action, self.cfg.constraints.min_km, self.cfg.constraints.max_km)
        return action

    def forward(self, x):
        mean = self.net(x)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        action = torch.normal(mean, std)
        action = self.bound_action(action)
        return action

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
    def __init__(self, cfg, logger):

        self.cfg = cfg
        self.logger = logger

        self.policy_net = PolicyNetwork(cfg)
        self.value_net = ValueNetwork(cfg)
        self.policy_optimizer = optim.AdamW(self.policy_net.parameters(), lr=cfg.method.actor_lr)
        self.value_optimizer = optim.AdamW(self.value_net.parameters(), lr=cfg.method.critic_lr)

    def collect_trajectory(self, env: KineticEnv):
        buf = TrajectoryBuffer()

        state = env.reset()
        for _ in range(self.cfg.training.max_steps_per_episode):
            mean, std = self.policy_net(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum()
            value = self.value_net(state)

            next_state, reward, done = env.step(action)

            buf.add(state, action, log_prob, value, reward, done)
            state = next_state
            if done:
                break

        trajectory = buf.to_tensors()
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
        advantages = torch.zeros_like(rewards)
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
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_grad_norm)
        self.policy_optimizer.step()
        self.value_optimizer.step()

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

                # get batch of data
                batch_idxs = idxs[start : start + self.cfg.training.batch_size]
                b_states  = states[batch_idxs] # (batch_size, p_size)
                b_actions = actions[batch_idxs] # (batch_size, p_size)
                b_oldlp   = old_logp[batch_idxs] # (batch_size,)
                b_advs    = advs[batch_idxs] # (batch_size,)
                b_returns = returns[batch_idxs] # (batch_size,)

                # policy forward
                mean, std = self.policy_net(b_states)
                dist = torch.distributions.Normal(mean, std)
                new_logp = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # ratio for clipped surrogate
                ratio = torch.exp(new_logp - b_oldlp)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.method.clip_eps, 1.0 + self.cfg.method.clip_eps) * b_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                new_values = self.value_net(b_states).squeeze(-1)
                value_loss = (b_returns - new_values).pow(2).mean()

                # combined loss
                loss = policy_loss \
                     + self.cfg.method.value_loss_weight * value_loss \
                     - self.cfg.method.entropy_loss_weight * entropy \
                     + 1e-3 * (self.policy_net.log_std**2).sum() # to prevent log_std from exploding

                # optimize
                self._optimize_policy(loss, self.cfg.training.max_grad_norm)

        return policy_loss.item(), value_loss.item(), entropy.item()