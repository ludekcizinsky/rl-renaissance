import torch
import torch.nn as nn
import torch.optim as optim

from helpers.buffers import TrajectoryBuffer
from helpers.env import KineticEnv, BatchKineticEnv

from typing import Dict, List

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
        print("[PPOAgent] Starting collect_trajectory (single env)")
        buf = TrajectoryBuffer()
        state = env.reset()
        print(f"[PPOAgent] Initial state: {state.shape} | {state[:5] if hasattr(state, '__getitem__') else state}")
        for step in range(self.cfg.training.max_steps_per_episode):
            mean, std = self.policy_net(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum()
            value = self.value_net(state)
            next_state, reward, done = env.step(action)
            buf.add(state, action, log_prob, value, reward, done)
            print(f"[PPOAgent] Step {step}: reward={reward}, done={done}")
            state = next_state
            if done:
                print(f"[PPOAgent] Episode finished at step {step}")
                break
        trajectory = buf.to_tensors()
        print(f"[PPOAgent] Trajectory collected: states={trajectory['states'].shape}, rewards={trajectory['rewards'].shape}")
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
            return [self.collect_batch_trajectory(env)]
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
        advantages = torch.zeros_like(rewards)
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
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_grad_norm)
        self.policy_optimizer.step()
        self.value_optimizer.step()

    def update(self, trajectories: List[Dict[str, torch.Tensor]]):
        print(f"[PPOAgent] update: {len(trajectories)} trajectories")
        # Concatenate all trajectories along time dimension
        states = torch.cat([traj["states"] for traj in trajectories], dim=0)      # (N, p_size)
        actions = torch.cat([traj["actions"] for traj in trajectories], dim=0)    # (N, p_size)
        old_logp = torch.cat([traj["log_probs"] for traj in trajectories], dim=0) # (N,)
        values = torch.cat([traj["values"].squeeze(-1) for traj in trajectories], dim=0) # (N,)
        rewards = torch.cat([traj["rewards"] for traj in trajectories], dim=0)    # (N,)
        dones = torch.cat([traj["dones"] for traj in trajectories], dim=0)        # (N,)
        print(f"[PPOAgent] update: states={states.shape}, actions={actions.shape}, rewards={rewards.shape}")
        # normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
        # Compute advantages & returns for each trajectory, then concatenate
        advs_list, returns_list = [], []
        idx = 0
        for traj in trajectories:
            T = traj["rewards"].size(0)
            with torch.no_grad():
                last_value = self.value_net(traj["states"][-1])  # Handles both batch and single
            advs, returns = self.compute_advantages(
                traj["rewards"], 
                traj["values"].squeeze(-1), 
                traj["dones"], 
                last_value
            )
            print(f"[PPOAgent] Trajectory {idx}: advs={advs.shape}, returns={returns.shape}")
            advs_list.append(advs)
            returns_list.append(returns)
            idx += T
        advs = torch.cat(advs_list, dim=0)
        returns = torch.cat(returns_list, dim=0)
        advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)
        N = rewards.size(0)
        for epoch in range(self.cfg.training.num_epochs):  
            idxs = torch.randperm(N) # shuffle indices
            for start in range(0, N, self.cfg.training.batch_size):
                batch_idxs = idxs[start : start + self.cfg.training.batch_size]
                b_states  = states[batch_idxs]
                b_actions = actions[batch_idxs]
                b_oldlp   = old_logp[batch_idxs]
                b_advs    = advs[batch_idxs]
                b_returns = returns[batch_idxs]
                mean, std = self.policy_net(b_states)
                dist = torch.distributions.Normal(mean, std)
                new_logp = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                ratio = torch.exp(new_logp - b_oldlp)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.method.clip_eps, 1.0 + self.cfg.method.clip_eps) * b_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                new_values = self.value_net(b_states).squeeze(-1)
                value_loss = (b_returns - new_values).pow(2).mean()
                loss = policy_loss \
                     + self.cfg.method.value_loss_weight * value_loss \
                     - self.cfg.method.entropy_loss_weight * entropy \
                     + 1e-3 * (self.policy_net.log_std**2).sum()
                print(f"[PPOAgent] Epoch {epoch}, batch {start//self.cfg.training.batch_size}: policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}, entropy={entropy.item():.4f}")
                self._optimize_policy(loss, self.cfg.training.max_grad_norm)
        print(f"[PPOAgent] update done: final policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}, entropy={entropy.item():.4f}")
        return policy_loss.item(), value_loss.item(), entropy.item()