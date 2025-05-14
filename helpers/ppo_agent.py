import torch
import torch.nn as nn
import torch.optim as optim

from helpers.buffers import TrajectoryBuffer
from helpers.env import KineticEnv


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
        return value

class PPOAgent:
    def __init__(self, cfg, logger):

        self.cfg = cfg
        self.logger = logger

        self.policy_net = PolicyNetwork(cfg)
        self.value_net = ValueNetwork(cfg)
        self.policy_optimizer = optim.AdamW(self.policy_net.parameters(), lr=cfg.training.actor_lr)
        self.value_optimizer = optim.AdamW(self.value_net.parameters(), lr=cfg.training.critic_lr)

    def collect_trajectory(self, env: KineticEnv):
        buf = TrajectoryBuffer()

        state = env.reset()
        for _ in range(self.cfg.training.max_steps_per_episode):
            mean, std = self.policy_net(state)
            action = torch.normal(mean, std)
            log_prob = (-((action - mean)**2) / (2*std**2)
                        - std.log() - 0.5*torch.log(2*torch.pi)).sum()
            value = self.value_net(state)

            next_state, reward, done = env.step(action)

            buf.add(state, action, log_prob, value, reward, done)
            state = next_state
            if done:
                break

        trajectory = buf.to_tensors()
        buf.clear()
        return trajectory

    def compute_advantages(self, rewards, values):
        # Compute advantages using GAE
        pass

    def update(self, trajectory):
        # Update policy and value networks
        pass