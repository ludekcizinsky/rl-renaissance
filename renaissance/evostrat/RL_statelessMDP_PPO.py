import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam
import pickle


## adapt later to the mlp which is used in original renaissance
class StatelessPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StatelessPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)
        std = torch.exp(self.log_std)
        return mean, std


class PPOAgent:
    def __init__(self, input_dim, hidden_dim, output_dim, reward_func, 
                 save_path, n_samples, lr=3e-4, clip_eps=0.2, epochs=10, batch_size=64):
        self.policy = StatelessPolicy(input_dim, hidden_dim, output_dim)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.reward_func = reward_func
        self.save_path = save_path
        self.n_samples = n_samples

    def select_action(self, dummy_input):
        # Generate a sample action and log probability
        with torch.no_grad():
            mean, std = self.policy(dummy_input)
            dist = Normal(mean, std)
            sample = dist.sample()
            log_prob = dist.log_prob(sample).sum(dim=-1)
        return sample, log_prob

    def evaluate_actions(self, dummy_input, actions):
        # Evaluate the actions and compute log probabilities and entropy
        mean, std = self.policy(dummy_input)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy

    def run(self, iterations):
        dummy_input = torch.zeros(1)  # stateless: constant input

        all_rewards = []
        for iteration in range(iterations):
            print("checkpoint 1")
            # Sample actions and compute rewards
            actions, log_probs_old = [], []
            for _ in range(self.n_samples):
                # Generate a batch of actions
                a, logp = self.select_action(dummy_input)
                actions.append(a)
                log_probs_old.append(logp)

            actions = torch.stack(actions)
            log_probs_old = torch.stack(log_probs_old).detach()

            print("checkpoint 1.5")
            rewards = torch.tensor([self.reward_func(a.numpy()) for a in actions], dtype=torch.float32)
            print("checkpoint 1.6")
            # Normalize rewards
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            for _ in range(self.epochs):
                print("checkpoint 2")
                # Shuffle the actions and rewards for mini-batch training
                for i in range(0, self.n_samples, self.batch_size):
                    # Update policy using mini-batch gradient descent
                    batch_actions = actions[i:i+self.batch_size]
                    batch_old_log_probs = log_probs_old[i:i+self.batch_size]
                    batch_rewards = rewards[i:i+self.batch_size]

                    new_log_probs, entropy = self.evaluate_actions(dummy_input, batch_actions)
                    ratio = (new_log_probs - batch_old_log_probs).exp()

                    surr1 = ratio * batch_rewards
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_rewards
                    loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            avg_reward = rewards.mean().item()
            print(f"[Iter {iteration}] Avg Reward: {avg_reward:.3f}")
            all_rewards.append(avg_reward)

            with open(os.path.join(self.save_path, f"policy_{iteration}.pkl"), 'wb') as f:
                pickle.dump(self.policy.state_dict(), f)

        return all_rewards
