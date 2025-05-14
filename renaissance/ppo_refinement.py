import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import os
import helper as hp
from configparser import ConfigParser

from kinetics.jacobian_solver import check_jacobian

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PPO Algorithm Components
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Actor(nn.Module):
    def __init__(self, state_dim, param_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.param_dim = param_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, param_dim)
        self.fc_log_std = nn.Linear(hidden_dim, param_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x)
        # Clamp log_std to avoid extremely small or large std values
        log_std = torch.clamp(log_std, -20, 2)
        return mu, log_std

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc_value(x)
        return value
    
class PPORefinement:
    def __init__(self, param_dim, noise_dim, reward_function,
                 min_x_bounds, max_x_bounds,
                 hidden_dim_actor=256, hidden_dim_critic=256,
                 actor_lr=3e-4, critic_lr=1e-3,
                 gamma=0.99, ppo_epochs=4, epsilon=0.2,
                 gae_lambda=0.95, T_horizon=20,
                 device=None):

        self.param_dim = param_dim
        self.noise_dim = noise_dim
        self.state_dim = noise_dim + param_dim
        self.reward_function = reward_function
        self.max_grad_norm = 0.5

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.min_x_bounds = torch.tensor(min_x_bounds, device=self.device, dtype=torch.float32)
        self.max_x_bounds = torch.tensor(max_x_bounds, device=self.device, dtype=torch.float32)
        if self.min_x_bounds.shape == (): # scalar
            self.min_x_bounds = self.min_x_bounds.repeat(param_dim)
        if self.max_x_bounds.shape == (): # scalar
            self.max_x_bounds = self.max_x_bounds.repeat(param_dim)


        self.gamma = gamma
        self.epsilon = epsilon
        self.ppo_epochs = ppo_epochs
        self.gae_lambda = gae_lambda
        self.T_horizon = T_horizon

        self.actor = Actor(self.state_dim, param_dim, hidden_dim_actor).to(self.device)
        self.critic = Critic(self.state_dim, hidden_dim_critic).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.mse_loss = nn.MSELoss()

    def _transform_to_bounded(self, params_raw):
        # Clip params_raw to the range [min_x_bounds, max_x_bounds]
        # return torch.clamp(params_raw, min=self.min_x_bounds, max=self.max_x_bounds)
        tanh_params = torch.tanh(params_raw)
        return self.min_x_bounds + (self.max_x_bounds - self.min_x_bounds) * (tanh_params + 1) / 2.0

    def _initialize_current_params_for_state(self):
        # Start with parameters uniformly in the range of min_x_bounds and max_x_bounds
        current_params_in_state = torch.rand(self.param_dim, device=self.device) * (self.max_x_bounds - self.min_x_bounds) + self.min_x_bounds
        # Or use: torch.randn(self.param_dim, device=self.device) * 0.1 # Small random noise
        # current_params_in_state = torch.randn(self.param_dim, device=self.device) * 0.1
        return current_params_in_state

    def collect_rollout_data(self):
        states, actions_raw, log_probs_raw, rewards, dones, values = [], [], [], [], [], []

        current_params_in_state = self._initialize_current_params_for_state().clone() # Shape: (param_dim)
        final_ode_params = None

        for t in range(self.T_horizon):
            noise = torch.randn(self.noise_dim, device=self.device) # Shape: (noise_dim)
            state_1d = torch.cat((noise, current_params_in_state.detach()), dim=0) # Shape: (state_dim)
            state_batch = state_1d.unsqueeze(0) # Shape: (1, state_dim) for actor/critic

            with torch.no_grad(): # During data collection, no grad needed for actor/critic forward pass
                mu_raw, log_std_raw = self.actor(state_batch) # mu_raw: (1, param_dim), log_std_raw: (1, param_dim)
                std_raw = torch.exp(log_std_raw)
                dist = Normal(mu_raw, std_raw)
                action_raw = dist.sample() # Shape: (1, param_dim)
                action_log_prob_raw = dist.log_prob(action_raw).sum(dim=-1) # Shape: (1)
                val = self.critic(state_batch) # Shape: (1, 1)

            ode_params = self._transform_to_bounded(action_raw) # Shape: (1, param_dim)

            if t == self.T_horizon - 1:
                # Squeeze action_raw to 1D tensor for reward function
                r = self.reward_function(ode_params.squeeze(0))
                d = True
                final_ode_params = ode_params.squeeze(0).detach()
            else:
                r = 0.0 # No intermediate rewards
                d = False

            states.append(state_1d) # Store 1D state
            actions_raw.append(action_raw.squeeze(0)) # Store 1D action_raw
            log_probs_raw.append(action_log_prob_raw.squeeze(0)) # Store scalar log_prob
            rewards.append(torch.tensor(r, device=self.device, dtype=torch.float32))
            dones.append(torch.tensor(d, device=self.device, dtype=torch.bool))
            values.append(val.squeeze()) # Store scalar value

            current_params_in_state = ode_params.squeeze(0) # Update for next state, Shape: (param_dim)

        # Print mean reward for the episode
        mean_reward = torch.mean(torch.stack(rewards)).item()
        # print(f"Mean reward for the episode: {mean_reward:.4f}")

        # Convert lists to tensors
        rollout_data = (
            torch.stack(states),
            torch.stack(actions_raw),
            torch.stack(log_probs_raw),
            torch.stack(rewards),
            torch.stack(dones),
            torch.stack(values)
        )
        return rollout_data, final_ode_params

    def update_policy(self, rollout_data):
        states, actions_raw, old_log_probs_raw, rewards, dones, old_values = rollout_data
        old_values = old_values.detach() # Ensure no gradients flow back to critic from here

        # Calculate advantages and returns using GAE
        advantages = torch.zeros_like(rewards, device=self.device)
        returns = torch.zeros_like(rewards, device=self.device)
        gae = 0
        # next_val for the last state is 0 because the episode terminates
        next_val = torch.tensor(0.0, device=self.device)

        for t in reversed(range(len(rewards))):
            # if dones[t] is True, (1.0 - dones[t].float()) will be 0
            delta = rewards[t] + self.gamma * next_val * (1.0 - dones[t].float()) - old_values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t].float()) * gae
            advantages[t] = gae
            returns[t] = gae + old_values[t] # Q_t = A_t + V(s_t)
            next_val = old_values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize policy and value network for K epochs
        for _ in range(self.ppo_epochs):
            # Actor update
            mu_raw, log_std_raw = self.actor(states)
            std_raw = torch.exp(log_std_raw)
            dist_new = Normal(mu_raw, std_raw)
            new_log_probs_raw = dist_new.log_prob(actions_raw).sum(dim=-1)

            ratios = torch.exp(new_log_probs_raw - old_log_probs_raw.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            # Optional: Add entropy bonus
            # entropy = dist_new.entropy().sum(-1).mean()
            # actor_loss -= 0.01 * entropy


            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Optional: Gradient clipping for actor
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Critic update
            current_values = self.critic(states).squeeze(-1) # Shape: (T_horizon)
            critic_loss = self.mse_loss(current_values, returns.detach())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # Optional: Gradient clipping for critic
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

    def train(self, num_training_iterations, output_path=None):
        print(f"Training on {self.device}")
        final_rewards = []
        for iteration in range(num_training_iterations):
            rollout_data_tuple, final_ode_params = self.collect_rollout_data()
            self.update_policy(rollout_data_tuple)

            if final_ode_params is not None:
                final_reward = self.reward_function(final_ode_params)
                final_rewards.append(final_reward)
                if (iteration + 1) % 1 == 0: # Log every iteration
                    print(f"Iteration {iteration+1}/{num_training_iterations}, Final Reward: {final_reward:.4f}")
                    # print(f"   Final ODE Params: {final_ode_params.cpu().numpy()}")
            # save the model every 2 iterations
            if (iteration + 1) % 100 == 0:
                torch.save(self.actor.state_dict(), output_path + f"actor_{iteration+1}.pth")
                torch.save(self.critic.state_dict(), output_path + f"critic_{iteration+1}.pth")
                print(f"Model saved at iteration {iteration+1}")

        # save final rewards
        if output_path is not None:
            np.save(output_path + "final_rewards.npy", final_rewards)
            print(f"Final rewards saved at {output_path}final_rewards.npy")

        print("Training finished.")
        # You can return the trained actor or save it
        return self.actor, final_rewards