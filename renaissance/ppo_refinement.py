import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x)
        # Clamp log_std for stability, common practice
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
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
                 n_trajectories=32,
                 device=None):

        self.param_dim = param_dim
        self.noise_dim = noise_dim
        self.state_dim = noise_dim + param_dim
        self.reward_function = reward_function
        self.max_grad_norm = 0.5
        self.n_trajectories = n_trajectories

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
        tanh_params = torch.tanh(params_raw)
        return self.min_x_bounds + (self.max_x_bounds - self.min_x_bounds) * (tanh_params + 1) / 2.0

    def _initialize_current_params_for_state(self):
        current_params_in_state = torch.rand(self.param_dim, device=self.device) * (self.max_x_bounds - self.min_x_bounds) + self.min_x_bounds
        return current_params_in_state

    def collect_rollout_data(self):
        states, actions_raw, log_probs_raw, rewards, dones, values = [], [], [], [], [], []
        current_params_in_state = self._initialize_current_params_for_state().clone()
        final_ode_params = None

        for t in range(self.T_horizon):
            noise = torch.randn(self.noise_dim, device=self.device)
            state_1d = torch.cat((noise, current_params_in_state.detach()), dim=0)
            state_batch = state_1d.unsqueeze(0)

            with torch.no_grad():
                mu_raw, log_std_raw = self.actor(state_batch)
                std_raw = torch.exp(log_std_raw)
                dist = Normal(mu_raw, std_raw)
                action_raw = dist.sample()
                action_log_prob_raw = dist.log_prob(action_raw).sum(dim=-1)
                val = self.critic(state_batch)

            ode_params = self._transform_to_bounded(action_raw)

            if t == self.T_horizon - 1:
                r = self.reward_function(ode_params.squeeze(0))
                d = True
                final_ode_params = ode_params.squeeze(0).detach()
            else:
                r = 0.0
                d = False

            states.append(state_1d)
            actions_raw.append(action_raw.squeeze(0))
            log_probs_raw.append(action_log_prob_raw.squeeze(0))
            rewards.append(torch.tensor(r, device=self.device, dtype=torch.float32))
            dones.append(torch.tensor(d, device=self.device, dtype=torch.bool))
            values.append(val.squeeze())
            current_params_in_state = ode_params.squeeze(0)
        
        # mean_reward_trajectory = torch.mean(torch.stack(rewards)).item() # Optional: for debugging single trajectory
        # print(f"Mean reward for this single trajectory: {mean_reward_trajectory:.4f}")


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
        old_values = old_values.detach()

        advantages = torch.zeros_like(rewards, device=self.device)
        returns = torch.zeros_like(rewards, device=self.device)
        gae = 0
        next_val = torch.tensor(0.0, device=self.device)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_val * (1.0 - dones[t].float()) - old_values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t].float()) * gae
            advantages[t] = gae
            returns[t] = gae + old_values[t]
            next_val = old_values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            mu_raw, log_std_raw = self.actor(states)
            std_raw = torch.exp(log_std_raw)
            dist_new = Normal(mu_raw, std_raw)
            new_log_probs_raw = dist_new.log_prob(actions_raw).sum(dim=-1)

            ratios = torch.exp(new_log_probs_raw - old_log_probs_raw.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            current_values = self.critic(states).squeeze(-1)
            critic_loss = self.mse_loss(current_values, returns.detach())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

    def train(self, num_training_iterations, output_path=None):
        print(f"Training on {self.device}. {self.n_trajectories} trajectories per update.")
        # Stores the average final reward of the batch of trajectories for each training iteration
        avg_final_rewards_per_iteration = []

        for iteration in range(num_training_iterations):
            # Lists to store data from multiple trajectories for the current update iteration
            batch_states_list, batch_actions_raw_list, batch_log_probs_raw_list = [], [], []
            batch_rewards_list, batch_dones_list, batch_values_list = [], [], []
            
            # Store final rewards from each trajectory in this batch for averaging
            current_batch_final_rewards = []

            for _ in range(self.n_trajectories):
                # collect_rollout_data collects one trajectory
                (states, actions_raw, log_probs_raw,
                 rewards, dones, values), final_ode_params = self.collect_rollout_data()

                batch_states_list.append(states)
                batch_actions_raw_list.append(actions_raw)
                batch_log_probs_raw_list.append(log_probs_raw)
                batch_rewards_list.append(rewards)
                batch_dones_list.append(dones)
                batch_values_list.append(values)

                if final_ode_params is not None:
                    # Calculate reward for this specific trajectory's final params
                    # .item() is used to get a Python number for aggregation
                    reward_val = self.reward_function(final_ode_params).item() 
                    current_batch_final_rewards.append(reward_val)
            
            if not batch_states_list: # Should only happen if self.n_trajectories is 0 or less
                print(f"Warning: No rollout data collected in iteration {iteration+1}. Skipping update.")
                continue
                
            # Concatenate all collected data to form a single large batch
            # Each tensor from collect_rollout_data has shape (T_horizon, ...),
            # so concatenating along dim=0 results in (self.n_trajectories * T_horizon, ...)
            batched_states = torch.cat(batch_states_list, dim=0)
            batched_actions_raw = torch.cat(batch_actions_raw_list, dim=0)
            batched_log_probs_raw = torch.cat(batch_log_probs_raw_list, dim=0)
            batched_rewards = torch.cat(batch_rewards_list, dim=0)
            batched_dones = torch.cat(batch_dones_list, dim=0)
            batched_values = torch.cat(batch_values_list, dim=0)

            batched_rollout_data_tuple = (
                batched_states, batched_actions_raw, batched_log_probs_raw,
                batched_rewards, batched_dones, batched_values
            )

            self.update_policy(batched_rollout_data_tuple)

            if current_batch_final_rewards:
                avg_final_reward_for_batch = sum(current_batch_final_rewards) / len(current_batch_final_rewards)
                avg_final_rewards_per_iteration.append(avg_final_reward_for_batch)
                # Log every iteration (or adjust frequency as needed)
                if (iteration + 1) % 1 == 0: 
                    print(f"Iteration {iteration+1}/{num_training_iterations}, Avg Batch Final Reward: {avg_final_reward_for_batch:.4f}")
            
            if output_path and (iteration + 1) % 100 == 0:
                torch.save(self.actor.state_dict(), output_path + f"actor_{iteration+1}.pth")
                torch.save(self.critic.state_dict(), output_path + f"critic_{iteration+1}.pth")
                print(f"Model saved at iteration {iteration+1}")

        if output_path:
            np.save(output_path + "avg_final_rewards_per_iteration.npy", np.array(avg_final_rewards_per_iteration))
            print(f"Average final rewards per iteration saved at {output_path}avg_final_rewards_per_iteration.npy")

        print("Training finished.")
        return self.actor, avg_final_rewards_per_iteration