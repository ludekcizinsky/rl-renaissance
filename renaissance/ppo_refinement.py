import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PPO Algorithm Components
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 512, 1024)):
        super(Actor, self).__init__()
        self.hidden_dims = hidden_dims 
        layers = []
        input_d = state_dim
        for hidden_d in hidden_dims:
            layers.append(nn.Linear(input_d, hidden_d))
            layers.append(nn.ReLU())
            input_d = hidden_d
        
        self.network = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5, dtype=torch.float32))

    def forward(self, state):
        x = self.network(state)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)
        if mean.ndim > 1 and std.ndim == 1 and std.shape[0] == mean.shape[1]:
             std = std.unsqueeze(0).expand_as(mean)
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dims=(256, 512, 1024)):
        super(Critic, self).__init__()
        self.hidden_dims = hidden_dims 
        layers = []
        input_d = state_dim
        for hidden_d in hidden_dims:
            layers.append(nn.Linear(input_d, hidden_d))
            layers.append(nn.ReLU())
            input_d = hidden_d
        
        self.network = nn.Sequential(*layers)
        self.value_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state):
        x = self.network(state)
        value = self.value_layer(x)
        return value

class PPORefinement:
    def __init__(self, param_dim, latent_dim, min_x_bounds, max_x_bounds, 
                 names_km_full, chk_jcbn,
                 actor_hidden_dims=(256, 512, 1024), critic_hidden_dims=(256, 512, 1024),
                 p0_init_std=1, actor_lr=1e-4, critic_lr=1e-4, 
                 gamma=0.99, epsilon=0.2, gae_lambda=0.95,
                 ppo_epochs=10, num_episodes_per_update=64, 
                 T_horizon=5, k_reward_steepness=1.0,
                 action_clip_range=(-0.1, 0.1),
                 entropy_coeff=0.01, max_grad_norm=0.5):
        
        self.param_dim = param_dim
        self.latent_dim = latent_dim # For z in state
        self.min_x_bounds = min_x_bounds
        self.max_x_bounds = max_x_bounds
        self.p0_init_std = p0_init_std

        self.names_km_full = names_km_full 
        self.chk_jcbn = chk_jcbn # Store the jacobian checker instance

        # State dim: p_t (param_dim) + z (latent_dim) + lambda_max (1) + t (1)
        self.state_dim = self.param_dim + self.latent_dim + 1 + 1
        self.action_dim = self.param_dim # Actor outputs updates to p_t

        self.actor = Actor(self.state_dim, self.action_dim, hidden_dims=actor_hidden_dims)
        self.critic = Critic(self.state_dim, hidden_dims=critic_hidden_dims)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.num_episodes_per_update = num_episodes_per_update
        self.T_horizon = T_horizon
        self.k_reward_steepness = k_reward_steepness 
        self.action_clip_range = action_clip_range 
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        
        self.eig_partition_final_reward = -2.5 

    def _get_lambda_max(self, p_tensor_single):
        p_numpy = p_tensor_single.detach().cpu().numpy()
        # Use the stored chk_jcbn instance
        self.chk_jcbn._prepare_parameters([p_numpy], self.names_km_full) 
        max_eig_list = self.chk_jcbn.calc_eigenvalues_recal_vmax()

        return max_eig_list[0] 

    def _compute_reward(self, lambda_max_val):
        intermediate_r = 1.0 / (1.0 + np.exp(self.k_reward_steepness * (lambda_max_val - (self.eig_partition_final_reward))))
        # TODO: Right now, we are not using the Incidence part of the reward.

        return intermediate_r

    def _collect_trajectories(self):
        batch_states = []
        batch_actions = []
        batch_log_probs_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        all_episode_total_rewards = []

        for i_episode in range(self.num_episodes_per_update):
            print(f"Collecting episode data {i_episode + 1}/{self.num_episodes_per_update}...")
            # Episode initialization
            # Generate p0: small random values, then clamp. No parameter fixing.
            p_curr_np = np.random.normal(0, self.p0_init_std, size=self.param_dim)
            p_curr_np = (p_curr_np - p_curr_np.min()) / (p_curr_np.max() - p_curr_np.min())  # Normalize to [0, 1]
            p_curr_np = p_curr_np * (self.max_x_bounds - self.min_x_bounds) + self.min_x_bounds  # Scale to [min_x_bounds, max_x_bounds]
            
            # Generate z for state
            z_curr_np = np.random.normal(0, 1, size=self.latent_dim)

            p_curr_torch = torch.tensor(p_curr_np, dtype=torch.float32)
            z_torch_ep = torch.tensor(z_curr_np, dtype=torch.float32)

            episode_total_reward = 0

            for t_s in range(self.T_horizon):
                # Use instance methods for lambda_max and reward
                lambda_max_pt_val = self._get_lambda_max(p_curr_torch) 
                
                state_torch_flat = torch.cat((
                    p_curr_torch, z_torch_ep,
                    torch.tensor([lambda_max_pt_val], dtype=torch.float32),
                    torch.tensor([t_s], dtype=torch.float32)
                ))

                with torch.no_grad():
                    action_mean, action_std = self.actor(state_torch_flat.unsqueeze(0))
                    dist = Normal(action_mean, action_std)
                    action = dist.sample()
                    log_prob_action = dist.log_prob(action).sum(dim=-1)

                batch_states.append(state_torch_flat)
                batch_actions.append(action.squeeze(0))
                batch_log_probs_actions.append(log_prob_action)

                action_clipped = torch.clamp(action.squeeze(0), self.action_clip_range[0], self.action_clip_range[1])
                p_next_torch = p_curr_torch + action_clipped
                p_next_torch = torch.clamp(p_next_torch, self.min_x_bounds, self.max_x_bounds)
                
                lambda_max_p_next_val = self._get_lambda_max(p_next_torch)
                is_final_step = (t_s == self.T_horizon - 1)
                # print(lambda_max_p_next_val) : for DEBUG
                reward_val = self._compute_reward(lambda_max_p_next_val)

                batch_rewards.append(torch.tensor([reward_val], dtype=torch.float32))
                episode_total_reward += reward_val

                next_state_torch_flat = torch.cat((
                    p_next_torch, z_torch_ep,
                    torch.tensor([lambda_max_p_next_val], dtype=torch.float32),
                    torch.tensor([t_s + 1], dtype=torch.float32)
                ))
                batch_next_states.append(next_state_torch_flat)
                batch_dones.append(torch.tensor([1.0 if is_final_step else 0.0], dtype=torch.float32))
                p_curr_torch = p_next_torch
            
            all_episode_total_rewards.append(episode_total_reward)

        # Concatenate collected data into batch tensors
        final_batch_states = torch.stack(batch_states) if batch_states else torch.empty(0, self.state_dim)
        final_batch_actions = torch.stack(batch_actions) if batch_actions else torch.empty(0, self.action_dim)
        final_batch_log_probs = torch.stack(batch_log_probs_actions) if batch_log_probs_actions else torch.empty(0,1)
        final_batch_rewards = torch.stack(batch_rewards) if batch_rewards else torch.empty(0,1)
        final_batch_next_states = torch.stack(batch_next_states) if batch_next_states else torch.empty(0, self.state_dim)
        final_batch_dones = torch.stack(batch_dones) if batch_dones else torch.empty(0,1)
        
        avg_episode_reward_val = np.mean(all_episode_total_rewards) if all_episode_total_rewards else 0

        return (final_batch_states, final_batch_actions, final_batch_log_probs, 
                final_batch_rewards, final_batch_next_states, final_batch_dones,
                avg_episode_reward_val)

    def _compute_gae(self, rewards, values, next_values, dones):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        if rewards.nelement() == 0: 
             return torch.zeros_like(rewards), torch.zeros_like(values) 

        for t in reversed(range(len(rewards))): 
            is_terminal_transition = dones[t].item() > 0.5 
            delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_advantage
        returns = advantages + values
        return advantages, returns

    def update(self, trajectories_data):
        states, actions, log_probs_old, rewards, next_states, dones, _ = trajectories_data
        
        if states.nelement() == 0: 
            return 0.0, 0.0

        with torch.no_grad():
            values = self.critic(states)          
            next_values = self.critic(next_states) 

        advantages, returns = self._compute_gae(rewards, values, next_values, dones)
        
        if advantages.nelement() == 0: 
             return 0.0, 0.0

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_total_loss_epoch = 0
        critic_total_loss_epoch = 0

        for _ in range(self.ppo_epochs):
            current_pi_mean, current_pi_std = self.actor(states)
            dist_new = Normal(current_pi_mean, current_pi_std)
            log_probs_new = dist_new.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist_new.entropy().mean()

            ratios = torch.exp(log_probs_new - log_probs_old.detach()) 
            
            surr1 = ratios * advantages.detach() 
            surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            actor_total_loss_epoch += actor_loss.item()

            values_pred = self.critic(states) 
            critic_loss = (returns.detach() - values_pred).pow(2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            critic_total_loss_epoch += critic_loss.item()
        
        avg_actor_loss = actor_total_loss_epoch / self.ppo_epochs
        avg_critic_loss = critic_total_loss_epoch / self.ppo_epochs
        return avg_actor_loss, avg_critic_loss


    def train(self, num_iterations, output_path_base="ppo_training_output"):
        import os
        os.makedirs(output_path_base, exist_ok=True)
        
        all_iter_avg_rewards = []
        print(f"Starting PPO training for {num_iterations} iterations (serial execution).") 
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}, Latent (z) dim: {self.latent_dim}")
        print(f"Num episodes per update: {self.num_episodes_per_update}, Horizon T: {self.T_horizon}")
        print(f"p0 initialized with N(0, {self.p0_init_std**2}) and clamped to [{self.min_x_bounds}, {self.max_x_bounds}]")


        for iteration in range(num_iterations):
            trajectories_data = self._collect_trajectories()
            avg_episode_reward = trajectories_data[-1] 
            
            if trajectories_data[0].nelement() == 0 and self.num_episodes_per_update > 0:
                print(f"Iter {iteration:04d}: No trajectories collected. Skipping update. Avg Ep Reward: {avg_episode_reward:.4f}")
                all_iter_avg_rewards.append(avg_episode_reward) 
                continue 

            actor_loss, critic_loss = self.update(trajectories_data)
            all_iter_avg_rewards.append(avg_episode_reward)
        
            print(f"Iter {iteration:04d}: Avg Ep Reward: {avg_episode_reward:.4f}, "
                    f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
            
            if iteration % 50 == 0 or iteration == num_iterations -1 : 
                actor_path = os.path.join(output_path_base, f"actor_iter_{iteration}.pth")
                critic_path = os.path.join(output_path_base, f"critic_iter_{iteration}.pth")
                torch.save(self.actor.state_dict(), actor_path)
                torch.save(self.critic.state_dict(), critic_path)
        
        print("Training finished.")
        return all_iter_avg_rewards
