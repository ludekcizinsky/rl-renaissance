import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import pandas as pd # For reading fixed parameter names CSV
import helper as hp # For hp.unscale_range

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initial Parameter Generator (PyTorch based)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class InitialGeneratorPT(nn.Module):
    def __init__(self, latent_dim, n_parameters, hidden_dims_gen_mlp, 
                 names_km_full, min_x, max_x, 
                 param_fixing_flag, fixed_param_names_path=None, fixed_param_ranges_path=None):
        super(InitialGeneratorPT, self).__init__()
        self.latent_dim = latent_dim
        self.n_parameters = n_parameters # Total number of parameters
        self.names_km_full = names_km_full # Full list of parameter names
        self.min_x = min_x # For scaling, e.g., lnminkm
        self.max_x = max_x # For scaling, e.g., lnmaxkm
        
        self.param_fixing_active = param_fixing_flag
        self.fixed_param_names = []
        self.fixed_ranges_np = np.array([])
        self.fixed_param_indices = []

        if self.param_fixing_active:
            if not fixed_param_names_path or not fixed_param_ranges_path:
                raise ValueError("Paths for fixed parameter names and ranges must be provided if param_fixing is active.")
            try:
                self.fixed_param_names = pd.read_csv(fixed_param_names_path).iloc[:, 1].values
                # Assuming fixed_param_ranges_path points to a .npy file as in original MLP
                # Original code: fixed_ranges = np.log(np.load(path_values) * 1e-3)
                # We need to ensure this transformation is applied if the stored .npy file is not already log-scaled.
                # For now, assume it's loaded as is, and if log-scaling is needed, it should be done when loading.
                # The problem description implies these are bounds for parameters like K_M, often log-transformed.
                # Let's assume the values in the .npy file are already in the correct (e.g., log) domain.
                self.fixed_ranges_np = np.load(fixed_param_ranges_path) 
                if len(self.fixed_param_names) != len(self.fixed_ranges_np):
                    raise ValueError("Mismatch between number of fixed parameter names and ranges.")
                
                # Get indices of fixed parameters
                name_to_idx = {name: i for i, name in enumerate(self.names_km_full)}
                self.fixed_param_indices = [name_to_idx[name] for name in self.fixed_param_names if name in name_to_idx]

            except Exception as e:
                raise RuntimeError(f"Error loading fixed parameter data: {e}")

        # Build the generator network
        # Input to MLP is just latent_dim, as cond_class is not needed
        self.network = self._build_network(self.latent_dim, self.n_parameters, hidden_dims_gen_mlp)

    def _build_network(self, input_dim, output_dim, hidden_dims):
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.5)) # Matching original Keras MLP dropout
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        # No final activation like tanh, raw outputs will be scaled.
        return nn.Sequential(*layers)

    def _param_fixer_pt(self, scaled_params_np_single_sample):
        """Applies parameter fixing to a single sample of scaled parameters."""
        if not self.param_fixing_active or len(self.fixed_param_indices) == 0:
            return scaled_params_np_single_sample

        fixed_params_copy = np.copy(scaled_params_np_single_sample)
        for i, param_idx_in_full_vector in enumerate(self.fixed_param_indices):
            # Sample from the range for this fixed parameter
            # self.fixed_ranges_np[i] is a pair [min_val, max_val] for the i-th fixed parameter
            min_val, max_val = self.fixed_ranges_np[i]
            fixed_params_copy[param_idx_in_full_vector] = np.random.uniform(min_val, max_val)
        return fixed_params_copy

    def sample_initial_parameters(self, n_samples=1):
        """Generates, scales, and optionally fixes initial parameters."""
        z_np = np.random.normal(0, 1, (n_samples, self.latent_dim))
        z_torch = torch.tensor(z_np, dtype=torch.float32)

        with torch.no_grad():
            unscaled_params_torch = self.network(z_torch)
        
        unscaled_params_np = unscaled_params_torch.cpu().numpy() # Shape (n_samples, n_parameters)

        # Scale parameters
        # hp.unscale_range expects gen_par, np.min(gen_par), np.max(gen_par), self.min_x, self.max_x
        # It scales based on the min/max of the *current batch* of unscaled_params_np.
        scaled_params_np = np.zeros_like(unscaled_params_np)
        for i in range(n_samples):
            # Scale each sample individually if min/max of batch is not desired for scaling each
            # Or scale based on overall min/max of the batch. Original MLP scales based on batch.
            # For n_samples=1, this is equivalent.
            sample_unscaled = unscaled_params_np[i, :]
            # If unscale_range is applied per sample, min/max should be per sample.
            # If applied to batch, min/max is of batch.
            # The original MLP's sample_parameters applies unscale_range to the whole batch `gen_par`.
            # Let's stick to that: scale the whole batch.
            if i == 0: # Perform scaling once for the whole batch
                 batch_scaled_params, _, _ = hp.unscale_range(unscaled_params_np, 
                                                              np.min(unscaled_params_np), 
                                                              np.max(unscaled_params_np), 
                                                              self.min_x, self.max_x)
            scaled_params_np[i,:] = batch_scaled_params[i,:]


        # Apply parameter fixing if active
        final_params_np = np.zeros_like(scaled_params_np)
        if self.param_fixing_active:
            for i in range(n_samples):
                final_params_np[i, :] = self._param_fixer_pt(scaled_params_np[i, :])
        else:
            final_params_np = scaled_params_np
        
        if n_samples == 1:
            return final_params_np.flatten(), z_np.flatten() # Return 1D arrays
        return final_params_np, z_np


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PPO Algorithm Components
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 512, 1024)):
        super(Actor, self).__init__()
        layers = []
        input_d = state_dim
        for hidden_d in hidden_dims:
            layers.append(nn.Linear(input_d, hidden_d))
            layers.append(nn.ReLU())
            input_d = hidden_d
        
        self.network = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        # Learnable log standard deviation for actions
        # Initialize to a small negative value for initial small exploration std dev
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5, dtype=torch.float32))

    def forward(self, state):
        x = self.network(state)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)
        # Expand std to match batch size of mean if it's not already
        if mean.ndim > 1 and std.ndim == 1 and std.shape[0] == mean.shape[1]:
             std = std.unsqueeze(0).expand_as(mean)
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dims=(256, 512, 1024)):
        super(Critic, self).__init__()
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
    def __init__(self, initial_generator_mlp, chk_jcbn, names_km,
                 # Dimensions will be inferred from initial_generator_mlp
                 actor_lr=1e-4, critic_lr=1e-4, # Adjusted learning rates
                 gamma=0.99, epsilon=0.2, gae_lambda=0.95,
                 ppo_epochs=10, num_episodes_per_update=64, # Renamed batch_size_ppo
                 T_horizon=5, k_reward_steepness=1.0,
                 action_clip_range=(-0.1, 0.1),
                 entropy_coeff=0.01, max_grad_norm=0.5):
        
        self.initial_generator_mlp = initial_generator_mlp # Instance of InitialGeneratorPT
        self.chk_jcbn = chk_jcbn
        self.names_km = names_km # Full list of parameter names, should match initial_generator_mlp.names_km_full

        # Get dimensions and bounds from the new InitialGeneratorPT
        self.param_dim = self.initial_generator_mlp.n_parameters
        self.z_dim = self.initial_generator_mlp.latent_dim 
        self.param_bounds = (self.initial_generator_mlp.min_x, self.initial_generator_mlp.max_x)

        # State dim: p_t (param_dim) + z (z_dim) + lambda_max (1) + t (1)
        self.state_dim = self.param_dim + self.z_dim + 1 + 1
        self.action_dim = self.param_dim

        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)
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
        
        # For final reward's simplified incidence part
        self.eig_partition_final_reward = -2.5 # Target for good eigenvalue
        self.n_samples_incidence = 10 # Reduced for speed in _compute_reward, user can tune

    def _get_lambda_max(self, p_tensor_single):
        # p_tensor_single should be a 1D tensor of shape (param_dim,)
        p_numpy = p_tensor_single.detach().cpu().numpy()
        
        # chk_jcbn._prepare_parameters expects a list of parameter arrays/lists
        # and names_km should be the specific names for *these* parameters.
        # Assuming names_km passed to PPORefinement is the full list,
        # and p_numpy corresponds to these names in order.
        self.chk_jcbn._prepare_parameters([p_numpy], self.names_km)
        max_eig_list = self.chk_jcbn.calc_eigenvalues_recal_vmax()
        return max_eig_list[0] # Assuming it returns the scalar max eigenvalue

    def _compute_reward(self, lambda_max_val, current_step_in_episode, is_final_param_set_pT):
        # current_step_in_episode is 1-indexed (1 to T)
        # lambda_max_val is for the parameter set p_current_step_in_episode
        
        intermediate_r = 1.0 / (1.0 + np.exp(self.k_reward_steepness * (lambda_max_val - (-2.5))))

        if is_final_param_set_pT: # This is p_T, reward is r_{T-1} but includes final term
            # Simplified incidence: 1 if lambda_max(p_T) is good, scaled penalty otherwise.
            # This is a proxy for "incidence(p_T)"
            if lambda_max_val <= self.eig_partition_final_reward:
                incidence_proxy_reward = 1.0 
            else:
                # Penalty similar to original reward_func, but for a single sample
                incidence_proxy_reward = 0.01 / (1 + np.exp(lambda_max_val - self.eig_partition_final_reward))
            return intermediate_r + incidence_proxy_reward
        return intermediate_r

    def _collect_trajectories(self):
        batch_states = []
        batch_actions = []
        batch_log_probs_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        all_episode_total_rewards = []

        for _ in range(self.num_episodes_per_update):
            # Episode initialization using InitialGeneratorPT
            # sample_initial_parameters returns (p0_1D_np_array, z_1D_np_array)
            p_current_numpy, z_numpy_for_state = self.initial_generator_mlp.sample_initial_parameters(n_samples=1)
            
            p_current_tensor = torch.tensor(p_current_numpy, dtype=torch.float32) # Should be 1D
            z_tensor = torch.tensor(z_numpy_for_state, dtype=torch.float32) # Should be 1D

            episode_total_reward = 0

            for t_step in range(self.T_horizon): # t_step from 0 to T-1
                # Current parameter set is p_t (p_current_tensor)
                # Calculate lambda_max(p_t) for state s_t
                lambda_max_pt_val = self._get_lambda_max(p_current_tensor)
                
                # State s_t = (p_t, z, lambda_max(p_t), t)
                state_tensor_flat = torch.cat((
                    p_current_tensor, 
                    z_tensor, 
                    torch.tensor([lambda_max_pt_val], dtype=torch.float32),
                    torch.tensor([t_step], dtype=torch.float32)
                )) # This is a 1D tensor

                # Select action a_t ~ pi(a_t | s_t)
                with torch.no_grad():
                    action_mean, action_std = self.actor(state_tensor_flat.unsqueeze(0)) # Add batch dim for actor
                    dist = Normal(action_mean, action_std)
                    action = dist.sample() # Shape (1, action_dim)
                    log_prob_action = dist.log_prob(action).sum(dim=-1) # Shape (1,)

                batch_states.append(state_tensor_flat) # Store 1D state
                batch_actions.append(action.squeeze(0)) # Store 1D action
                batch_log_probs_actions.append(log_prob_action) # Store scalar log_prob

                # Apply action: p_{t+1} = p_t + a_t
                action_clipped = torch.clamp(action.squeeze(0), self.action_clip_range[0], self.action_clip_range[1])
                p_next_tensor = p_current_tensor + action_clipped
                p_next_tensor = torch.clamp(p_next_tensor, self.param_bounds[0], self.param_bounds[1])

                # Calculate lambda_max(p_{t+1}) for reward r_t and next state s_{t+1}
                lambda_max_p_next_val = self._get_lambda_max(p_next_tensor)
                
                is_final_param_set = (t_step == self.T_horizon - 1) # p_next_tensor is p_T
                reward_val = self._compute_reward(lambda_max_p_next_val,
                                                  current_step_in_episode=t_step + 1,
                                                  is_final_param_set_pT=is_final_param_set)
                
                batch_rewards.append(torch.tensor([reward_val], dtype=torch.float32))
                episode_total_reward += reward_val

                # Next state s_{t+1} = (p_{t+1}, z, lambda_max(p_{t+1}), t+1)
                next_state_tensor_flat = torch.cat((
                    p_next_tensor,
                    z_tensor,
                    torch.tensor([lambda_max_p_next_val], dtype=torch.float32),
                    torch.tensor([t_step + 1], dtype=torch.float32)
                ))
                batch_next_states.append(next_state_tensor_flat)
                
                done_val = 1.0 if is_final_param_set else 0.0
                batch_dones.append(torch.tensor([done_val], dtype=torch.float32))

                p_current_tensor = p_next_tensor
            
            all_episode_total_rewards.append(episode_total_reward)

        avg_episode_reward = np.mean(all_episode_total_rewards) if all_episode_total_rewards else 0
        # print(f"Collected trajectories. Avg episode reward: {avg_episode_reward:.4f}")

        return (torch.stack(batch_states), torch.stack(batch_actions), 
                torch.stack(batch_log_probs_actions), torch.stack(batch_rewards), 
                torch.stack(batch_next_states), torch.stack(batch_dones),
                avg_episode_reward)


    def _compute_gae(self, rewards, values, next_values, dones):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))): # Iterate over total steps in batch
            # Check if this step is a terminal step for an episode
            # If dones[t] is 1, it means current s_t led to s_{t+1} which was terminal.
            # So V(s_{t+1}) should be 0 if s_{t+1} is terminal.
            # next_values[t] is V(s_{t+1})
            is_terminal_transition = dones[t].item() > 0.5 
            
            # If s_{t+1} is from a terminal state, its value is 0.
            # However, our 'dones' signal end of episode (T steps).
            # The value of s_T (next_state for t=T-1) is what critic estimates.
            # For GAE, if it's a true terminal state, V(s_term) = 0.
            # Here, episode ends due to horizon. So V(s_T) is not necessarily 0.
            
            delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_advantage
        returns = advantages + values
        return advantages, returns

    def update(self, trajectories_data):
        states, actions, log_probs_old, rewards, next_states, dones, _ = trajectories_data

        with torch.no_grad():
            values = self.critic(states)          # V(s_t)
            next_values = self.critic(next_states) # V(s_{t+1})

        advantages, returns = self._compute_gae(rewards, values, next_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_total_loss_epoch = 0
        critic_total_loss_epoch = 0

        for _ in range(self.ppo_epochs):
            # Actor update
            current_pi_mean, current_pi_std = self.actor(states)
            dist_new = Normal(current_pi_mean, current_pi_std)
            log_probs_new = dist_new.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist_new.entropy().mean()

            ratios = torch.exp(log_probs_new - log_probs_old.detach()) # log_probs_old needs to be (N,1)
            
            surr1 = ratios * advantages.detach() # advantages needs to be (N,1)
            surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            actor_total_loss_epoch += actor_loss.item()

            # Critic update
            values_pred = self.critic(states) # Re-evaluate V(s_t) with current critic
            critic_loss = (returns.detach() - values_pred).pow(2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            critic_total_loss_epoch += critic_loss.item()
        
        avg_actor_loss = actor_total_loss_epoch / self.ppo_epochs
        avg_critic_loss = critic_total_loss_epoch / self.ppo_epochs
        # print(f"Update: Avg Actor Loss: {avg_actor_loss:.4f}, Avg Critic Loss: {avg_critic_loss:.4f}")
        return avg_actor_loss, avg_critic_loss


    def train(self, num_iterations, output_path_base="ppo_training_output"):
        import os
        os.makedirs(output_path_base, exist_ok=True)
        
        all_iter_avg_rewards = []
        print(f"Starting PPO training for {num_iterations} iterations.")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}, z_dim: {self.z_dim}")
        print(f"Num episodes per update: {self.num_episodes_per_update}, Horizon T: {self.T_horizon}")

        for iteration in range(num_iterations):
            trajectories_data = self._collect_trajectories()
            avg_episode_reward = trajectories_data[-1] # Last element is avg_episode_reward
            
            actor_loss, critic_loss = self.update(trajectories_data)
            all_iter_avg_rewards.append(avg_episode_reward)
            
            if iteration % 10 == 0 or iteration == num_iterations -1 : # Log progress
                print(f"Iter {iteration:04d}: Avg Ep Reward: {avg_episode_reward:.4f}, "
                      f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
            
            if iteration % 50 == 0 or iteration == num_iterations -1 : # Save models periodically
                actor_path = os.path.join(output_path_base, f"actor_iter_{iteration}.pth")
                critic_path = os.path.join(output_path_base, f"critic_iter_{iteration}.pth")
                torch.save(self.actor.state_dict(), actor_path)
                torch.save(self.critic.state_dict(), critic_path)
                # print(f"Saved models at iteration {iteration}")
        
        print("Training finished.")
        return all_iter_avg_rewards
