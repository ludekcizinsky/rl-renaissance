import os
import numpy as np
import helper as hp
import multiprocessing as mp
from configparser import ConfigParser
import torch
from helpers.ppo_refinement import PPORefinement
from helpers.ppo_refinement import evaluate_policy_incidence

from kinetics.jacobian_solver import check_jacobian

#Parse arguments from configfile
configs = ConfigParser()
configs.read('configfile.ini')

n_samples = int(configs['MLP']['n_samples']) # Used by MLP for its internal sampling if any, and for p0 generation.

lnminkm = float(configs['CONSTRAINTS']['min_km'])
lnmaxkm = float(configs['CONSTRAINTS']['max_km'])

repeats = int(configs['EVOSTRAT']['repeats'])
generations = int(configs['EVOSTRAT']['generations']) # Will be used as num_iterations for PPO
ss_idx = int(configs['EVOSTRAT']['ss_idx'])
# n_threads = int(configs['EVOSTRAT']['n_threads']) # PPO collection is currently single-threaded

output_path = configs['PATHS']['output_path']
met_model = configs['PATHS']['met_model']
names_km_config = hp.load_pkl(f'models/{met_model}/parameter_names_km_fdp1.pkl') # Full list of param names

# Parameters needed directly by PPORefinement
param_dim_config = int(configs['MLP']['no_kms'])
latent_dim_config = int(configs['MLP']['latent_dim']) # For z vector in state


# Call solvers from SKimPy (Used only for initial messages now)
chk_jcbn = check_jacobian()

# Integrate data
print('---- Load kinetic and thermodynamic data')
chk_jcbn._load_ktmodels(met_model, 'fdp1')           ## Load kinetic and thermodynamic data
print('---- Load steady state data')
chk_jcbn._load_ssprofile(met_model, 'fdp1', ss_idx)  ## Integrate steady state information

print('--- Begin PPO refinement strategy')
for rep in range(repeats):
    this_savepath = f'{output_path}/ppo_repeat_{rep}/' 
    os.makedirs(this_savepath, exist_ok=True)

    # Instantiate PPORefinement agent with direct parameters for serial execution
    ppo_agent = PPORefinement(
        param_dim=param_dim_config,
        latent_dim=latent_dim_config,
        min_x_bounds=lnminkm,
        max_x_bounds=lnmaxkm,
        names_km_full=names_km_config, # Pass the full list of names
        chk_jcbn=chk_jcbn,             # Pass the jacobian checker instance
        # Optional: Pass other PPO hyperparameters if needed, e.g.,
        p0_init_std=1, # Default is 0.01
        # actor_lr=1e-4, # Default is 1e-4
        # critic_lr=1e-4, # Default is 1e-4
        # T_horizon=5, # Default is 5
        # num_episodes_per_update=64 # Default is 64
    )
    
    print(f"Repeat {rep}: Starting PPO training for {generations} iterations (serial execution).")
    ppo_iteration_rewards = ppo_agent.train(
        num_iterations=generations, # Use 'generations' from config as PPO iterations
        output_path_base=this_savepath
    )
    
    hp.save_pkl(f'{this_savepath}/ppo_iteration_rewards.pkl', ppo_iteration_rewards)
    print(f"Repeat {rep}: PPO training finished. Rewards log saved to {this_savepath}")
    
    best_actor_path = os.path.join(this_savepath, "best_actor.pth")
    print(f"Repeat {rep}: Evaluating policy incidence using {best_actor_path}...")
    incidence_rate, all_final_params = evaluate_policy_incidence(ppo_agent, best_actor_path, num_trials=50)
    print(f"Repeat {rep}: Incidence Rate: {incidence_rate:.4f}")
    hp.save_pkl(f'{this_savepath}/policy_incidence_results.pkl', {
        "incidence_rate": incidence_rate,
        "all_final_params": all_final_params
    })