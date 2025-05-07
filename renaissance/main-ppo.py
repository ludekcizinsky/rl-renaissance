import os
import numpy as np
import helper as hp
import multiprocessing as mp # Retained for now, though PPO doesn't use it directly.
from configparser import ConfigParser
import torch # Added for PyTorch

# from evostrat.init_mlp import MLP # Removed
from ppo_refinement import PPORefinement, InitialGeneratorPT # Added InitialGeneratorPT

from kinetics.jacobian_solver import check_jacobian

# ES-specific reward_func and its globals have been removed.
# PPO uses its own internal reward computation logic.

#Parse arguments from configfile
configs = ConfigParser()
configs.read('configfile.ini')

n_samples = int(configs['MLP']['n_samples']) # Used by MLP for its internal sampling if any, and for p0 generation.

lnminkm = float(configs['CONSTRAINTS']['min_km'])
lnmaxkm = float(configs['CONSTRAINTS']['max_km'])

repeats = int(configs['EVOSTRAT']['repeats'])
# save_step = int(configs['EVOSTRAT']['save_step']) # PPO has internal saving logic based on iterations
generations = int(configs['EVOSTRAT']['generations']) # Will be used as num_iterations for PPO
# pop_size = int(configs['EVOSTRAT']['pop_size']) # PPO uses num_episodes_per_update
# noise = float(configs['EVOSTRAT']['noise']) # PPO has its own exploration mechanism (std dev of policy)
# lr = float(configs['EVOSTRAT']['lr']) # PPO has actor_lr, critic_lr
# decay = float(configs['EVOSTRAT']['decay']) # PPO optimizers might have weight_decay if configured
ss_idx = int(configs['EVOSTRAT']['ss_idx'])
# n_threads = int(configs['EVOSTRAT']['n_threads']) # PPO collection is currently single-threaded

output_path = configs['PATHS']['output_path']
met_model = configs['PATHS']['met_model']
names_km_config = hp.load_pkl(f'models/{met_model}/parameter_names_km_fdp1.pkl') # Full list of param names

# Parameters for InitialGeneratorPT
latent_dim_gen = int(configs['MLP']['latent_dim'])
n_params_total = int(configs['MLP']['no_kms'])
gen_mlp_layer_1 = int(configs['MLP']['layer_1'])
gen_mlp_layer_2 = int(configs['MLP']['layer_2'])
gen_mlp_layer_3 = int(configs['MLP']['layer_3'])
hidden_dims_gen_mlp = (gen_mlp_layer_1, gen_mlp_layer_2, gen_mlp_layer_3)

pf_flag = int(configs['PARAMETER_FIXING']['pf_flag'])
fixed_param_names_path_config = configs['PARAMETER_FIXING']['fixed_parameter_names']
fixed_param_ranges_path_config = configs['PARAMETER_FIXING']['fixed_parameter_ranges']


# Call solvers from SKimPy
chk_jcbn = check_jacobian()

# Integrate data
print('---- Load kinetic and thermodynamic data')
chk_jcbn._load_ktmodels(met_model, 'fdp1')           ## Load kinetic and thermodynamic data
print('---- Load steady state data')
chk_jcbn._load_ssprofile(met_model, 'fdp1', ss_idx)  ## Integrate steady state information

print('--- Begin PPO refinement strategy')
for rep in range(repeats):
    # Instantiate the new PyTorch-based InitialGeneratorPT
    initial_gen_pt = InitialGeneratorPT(
        latent_dim=latent_dim_gen,
        n_parameters=n_params_total,
        hidden_dims_gen_mlp=hidden_dims_gen_mlp,
        names_km_full=names_km_config,
        min_x=lnminkm,
        max_x=lnmaxkm,
        param_fixing_flag=bool(pf_flag), # Ensure it's boolean
        fixed_param_names_path=fixed_param_names_path_config if bool(pf_flag) else None,
        fixed_param_ranges_path=fixed_param_ranges_path_config if bool(pf_flag) else None
    )

    this_savepath = f'{output_path}/ppo_repeat_{rep}/' # Modified path to distinguish from ES runs
    os.makedirs(this_savepath, exist_ok=True)

    # Instantiate PPORefinement agent with the new InitialGeneratorPT
    ppo_agent = PPORefinement(
        initial_generator_mlp=initial_gen_pt, 
        chk_jcbn=chk_jcbn,
        names_km=names_km_config, # Pass the full list of names
        # PPO hyperparameters can be tuned here or loaded from a new config section
    )
    
    print(f"Repeat {rep}: Starting PPO training for {generations} iterations.")
    # The train method in PPORefinement will save models to output_path_base (this_savepath)
    ppo_iteration_rewards = ppo_agent.train(
        num_iterations=generations, # Use 'generations' from config as PPO iterations
        output_path_base=this_savepath
    )
    
    hp.save_pkl(f'{this_savepath}/ppo_iteration_rewards.pkl', ppo_iteration_rewards)
    print(f"Repeat {rep}: PPO training finished. Rewards log saved to {this_savepath}")
