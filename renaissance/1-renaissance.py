import os
import numpy as np
import helper as hp
import multiprocessing as mp
from configparser import ConfigParser

from evostrat.init_mlp import MLP
from evostrat.evolution_strategy import EvolutionStrategy

from kinetics.jacobian_solver import check_jacobian

# declare reward functions

def reward_func(weights):
    """
    evaluate reward for a set of generated kinetic parameter sets
    :param gen_params: agent generated kinetic parameter sets
    :return: reward
    """

    global calc_eig
    global n_samples
    global eig_partition
    global n_threads
    global names_km
    global reward_flag
    global n_consider

    def calc_eig(gen_param):
        chk_jcbn._prepare_parameters(gen_param, names_km)
        max_eig = chk_jcbn.calc_eigenvalues_recal_vmax()
        return max_eig

    pool = mp.Pool(n_threads)
    mlp.generator.set_weights(weights)
    gen_params = mlp.sample_parameters()
    gen_params = [[params] for params in gen_params]
    max_eig = pool.map(calc_eig, gen_params)
    max_eig = np.array([this_eig for eig in max_eig for this_eig in eig])

    if reward_flag == 0:
        max_neg_eig = np.min(max_eig)
        if max_neg_eig > eig_partition:
            this_reward = 0.01 / (1 + np.exp(max_neg_eig - eig_partition))
        else:
            this_reward = len(np.where(max_eig <= eig_partition)[0]) / n_samples
    elif reward_flag == 1:
        max_eig.sort()
        considered_avg = sum(max_eig[:n_consider]) / n_consider
        this_reward = np.exp(-0.1 * considered_avg) / 2
    
    pool.close()
    pool.join()

    return this_reward

#Parse arguments from configfile
configs = ConfigParser()
configs.read('configfile.ini')

n_samples = int(configs['MLP']['n_samples'])

lnminkm = float(configs['CONSTRAINTS']['min_km'])
lnmaxkm = float(configs['CONSTRAINTS']['max_km'])

repeats = int(configs['EVOSTRAT']['repeats'])
save_step = int(configs['EVOSTRAT']['save_step'])
generations = int(configs['EVOSTRAT']['generations'])
pop_size = int(configs['EVOSTRAT']['pop_size'])
noise = float(configs['EVOSTRAT']['noise'])
lr = float(configs['EVOSTRAT']['lr'])
decay = float(configs['EVOSTRAT']['decay'])
ss_idx = int(configs['EVOSTRAT']['ss_idx'])
n_threads = int(configs['EVOSTRAT']['n_threads'])

output_path = configs['PATHS']['output_path']
met_model = configs['PATHS']['met_model']
names_km = hp.load_pkl(f'data/{met_model}/parameter_names_km_fdp1.pkl')

reward_flag = int(configs['REWARDS']['reward_flag'])
eig_partition = float(configs['REWARDS']['eig_partition'])
n_consider = int(configs['REWARDS']['n_consider'])


# Call solvers from SKimPy
chk_jcbn = check_jacobian()

# Integrate data
print('---- Load kinetic and thermodynamic data')
chk_jcbn._load_ktmodels(met_model, 'fdp1')           ## Load kinetic and thermodynamic data
print('---- Load steady state data')
chk_jcbn._load_ssprofile(met_model, 'fdp1', ss_idx)  ## Integrate steady state information

print('--- Begin evolution strategy')
for rep in range(repeats):
    # Call neural network agent
    mlp = MLP(lnminkm, lnmaxkm, n_samples, names_km)
    init_dict = mlp.generator.get_weights()

    this_savepath = f'{output_path}/repeat_{rep}/'
    os.makedirs(this_savepath, exist_ok=True)

    es = EvolutionStrategy(mlp.generator.get_weights(),
                           reward_func, this_savepath,
                           population_size= pop_size,
                           sigma= noise,  # noise std deviation
                           learning_rate=lr,
                           decay=decay,
                           num_threads=1,#TODO: change to n_threads when we resolve the multiprocessing issue
                           n_samples=n_samples) 

    rewards = es.run(generations, print_step=save_step)
    hp.save_pkl(f'{this_savepath}/rewards', rewards)

