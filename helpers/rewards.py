def reward_func(cfg, weights):
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
        pool.close()
    elif reward_flag == 1:
        max_eig.sort()
        considered_avg = sum(max_eig[:n_consider]) / n_consider
        this_reward = np.exp(-0.1 * considered_avg) / 2
        pool.close()

    return this_reward