from __future__ import print_function
import numpy as np
import multiprocessing as mp
import pickle
import time
np.random.seed(0)


def worker_process(arg):
    get_reward_func, weights = arg
    return get_reward_func(weights)


class EvolutionStrategy(object):
    def __init__(self, weights, get_reward_func, savepath, population_size=50, sigma=0.1,
                 learning_rate=0.03, decay=0.999, num_threads=1, n_samples=100):

        self.weights = weights
        self.get_reward = get_reward_func
        self.save_path = savepath
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads
        self.n_samples = n_samples

    def _get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA * i
            weights_try.append(w[index] + jittered)
        return weights_try

    def get_weights(self):
        return self.weights

    def _get_population(self):
        population = []
        for i in range(self.POPULATION_SIZE):
            x = []
            for w in self.weights:
                x.append(np.random.randn(*w.shape))
            population.append(x)
        return population

    def _get_rewards(self, pool, population):
        if pool is not None:
            worker_args = ((self.get_reward, self._get_weights_try(self.weights, p)) for p in population)
            rewards = pool.map(worker_process, worker_args)

        else:
            rewards = []
            start = time.time()
            for p in population:
                weights_try = self._get_weights_try(self.weights, p)
                rewards.append(self.get_reward(weights_try))
        rewards = np.array(rewards)
        return rewards

    def _update_weights(self, rewards, population):
        std = rewards.std()
        if std == 0:
            return
        rewards = (rewards - rewards.mean()) / std
        for index, w in enumerate(self.weights):
            layer_population = np.array([p[index] for p in population])
            update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
            self.weights[index] = w + update_factor * np.dot(layer_population.T, rewards).T
        self.learning_rate *= self.decay

    def run(self, iterations, print_step=1):
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        print('starting evolution strategy')
        start = time.time()
        all_rewards = []

        for iteration in range(iterations):

            population = self._get_population()

            print(f'[{iteration}] getting rewards for the population ({len(population)*self.n_samples} reward calculations in total)')
            rewards = self._get_rewards(pool, population)

            print(f'[{iteration}] updating weights')
            self._update_weights(rewards, population)

            print(f'[{iteration}] getting reward for the updated weights ({self.n_samples} reward calculations in total)')
            this_reward = self.get_reward(self.weights)

            print(f'[{iteration}] saving weights')
            #save weights
            with open(f'{self.save_path}/weights_{iteration}.pkl', 'wb') as f:
                    pickle.dump(self.weights, f)

            if (iteration + 1) % print_step == 0:
                this_end = time.time()
                print(f'[{iteration}] reward: {this_reward:.3f}, time elapsed: {(this_end-start)/60:.2f} minutes')

            all_rewards.append(this_reward)
        if pool is not None:
            pool.close()
            pool.join()

        return np.array(all_rewards)
