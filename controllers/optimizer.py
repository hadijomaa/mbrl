import copy

import numpy as np
from scipy import stats


class RandomShooter(object):
    """
    Random Shooting Class
    """
    def __init__(self, num_random_trajectories, seed=0):
        """
        Initialize the Optimizer Class

        Args:
            num_random_trajectories (int): The number of candidate solutions to be sampled at every iteration
        """

        self.num_random_trajectories = num_random_trajectories
        self.randomizer = np.random.RandomState(seed=seed)

    def shoot(self, cost_function, candidate_pool, horizon, utility_function, apply_lookahead=False):
        """
        Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            cost_function (lambda func): A function for computing costs over a batch of candidate solutions.
            candidate_pool (np.array): array of available candidates to be evaluated
            horizon (int): number of steps to investigate
            apply_lookahead (bool): indicator if we employ lookahead to select the best candidate
            utility_function (lambda func):
        """

        if horizon > 1:
            indexes = []
            for _ in range(self.num_random_trajectories):
                self.randomizer.shuffle(candidate_pool)
                indexes.append(copy.deepcopy(candidate_pool[:horizon]))
        else:
            indexes = candidate_pool.reshape(-1, 1)

        # get hyperparameter representation from indexes
        samples = utility_function(np.array(indexes))
        # perform simulated rollout
        info = dict()
        action_particle_regret = cost_function(samples)
        # action regret is num_random_trajectories x horizon x particles
        # average over particles --> num_random_trajectories x horizon
        # calculate regret here
        action_regret = action_particle_regret.mean(axis=-1)
        trajectory_cost = action_regret.min(axis=1)
        info["max_trajectory_regret"] = max(trajectory_cost)
        info["best_expected_horizon"] = int(stats.mode(action_regret.argmin(axis=1), keepdims=False).mode.item())
        lowest_cost = np.argmin(trajectory_cost)
        best_trajectory_indexes = indexes[lowest_cost]
        info["expected_regret"] = action_regret[lowest_cost][0]
        info["rollout_regret"] = np.min(action_regret[lowest_cost])
        if apply_lookahead:
            n = np.argmin(best_trajectory_indexes)
        else:
            n = 0
        return best_trajectory_indexes[n], info
