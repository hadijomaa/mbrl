import copy

import numpy as np


class RandomShooter(object):

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
        costs, trajectory_regret = cost_function(samples)
        if apply_lookahead:
            min_regret, min_idx = np.min(trajectory_regret, 1), np.argmin(trajectory_regret, 1)
            min_between_traj = np.argmin(min_regret)
            indexes = np.array(indexes)[min_between_traj, min_idx[min_between_traj]]
            samples = utility_function(np.array(indexes.reshape(-1, 1)))
            return samples[0][None], [indexes]

        return samples[np.argmin(costs)][None], indexes[np.argmin(costs)]
