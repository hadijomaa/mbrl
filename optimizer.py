from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import scipy.stats as stats
from sklearn.metrics import euclidean_distances
RandomGenerator = np.random.RandomState(3019)
import copy
class Optimizer(object):

    def __init__(self, sol_dim, max_iters, popsize, num_elites,best_in_traj,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """

        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites
        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha
        self.best_in_traj = best_in_traj
        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        self.num_opt_iters, self.mean, self.var = None, None, None

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var,cost_fn,verbose=False):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        mean, var, t = init_mean, init_var, 0
        X = stats.norm(loc=np.zeros_like(mean), scale=np.ones_like(mean))
        X.random_state=np.random.RandomState(2934)
        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
            costs = cost_fn(samples)
            elites = samples[np.argsort(costs)][:self.num_elites]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)
            if verbose:
                print("Cost:{y:.4f}, X:{X}".format(y=min(costs),X=elites[0]))
            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var
            t += 1

        return mean
    
    def obtain_discrete_solution(self, cost_fn, card, nonavailable, plan_hor,
                                 init_mean,init_var, dK, data):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            dK (list): A list that includes the number of subactions per hyperparameter
            
        Remarks
        -------    
        In a setting where action repetition is allowed, this would be the correct
        implementation. However, since for HPO, it does not make any sense to 
        select a hyper-parameter multiple times, we resort to the third approach.
        
        """
        mean, var, t = init_mean, init_var, 0
        X = stats.norm(loc=np.zeros_like(mean), scale=np.ones_like(mean))
        X.random_state=np.random.RandomState(2934)        
        available = np.setdiff1d(np.arange(card),nonavailable)
        
        while (t < self.max_iters):
            lb_dist, ub_dist = mean - np.zeros_like(mean), np.ones_like(mean) - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            samples = X.rvs(size=[self.popsize, len(mean)]) * np.sqrt(constrained_var) + mean
            sample_reshape = samples.reshape(self.popsize, plan_hor, dK) 
            samples,indices = hpo(sample_reshape, data, available)
            samples = samples.reshape(self.popsize,-1)
            indices = np.array(indices).reshape(self.popsize,-1)
            costs, trajectory_regret = cost_fn(samples)
            elites = samples[np.argsort(costs)][:self.num_elites]
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)
            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var
            t += 1
        mean ,indexes = hpo(mean.reshape(1,plan_hor,dK) , data , available)
        mean = mean.reshape(-1,)
        if not self.best_in_traj:
            indexes = indexes[0] 
            return mean, indexes
        else:
            _, trajectory_regret = cost_fn(mean.reshape(1,-1))
            min_regret,min_idx = np.min(trajectory_regret,1),np.argmin(trajectory_regret,1)
            min_between_traj = np.argmin(min_regret)
            indexes = np.array([np.array(indexes)[min_between_traj,min_idx[min_between_traj]]])
            mean = np.concatenate([mean[min_idx[min_between_traj]*dK:(min_idx[min_between_traj]+1)*dK],
                                   np.zeros(dK*(plan_hor-1))])
            return mean,indexes

    def obtain_random_solution(self, cost_fn, card, nonavailable, plan_hor, utility_fn):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            card (int): cardinality of the hyperparameter space
            nonavailable (np.array): list of hyperparameter that have been selected
            plan_hor (int): total number of actions allowed
            
        """
        available = np.setdiff1d(np.arange(card),nonavailable)
        if plan_hor>1:
            indexes = []
            for _ in range(self.popsize):
                RandomGenerator.shuffle(available)
                indexes.append(copy.deepcopy(available[:plan_hor]))
        else:
            indexes = available.reshape(-1,plan_hor)
        samples = utility_fn(np.array(indexes))
        costs, trajectory_regret = cost_fn(samples)
        if not self.best_in_traj:
            return samples[np.argmin(costs)][None],indexes[np.argmin(costs)]
        else:
            min_regret,min_idx = np.min(trajectory_regret,1),np.argmin(trajectory_regret,1)
            min_between_traj = np.argmin(min_regret)
            indexes = np.array(indexes)[min_between_traj,min_idx[min_between_traj]]
            samples = utility_fn(np.array(indexes.reshape(-1,1)))
            return samples[0][None],[indexes]
    

# class JointMultinomial:
    
#     def __init__(self,mean,num_config_per_hyp):
#         X, z, i = [], 0, 0
#         for j in num_config_per_hyp:
#             z+=j
#             X.append(stats.multinomial(1,mean[i:z]))
#             X[-1].random_state = np.random.RandomState(391)
#             i+=j
#         self.X = X
        
#     def rvs(self,size):
#         output = []
#         for X in self.X:
#             output.append(X.rvs(size=size))
#         return np.concatenate(output,axis=-1)

def hpo(x, X, available):
    ## return unique popsize x plan_hor x dK from available
    nx = []
    nx_idx = []
    X = X.X[0][np.array(available)]
    for pop in x:
        pop_available = np.arange(len(available)).tolist()
        # sort by nearest value to grid
        indices = euclidean_distances(pop,X).argsort().tolist()
        px = [indices[0][0]]
        for hor_idx in indices[1:]:
            pop_available.pop(pop_available.index(px[-1]))
            while hor_idx[0] not in pop_available:
                hor_idx.pop(0)
            px.append(hor_idx[0])
        nx_idx.append(np.array(available)[px])
        nx.append(X[px])
    return np.array(nx),nx_idx 
        
if __name__=="__main__":
    def branin(x):
        # the Branin function (2D)
        # https://www.sfu.ca/~ssurjano/branin.html
        x1 = x[:, 0]
        x2 = x[:, 1]
    
        # parameters
        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
    
        bra = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    
        return bra
    plan_hor = 1
    ndim = 2
    lb = np.array([[-5,0]])
    ub = np.array([[0,15]])
    broptimizer = Optimizer(ndim*plan_hor,1000,1000,100,upper_bound=np.tile(ub,[plan_hor]),lower_bound=np.tile(lb,[plan_hor]))
    random = np.random.RandomState(301)
    init_mean = random.uniform(lb,ub)
    init_var  = np.ones(lb.shape) 
    broptimizer.obtain_solution(np.tile(init_mean,[plan_hor]),np.tile(init_var,[plan_hor]),branin,True)
