from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
random = np.random.RandomState(12393)
from optimizer import Optimizer


class MPC(object):

    def __init__(self, model, params, data, x, dataset_id):
        
        self.dU = params["dU"]
        self.CEM = params["opt_mode"]=="CEM"
        self.model = model
        self.dataset_id = dataset_id
        self.num_particles = params["num_particles"]
        self.data = data
        self.card = data.X[0].shape[0]
        self.plan_hor = params["plan_hor"]

        # Create action sequence optimizer
        self.optimizer = Optimizer(
            sol_dim=None,
            max_iters = params["max_iters"],
            num_elites = params["num_elites"],
            popsize = params["popsize"],
            best_in_traj = params["best_in_traj"],
        )
        # Controller state variables
        self.has_been_trained = True
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.nonavailable = x
        self.ac_ub = np.ones(self.dU)
        self.ac_lb = np.zeros(self.dU)
        self.prev_sol = self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])

        self.reset()
        
    def train(self, *args):
        self.has_been_trained = True

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).
        Returns: None
        """
        self.optimizer.reset()#

    def act(self, obs, trajectory=None):
        """Returns the action that this controller would take at time t given observation obs.
        Arguments:
            obs: The current observation
            data: (DataGenerator) class that contains all tasks of a meta-dataset
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.
        Returns: An action (and possibly the predicted cost)
        """
        
        if self.ac_buf.shape[0] > 0: 
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            return action, trajectory
        cost_fn = lambda acs: self._compile_cost(acs, obs,self.data)
        if not self.CEM:
            mat2rep = lambda x: matrix_to_rep(x,self.data,self.dataset_id)
            soln,index = self.optimizer.obtain_random_solution(cost_fn, self.card, self.nonavailable, self.plan_hor, mat2rep) #### plan_hor x dU
        else:
            soln, index = self.optimizer.obtain_discrete_solution(cost_fn, self.card,
                                                                             self.nonavailable, self.plan_hor,
                                                                             self.prev_sol,self.init_var, self.dU, self.data) #### plan_hor x dU        
            self.prev_sol = np.concatenate([np.copy(soln)[self.dU:], np.zeros(self.dU)])
        self.nonavailable.append(index[0])
        self.ac_buf = soln.reshape(-1,)[:self.dU].reshape(-1, self.dU)
        return self.act(obs, index)

    def _compile_cost(self, ac_seqs, obs, data):
        t, nopt = 0, ac_seqs.shape[0]   # 1 x (dUxplan_hor) # t = obs.shape[0]
        pred_regret = np.ones([nopt, self.num_particles])
        actual_regret = np.ones([nopt, self.plan_hor])
        ac_seqs = np.reshape(ac_seqs, [-1, self.plan_hor, self.dU]) ### nopt x plan_hor x dU
        ac_seqs = np.reshape(np.tile(
            np.transpose(ac_seqs, [1, 0, 2])[:, :, None], ### plan_hor x nopt x 1 x dU
            [1, 1, self.num_particles, 1] ### plan_hor x nopt x npart x dU
        ), [self.plan_hor, -1, self.dU]) ##### plan_hor x (npart*nopt) x dU
        obs = np.tile(obs[None], [nopt * self.num_particles, 1, 1]) #### (nopt*npart)xtx(dU+1) ---> batch ? 
        def continue_prediction(t, *args):
            return np.less(t, self.plan_hor)

        def iteration(t, cur_regret, cur_obs,act_regret):
            cur_acs = ac_seqs[t]
            next_obs,next_perf = self._predict_next_obs(cur_obs, cur_acs)
            next_perf = np.reshape(
                    next_perf, [-1, self.num_particles] #### nopt x npart
                )
            next_regret = np.maximum(np.minimum(1 - next_perf, 1),0) #### regret bounded between 0 and 1
            act_regret[:,t:t+1] = np.mean(next_regret,1,keepdims=True)
            return t + 1, np.minimum(cur_regret,next_regret), next_obs,act_regret
        while continue_prediction(t):
            t,pred_regret,obs,actual_regret = iteration(t,pred_regret,obs,actual_regret)
        return np.mean(pred_regret, axis=1),actual_regret

    def _predict_next_obs(self, obs, acs):
        # set up model input
        # acs --> (npart*nopt) x dU
        # obs --> (nopt*npart)xtx(dU+1)
        e = obs[:,:,:-1,None]
        r = obs[:,:,-1:]
        # true_reward = get_res(acs,self.data)
        acs = acs[:,:,None]
        metafeatures = np.repeat(self.data.Z[0][0][None],acs.shape[0],0)
        inputs = (e, r, acs, metafeatures)
        output = self.model(inputs)# ----> (nopt*npart)x1
        assert((e[0]==e[1]).all())
        assert((acs[0]==acs[1]).all())
        mean, var = np.split(output.numpy(),2,1)
        predictions = mean + random.normal(size=np.shape(mean), loc=0, scale=1) * np.sqrt(var)
        
        model_out_dim = predictions.shape[-1]

        predictions = np.reshape(predictions, [-1, self.num_particles, model_out_dim]) # ----> noptx npart x 1
        prediction_mean = np.mean(predictions, axis=1, keepdims=True) # ----> noptx 1 x 1
        prediction_var = np.mean(np.square(predictions - prediction_mean), axis=1, keepdims=True)
        z = random.normal(size=np.shape(predictions), loc=0, scale=1)
        samples = prediction_mean + z * np.sqrt(prediction_var) # ----> noptx npart x 1
        predictions = np.reshape(samples, [-1, model_out_dim])# ----> (noptx npart) x 1
        
        new_state = np.concatenate([acs.squeeze(axis=-1),predictions],axis=-1)[:,None] ### ---> (npart*nopt) x 1 x (dU+1)
        # new_state = np.concatenate([acs.squeeze(axis=-1),true_reward.reshape(-1,1)],axis=-1)[:,None] ### ---> (npart*nopt) x 1 x (dU+1)
        next_obs  = np.concatenate([obs,new_state],axis=1)### ---> (npart*nopt) x (t+1) x (dU+1)
        
        return next_obs,predictions.reshape(model_out_dim,-1)

# get_res = lambda x, d: hpo(x,d,0)[0]

_gather = lambda x,X: np.where(np.all(x == X, axis=1))[0]

def index_to_rep(x,data, dataset):
    assert(x.ndim==1)
    return data.X[dataset][x]

def matrix_to_rep(x,data,dataset):
    assert(x.ndim==2)
    output = []
    for i in x:
        output.append(index_to_rep(i,data,dataset).reshape(1,-1))
    return np.vstack(output)
    
def hpo(x, data, dataset, return_index=False):
    X = data.X[dataset]
    Y = data.Y[dataset]
    no_duplicates = pd.DataFrame(X).drop_duplicates().index.tolist()
    idx = []
    X = X[no_duplicates]
    Y = Y[no_duplicates]
    for _x in x:
        loc = _gather(_x,X)
        idx.append(loc)
    idx = np.concatenate([idx])
    ret = Y[idx].reshape(-1,1)
    assert np.size(ret) == x.shape[0]
    if not return_index:
        return ret, X[idx].squeeze(1)
    else:
        return ret, X[idx].squeeze(1), idx

def sparse_hpo(x, data, dataset):
    '''
    Function that return the actual response and encoded hyper-parameter 
    configuration from the sparse hyper-parameter vector.

    Parameters
    ----------
    x : (np.array) (mxn_sparse) one-hot encoded hyperparameter configuration
    data : (DataGenerator) class that includes the meta-datasets.
    dataset : (int) index of the test task in the list

    Returns
    -------
    tuple((mx1),(mxn)) Response of the hyper-parameter(s) and the encoded
    hyper-parameter(s) representation.
    
    Remarks
    -------    
    Some hyper-parameter configurations in Optimization/Layout Md are not 
    available in the grid. For example, configurations with 1 LAYER and ASC 
    ARCHITECTURE. This is because the model looks exactly like the one with 
    1 LAYER and SQU ARCHITECTURE. This was done to remove redundancy in an 
    earlier paper. This gives rise to a problem when sampling from the 
    categorical distributions gives us 1 LAYER and ASC ARCHITECTURE. As a 
    workaround, we simply switch the ASC ARCHITECTURE with the SQU ARCHITECTURE
    every time 1 LAYER is selected. 
    
    '''
    X = data.X[dataset]
    Y = data.Y[dataset]
    arch_cols = [i  for i,_ in enumerate(data.X_sparse[dataset].columns.tolist()) \
                 if _.startswith("ARCHITE") and "SQU" not in _]
    squ_index = [i  for i,_ in enumerate(data.X_sparse[dataset].columns.tolist()) \
                 if "SQU" in _][0]
    layer_index = [i  for i,_ in enumerate(data.X_sparse[dataset].columns.tolist()) \
                   if _=="LAYERS_1"][0]
    X_sparse = np.array(data.X_sparse[dataset])
    idx = []
    for _x in x:
        loc = _gather(_x,X_sparse)
        if np.size(loc)==0:
            assert(sum(_x[arch_cols])==1) ### make sure that non-SQU Architecture
            assert(_x[layer_index]==1) ### make sure LAYER_1
            _x[arch_cols] = 0
            _x[squ_index]=1
            loc = _gather(_x,X_sparse)
        idx.append(loc)
    idx = np.concatenate([idx])
    ret = Y[idx].reshape(-1,1)
    assert np.size(ret) == x.shape[0]
    return ret, X[idx].squeeze(1)

def dense_hpo(x,t,data):
    '''
    Parameters
    ----------    
    x : (mx(txn_sparse)) sequence of (t) one-hot encoded hyper-parameters
    t: (int) length of the sequence.
    data: (DataGenerator) class that contains all tasks of a meta-dataset

    Returns
    -------
    x_t : (mx(txn)) sequence of (t) encoded hyper-parameters.

    '''
    x   = np.split(x,t,axis=1)
    x_t = [sparse_hpo(_,data,0)[1] for _ in x]
    x_t = np.concatenate(x_t,axis=1)
    return x_t

if __name__=="__main__":
    from modules.net import deepset as net
    import pandas as pd
    import os
    from batchloader import DataGenerator
    
    testparams = {}
    testparams["index"] = 0
    testparams["split"] = 0
    testparams["searchspace"] = "a"
    rootdir         = os.path.dirname(os.path.realpath(__file__))
    test_dataset_id = pd.read_csv(os.path.join(rootdir,"dataset_id_splits.csv"),index_col=0)[f"test-{testparams['split']}"].dropna().astype(int).ravel()[testparams['index']]
    metafeatures = pd.read_csv(os.path.join(rootdir,"metafeatures",f"meta-features-split-{testparams['split']}.csv"),index_col=0)
    
    searchspace = {"a":"Layout Md", # ----> dU_sparse = 10
                   "b":"Regularization Md",
                   "c":"Optimization Md"}
    
    datagenerator = DataGenerator(rootdir, [test_dataset_id],metafeatures,shuffle=True,transformation=testparams["searchspace"])
    
    configuration = {}
    configuration["output_size_D"] = [128]*4
    configuration["units_C"] = 128
    configuration["output_size_B"] = [128]*2  + [2]    
    model = net(configuration)
    
    params = {}
    params["dU"]   = datagenerator.X[0].shape[1]
    params["npart"]= 100
    
    params["opt_mode"] = "CEM"
    params["plan_hor"] = 1
    
    params["log_traj_preds"] = False
    params["log_particles"]  = False
    
    params["max_iters"] = 0
    params["popsize"] = 20
    params["num_elites"] = 1
    params["subaction_dim"] = [0]
    dataset_id = 0
    obs = random.rand(3,params["dU"]+1)
    controller = MPC(model,params,datagenerator,x=random.randint(0,100,3).tolist(), dataset_id=dataset_id)
    action = controller.act(obs)
    print("The observed reward for the action {action} is {reward}".format(action=action,reward=hpo(x=action[None], data=datagenerator,dataset=dataset_id)[0].item()))
