import numpy as np
import pandas as pd
import os
random = np.random.RandomState(1039)
IndependentNormal = random.normal
from scipy.stats import norm

def QuadraticKernel(x1,x2,rho):
    def delta(x):
        return 0.75*(1-x**2) if x<=1 else 0
    
    y = np.linalg.norm(x1-x2)/rho
    return delta(y)

def makerank(responses):
    t = responses.shape[0]
    mf = np.zeros(shape=t**2)
    for i,vi in enumerate(responses):
        for j,vj in enumerate(responses):
            mf[i+(j)*t] = np.asscalar(vi>vj)
    return mf

def regret(output,response):
    incumbent   = output[0]
    best_output = []
    for _ in output:
        incumbent = _ if _ > incumbent else incumbent
        best_output.append(incumbent)
    opt       = max(response)
    orde      = list(np.sort(np.unique(response))[::-1])
    tmp       = pd.DataFrame(best_output,columns=['regret_validation'])
    
    tmp['rank_valid']        = tmp['regret_validation'].map(lambda x : orde.index(x))
    tmp['regret_validation'] = opt - tmp['regret_validation']
    return tmp

def propose(optimal, yhat,support):
    mu,var = np.split(yhat,2,axis=1)
    mu = mu.ravel()
    stddev = np.sqrt(var.ravel())
    with np.errstate(divide='warn'):
        imp = mu - optimal
        Z = imp / stddev
        ei = imp * norm.cdf(Z) + stddev * norm.pdf(Z)
    for _ in range(len(ei)):
        if _ in support:
            ei[_] = 0
    return np.argmax(ei)

def make_obs(searchspace,support,response):
    s_t = np.array([searchspace[support]])
    r_t = np.array([response])
    return np.concatenate([s_t,r_t],axis=-1)


def prepare_data(indexes,support,Lambda,response,metafeatures):
    X,E,Z,y,r = [],[],[],[],[]
    for dim in indexes:
        X.append(Lambda[dim])
        y.append(response[dim])
        Z.append(metafeatures)
        E.append(Lambda[support])
        r.append(response[support][:,None])
    X = np.array(X);E = np.array(E);Z = np.array(Z);y = np.array(y);r = np.array(r).astype(np.float32)
    return (np.expand_dims(E,axis=-1),r.squeeze(-1),np.expand_dims(X,axis=-1),Z), y

def get_filepaths(checkpoint_dir, members, random_seed=None):
    filepaths = []
    for _ in range(members):
        files = [_ for _ in os.listdir(os.path.join(checkpoint_dir,f"seed-{_}")) if _.startswith("model-") and _.endswith(".index")]
        assert(len(files)==1)
        filepaths.append(os.path.join(checkpoint_dir,f"seed-{_}",files[0].replace(".index","")))
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(filepaths)
    return filepaths[:members]
        

class Metric(object):
    def __init__(self, save_dir):
        import tensorflow as tf
        os.makedirs(save_dir,exist_ok=True)
        self.writer = tf.summary.create_file_writer(save_dir)
        self.loss_track = tf.keras.metrics.Mean(name="loss")
        self.log_var_track = tf.keras.metrics.Mean("log_var")
        self.mse_track = tf.keras.metrics.Mean("mse")
        self.scalar_fn = tf.summary.scalar
        self.message="loss: {loss:.2f} - log_var: {log_var:.2f} - mse: {mse:.2f}"
        
    def update(self,loss,logval,mean_squared_error):
        self.loss_track(loss)
        self.log_var_track(logval)
        self.mse_track(mean_squared_error)
    
    def reset(self,):
        self.loss_track.reset_states()
        self.log_var_track.reset_states()
        self.mse_track.reset_states()
    
    def report(self):
        return self.message.format(loss=self.loss_track.result().numpy(),
                            log_var=self.log_var_track.result().numpy(),
                            mse=self.mse_track.result().numpy(),)
    
    def get(self):
        return {"loss":self.loss_track.result().numpy(),
                "log_var":self.log_var_track.result().numpy(),
                "mse":self.mse_track.result().numpy()}
    
    def write(self, step):
        with self.writer.as_default():
            for key,val in self.get().items():
                self.scalar_fn(key, val, step=step)

class StandardMetric(object):
    def __init__(self, name, save_dir):
        import tensorflow as tf
        os.makedirs(save_dir,exist_ok=True)
        self.name = name
        self.writer = tf.summary.create_file_writer(save_dir)
        self.loss_track = tf.keras.metrics.Mean(name=name)
        self.scalar_fn = tf.summary.scalar
        
    def update(self,loss):
        self.loss_track(loss)
    
    def reset(self,):
        self.loss_track.reset_states()
    
    def report(self):
        raise NotImplementedError
    
    def get(self):
        return {self.name:self.loss_track.result().numpy()}
    
    def write(self, step):
        with self.writer.as_default():
            for key,val in self.get().items():
                self.scalar_fn(key, val, step=step)
                
def ptp(X):
    param_nos_to_extract = X.shape[1]
    domain = np.zeros((param_nos_to_extract, 2))
    for i in range(param_nos_to_extract):
        domain[i, 0] = np.min(X[:, i])
        domain[i, 1] = np.max(X[:, i])
    X = (X - domain[:, 0]) / np.ptp(domain, axis=1)
    return X                
