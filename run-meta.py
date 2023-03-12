import pandas as pd
import os
from batchloader import DataGenerator
from modules.net import deepset as net
from net_utils import mse,nll,log_var
import datetime
import numpy as np
import argparse
import tqdm 
import tensorflow as tf
# create parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--split', help='Select training fold', type=int,default=0)
parser.add_argument('--searchspace', help='Select searchspace ', type=str,default="a")
parser.add_argument('--seed', help='Select seed ', type=int,default=0)
args    = parser.parse_args()
random = np.random.RandomState(args.seed)
RandomIndexGenerator= np.random.RandomState(args.seed)
rootdir     = os.path.dirname(os.path.realpath(__file__))

training_dataset_ids = pd.read_csv(os.path.join(rootdir,"dataset_id_splits.csv"),index_col=0)[f"train-{args.split}"].dropna().astype(int).ravel()
validation_dataset_ids = pd.read_csv(os.path.join(rootdir,"dataset_id_splits.csv"),index_col=0)[f"valid-{args.split}"].dropna().astype(int).ravel()
metafeatures = pd.read_csv(os.path.join(rootdir,"metafeatures",f"meta-features-split-{args.split}.csv"),index_col=0)

searchspace = {"a":"Layout Md",
               "b":"Regularization Md",
               "c":"Optimization Md",
               "d":"Tree Md"}
datagenerator = DataGenerator(rootdir,training_dataset_ids,metafeatures,shuffle=True,transformation=args.searchspace)

validgenerator = DataGenerator(rootdir,validation_dataset_ids,metafeatures,transformation=args.searchspace,fixed_history=5,
                               batch_size=None,shuffle=False)

configuration = {}
configuration["output_size_D"] = [128]*4
configuration["units_C"] = 128
configuration["output_size_B"] = [128]*2  + [2]

model = net(configuration)
backend = net(configuration)
compiled_model = net(configuration)
model.directory = os.path.join(rootdir,"checkpoints",searchspace[args.searchspace],str(args.seed),f"split-{args.split}",datetime.datetime.now().strftime('meta-v2-%Y-%m-%d-%H-%M-%S-%f'))
filepath    = os.path.join(model.directory,"model-{epoch:02d}")
##### do one dummy forward pass
compiled_model.compile(loss=nll,optimizer="adam",metrics=[log_var,mse])
dummy,_ = datagenerator.__getmetaitem__(0,np.arange(32))
_ = model(dummy)
_ = backend(dummy)
_ = compiled_model(dummy)
backend.set_weights(model.get_weights())
compiled_model.set_weights(model.get_weights())




inner_optimizer=tf.keras.optimizers.Adam(1e-3,beta_1=0.)
outer_optimizer=tf.keras.optimizers.SGD(1e-3)
os.makedirs(model.directory,exist_ok=True)

p = np.arange(len(training_dataset_ids))
epochs         = 1000
n_inner_steps  = 5
minibatch_size = None ## not used
task_batchsize = 8
ntaskbatch = p.shape[0]//task_batchsize

configuration.update(vars(args))
configuration["n_inner_steps"] = n_inner_steps
configuration["minibatch_size"] = minibatch_size
configuration["task_batchsize"] = task_batchsize
configuration["ntaskbatch"] = ntaskbatch

import json
json.dump(configuration, open(os.path.join(model.directory,"configuration.json"),"w"))
class Metric(object):
    def __init__(self,):
        self.loss_track = tf.keras.metrics.Mean(name="loss")
        self.log_var_track = tf.keras.metrics.Mean("log_var")
        self.mse_track = tf.keras.metrics.Mean("mse")
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
    
tracker = Metric()
epochBar = tqdm.tqdm(range(epochs))
train_log_dir = os.path.join(model.directory,"train")
os.makedirs(train_log_dir,exist_ok=True)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

valid_log_dir = os.path.join(model.directory,"valid")
os.makedirs(valid_log_dir,exist_ok=True)
valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

for epoch in epochBar:
    tracker.reset()
    random.shuffle(p)
    for taskbatchID in range(ntaskbatch): ### task batch size equals to number of algorithms
        taskbatch = p[taskbatchID*task_batchsize:(taskbatchID+1)*task_batchsize]
        #### store weights
        theta =  model.get_weights() 
        theta_i = []
        for taskID in taskbatch:
            model.set_weights(theta)
            file = taskID
            ##### reset weights
            for inner_steps in range(n_inner_steps):
                minibatch,targetminibatch = datagenerator.__getmetaitem__(file,np.arange(datagenerator.X[0].shape[0]))
                
                with tf.GradientTape() as tape:
                    output = model(minibatch,training=True)
                    loss = nll(targetminibatch,output)
                grads = tape.gradient(loss, model.trainable_variables)
                inner_optimizer.apply_gradients(zip(grads, model.trainable_variables))
                tracker.update(loss,log_var(targetminibatch,output),mse(targetminibatch,output))
            theta_i.append(model.get_weights())
        theta = [tf.reduce_mean(tf.stack(_,axis=0),axis=0) for _ in zip(*theta_i)]
        model.set_weights(theta)            

        # update the weights of the original model
        meta_grads = [tf.subtract(old, new) for old,new in zip(backend.trainable_variables,
                                                                model.trainable_variables)]
        # \phi = \phi - 1/n (\phi-W)
        # call the optimizer on the original vars based on the meta gradient
        outer_optimizer.apply_gradients(zip(meta_grads, backend.trainable_variables))
        
        model.set_weights(backend.get_weights())
    compiled_model.set_weights(model.get_weights())
    validation = compiled_model.evaluate(validgenerator,verbose=False)
    model.save_weights(filepath.format(epoch=epoch+1))
    epochBar.set_description(tracker.report())
    epochBar.refresh() # to show immediately the update       
    train_scores = tracker.get()
    with train_summary_writer.as_default():
        for key,val in train_scores.items():
            tf.summary.scalar(key, val, step=epoch)

    with valid_summary_writer.as_default():
        for i,key in enumerate(train_scores.keys()):
            tf.summary.scalar(key, validation[i], step=epoch)
