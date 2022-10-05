import pandas as pd
import os
import tqdm
from net import deepsetEnsemble as net
from net_utils import _map
from batchloader import DataGenerator
import argparse
from net_utils import nll
from helper_fn import regret,make_obs,get_filepaths,prepare_data
import tensorflow as tf
import math
import numpy as np
tf.math.pi = math.pi
# create parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--split', help='Select training fold', type=int,default=0)
parser.add_argument('--index', help='Dataset index in the fold', type=int,default=4)
parser.add_argument('--shots', help='Number of random trajectories to investigate', type=int,default=1000)
parser.add_argument('--plan_hor', help='Planning Horizon', type=int,default=5)
parser.add_argument('--npart', help='Number of particles', type=int,default=20)
parser.add_argument('--fold', help='Select fold', type=str,default="test")
parser.add_argument('--searchspace', help='Select searchspace ', type=str,default="a")
parser.add_argument('--k', help='number of starting points', type=int,default=3)
parser.add_argument('--members', help='number of models to load', type=int,default=5)
parser.add_argument('--elites', help='fraction of elites to average', type=float,default=0.2)
parser.add_argument('--max_iters', help='maximum CEM iterations', type=int,default=10)
parser.add_argument('--RS', help='Random Shooting', type=str,choices=["True","False"],default="True")
parser.add_argument('--best_in_traj', help='Select best action in trajectory', type=str,choices=["True","False"],default="True")
parser.add_argument('--fit', help='Fine tune', type=str,choices=["True","False"],default="False")
args    = parser.parse_args()
args.RS = eval(args.RS)
args.fit = eval(args.fit)
args.best_in_traj = eval(args.best_in_traj)
rootdir     = os.path.dirname(os.path.realpath(__file__))

test_dataset_id = pd.read_csv(os.path.join(rootdir,"dataset_id_splits.csv"),index_col=0)[f"test-{args.split}"].dropna().astype(int).ravel()[args.index]
metafeatures = pd.read_csv(os.path.join(rootdir,"metafeatures",f"meta-features-split-{args.split}.csv"),index_col=0)

searchspace = {"a":"Layout Md",
               "b":"Regularization Md",
               "c":"Optimization Md",
               "d":"Tree Md"}

datagenerator = DataGenerator(rootdir,[test_dataset_id],metafeatures,shuffle=True,transformation=args.searchspace)

configuration = {}
configuration["output_size_D"] = [128]*4
configuration["units_C"] = 128
configuration["output_size_B"] = [128]*2  + [2]
configuration["seed"] = 34
model = net(configuration, members=args.members)

epochs    = 50
if not args.RS:
    extra = f"-elites-{args.elites}-max_iters-{args.max_iters}-"
else:
    extra = "" if not args.best_in_traj else "-TRAJ-"
for args.seed in range(100, 105):
    random = np.random.RandomState(args.seed)
    filename = f"pets-shots-{args.shots if args.plan_hor!=1 else 'none'}-plan_hor-{args.plan_hor}-particles-{args.npart}{extra}members-{args.members}{'-fit' if args.fit else ''}"
    savedir     = os.path.join(rootdir,"results-master",f"init-{args.k}",f"seed-{args.seed}",filename,f"searchspace-{args.searchspace}",f"split-{args.split}",args.fold)
    os.makedirs(savedir,exist_ok=True)
    if not os.path.isfile(os.path.join(savedir,f"{_map[test_dataset_id]}.csv")):
        Lambda,response,Z = datagenerator.X[0],datagenerator.Y[0],datagenerator.Z[0]
        card,dim = Lambda.shape
        x        = random.choice(np.arange(0,card),size=args.k,replace=False)
        
        y = [response[_] for _ in x]
          
        q = np.vstack([Lambda[_] for _ in x])
        
        x = list(x)
        
        opt = max(response)
        
        checkpoint_dir = os.path.join(rootdir,"tensorboard", searchspace[args.searchspace], f"split-{args.split}")
        from pets import MPC,hpo
        params = {}
        params["dU"]   = datagenerator.X[0].shape[1]
        params["num_particles"]= args.npart
        params["opt_mode"] = "Random" if args.RS else "CEM"
        params["plan_hor"] = args.plan_hor
        params["popsize"] = args.shots  # also not important with horizon 1 because all grid is considered
        params["max_iters"] = args.max_iters # doesnt matter
        params["num_elites"] =  int(args.shots*args.elites) # doesnt matter
        params["best_in_traj"] =  args.best_in_traj
        # model, params, data, x, dataset_id
        controller = MPC(model,params,datagenerator,x, 0)
        trajectories = []
        trajectory = []
        epochsbar = tqdm.tqdm(range(epochs), desc='Epochs',leave=True)        
        for e in epochsbar:
            if opt in y:
                break
            filepaths    = get_filepaths(checkpoint_dir, members= args.members, random_seed=e)
            model.load_models(filepaths)
            input, output = prepare_data(x,x,Lambda,response,Z[0])
            earlystopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss', min_delta=0, patience=100, verbose=0,
                mode='auto', baseline=None, restore_best_weights=True
            )
            model._compile_(loss=nll,optimizer="adam")
            if args.fit:
                model.fit(input,output,epochs=1000,callbacks=[earlystopping])
            obs = make_obs(Lambda,x,y)[0]
            action, trajectory = controller.act(obs)
            if len(trajectory)>1:
                trajectories.append(regret(response[trajectory].squeeze(),response)["regret_validation"].ravel())
            reward, action_, action_idx = hpo(action[None],datagenerator,0,return_index=True)
            q = Lambda[x]
            y = response[x]
        results            = regret(y,response)
        results["indices"] = x
        results["actual"] = response[x]
        results.to_csv(os.path.join(savedir,f"{_map[test_dataset_id]}.csv"))
        if len(trajectory)>1:
            pd.DataFrame(np.array(trajectories)).to_csv(os.path.join(savedir,f"trajectories-{_map[test_dataset_id]}.csv"))
