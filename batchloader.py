import numpy as np
from net_utils import _map
from tensorflow.keras.utils import Sequence
import pickle as pkl
import os
# from ismlldataset.datasets import get_metadataset
# import pandas as pd
rng     = np.random.RandomState(23)
emb_rng = np.random.RandomState(53)
from sklearn.preprocessing import MinMaxScaler
def ptp(X):
    param_nos_to_extract = X.shape[1]
    domain = np.zeros((param_nos_to_extract, 2))
    for i in range(param_nos_to_extract):
        domain[i, 0] = np.min(X[:, i])
        domain[i, 1] = np.max(X[:, i])
    X = (X - domain[:, 0]) / np.ptp(domain, axis=1)
    return X

class DataGenerator(Sequence):
    def __init__(self,rootdir,dataset_ids,metafeatures,transformation="a",
                 batch_size=16,shuffle=True,normalize_support_response=False,fixed_history=None):
        self.shuffle = shuffle
        self.t =fixed_history
        self.normalize_support_response = normalize_support_response
        self.X = []
        self.X_sparse  = []
        self.Y = []
        self.Z = []
        with open(os.path.join(rootdir,"pickled",f"searchspace-{transformation}.pkl"), "rb") as f:
                hpo_data = pkl.load(f)          
        for dataset_id in dataset_ids:
            # loaded = False
            # while not loaded:
            #     try:            
            #         md = get_metadataset(dataset_id)
            #         loaded=True
            #     except Exception as e:
            #         print(e)
            # md.apply_special_transformation(transformation)
            # md.normalize_response()
            # x,y = md.get_meta_data()
            # self.subaction_dim = x.nunique().tolist()
            # self.X_sparse.append(pd.get_dummies(x,columns=x.columns))
            # x = np.array(pd.get_dummies(pd.DataFrame(x)))
            # y = np.array(y).reshape(-1,1)
            x = hpo_data[_map[dataset_id]]["X"]
            y = hpo_data[_map[dataset_id]]["Y"].reshape(-1,1)            
            z = metafeatures.loc[_map[dataset_id]].ravel()
            z = np.repeat(z[np.newaxis,:],y.shape[0],axis=0)
            self.X.append(ptp(x))
            self.Y.append(y)
            self.Z.append(z)
        self.batch_size=batch_size if batch_size is not None else x.shape[0] #### entire search space
        self.on_epoch_end()
# 
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return len(self.indexes)//self.batch_size#sum([int(np.floor(np.array(x[0]).shape[0] / self.batch_size)) for x in self.X])

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        X,E,Z,y,r = [],[],[],[],[] # X is the action # E is the history 
        t = emb_rng.randint(low=1,high=99) if self.t is None else self.t # evidence
        p = emb_rng.choice(np.arange(self.X[0].shape[0]),size=t,replace=False) # select random number of support configurations
        for dim,file in indexes:
            X.append(self.X[file][dim])
            y.append(self.Y[file][dim])
            Z.append(self.Z[file][dim])
            E.append(self.X[file][p])
            r.append(self.Y[file][p])
        X = np.array(X);E = np.array(E);Z = np.array(Z);y = np.array(y);r = np.array(r)
        if self.normalize_support_response:
            scaler = MinMaxScaler()
            r = np.expand_dims(scaler.fit_transform(r.squeeze(axis=-1).transpose()).transpose(),axis=-1)           
        return (np.expand_dims(E,axis=-1),r,np.expand_dims(X,axis=-1),Z), y

    def __getmetaitem__(self, file,indexes, shared_context=True, context_size = None):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting
        ----------
        Notes: shared context across all the batch
        """
        # Generate indexes of the batch
        X,E,Z,y,r = [],[],[],[],[]
        t = emb_rng.randint(low=1,high=99) if context_size is None else context_size
        # print(indexes)
        p = emb_rng.choice(np.arange(self.X[0].shape[0]),size=t,replace=False) # select random number of support configurations
        for dim in indexes:
            X.append(self.X[file][dim])
            y.append(self.Y[file][dim])
            Z.append(self.Z[file][dim])
            E.append(self.X[file][p])
            r.append(self.Y[file][p])
            if not shared_context:
                p = emb_rng.choice(np.arange(self.X[0].shape[0]),size=t,replace=False)            
        X = np.array(X);E = np.array(E);Z = np.array(Z);y = np.array(y);r = np.array(r)
        if self.normalize_support_response:
            scaler = MinMaxScaler()
            r = np.expand_dims(scaler.fit_transform(r.squeeze(axis=-1).transpose()).transpose(),axis=-1)        
        return (np.expand_dims(E,axis=-1),r,np.expand_dims(X,axis=-1),Z), y
    
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = []
        
        nfiles = len(self.X) # number of files
        dim    = self.X[0].shape[0] # number of configurations for algorithm x
        p1,p2  = np.meshgrid(np.arange(dim),np.arange(nfiles))
        p1,p2  = p1.ravel()[:,np.newaxis],p2.ravel()[:,np.newaxis]
        p = np.arange(len(p1)) # nfiles\times dim
        if self.shuffle:
            rng.shuffle(p)
            p1 = p1[p];p2 = p2[p];
        self.indexes = np.concatenate([p1,p2],axis=1).tolist()
        self.indexes = self.indexes[:int(np.floor((len(p) / self.batch_size))*self.batch_size)]
            
class BraninGenerator(Sequence):
    def __init__(self,rootdir,dataset_ids,
                 batch_size=16,shuffle=True,normalize_support_response=False,fixed_history=None):
        self.shuffle = shuffle
        self.t =fixed_history
        self.normalize_support_response = normalize_support_response
        self.X = []
        self.X_sparse  = []
        self.Y = []
        self.Z = []
        with open(os.path.join(rootdir,"pickled","branin.pkl"), "rb") as f:
                hpo_data = pkl.load(f)          
        for dataset_id in dataset_ids:
            x = hpo_data["X"][dataset_id]
            y = hpo_data["Y"][dataset_id].reshape(-1,1)            
            z = np.random.rand(32)
            z = np.repeat(z[np.newaxis,:],y.shape[0],axis=0)
            self.X.append(x)
            self.Y.append(y)
            self.Z.append(z)
        self.batch_size=batch_size if batch_size is not None else x.shape[0] #### entire search space
        self.on_epoch_end()
# 
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return len(self.indexes)//self.batch_size#sum([int(np.floor(np.array(x[0]).shape[0] / self.batch_size)) for x in self.X])

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        X,E,Z,y,r = [],[],[],[],[] # X is the action # E is the history 
        t = emb_rng.randint(low=1,high=99) if self.t is None else self.t # evidence
        p = emb_rng.choice(np.arange(self.X[0].shape[0]),size=t,replace=False) # select random number of support configurations
        for dim,file in indexes:
            X.append(self.X[file][dim])
            y.append(self.Y[file][dim])
            Z.append(self.Z[file][dim])
            E.append(self.X[file][p])
            r.append(self.Y[file][p])
        X = np.array(X);E = np.array(E);Z = np.array(Z);y = np.array(y);r = np.array(r)
        if self.normalize_support_response:
            scaler = MinMaxScaler()
            r = np.expand_dims(scaler.fit_transform(r.squeeze(axis=-1).transpose()).transpose(),axis=-1)           
        return (np.expand_dims(E,axis=-1),r,np.expand_dims(X,axis=-1),Z), y

    def __getmetaitem__(self, file,indexes, shared_context=True, context_size = None):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting
        ----------
        Notes: shared context across all the batch
        """
        # Generate indexes of the batch
        X,E,Z,y,r = [],[],[],[],[]
        t = emb_rng.randint(low=1,high=99) if context_size is None else context_size
        # print(indexes)
        p = emb_rng.choice(np.arange(self.X[0].shape[0]),size=t,replace=False) # select random number of support configurations
        for dim in indexes:
            X.append(self.X[file][dim])
            y.append(self.Y[file][dim])
            Z.append(self.Z[file][dim])
            E.append(self.X[file][p])
            r.append(self.Y[file][p])
            if not shared_context:
                p = emb_rng.choice(np.arange(self.X[0].shape[0]),size=t,replace=False)            
        X = np.array(X);E = np.array(E);Z = np.array(Z);y = np.array(y);r = np.array(r)
        if self.normalize_support_response:
            scaler = MinMaxScaler()
            r = np.expand_dims(scaler.fit_transform(r.squeeze(axis=-1).transpose()).transpose(),axis=-1)        
        return (np.expand_dims(E,axis=-1),r,np.expand_dims(X,axis=-1),Z), y
    
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = []
        
        nfiles = len(self.X) # number of files
        dim    = self.X[0].shape[0] # number of configurations for algorithm x
        p1,p2  = np.meshgrid(np.arange(dim),np.arange(nfiles))
        p1,p2  = p1.ravel()[:,np.newaxis],p2.ravel()[:,np.newaxis]
        p = np.arange(len(p1)) # nfiles\times dim
        if self.shuffle:
            rng.shuffle(p)
            p1 = p1[p];p2 = p2[p];
        self.indexes = np.concatenate([p1,p2],axis=1).tolist()
        self.indexes = self.indexes[:int(np.floor((len(p) / self.batch_size))*self.batch_size)]
        
# import os
# rootdir     = os.path.dirname(os.path.realpath(__file__))
# split = 0
# dataset_ids = pd.read_csv(os.path.join(rootdir,"dataset_id_splits.csv"),index_col=0)[f"train-{split}"].dropna().astype(int).ravel()[:10]
# metafeatures = pd.read_csv(os.path.join(rootdir,"metafeatures",f"meta-features-split-{split}.csv"),index_col=0)
# datagen = DataGenerator(dataset_ids,metafeatures)

# X,y = next(iter(datagen))
# from net import net
# configuration = {}
# configuration["dim"] = 32
# configuration["num_heads"] = 4
# configuration["num_inds"] = 32
# configuration["ln"]    = False
# configuration["arch"] = (32,4,"relu","SQU")
# model = net(configuration)

# o = model(X)
