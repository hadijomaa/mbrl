import tensorflow as tf
from net_utils import mse,log_var, nll
tf.random.set_seed(247)#246#245#244#243
import numpy as np
import tqdm
ARCHITECTURES = ['SQU','SYM','ENC']
def get_units(idx,neurons,architecture,layers=None):
    assert architecture in ARCHITECTURES
    if architecture == 'SQU':
        return neurons
    elif architecture=='SYM':
        assert (layers is not None and layers > 2)
        if layers%2==1:
            return neurons*2**(int(layers/2) - abs(int(layers/2)-idx))
        else:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx -1
            return neurons*2**(int(layers/2) - abs(int(layers/2)-idx))
        
    elif architecture=='ENC':
        assert (layers is not None and layers > 2)
        if layers%2==0:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx -1            
            return neurons*2**(int(layers/2)-1 -idx)
        else:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx
            return neurons*2**(int(layers/2) -idx)

class DropoutLayerwActivation(tf.keras.layers.Layer):
    
    def __init__(self, units,rate=0.1, nonlinearity="relu"):
        
        super(DropoutLayerwActivation, self).__init__()    
        self.units        = units        
        self.nonlinearity = tf.keras.layers.Activation(nonlinearity)
        self.dense        = tf.keras.layers.Dense(units=self.units)
        self.dropout      = tf.keras.layers.Dropout(rate)
        
    def call(self,x):
        return self.nonlinearity(self.dropout(self.dense(x)))
    
class DenseLayerwActivation(tf.keras.layers.Layer):
    
    def __init__(self, units, nonlinearity="relu"):
        
        super(DenseLayerwActivation, self).__init__()    
        self.units        = units        
        self.nonlinearity = tf.keras.layers.Activation(nonlinearity)
        self.dense        = tf.keras.layers.Dense(units=self.units)
        
    def call(self,x):
        return self.nonlinearity(self.dense(x))
    
class batch_mlp(tf.keras.layers.Layer):
    def __init__(self, output_sizes):
        
        super(batch_mlp, self).__init__()
        self.block = []
        for units in output_sizes[:-1]:
            self.block += [DenseLayerwActivation(units=units,nonlinearity="relu")]
        self.block += [tf.keras.layers.Dense(units=output_sizes[-1])]
        self.output_ = output_sizes[-1]
        
    def call(self,x):
        
        for fc in self.block:
            x = fc(x)
        return x
    
class deepset(tf.keras.Model):
    def __init__(self, configuration):
        
        super(deepset, self).__init__()
        # project any hyperparameter configuration to same dimension
        # units_fv_bar 
        self.D = batch_mlp(configuration["output_size_D"])
        self.C = DenseLayerwActivation(units=configuration["units_C"],nonlinearity="relu")
        self.B = batch_mlp(configuration["output_size_B"])
        from policies import BufferSequence
        self.make_buffer_sequence = lambda data, batch_size : BufferSequence(data, batch_size)
    def call(self, x):
        # e is the embedding, c is the training sample
        # e NxTxKx1
        # r NxTx1
        # z Nx32
        # x NxKx1
        e,r,x,z = x
        e = tf.squeeze(e,axis=-1) # --> NxTxK
        x = tf.squeeze(x,axis=-1) # --> NxK
        input_d  = tf.concat([e,r],axis=-1) # NxTx(units_f_knr+units_g_knr)
        output_d = self.D(input_d) # NxTxD
        #### pool across time (T)
        output_d = tf.reduce_mean(output_d,axis=1)### NxD
        output_c = self.C(output_d)### NxC
        
        input_b = tf.concat([x,output_c],axis=-1) ###Nx(units_f_knt+units_g_knt)
        output_b = self.B(input_b) ###NxB 

        # Get the mean an the variance
        mu, log_sigma = tf.split(output_b, 2, axis=-1)
    
        # Bound the variance
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)
        
        return tf.concat([mu,sigma],axis=-1)
    
    def train(self, oldbuffer, newbuffer, epochs, fraction_use_new, optimizer, batch_size, episode_length, trackers, verbose, 
              num_aggregation_iters):
        npr = np.random.RandomState(39)
        
        assert(len(newbuffer)%episode_length==0)

        # #init vars
        
        num_new_pts = len(newbuffer)//episode_length # how many new rollouts
        #how much of new data to use per batch
        if(num_new_pts<(batch_size*fraction_use_new)):
            batchsize_new_pts = num_new_pts #use all of the new ones, i.e. all rollouts
        else:
            batchsize_new_pts = int(batch_size*fraction_use_new)        

        #how much of old data to use per batch
        batchsize_old_pts = int(batch_size - batchsize_new_pts)
        
        if num_new_pts > 0:
            new_data = self.make_buffer_sequence(newbuffer, batchsize_new_pts)
        progressbar = tqdm.tqdm(range(epochs)) 
        for i in progressbar:
            #reset to 0
            trackers["current"].reset()
            #train from both old and new dataset
            oldbuffer.generate()
            old_data = self.make_buffer_sequence(oldbuffer, batchsize_old_pts)
            if(batchsize_old_pts>0): 
                
                #get through the full old dataset
                for batch in old_data:

                    #walk through the randomly reordered "old data"
                    dataX_old_batch, dataZ_old_batch = batch
                    # get time step
                    step = dataX_old_batch[0][0].shape[0]    - old_data.low             
                    #randomly sample points from new dataset
                    if(num_new_pts>0):
                        
                        new_episode_begins = np.arange(0,num_new_pts*episode_length,episode_length)
                        npr.shuffle(new_episode_begins)
                        new_indeces = new_episode_begins+step
                        dataX_new_batch, dataZ_new_batch = newbuffer.get(new_indeces)
                        
                        #combine the old and new data
                        
                        dataX_batch = ()
                        for a,b in zip(dataX_old_batch, dataX_new_batch):
                            dataX_batch += (np.concatenate((a, b)),)
                        dataZ_batch = np.concatenate((dataZ_old_batch, dataZ_new_batch))
                        
                    else:
                        #combine the old and new data
                        dataX_batch = dataX_old_batch
                        dataZ_batch = dataZ_old_batch
                        
                    #one iteration of feedforward training
                    with tf.GradientTape() as tape:
                        output  = self(dataX_batch,training=True)
                        loss    = nll(dataZ_batch,output)
                    grads = tape.gradient(loss, self.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.trainable_variables))
                    trackers["current"].update(loss, log_var(dataZ_batch, output), mse(dataZ_batch, output))

            #train completely from new set
            else: 
                for batch in new_data:
                    
                    #walk through the shuffled new data
                    dataX_batch, dataZ_batch = batch

                    #one iteration of feedforward training
                    with tf.GradientTape() as tape:
                        output  = self(dataX_batch,training=True)
                        loss    = nll(dataZ_batch,output)
                    grads = tape.gradient(loss, self.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.trainable_variables))   
                    trackers["current"].update(loss, log_var(dataZ_batch, output), mse(dataZ_batch, output))
            if num_new_pts>0:
                new_data.on_epoch_end()
            old_data.on_epoch_end()
            trackers["current"].write(i + num_aggregation_iters*epochs )
            #save losses after an epoch
            
            if(verbose):
                if((i%10)==0):
                    print("\n=== Epoch {} ===".format(i))
                    print("\n"+trackers["current"].report())

        #get loss of curr model on old dataset
        trackers["old"].reset()      
        oldbuffer.generate()
        old_data = self.make_buffer_sequence(oldbuffer, batch_size)
        for batch in old_data:
            # Batch the training data
            dataX_batch, dataZ_batch = batch
            #one iteration of feedforward training
            output  = self(dataX_batch,training=False)
            trackers["old"].update(nll(dataZ_batch,output), log_var(dataZ_batch, output), mse(dataZ_batch, output))
        trackers["old"].write(num_aggregation_iters)
        if len(newbuffer)>0:
            trackers["new"].reset()
            #get loss of curr model on new dataset
            new_data = self.make_buffer_sequence(newbuffer, batch_size)
            for batch in new_data:
                # Batch the training data
                dataX_batch, dataZ_batch = batch
                #one iteration of feedforward training
                output  = self(dataX_batch,training=False)
                trackers["new"].update(nll(dataZ_batch,output), log_var(dataZ_batch, output), mse(dataZ_batch, output))
            trackers["new"].write(num_aggregation_iters)

    def run_validation(self, valbuffer, batch_size,  trackers, num_aggregation_iters):
        trackers["val"].reset()
        #get loss of curr model on validation dataset given the actual states
        val_data = self.make_buffer_sequence(valbuffer, batch_size)
        for batch in val_data:
            # Batch the training data
            dataX_batch, dataZ_batch = batch
            #one iteration of feedforward training
            output  = self(dataX_batch,training=False)
            trackers["val"].update(nll(dataZ_batch,output), log_var(dataZ_batch, output), mse(dataZ_batch, output))
        trackers["val"].write(num_aggregation_iters)

    def do_forward_sim(self, valbuffer, batch_size,  trackers):
        # TODO #get loss of curr model on validation dataset given the predicted states
        raise ("Not Implemented")
            
    
class deepsetEnsemble(tf.keras.Model):

    def __init__(self, configuration, members):
        
        super(deepsetEnsemble, self).__init__()
        self.ensemble = []
        for _ in range(members):
            self.ensemble.append(deepset(configuration))
            
        self.members = members
        
    def load_models(self,   filepaths):
        assert(len(filepaths)==self.members)
        
        for filepath,model in zip(filepaths,self.ensemble):
            model.load_weights(filepath)
    
    def call(self, x):
        ensemble_mean,ensemble_var = [],[]
        for model in self.ensemble:
            output = model(x)
            mu, var = tf.split(output, 2, axis=-1)
            ensemble_mean.append(mu)
            ensemble_var.append(var)
        mu = tf.reduce_mean(tf.stack(ensemble_mean,axis=0),axis=0)
        var = 0
        for model_var,model_mean in zip(ensemble_var,ensemble_mean):
        	var = var + model_var + tf.square(model_mean)
        var = tf.divide(var , self.members) - tf.square(mu)
        
        return tf.concat([mu,var],axis=-1)    
            
    def fit(self, x,y,epochs,callbacks=None):
        for model in self.ensemble:
            model.fit(x,y,epochs=epochs,callbacks=callbacks,verbose=False)

    def _compile_(self, loss, optimizer):
        for model in self.ensemble:
            model.compile(loss=loss, optimizer=optimizer)
            
class deepsetWMeta(deepset):
    def __init__(self, configuration):
        
        super(deepset, self).__init__()
        # project any hyperparameter configuration to same dimension
        # units_fv_bar 
        self.D = batch_mlp(configuration["output_size_D"])
        self.C = DenseLayerwActivation(units=configuration["units_C"],nonlinearity="relu")
        self.B = batch_mlp(configuration["output_size_B"])
        from policies import BufferSequence
        self.make_buffer_sequence = lambda data, batch_size : BufferSequence(data, batch_size)
    def call(self, x):
        # e is the embedding, c is the training sample
        # e NxTxKx1
        # r NxTx1
        # z Nx32
        # x NxKx1
        e,r,x,z = x
        e = tf.squeeze(e,axis=-1) # --> NxTxK
        x = tf.squeeze(x,axis=-1) # --> NxK
        input_d  = tf.concat([e,r],axis=-1) # NxTx(units_f_knr+units_g_knr)
        output_d = self.D(input_d) # NxTxD
        #### pool across time (T)
        output_d = tf.reduce_mean(output_d,axis=1)### NxD
        output_c = self.C(tf.concat([output_d,z],1))### NxC
        
        input_b = tf.concat([x,output_c],axis=-1) ###Nx(units_f_knt+units_g_knt)
        output_b = self.B(input_b) ###NxB 

        # Get the mean an the variance
        mu, log_sigma = tf.split(output_b, 2, axis=-1)
    
        # Bound the variance
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)
        
        return tf.concat([mu,sigma],axis=-1)

class deepsetEnsembleWMeta(tf.keras.Model):

    def __init__(self, configuration, members):
        
        super(deepsetEnsembleWMeta, self).__init__()
        self.ensemble = []
        for _ in range(members):
            self.ensemble.append(deepsetWMeta(configuration))
            
        self.members = members
        
    def load_models(self,   filepaths):
        assert(len(filepaths)==self.members)
        
        for filepath,model in zip(filepaths,self.ensemble):
            model.load_weights(filepath)
    
    def call(self, x):
        ensemble_mean,ensemble_var = [],[]
        for model in self.ensemble:
            output = model(x)
            mu, var = tf.split(output, 2, axis=-1)
            ensemble_mean.append(mu)
            ensemble_var.append(var)
        mu = tf.reduce_mean(tf.stack(ensemble_mean,axis=0),axis=0)
        var = 0
        for model_var,model_mean in zip(ensemble_var,ensemble_mean):
        	var = var + model_var + tf.square(model_mean)
        var = tf.divide(var , self.members) - tf.square(mu)
        
        return tf.concat([mu,var],axis=-1)    
            
    def fit(self, x,y,epochs,callbacks=None):
        for model in self.ensemble:
            model.fit(x,y,epochs=epochs,callbacks=callbacks,verbose=False)

    def _compile_(self, loss, optimizer):
        for model in self.ensemble:
            model.compile(loss=loss, optimizer=optimizer)
                    
