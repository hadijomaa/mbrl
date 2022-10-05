import numpy as np
from namedlist import namedlist
import copy
from helper_fn import make_obs
from pets import hpo
from tensorflow.keras.utils import Sequence
Transition = namedlist("Transition", ["context_x", "target_x", "context_y", "target_y", "metafeatures", "task", "action"])

class Policy(object):
    def __init__(self, env, init_state, seed):

        self.env = env
        self.shuffler = np.random.RandomState(seed)
        self.init_state = init_state
        self.action_space = self.env.X[0]
        card, dim = self.env.X[0].shape
        ntasks = len(self.env.X)
        taskGenerator = np.random.RandomState(seed)
        self.task_generator = lambda : taskGenerator.choice(range(ntasks))
        self.default_trajectory = np.arange(card)
        self.response_fn = lambda task, actions : self.env.Y[task][actions]
        self.observation_fn = lambda actions, responses: make_obs(self.action_space, actions, responses)[0]
        self.metafeatures_fn = lambda task, actions: self.env.Z[task][actions]

    def get_action(self, x, task)        :
        raise Exception("not implemented")

    def perform_rollout(self, x)        :
        raise Exception("not implemented")
        
class Policy_Random(Policy):
    
    def __init__(self, env, init_state, seed=413):
        super().__init__(env, init_state, seed)
        
    def get_action(self, x, task):
        observed = x.shape[0]
        return self.default_trajectory[observed]
    
    def perform_rollout(self, nrollout, task=None):
        if task is None:
            task = self.task_generator()
        self.shuffler.shuffle(self.default_trajectory)
        x = copy.deepcopy(self.default_trajectory)[:self.init_state].tolist()
        rollout = []
        for _ in range(self.init_state, nrollout):
            y = self.response_fn(task, x)
            context = self.observation_fn(x , y)
            context_x = context[:,:-1,None]
            context_y = context[:,-1:]
            step = self.get_action(context, task)
            target_x  = self.action_space[step][:,None]
            target_y  = self.response_fn(task, step)
            metafeatures = self.metafeatures_fn(task, step)
            rollout.append(Transition(context_x, target_x, context_y, target_y, metafeatures, task, step))
            x.append(step)
        assert((np.array(x)==self.default_trajectory[:len(x)]).all())
        return rollout
    

class Policy_MPC(Policy):
    def __init__(self, controller_fn, model,  env, init_state, seed=413):
        super().__init__(env, init_state, seed)
        self.controller_fn = controller_fn
        self.backbone = model
        self.seed =      seed
    def get_action(self, x, task):
        action = self.controller.act(x)
        target_y, _, target_x_idx = hpo(action[None],self.env,task,return_index=True)
        return target_x_idx[0][0]
    
    def perform_rollout(self, nrollout, task=None, x = None, return_actions=False):
        if task is None:
            task = self.task_generator()
        self.shuffler.shuffle(self.default_trajectory)
        x = copy.deepcopy(self.default_trajectory)[:self.init_state].tolist() if x is None else copy.deepcopy(x)
        self.controller = self.controller_fn(x, task, self.env, self.backbone )
        rollout = []
        rewards = 0
        for _ in range(nrollout-self.init_state):
            y = self.response_fn(task, x)
            context = self.observation_fn(x , y)
            context_x = context[:,:-1,None]
            context_y = context[:,-1:]
            step = self.get_action(context, task) ### x is automatically updated in the backgroung
            target_x  = self.action_space[step][:,None]
            target_y  = self.response_fn(task, step)
            metafeatures = self.metafeatures_fn(task, step)
            rollout.append(Transition(context_x, target_x, context_y, target_y, metafeatures, task, step))
            rewards = max(rewards, target_y)
        assert(len(np.unique(x))==len(x))
        if not return_actions:
            return rollout,rewards
        else:
            return rollout,rewards,x
    
class Buffer(object):
    def __init__(self,seed=413):
        self.shuffler = np.random.RandomState(seed)
        self.clear()
        
    def __len__(self):
        return len(self.transitions)
    
    def clear(self):
        self.context_size = []
        self.transitions  = []
        
    def append(self, transition):
        self.transitions.append(transition)
        self.context_size.append(transition.context_x.shape)
    
    def sample(self, batch_size, context_size):
        indexes = np.where(np.array(self.context_size)==context_size)[0]
        indexes = self.shuffler.choice(indexes, batch_size, replace=False).tolist()
        minibatch = [self.transitions[i] for i in indexes]
        transitions = Transition(*zip(*minibatch))
        return transitions
    
    def get(self, indexes):
        minibatch = [self.transitions[i] for i in indexes]
        minibatch = Transition(*zip(*minibatch))
        E  = minibatch.context_x
        r  = minibatch.context_y
        X  = minibatch.target_x
        Z = minibatch.metafeatures
        y = minibatch.target_y
        return (np.stack(E),np.stack(r),np.stack(X),np.vstack(Z)), np.vstack(y)

    def _generate(self, policy, num_rollouts, rollout_length):
        self.clear()
        for _ in range(num_rollouts):
            random_rollouts = policy.perform_rollout(rollout_length)
            [self.append(rand_rol)   for rand_rol in random_rollouts]        
    
class BufferSequence(Sequence):

    def __init__(self, buffer, batch_size,shuffle=True, seed=100, init_state_size=3, final_state_size=100):
        self.rng = np.random.RandomState(seed=seed)
        self.low = init_state_size
        self.high = final_state_size
        self.buffer = buffer
        self.episode_length = self.high-self.low
        self.batch_size=batch_size
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return len(self.indexes)//self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.buffer.get(indexes)

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        assert((len(self.buffer)%self.episode_length) == 0)
        num_rollouts = len(self.buffer)//self.episode_length
        episode_begins = np.arange(0,num_rollouts*self.episode_length,self.episode_length)
        state_size = np.arange(self.episode_length)
        self.rng.shuffle(state_size)
        indexes = []
        num_rollouts_divisible_by_batch_size = num_rollouts//self.batch_size
        for t in state_size:
            abc = episode_begins+t
            self.rng.shuffle(abc)
            indexes+=abc[:num_rollouts_divisible_by_batch_size*self.batch_size].tolist()
        self.indexes = indexes
        assert(len(indexes)%self.batch_size==0)
