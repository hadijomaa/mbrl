import copy
import numpy as np


class MPC(object):
    """
    Model Predictive Control Class
    """

    def __init__(self, model, candidate_pool, input_dim, num_particles, horizon, optimizer, utility_function,
                 apply_lookahead=True, seed=0):
        """
        Initialize MPC Class

        Args:
            model (tf.keras.model): neural network model
            candidate_pool (list): collection of actions that can be taken
            input_dim (int): dimensionality of input space
            num_particles (int): number of particles to be sampled
            horizon (int): length of simulated rollout
            optimizer (object): optimizer that produces a candidate solution
            utility_function (lambda fn): function that maps vectors to indices
            apply_lookahead (bool): Indicator to apply LookAhead
            seed (int): random seed
        """
        self.candidate_pool = copy.deepcopy(candidate_pool)
        self.apply_lookahead = apply_lookahead
        self.action_buffer = None
        self.model = model
        self.optimizer = optimizer
        self.utility_function = utility_function
        self.input_dim = input_dim
        self.num_particles = num_particles
        self.horizon = horizon

        self.randomizer = np.random.RandomState(seed=seed)

    def act(self, observation):
        """
        Returns the action that this controller would take at time t given observation obs.

        Args:
            observation (np.array): The current state of evaluated hyperparameters and their response
        Returns:
            An action (and possibly the predicted cost)
        """

        cost_function = lambda acs: self._compile_cost(action_sequence=acs, observation=observation)
        hp_index, info = self.optimizer.shoot(cost_function, candidate_pool=copy.deepcopy(self.candidate_pool),
                                              horizon=self.horizon,
                                              utility_function=self.utility_function,
                                              apply_lookahead=self.apply_lookahead)

        self.candidate_pool.pop(self.candidate_pool.index(hp_index))
        return hp_index, info

    def _compile_cost(self, action_sequence, observation):
        # action_sequence --> 1 x (input_dim x horizon)
        # t = obs.shape[0]
        # nopt = number of random trajectories
        nopt = action_sequence.shape[0]
        pred_regret = np.ones([nopt, self.num_particles])
        actual_regret = np.ones([nopt, self.horizon, self.num_particles])
        # nopt x plan_hor x input_dim ------ nopt = 1
        action_sequence = np.reshape(action_sequence, [-1, self.horizon, self.input_dim])
        # horizon x nopt x 1 x input_dim
        transpose_action_sequence = np.transpose(action_sequence, [1, 0, 2])[:, :, None]
        # horizon x nopt x particles x input_dim
        tile_action_sequence = np.tile(transpose_action_sequence, [1, 1, self.num_particles, 1])
        # horizon x (particles*nopt) x input_dim
        action_sequence = np.reshape(tile_action_sequence, [self.horizon, -1, self.input_dim])
        # (nopt*particles)xtx(input_dim+1) ---> batch ?
        observation = np.tile(observation[None], [nopt * self.num_particles, 1, 1])

        def continue_prediction(t):
            return np.less(t, self.horizon)

        def iteration(current_time_step, current_regret, current_observation, action_regret):
            """
            Args:
                current_time_step (int): indicates the number of steps taken so far
                current_regret (float): current regret observed so far
                current_observation (np.array): current state
                action_regret (float): regret observed so far
            """
            current_action_sequence = action_sequence[current_time_step]
            next_observation, next_performance = self._predict_next_obs(current_observation, current_action_sequence)
            # reshape performance nopt x particles
            next_performance = np.reshape(next_performance, [-1, self.num_particles])
            # regret bounded between 0 and 1
            next_regret = np.clip(1 - next_performance, a_min=0, a_max=1)
            action_regret[:, current_time_step:current_time_step + 1, :] = np.expand_dims(next_regret, axis=1)
            return current_time_step + 1, np.minimum(current_regret, next_regret), \
                next_observation, action_regret

        t = 0
        while continue_prediction(t):
            t, pred_regret, observation, actual_regret = iteration(t, pred_regret, observation, actual_regret)
        return actual_regret

    def _predict_next_obs(self, observation, action_sequence):
        # acs --> (particles*nopt) x input_dim
        # obs --> (nopt*particles)xtx(input_dim+1)
        action_sequence = np.expand_dims(action_sequence, axis=1)
        inputs = (observation, action_sequence)
        # output shape ----> (nopt*particles)x1
        output = self.model(inputs)
        assert ((action_sequence[0] == action_sequence[1]).all())
        mean, var = np.split(output.numpy(), 2, 1)
        predictions = mean + self.randomizer.normal(size=np.shape(mean), loc=0, scale=1) * np.sqrt(var)

        model_out_dim = predictions.shape[-1]

        # predictions --> nopt x particles x 1
        predictions = np.reshape(predictions, [-1, self.num_particles, model_out_dim])
        prediction_mean = np.mean(predictions, axis=1, keepdims=True)
        prediction_var = np.mean(np.square(predictions - prediction_mean), axis=1, keepdims=True)

        z = self.randomizer.normal(size=np.shape(predictions), loc=0, scale=1)
        # samples --> nopt x particles x 1
        samples = prediction_mean + z * np.sqrt(prediction_var)
        predictions = np.reshape(samples, [-1, model_out_dim])
        # new_state --> (particles*nopt) x 1 x (input_dim+1)
        new_state = np.concatenate([action_sequence.squeeze(axis=1), predictions], axis=-1)[:, None]
        # next_observation --> (particles*nopt) x (t+1) x (input_dim+1)
        next_observation = np.concatenate([observation, new_state], axis=1)

        return next_observation, predictions.reshape(model_out_dim, -1)
