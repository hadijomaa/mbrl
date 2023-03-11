import tensorflow as tf
import copy


class ReptileCallback(tf.keras.callbacks.Callback):

    def __init__(self, inner_steps, meta_batch_size, outer_optimizer):
        super(ReptileCallback, self).__init__()
        self.new_theta = None
        self.theta = None
        self.theta_i = []
        self.inner_steps = inner_steps
        self.meta_batch_size = meta_batch_size
        self.outer_optimizer = outer_optimizer
        self.meta_batch_index = 1
        self.inner_step_index = 1
        self.backend_model = None

    def update_weights_list(self, weights):
        self.theta_i.append(weights)

    def reset_weights(self):
        self.model.set_weights(self.theta)

    def pool_weights(self):
        self.new_theta = [tf.reduce_mean(tf.stack(_, axis=0), axis=0) for _ in zip(*self.theta_i)]
        self.model.set_weights(self.new_theta)

    def perform_outer_gradient_step(self):
        # update the weights of the original model
        meta_grads = [tf.subtract(old, new) for old, new in
                      zip(self.backend_model.trainable_variables, self.model.trainable_variables)]
        # \phi = \phi - 1/n (\phi-W)
        # call the optimizer on the original vars based on the meta gradient
        self.outer_optimizer.apply_gradients(zip(meta_grads, self.backend_model.trainable_variables))
        self.model.set_weights(copy.deepcopy(self.backend_model.get_weights()))

    def on_train_batch_end(self, batch, logs=None):
        """
        Implementation of the first-order meta-learning logic

        Notes:
            With every batch, the model is optimized on the chosen
            task, and the self.inner_step_index is incremented by 1. Every self.inner_steps batches, the model weights
            are stored, reset and a new task is chosen. The self.meta_batch_index is then incremented by 1. Every
            self.meta_batch_size increments, the weights of the model are aggregated and set for the new meta batch of
            tasks. An epoch ends after all meta-tasks have been
            seen.
        """
        self.inner_step_index += 1

        if self.inner_steps % self.inner_step_index == 0:
            self.inner_step_index = 0
            self.update_weights_list(copy.deepcopy(self.model.get_weights()))
            self.reset_weights()
            self.meta_batch_index += 1

        if self.meta_batch_size % self.meta_batch_index == 0:
            assert len(self.theta_i) == self.meta_batch_size
            self.meta_batch_index = 0
            self.pool_weights()
            self.perform_outer_gradient_step()
            self.theta_i = []

        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_train_begin(self, logs=None):
        self.theta = copy.deepcopy(self.model.get_weights())
        self.backend_model = copy.deepcopy(self.model)
        self.backend_model.set_weights(self.theta)
