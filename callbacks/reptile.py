import tensorflow as tf
import copy


class ReptileCallback(tf.keras.callbacks.Callback):

    def __init__(self, inner_steps, meta_batch_size, outer_optimizer, verbose=False):
        """
        Callback that implements the first-order meta-learning logic in the .fit() function.

        Args:
            inner_steps (int): number of inner steps per task
            meta_batch_size (int): size of tasks sampled by a meta-batch
            outer_optimizer (tf.keras.optimizer): meta-optimizer required to update the weights of the model
                                                after every meta-batch
            verbose (bool): indicator to print out info
        """
        super(ReptileCallback, self).__init__()
        self.meta_batch_index = None
        self.inner_step_index = None
        self.new_theta = None
        self.theta = None

        self.theta_i = []
        self.inner_steps = inner_steps
        self.meta_batch_size = meta_batch_size
        self.outer_optimizer = outer_optimizer
        self.verbose = verbose
        self.reset_meta_batch_index()
        self.reset_inner_step_index()

    def reset_inner_step_index(self):
        self.inner_step_index = 0

    def reset_meta_batch_index(self):
        self.meta_batch_index = 1

    def update_weights_list(self, weights):
        print("\nUpdating meta-batch intermediate weights")
        self.theta_i.append(weights)

    def reset_weights(self):
        self.model.set_weights(copy.deepcopy(self.theta))

    def pool_weights(self):
        print("Pooling weights resulting from the meta-batch")
        self.new_theta = [tf.reduce_mean(tf.stack(_, axis=0), axis=0) for _ in zip(*self.theta_i)]
        self.model.set_weights(self.new_theta)
        self.theta_i = []

    def perform_outer_gradient_step(self):
        print("Performing one meta-gradient optimization step")
        # calculate the meta-gradient. self.theta represents the
        # old weight stored at self.on_begin_train and later in
        # self.perform_outer_gradient_step
        meta_grads = [tf.subtract(old, new) for old, new in
                      zip(self.theta, self.model.get_weights())]
        # reset the model weights to self.theta
        self.model.set_weights(self.theta)
        # \phi = \phi - 1/n (\phi-W)
        # call the optimizer on the original vars based on the meta gradient
        self.outer_optimizer.apply_gradients(zip(meta_grads, self.model.trainable_variables))
        # we have an updated backend model
        # store the new weights in self.theta and reset self.new_theta
        self.theta = copy.deepcopy(self.model.get_weights())
        self.new_theta = None

    def on_train_batch_end(self, batch, logs=None):
        """
        Implementation of the first-order meta-learning logic

        Notes:
            With every batch, the model is optimized on the chosen task, and the self.inner_step_index is incremented
            by 1. Every self.inner_steps batches, the model weights are stored, reset and a new task is chosen. The
            self.meta_batch_index is then incremented by 1. Every self.meta_batch_size increments, the weights of the
            model are aggregated and set for the new meta batch of tasks. An epoch ends after all meta-tasks have been
            seen.
        """
        self.inner_step_index += 1
        if self.verbose:
            print(f"\nInner step {self.inner_step_index}/{self.inner_steps}")
            print(f"Task {self.meta_batch_index}/{self.meta_batch_size}")
        if self.inner_steps == self.inner_step_index:
            self.reset_inner_step_index()
            # update list of weights self.theta_i
            self.update_weights_list(copy.deepcopy(self.model.get_weights()))
            # reset model weights to self.theta
            self.reset_weights()
            self.meta_batch_index += 1

        if self.meta_batch_size == self.meta_batch_index - 1:
            assert len(self.theta_i) == self.meta_batch_size
            self.reset_meta_batch_index()
            # average theta_i across meta_batch tasks producing self.new_theta
            self.pool_weights()
            # one meta-learning update step
            self.perform_outer_gradient_step()
            print("Outer Gradient Step Complete")

    def on_train_begin(self, logs=None):
        """
        Function that saves a copy of the model and its weights
        """
        print("Setting up backend model!")
        self.theta = copy.deepcopy(self.model.get_weights())
