import tensorflow as tf
import math

tf.math.pi = math.pi


class NegativeLogLikelihood(tf.keras.losses.Loss):

    def __init__(self, name="negative_log_likelihood"):
        super().__init__(name=name)

    def __call__(self, labels, predications, sample_weight=None):
        raw_mu, var_values = tf.split(predications, 2, -1)
        y_diff = tf.subtract(raw_mu, labels)
        loss = tf.reduce_mean(0.5 * tf.math.log(var_values + 1e-6) + 0.5 * tf.divide(tf.square(y_diff),
                                                                                     var_values + 1e-6)) + 0.5 * tf.math.log(
            2 * tf.math.pi)
        return loss


class MeanSquaredError(tf.keras.losses.Loss):

    def __init__(self, name="mean_squared_error"):
        super().__init__(name=name)

    def __call__(self, labels, predictions, sample_weight=None):
        raw_mu, _ = tf.split(predictions, 2, -1)
        return tf.keras.losses.MeanSquaredError()(labels, raw_mu)


class LogVar(tf.keras.losses.Loss):

    def __init__(self, name="log_var"):
        super().__init__(name=name)

    def __call__(self, labels, predictions, sample_weight=None):
        _, var_values = tf.split(predictions, 2, -1)
        return tf.reduce_mean(tf.math.log(var_values + 1e-6))
