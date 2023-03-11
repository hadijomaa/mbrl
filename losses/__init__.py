import tensorflow as tf
import math

tf.math.pi = math.pi


def nll(y_true, y_hat):
    raw_mu, var_values = tf.split(y_hat, 2, -1)
    y_diff = tf.subtract(raw_mu, y_true)
    loss = tf.reduce_mean(0.5 * tf.math.log(var_values + 1e-6) + 0.5 * tf.divide(tf.square(y_diff),
                                                                                 var_values + 1e-6)) + 0.5 * tf.math.log(
        2 * tf.math.pi)
    return loss


def mse(y_true, y_hat):
    raw_mu, var_values = tf.split(y_hat, 2, -1)
    return tf.keras.losses.MeanSquaredError()(y_true, raw_mu)


def log_var(y_true, y_hat):
    raw_mu, var_values = tf.split(y_hat, 2, -1)
    return tf.reduce_mean(tf.math.log(var_values + 1e-6))
