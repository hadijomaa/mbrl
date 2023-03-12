import tensorflow as tf
from modules.utils import get_units
from modules import networks


class DeepSet(tf.keras.Model):
    def __init__(self, layers_encoder, units, layers_decoder, architecture, dropout_rate=0.1):
        super(DeepSet, self).__init__()

        self.encoder = tf.keras.Sequential()
        fn = lambda x: get_units(idx=x, neurons=units, layers=layers_encoder, architecture=architecture)
        for l in range(layers_encoder):
            self.encoder.add(tf.keras.layers.Dense(fn(l), activation='relu'))
            self.encoder.add(tf.keras.layers.Dropout(dropout_rate))

        self.encoder_pooled = tf.keras.Sequential(
            tf.keras.layers.Dense(units, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate)
        )

        self.x_transform = tf.keras.Sequential(
            tf.keras.layers.Dense(units, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate)
        )

        self.decoder = tf.keras.Sequential()
        for l in range(layers_decoder):
            self.decoder.add(tf.keras.layers.Dense(fn(l), activation='relu'))
            self.decoder.add(tf.keras.layers.Dropout(dropout_rate))

        self.decoder.add(networks.VariationalMLP(d_model=units, dff=1, dropout_rate=dropout_rate))

    def call(self, inputs):

        context, x = inputs

        context = self.encoder(context)
        context = tf.reduce_mean(context, axis=1)
        context = self.encoder_pooled(context)

        x = self.x_transform(tf.squeeze(x, axis=1))

        x = tf.concat([context, x], axis=-1)

        logits = self.decoder(x)

        return logits
