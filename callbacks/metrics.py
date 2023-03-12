import os

import pandas as pd
import tensorflow as tf


class SaveLogsCallback(tf.keras.callbacks.Callback):

    def __init__(self, checkpoint_path, has_validation):
        super(SaveLogsCallback, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.has_validation = has_validation

        columns = ["loss", "log_var", "mse"]
        if has_validation:
            columns += [f"val_{c}" for c in columns]

        self.columns = columns
        self.metrics = pd.DataFrame(columns=columns)

    def on_epoch_end(self, epoch, logs=None):
        epoch_logs = pd.DataFrame([logs[c] for c in self.columns]).transpose()
        epoch_logs.columns = self.columns
        self.metrics = pd.concat([self.metrics, epoch_logs], axis=0).reset_index(drop=True)
        self.metrics.to_csv(os.path.join(self.checkpoint_path, "metrics.csv"))

