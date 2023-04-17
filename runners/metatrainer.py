import copy
import json
import os

import tensorflow as tf
import tensorflow_addons as tfa

import losses
from callbacks.reptile import ReptileCallback
from loaders.HPGenerator import MetaTaskGenerator
from runners import Runner


class MetaTrainer(Runner):
    inference = False

    def __init__(self, args):
        super(MetaTrainer, self).__init__(args)
        self.meta_batch_size = args.meta_batch_size
        self.generate_meta_tasks()
        self.initialize_model()
        self.setup_model_path()
        with open(os.path.join(self.model_path, "config.json"), 'w') as f:
            json.dump(self.config, f)

    def setup_model_path(self):
        self.model_path = os.path.join(self.save_path, self.search_space, "reptile" if self.is_reptile else "joint",
                                       self.job_start_date if not self.tuning_job else f"seed-{self.cs_seed}")
        self.clear_path(self.model_path)

    def generate_meta_tasks(self):
        self.meta_train_tasks = MetaTaskGenerator(data_directory=self.data_directory, search_space_id=self.search_space,
                                                  seed=self.seed, batch_size=self.batch_size, shuffle=True,
                                                  inner_steps=self.inner_steps)

        self.meta_valid_tasks = copy.deepcopy(self.meta_train_tasks)
        self.meta_valid_tasks.fixed_context = True
        self.meta_valid_tasks.mode = "validation"
        self.meta_valid_tasks.on_epoch_end()

    def fit(self):
        callbacks = self.prepare_callbacks()

        if self.is_reptile:
            if self.meta_optimizer == "adam":
                meta_optimizer = tf.keras.optimizers.Adam(learning_rate=self.meta_learning_rate)
            elif self.meta_optimizer == "radam":
                total_steps = self.epochs * len(self.meta_train_tasks.files["train"]) // self.inner_steps
                meta_optimizer = tfa.optimizers.RectifiedAdam(lr=self.meta_learning_rate, total_steps=total_steps)
            else:
                meta_optimizer = tf.keras.optimizers.SGD(learning_rate=self.meta_learning_rate)

            callbacks += [ReptileCallback(inner_steps=self.inner_steps, meta_batch_size=self.meta_batch_size,
                                          outer_optimizer=meta_optimizer)]

        self.model.fit(self.meta_train_tasks, validation_data=self.meta_valid_tasks, callbacks=callbacks,
                       epochs=self.epochs, verbose=0)

    @property
    def n_features(self):
        return self.meta_train_tasks.n_features

    def compile_model(self):
        optimizer = self.get_optimizer()
        self.model.compile(loss=losses.NegativeLogLikelihood(), optimizer=optimizer,
                           metrics=[losses.LogVar(), losses.MeanSquaredError()])
