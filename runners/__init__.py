import json
import os
from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa

import losses
from callbacks.metrics import SaveLogsCallback
from callbacks.reptile import ReptileCallback
from loaders.HPGenerator import MetaTaskGenerator
from modules.transformers import Transformer


class Runner:

    def __init__(self, args):
        self.model_path = None
        self.meta_valid_tasks = None
        self.meta_train_tasks = None
        self.model = None

        self.data_directory = args.data_directory
        self.search_space = args.search_space
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.meta_batch_size = args.meta_batch_size
        self.inner_steps = args.inner_steps
        self.num_layers = args.num_layers
        self.dropout_rate = args.dropout_rate
        self.num_heads = args.num_heads
        self.d_model = args.d_model
        self.dff = args.dff

        self.apply_scheduler = args.apply_scheduler
        self.learning_rate = args.learning_rate
        self.meta_learning_rate = args.meta_learning_rate
        self.meta_optimizer = args.meta_optimizer
        self.optimizer = args.optimizer

        self.epochs = args.epochs

        self.job_start_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')

        rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
        self.save_path = os.path.join(rootdir, args.save_path)

        self.config = vars(args)
        self.setup_model_path()
        self.generate_meta_tasks()
        self.initialize_model()
        with open(os.path.join(self.save_path, "config.json"), 'w') as f:
            json.dump(self.config, f)


    def generate_meta_tasks(self):
        self.meta_train_tasks = MetaTaskGenerator(data_directory=self.data_directory, search_space_id=self.search_space,
                                                  seed=self.seed, batch_size=self.batch_size, shuffle=True,
                                                  inner_steps=self.inner_steps, mode="train")

        self.meta_valid_tasks = MetaTaskGenerator(data_directory=self.data_directory, search_space_id=self.search_space,
                                                  seed=self.seed, batch_size=self.batch_size, shuffle=True,
                                                  inner_steps=self.inner_steps, mode="validation", fixed_context=True)

    def initialize_model(self):
        x = tf.keras.layers.Input(shape=(None, self.meta_train_tasks.n_features,))
        context = tf.keras.layers.Input(shape=(None, self.meta_train_tasks.n_features + 1,))
        transformer = Transformer(num_layers=self.num_layers, num_heads=self.num_heads, dropout_rate=self.dropout_rate,
                                  dff=self.dff, d_model=self.d_model, num_latent=1)([context, x])
        self.model = tf.keras.Model(inputs=[context, x], outputs=transformer)

    def compile_model(self):
        if self.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

        self.model.compile(loss=losses.nll, optimizer=optimizer, metrics=[losses.log_var, losses.mse])

    @property
    def is_reptile(self):
        return self.inner_steps > 1

    def fit(self):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.model_path, "model"), save_weights_only=True,
                                               monitor='val_loss', mode='min', save_best_only=True),
                     SaveLogsCallback(checkpoint_path=self.model_path, has_validation=True)]

        if self.apply_scheduler == "polynomial":
            callbacks += [tf.keras.callbacks.LearningRateScheduler(
                tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=self.learning_rate,
                                                              end_learning_rate=self.learning_rate / self.epochs,
                                                              decay_steps=self.epochs, power=2, cycle=False),
                verbose=0)]
        elif self.apply_scheduler == "cosine":
            callbacks += [tf.keras.callbacks.LearningRateScheduler(
                tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=self.learning_rate,
                                                                  first_decay_steps=self.epochs),
                verbose=0)]
        else:
            pass

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
                       epochs=self.epochs)

    def setup_model_path(self):
        self.model_path = os.path.join(self.save_path, self.search_space, "reptile" if self.is_reptile else "joint",
                                       self.job_start_date)
        os.makedirs(self.model_path, exist_ok=True)
