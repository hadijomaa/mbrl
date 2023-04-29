import os
from datetime import datetime

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import tensorflow as tf

from callbacks.metrics import SaveLogsCallback
from modules.transformers import Transformer
import tensorflow_addons as tfa


class Runner(object):
    inference = False

    def __init__(self, args):
        self.model_path = None
        self.meta_valid_tasks = None
        self.meta_train_tasks = None
        self.model = None
        self.cs_seed = None

        self.data_directory = args.data_directory
        self.search_space = args.search_space
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.is_reptile = args.reptile

        self.job_start_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')

        self.rootdir = rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
        self.save_path = os.path.join(rootdir, args.save_path)

        self.config = vars(args)
        self.tuning_job = "cs_seed" in self.config
        if self.tuning_job:
            cs = self.get_configspace(seed=self.config["cs_seed"])
            cs = cs.sample_configuration().get_dictionary()
            self.config.update(cs)
            self.cs_seed = self.config["cs_seed"]

        # hyperparameters
        self.num_layers_encoder = self.config["num_layers_encoder"]
        self.num_layers_decoder = self.config["num_layers_decoder"]
        self.dropout_rate = self.config["dropout_rate"]
        self.num_heads = 2 ** self.config["num_heads"]
        self.d_model = 2 ** self.config["d_model"]
        self.dff = 2 ** self.config["dff"]
        self.apply_scheduler = self.config["apply_scheduler"]
        self.learning_rate = self.config["learning_rate"]
        self.optimizer = self.config["optimizer"]
        self.inner_steps = 1
        if self.is_reptile:
            self.inner_steps = self.config["inner_steps"]
            self.meta_learning_rate = self.config["meta_learning_rate"]
            self.meta_optimizer = self.config["meta_optimizer"]

    def initialize_model(self):
        x = tf.keras.layers.Input(shape=(None, self.n_features,))
        context = tf.keras.layers.Input(shape=(None, self.n_features + 1,))
        transformer = Transformer(num_layers_encoder=self.num_layers_encoder,
                                  num_layers_decoder=self.num_layers_decoder, num_heads=self.num_heads,
                                  dropout_rate=self.dropout_rate, hidden_units=self.dff, d_model=self.d_model,
                                  num_latent=1)([context, x])
        self.model = tf.keras.Model(inputs=[context, x], outputs=transformer)

    def compile_model(self):
        raise NotImplementedError

    def get_optimizer(self, optimizer=None, learning_rate=None):
        if not self.inference:
            optimizer = self.optimizer
            learning_rate = self.learning_rate
            beta_1 = 0.
        else:
            beta_1 = 0.9

        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)
        elif optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == "radam":
            optimizer = tfa.optimizers.RectifiedAdam(lr=learning_rate, total_steps=self.epochs)
        else:
            raise f"{optimizer} not recognized"
        return optimizer

    def get_configspace(self, seed=None):
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.

        Args:
            seed (int): random seed for configuration space

        Returns:
            cs (CS.ConfigurationSpace) Configuration space of the backbone module
        """

        cs = CS.ConfigurationSpace(seed=seed)
        num_layers_decoder = CSH.UniformIntegerHyperparameter('num_layers_decoder', lower=1, upper=6, default_value=2)
        num_layers_encoder = CSH.UniformIntegerHyperparameter('num_layers_encoder', lower=1, upper=6, default_value=2)
        d_model = CSH.UniformIntegerHyperparameter('d_model', lower=5, upper=10, default_value=6)
        dff = CSH.UniformIntegerHyperparameter('dff', lower=5, upper=10, default_value=6)
        num_heads = CSH.UniformIntegerHyperparameter('num_heads', lower=1, upper=4, default_value=3)
        cs.add_hyperparameters([num_layers_decoder, d_model, num_heads, dff, num_layers_encoder])

        apply_scheduler = CSH.CategoricalHyperparameter('apply_scheduler', choices=["polynomial", "cosine", "None"])
        optimizer = CSH.CategoricalHyperparameter('optimizer', choices=["adam", "sgd"])
        rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0., upper=0.5, default_value=0.2)
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=1e-6, upper=1e-1, default_value=1e-3,
                                                       log=True)

        cs.add_hyperparameters([rate, optimizer, learning_rate, apply_scheduler])

        if self.is_reptile:
            meta_optimizer = CSH.CategoricalHyperparameter('meta_optimizer', choices=["adam", "radam", "sgd"])
            meta_learning_rate = CSH.UniformFloatHyperparameter('meta_learning_rate', lower=1e-6, upper=1e-1,
                                                                default_value=1e-3, log=True)
            inner_steps = CSH.UniformIntegerHyperparameter('inner_steps', lower=2, upper=10, default_value=1)
            cs.add_hyperparameters([meta_optimizer, meta_learning_rate, inner_steps])
        return cs

    @staticmethod
    def clear_path(model_path):
        os.makedirs(model_path, exist_ok=True)

        for file_name in os.listdir(model_path):
            # construct full file path
            file = os.path.join(model_path, file_name)
            if os.path.isfile(file):
                print('Deleting file:', file)
                os.remove(file)

    def prepare_callbacks(self, monitor="val_loss", has_validation=True, save_model=True, path=None,
                          early_stopping=False):

        path = self.model_path if path is None else path
        callbacks = [SaveLogsCallback(checkpoint_path=path, has_validation=has_validation)]

        if early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=16, restore_best_weights=True,
                                                              min_delta=1e-2))

        if save_model:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(path, "model"),
                                                                save_weights_only=True, monitor=monitor, mode='min',
                                                                save_best_only=True))

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

        return callbacks

    @property
    def n_features(self):
        raise NotImplementedError
