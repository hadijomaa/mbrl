import tensorflow as tf
import numpy as np

SEARCH_SPACE_IDS = ['4796', '5527', '5636', '5859', '5860',
                    '5891', '5906', '5965', '5970', '5971', '6766',
                    '6767', '6794', '7607', '7609', '5889']

PHASE_BOUNDS = (0, 2)
AMPLITUDE_BOUNDS = (0.1, 5)
GRANULARITY = 0.1
DOMAIN_BOUNDS = (-5, 5)

TRAIN = 'train'
VAL = 'validation'
TEST = 'test'

PARTS = [TRAIN, VAL, TEST]


class Generator(tf.keras.utils.Sequence):

    def __init__(self, seed, inner_steps, shuffle=True):
        """
        Constructor for the Meta-Task Generator

        Args:
            seed (int): random seed
            inner_steps (int): number of inner steps per task
            shuffle (bool): shuffle instances after each epoch
        """
        self.files = None
        self.meta_data = None
        self.indexes = None

        self.inner_steps = inner_steps
        self.seed = seed
        self.batch_size = 1
        self.shuffle = shuffle

        self.randomizer = np.random.RandomState(seed=seed)

    def __len__(self):
        """
        Number of task instances
        """
        # e.g. t0 t0 t0 t0 t0 .... tN tN tN tN tN
        return len(self.files[self.mode]) * self.inner_steps

    def __getitem__(self, index):
        """
        Query meta-instances from meta-dataset

        Args:
            index (int): index of instance in meta-dataset

        Returns:
            meta_item (list):  list of tasks.
        """
        # select task
        meta_item = self.meta_data[self.mode][self.files[self.mode][self.indexes[index]]]
        # shuffle sequence
        meta_item.on_epoch_end()
        # query task-specific batch
        return next(iter(meta_item))

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """

        self.indexes = np.arange(len(self.files[self.mode]))
        if self.shuffle:
            self.randomizer.shuffle(self.indexes)
        self.indexes = np.repeat(self.indexes, self.inner_steps).tolist()

    @property
    def n_features(self):
        return self.meta_data[self.mode][self.files[self.mode][0]].n_features


class Task(tf.keras.utils.Sequence):
    def __init__(self, seed=0, batch_size=0, shuffle=True):
        """
        Constructor for the Task

        Args:
            seed (int): random seed
            batch_size (int): size of batch
            shuffle (bool): shuffle instances after each epoch
        """
        # the mode can be only changed by user
        self.mode = "meta"
        self.indexes = dict()
        self.data = None
        self.targets = None
        self.name = None
        self.target_info = None

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.randomizer = np.random.RandomState(seed=seed)
        self.context_length_randomizer = np.random.RandomState(seed=seed)
        self.context_choice_randomizer = np.random.RandomState(seed=seed)

    def __getitem__(self, index):
        """
        Query instance from dataset
        """
        indexes = self.indexes[self.mode][index * self.batch_size:(index + 1) * self.batch_size]
        X = self.data[self.mode][indexes].astype(np.float32)
        y = self.targets[self.mode][indexes]
        if self.mode in ["meta", "hpo"]:
            context_size_indices = self.get_context_indices(indexes)
            context_X = np.stack([self.data[self.mode][i] for i in context_size_indices])
            context_y = np.stack([self.targets[self.mode][i] for i in context_size_indices])
            context = np.concatenate([context_X, context_y], axis=-1)
            return (context, np.expand_dims(X, axis=1)), y

        return X, y

    def __len__(self):
        """
        Number of instances
        """
        if self.batch_size == 0:
            self.batch_size = self.data[self.mode].shape[0]
        self.batch_size = min(self.batch_size, self.data[self.mode].shape[0])
        return self.data[self.mode].shape[0] // self.batch_size

    @property
    def n_features(self) -> int:
        return self.data[self.mode].shape[1]

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes[self.mode] = np.arange(self.data[self.mode].shape[0]).tolist()
        if self.shuffle:
            self.randomizer.shuffle(self.indexes[self.mode])

    def get_context_indices(self, indexes):
        context_size = self.context_length_randomizer.choice(100)
        n_total = self.data[self.mode].shape[0]
        context_size_indices = list(map(
            lambda value: self.context_choice_randomizer.choice(np.setdiff1d(np.arange(n_total), value),
                                                                size=min(context_size, n_total), replace=False),
            indexes))
        return context_size_indices
