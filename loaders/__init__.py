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

    def __init__(self, seed, meta_batch_size, shuffle=True):
        """
        Constructor for the Meta-Task Generator

        Args:
            seed (int): random seed
            meta_batch_size (int): size of batch
            shuffle (bool): shuffle instances after each epoch
        """
        self.files = None
        self.meta_data = None
        self.indexes = None

        self.seed = seed
        self.meta_batch_size = meta_batch_size
        self.shuffle = shuffle

        self.randomizer = np.random.RandomState(seed=seed)

    def __len__(self):
        """
        Number of task instances
        """
        return len(self.files[self.mode]) // self.meta_batch_size

    def __getitem__(self, index):
        """
        Query meta-instances from meta-dataset

        Args:
            index (int): index of instance in meta-dataset

        Returns:
            meta_item (list):  list of tasks.
        """
        indexes = self.indexes[index * self.meta_batch_size:(index + 1) * self.meta_batch_size]
        meta_item = [self.meta_data[self.mode][self.files[self.mode][index]] for index in indexes]
        return meta_item

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """

        self.indexes = np.arange(len(self.files[self.mode]))
        if self.shuffle:
            self.randomizer.shuffle(self.indexes)
        self.indexes = self.indexes.tolist()


class Task(tf.keras.utils.Sequence):
    def __init__(self, seed=0, batch_size=1, shuffle=True):
        """
        Constructor for the Task

        Args:
            seed (int): random seed
            batch_size (int): size of batch
            shuffle (bool): shuffle instances after each epoch
        """
        self.indexes = None
        self.data = None
        self.targets = None
        self.name = None
        self.target_info = None

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.randomizer = np.random.RandomState(seed=seed)

    def __getitem__(self, index):
        """
        Query instance from dataset
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X = self.data[self.mode][indexes].astype(np.float32)
        y = self.targets[self.mode][indexes]
        return X, y

    def __len__(self):
        """
        Number of instances
        """
        if self.batch_size is None:
            self.batch_size = self.data[self.mode].shape[0]
        return self.data[self.mode].shape[0] // self.batch_size

    @property
    def n_features(self) -> int:
        return self.data[self.mode].shape[1]

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """

        self.indexes = np.arange(self.data[self.mode].shape[0])
        if self.shuffle:
            self.randomizer.shuffle(self.indexes)
        self.indexes = self.indexes.tolist()
