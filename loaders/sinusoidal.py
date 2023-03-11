import numpy as np
from sklearn.model_selection import KFold
from loaders import Task


def split(X, seed):
    train_index = test_index = None
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        if i == seed:
            break
    return train_index, test_index


class SinusoidalTask(Task):

    def __init__(self, amplitude, phase, bounds, granularity=0.1, seed=0, batch_size=1, shuffle=True):
        """
        Constructor for the Sinusoidal Task

        Args:
            amplitude (float): amplitude of the sinusoidal function
            phase (float): phase of the sinusoidal function
            bounds (tuple): domain of the sinusoidal function
            granularity (float): the degree of separation between data points
            seed (int): random seed
            batch_size (int): size of batch
            shuffle (bool): shuffle instances after each epoch
        """
        super(SinusoidalTask, self).__init__(seed=seed, shuffle=shuffle, batch_size=batch_size)

        self.mode = "all"
        self.lower, self.upper = bounds
        self.granularity = granularity
        self.amplitude = amplitude
        self.phase = phase

        self.name = f"{self.amplitude}_{self.phase}_{self.lower}_{self.upper}"
        X = np.arange(self.lower, self.upper, self.granularity).reshape(-1, 1)
        func = lambda x: np.sin(x + phase) * amplitude + np.sin(np.pi * x)
        y = func(X)
        self.test_index, self.tr_index = split(seed=seed, X=X)
        self.data = X  # {"all": X, "train": X[self.tr_index], "test": X[self.test_index]}
        self.targets = y  # {"all": y, "train": y[self.tr_index], "test": y[self.test_index]}
        self.target_info = {"y_min": y.min(),
                            "y_max": y.max()}
        self.on_epoch_end()


if __name__ == "__main__":
    task = SinusoidalTask(amplitude=0.2, phase=1, bounds=(-5, 5), granularity=0.1, seed=0, batch_size=16)
    x, y = next(iter(task))
