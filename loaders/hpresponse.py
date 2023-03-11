import numpy as np
import os
from loaders import Task
import logging
import pickle
import xgboost as xgb

logging.basicConfig(level=logging.INFO)


class HPOTask(Task):
    """
    HPO Task Class
    """

    def __init__(self, dataset_id, search_space_id, output_folder, seed, shuffle, batch_size, **kwargs):
        """
        Constructor for the Task Class

        Args:
            dataset_id (str): dataset id of interest, i.e. task id
            search_space_id (str): search space id of interest
            output_folder (str): location to cache processed task
            seed (int): random seed
            batch_size (int): size of batch
            shuffle (bool): shuffle instances after each epoch

        Keyword Args:
            data (dict): dictionary that includes the configuration space "X" and function evaluations "y"
            seeds (dict): dictionary that includes initializations for Bayesian Optimization
            stats (dict): dictionary that includes the statistics about surrogate
            surrogate_directory (str): directory of pre-trained surrogate models

        """
        super(HPOTask, self).__init__(seed=seed, shuffle=shuffle, batch_size=batch_size)
        logging.info(f"Processing dataset {dataset_id} from search space {search_space_id}")

        self.dataset_id = dataset_id
        self.search_space_id = search_space_id
        self.folder = output_folder
        self.name = f"{self.search_space_id}-{dataset_id}"
        self.surrogate_name = f"surrogate-{self.name}"

        cache_path = \
            os.path.join(self.folder,
                         f'build_Xy__{self.name}.pickle') if self.folder else None

        self.cache_path = cache_path
        if cache_path and os.path.exists(cache_path):
            logging.info("Loading cached data")
            (X, self.info, y, self.target_info, self.initializations, self.surrogate_stats,
             self.surrogate) = self.from_pickled_id()
        else:
            logging.info("Creating dataset")
            assert "data" in kwargs, "If the task is not cached, the HPO-B database must be passed"
            assert "seeds" in kwargs, "If the task is not cached, the HPO-B BO initializations must be passed"
            assert "stats" in kwargs, "If the task is not cached, the HPO-B surrogate statistics must be passed"
            assert "surrogate_directory" in kwargs, "If the task is not cached, the HPO-B surrogate directory " \
                                                    "must be passed"
            data = kwargs["data"]
            seeds = kwargs["seeds"]
            stats = kwargs["stats"]
            surrogate_directory = kwargs["surrogate_directory"]
            (X, self.info, y, self.target_info, self.initializations, self.surrogate_stats,
             self.surrogate) = self.from_task_id(data, seeds, stats, surrogate_directory)
            logging.info("Saving data")
            with open(cache_path, 'wb') as f:
                pickle.dump((X, self.info, y, self.target_info, self.initializations,
                             self.surrogate_stats, self.surrogate), f)
        self.data = X
        self.targets = y
        self.on_epoch_end()

    def from_pickled_id(self):
        logging.info(f"Using cached task: {self.cache_path}")
        with open(self.cache_path, 'rb') as f:
            return pickle.load(f)

    def from_task_id(self, data, seeds, stats, surrogate_directory):
        """
         Get internal dataset splits
         """
        X = np.asarray(data["X"])
        info = {
            "name": self.name,
            "dataset_id": self.dataset_id,
            "search_space_id": self.search_space_id,
            "dim": X.shape[1],
            "size": X.shape[0],
        }
        targets = np.asarray(data["y"]).reshape(-1, 1)
        target_info = {"y_min": targets.min(),
                       "y_max": targets.max()}
        initializations = seeds

        model = xgb.Booster()
        model.load_model(os.path.join(surrogate_directory, f"{self.surrogate_name}.json"))
        surrogate_info = stats[self.surrogate_name]
        return X, info, targets, target_info, initializations, surrogate_info, model


if __name__ == "__main__":
    output_folder = "../hpob/processed"
    search_space_id = "4796"
    dataset_id = "3561"
    task = HPOTask(dataset_id=dataset_id, search_space_id=search_space_id, output_folder=output_folder, seed=0,
                   batch_size=16, shuffle=True)
    x, y = next(iter(task))
