import json
import logging
import os
import timeit

from hpresponse import HPOTask
from loaders import Generator, SEARCH_SPACE_IDS, PARTS

logging.basicConfig(level=logging.INFO)


class MetaTaskGenerator(Generator):
    """
    Metatask Class.
    """

    def __init__(self, data_directory, search_space_id, seed=0, batch_size=None, meta_batch_size=8, shuffle=True):
        """
        Constructor for the Metatask Class

        Args:
            data_directory (str): directory of datasets;
                                directory should contain downloaded data in the following subfolders:
                                - data
                                - splits
                                - surrogates
            search_space_id (str): search space id of interest
            seed (int): random seed
            meta_batch_size (int): size of batch
            batch_size (int): size of batch
            shuffle (bool): shuffle instances after each epoch
        """
        super(MetaTaskGenerator, self).__init__(seed=seed, meta_batch_size=meta_batch_size, shuffle=shuffle)

        assert search_space_id in SEARCH_SPACE_IDS, f"{search_space_id} not in {SEARCH_SPACE_IDS}"

        self.search_space_id = search_space_id
        self.mode = "train"

        tic = timeit.default_timer()

        logging.info("Reading HPO-B database")
        with open(os.path.join(data_directory, "data", "hpob.json"), "r") as f:
            self.hpob = json.load(f)[search_space_id]
        logging.info("Reading HPO-B BO initializations")
        with open(os.path.join(data_directory, "data", "bo-initializations.json"), "r") as f:
            self.hpob_seeds = json.load(f)[search_space_id]
        logging.info("Reading HPO-B surrogate statistics")
        with open(os.path.join(data_directory, "surrogates", "summary-stats.json"), "r") as f:
            self.hpob_surrogate_stats = json.load(f)

        # read train/validation/test dataset ids
        self.files = dict()
        for part in PARTS:
            with open(os.path.join(data_directory, "splits", f"{part}.json"), "r") as f:
                self.files[part] = json.load(f)[search_space_id]

        logging.info("Generating meta-tasks ...")
        self.meta_data = {mode: {dataset_id: HPOTask(dataset_id=dataset_id, search_space_id=search_space_id,
                                                     output_folder=os.path.join(data_directory, "processed"),
                                                     data=self.hpob[dataset_id],
                                                     seeds=self.hpob_seeds[dataset_id],
                                                     stats=self.hpob_surrogate_stats,
                                                     surrogate_directory=os.path.join(data_directory, "surrogates"),
                                                     seed=seed, shuffle=shuffle, batch_size=batch_size)

                                 for dataset_id in self.files[mode]}
                          for mode in PARTS}
        self.on_epoch_end()
        toc = timeit.default_timer()
        logging.info(f"Initialization time {toc - tic:.2f} seconds")


if __name__ == "__main__":
    data_directory = "../hpob"
    search_space_id = "4796"
    generator = MetaTaskGenerator(data_directory=data_directory, search_space_id=search_space_id)
    tasks = next(iter(generator))