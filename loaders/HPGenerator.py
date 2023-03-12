import json
import logging
import os
import timeit

from loaders.hpresponse import HPOTask
from loaders import Generator, SEARCH_SPACE_IDS, PARTS

logging.basicConfig(level=logging.INFO)


class MetaTaskGenerator(Generator):
    """
    Metatask Class.
    """

    def __init__(self, data_directory, search_space_id, mode=None, inner_steps=1, seed=0, batch_size=None,
                 shuffle=True, fixed_context=False):
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
            batch_size (int): size of batch
            shuffle (bool): shuffle instances after each epoch
            inner_steps (int): number of inner steps per task
            mode (str): mode of generator (train/test/valid)
            fixed_context (bool): indicator if we use the initialization seeds as context

        """
        super(MetaTaskGenerator, self).__init__(seed=seed, inner_steps=inner_steps, shuffle=shuffle)

        assert search_space_id in SEARCH_SPACE_IDS, f"{search_space_id} not in {SEARCH_SPACE_IDS}"

        self.search_space_id = search_space_id
        self.mode = mode if mode is not None else "train"
        self.partitions = PARTS if mode is None else [mode]
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
        for part in self.partitions:
            with open(os.path.join(data_directory, "splits", f"{part}.json"), "r") as f:
                self.files[part] = json.load(f)[search_space_id]

        logging.info("Generating meta-tasks ...")
        self.meta_data = {mode: {dataset_id: HPOTask(dataset_id=dataset_id, search_space_id=search_space_id,
                                                     output_folder=os.path.join(data_directory, "processed"),
                                                     data=self.hpob[dataset_id],
                                                     seeds=self.hpob_seeds[dataset_id],
                                                     stats=self.hpob_surrogate_stats,
                                                     surrogate_directory=os.path.join(data_directory, "surrogates"),
                                                     seed=seed, shuffle=shuffle, batch_size=batch_size,
                                                     fixed_context=fixed_context)

                                 for dataset_id in self.files[mode]}
                          for mode in self.partitions}
        self.on_epoch_end()
        toc = timeit.default_timer()
        logging.info(f"Initialization time {toc - tic:.2f} seconds")


if __name__ == "__main__":
    data_directory = "../hpob"
    search_space_id = "4796"
    generator = MetaTaskGenerator(data_directory=data_directory, search_space_id=search_space_id,
                                  seed=0, batch_size=0,
                                  shuffle=True,
                                  inner_steps=5, mode="train", fixed_context=True)
    x, y = next(iter(generator))
    # for x,y in generator:
    #     pass
