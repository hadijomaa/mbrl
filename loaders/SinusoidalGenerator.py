import logging
import timeit

import numpy as np

from loaders import Generator, PHASE_BOUNDS, PARTS, AMPLITUDE_BOUNDS, GRANULARITY, DOMAIN_BOUNDS
from sinusoidal import SinusoidalTask

logging.basicConfig(level=logging.INFO)


class MetaSinusoidalGenerator(Generator):
    """
    Metatoy task Class.
    """

    def __init__(self, number_of_tasks, phase_bounds=PHASE_BOUNDS, amplitude_bounds=AMPLITUDE_BOUNDS, batch_size=None,
                 domain_bounds=DOMAIN_BOUNDS, granularity=GRANULARITY, seed=0, meta_batch_size=8, shuffle=True):
        """
        Constructor for the Metatoy task Class

        Args
            number_of_tasks (dict): number of tasks for meta-training, meta-validation and meta-testing
            phase_bounds (tuple): includes the bounds of the phase variable of the Sinusoidal Tasks
            amplitude_bounds (tuple): includes the bounds of the phase variable of the Sinusoidal Tasks
            domain_bounds (tuple): domain of the sinusoidal function
            granularity (float): the degree of separation between data points
            seed (int): random seed
            meta_batch_size (int): size of batch
            batch_size (int): size of batch
            shuffle (bool): shuffle instances after each epoch
        """
        super(MetaSinusoidalGenerator, self).__init__(seed=seed, meta_batch_size=meta_batch_size, shuffle=shuffle)
        self.mode = "train"
        self.phase_low, self.phase_high = phase_bounds
        self.amplitude_low, self.amplitude_high = amplitude_bounds
        self.bounds = domain_bounds
        self.granularity = granularity
        self.random_task_generator = np.random.RandomState(seed=seed)

        tic = timeit.default_timer()

        logging.info("Generating Sinusoidal Tasks")

        self.files = dict()
        for part in PARTS:
            part_tasks = []
            for task in range(number_of_tasks[part]):
                phase = self.random_task_generator.uniform(low=self.phase_low, high=self.phase_high * np.pi)
                amplitude = self.random_task_generator.uniform(self.amplitude_low, self.amplitude_high)
                task_name = f"{phase:.3f}_{amplitude:.3f}"
                part_tasks.append(task_name)
            self.files[part] = part_tasks

        logging.info("Generating meta-tasks ...")
        self.meta_data = {mode: {
            dataset_id: SinusoidalTask(phase=float(dataset_id.split("_")[0]), amplitude=float(dataset_id.split("_")[1]),
                                       granularity=granularity, bounds=domain_bounds, seed=seed, batch_size=batch_size,
                                       shuffle=shuffle)
            for dataset_id in self.files[mode]}
            for mode in PARTS}
        toc = timeit.default_timer()
        logging.info(f"Initialization time {toc - tic:.2f} seconds")
        self.on_epoch_end()


if __name__ == "__main__":
    tasks = MetaSinusoidalGenerator(phase_bounds=(0, 2), amplitude_bounds=(0.1, 5),
                                    number_of_tasks={"train": 64, "validation": 16, "test": 20}, granularity=0.1,
                                    domain_bounds=(-5, 5), batch_size=None, meta_batch_size=16, shuffle=True)
    x = next(iter(tasks))
