import json
import os

from loaders.hpresponse import HPOTask
from runners import Runner

OUTPUT_FOLDER = "hpob/processed"


class Tester(Runner):

    def __init__(self, args):
        super(Tester, self).__init__(args)
        self.task = None
        self.dataset_id = args.dataset_id
        self.setup_model_path()
        self.generate_tasks()
        with open(os.path.join(self.model_path, "config.json"), 'w') as f:
            json.dump(self.config, f)

    def generate_tasks(self):
        self.task = HPOTask(self.dataset_id, self.search_space, output_folder=os.path.join(self.rootdir, OUTPUT_FOLDER),
                            seed=self.seed, shuffle=True, batch_size=self.batch_size)

    def setup_model_path(self):
        self.model_path = os.path.join(self.save_path, self.search_space, "reptile" if self.is_reptile else "joint",
                                       self.job_start_date if not self.tuning_job else f"seed-{self.cs_seed}",
                                       "inference")
        self.clear_path(self.model_path)

    def load_model(self):
        # todo load model
        pass

    def fit(self):
        callbacks = self.prepare_callbacks(monitor="loss", has_validation=False)
        self.model.fit(self.task, callbacks=callbacks, epochs=self.epochs)

    def perform_hpo(self, number_of_trials, controller):
        self.task.mode = "hpo"
        self.task.update_hpo_mode(seed=0)
        for t in range(number_of_trials):
            self.task.on_epoch_end()
            self.initialize_model()
            self.load_model()
            self.compile_model()
            self.fit()
            # action, trajectory = controller.act(self.task.state)

    @property
    def n_features(self):
        return self.task.n_features