import json
import os
import copy

import pandas as pd
import losses

from controllers.optimizer import RandomShooter
from controllers.mpc import MPC
from loaders.hpresponse import HPOTask
from runners import Runner

OUTPUT_FOLDER = "hpob/processed"


class Tester(Runner):

    inference = True

    def __init__(self, args):
        super(Tester, self).__init__(args)
        self.random_weights = None
        self.shooter = None
        self.utility_function = None
        self.controller = None
        self.task = None
        self.num_random_trajectories = args.num_random_trajectories
        self.inference_optimizer = args.inference_optimizer
        self.inference_learning_rate = args.inference_learning_rate
        self.inference_epochs = args.inference_epochs
        self.mpc_seed = args.mpc_seed
        self.num_particles = args.num_particles
        self.horizon = args.horizon
        self.apply_lookahead = args.apply_lookahead
        self.dataset_id = args.dataset_id
        self.load_pretrained = args.load_pretrained
        self.setup_model_path()
        self.generate_tasks()
        self.log_path = os.path.join(self.rootdir, args.log_path,
                                     "pre-trained" if self.load_pretrained else "random-initialization",
                                     self.search_space, f"horizon-{self.horizon}",
                                     f"trajectories-{self.num_random_trajectories}", f"particles-{self.num_particles}",
                                     f"{'LookAhead' if self.apply_lookahead else 'MPC'}", self.inference_optimizer,
                                     f"lr-{self.inference_learning_rate}", f"mpc-{self.mpc_seed}", self.dataset_id)
        os.makedirs(self.log_path, exist_ok=True)
        with open(os.path.join(self.log_path, "config.json"), 'w') as f:
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
        if self.random_weights is None:
            self.random_weights = copy.deepcopy(self.model.get_weights())

        if self.load_pretrained:
            print("Loading pre-trained model")
            self.model.load_weights(os.path.join(self.model_path, "..", "model"))
        else:
            print("Loading random initialization")
            self.model.set_weights(self.random_weights)

    def fit(self):
        print("Model being trained ...")
        callbacks = self.prepare_callbacks(monitor="log_var", has_validation=False, save_model=False,
                                           path=self.log_path, early_stopping=True)
        self.model.fit(self.task, callbacks=callbacks, epochs=self.inference_epochs, verbose=0)

    def design_controller(self):
        self.shooter = RandomShooter(num_random_trajectories=self.num_random_trajectories, seed=self.mpc_seed)
        self.utility_function = lambda x: self.task.utility_function(index=x)
        self.controller = MPC(model=self.model, input_dim=self.n_features, num_particles=self.num_particles,
                              horizon=self.horizon, optimizer=self.shooter, seed=self.mpc_seed,
                              apply_lookahead=self.apply_lookahead, utility_function=self.utility_function,
                              candidate_pool=self.task.candidate_pool)

    def perform_hpo(self, number_of_trials):
        self.task.mode = "hpo"
        print("------------------")
        print(f"Starting HPO for Dataset Id: {self.dataset_id}")
        results = pd.DataFrame()
        actions = pd.DataFrame()
        expectations = pd.DataFrame()
        self.initialize_model()
        for seed in range(5):
            self.task.update_hpo_mode(seed=seed)
            self.design_controller()
            seed_run_info = {}
            for t in range(number_of_trials):
                print(f"Trial {t + 1}/{number_of_trials} | Seed {seed + 1}/5")
                self.task.on_epoch_end()
                self.load_model()
                self.compile_model()
                self.fit()
                action, info = self.controller.act(self.task.state)
                self.task.log_evaluation(action)
                print("Evaluating suggested hyperparameter")
                self.task.do_trial()
                info.update({"regret": self.task.regret[-1]})
                for k, v in info.items():
                    if "horizon" in k:
                        print(f"{k}: {v}")
                    else:
                        print(f"{k}: {100 * v:.2f}")
                if self.task.regret[-1] == 0:
                    print("Breaking optimization because Global optimum found!")
                    break
                seed_run_info.update({t: info})
                print("------------------")
            results = pd.concat([results, pd.DataFrame(self.task.regret, columns=[f"seed-{seed}"])], axis=1)
            actions = pd.concat([actions, pd.DataFrame(self.task.evaluated_hps, columns=[f"seed-{seed}"])], axis=1)
            seed_run_info = pd.DataFrame.from_dict(seed_run_info).transpose()
            seed_run_info["seed"] = seed
            expectations = pd.concat([expectations, seed_run_info], axis=0)
        print(f"Saving to {self.log_path}!")
        results.to_csv(os.path.join(self.log_path, "results.csv"))
        actions.to_csv(os.path.join(self.log_path, "actions.csv"))
        expectations.to_csv(os.path.join(self.log_path, "expectations.csv"))

    @property
    def n_features(self):
        return self.task.n_features

    def compile_model(self):
        optimizer = self.get_optimizer(optimizer=self.inference_optimizer, learning_rate=self.inference_learning_rate)
        self.model.compile(loss=losses.NegativeLogLikelihood(), optimizer=optimizer,
                           metrics=[losses.LogVar(), losses.MeanSquaredError()])
