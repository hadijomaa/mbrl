import json
import os
import pickle

import numpy as np
import pandas as pd

from helpers.parsers import get_hp_parser
from legacy_code import legacy_helpers

if __name__ == "__main__":
    parser = get_hp_parser()
    args = parser.parse_args()

    rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
    search_space_id = args.search_space
    data_directory = os.path.join(rootdir, "hpob")

    part = "test"

    # load initializations
    with open(os.path.join(data_directory, "data", "bo-initializations.json"), "r") as f:
        hpob_seeds = json.load(f)[search_space_id]

    with open(os.path.join(data_directory, "data", "hpob.json"), "r") as f:
        search_space_tasks = json.load(f)[search_space_id]

    with open(os.path.join(data_directory, "legacy_results_v0", "originals.pkl"), "rb") as f:
        initial_seed_results = pickle.load(f)

    # get part files
    with open(os.path.join(data_directory, "splits", f"{part}.json"), "r") as f:
        files = json.load(f)[search_space_id]

    # get benchmark methods
    single_task_methods = {"rs": "Random", "Bohamiann": "Bohamiann", "gpy": "GP", "DNGO": "DNGO",
                           "Deep Kernel GP": "Deep Kernel GP"}
    transfer_task_methods = {"RGPE": "RGPE", "quantile": "GCP+Prior", "DKLM": "DKLM", "FSBO": "FSBO",
                             "DKLM (RI)": "DKLM (RI)", "ABLR": "ABLR", "TST-R": "TST-R", "TAF-R": "TAF-R", }

    methods_to_evaluate = transfer_task_methods
    methods_to_evaluate.update(single_task_methods)

    # initialize search space results
    search_space_results = []
    search_space_results_rank = []

    for hpo_seed in [0, 1, 2, 3, 4]:
        # initialize seed lists
        seed_results = []
        seed_results_rank = []

        # get individual task names
        for task in files:
            method_results = pd.DataFrame()
            # iterate over methods
            for method in methods_to_evaluate:
                # get results directory of task
                results_directory = legacy_helpers.get_hpo_results_directory(rootdir=data_directory,
                                                                             search_space=search_space_id,
                                                                             method=method, hpo_seed=hpo_seed)

                # get results on task
                task_result = legacy_helpers.get_hpo_results(results_directory=results_directory, task=task,
                                                             search_space=search_space_id, method=method,
                                                             hpo_seed=hpo_seed)
                task_result = 100 * np.concatenate([initial_seed_results[search_space_id][hpo_seed][task], task_result])

                # collect method results
                method_results = pd.concat(
                    [method_results, pd.DataFrame(task_result, columns=[methods_to_evaluate[method]])], axis=1)

            assert method_results.shape[1] == len(methods_to_evaluate)

            # make directory and save results in updated format
            search_space_seed_path = os.path.join(data_directory, "legacy_results", search_space_id, f"{hpo_seed}")
            os.makedirs(search_space_seed_path, exist_ok=True)
            method_results.to_csv(os.path.join(search_space_seed_path, "results.csv"))

            method_results = method_results.reindex(range(101)).fillna(0).round(6)
            method_results_rank = method_results.rank(1, method="min")

            seed_results_rank.append(method_results_rank)
            seed_results.append(method_results)

        seed_results = pd.DataFrame(np.array(seed_results).mean(axis=0)[:100], columns=method_results.columns.tolist())

        seed_results_rank = pd.DataFrame(np.array(seed_results_rank).mean(axis=0)[:100],
                                         columns=method_results.columns.tolist())
        search_space_results.append(seed_results)
        search_space_results_rank.append(seed_results_rank)

    baselines_results_rank_std = pd.DataFrame(np.array(search_space_results_rank).std(axis=0),
                                              columns=method_results.columns.tolist())
    baselines_results_std = pd.DataFrame(np.array(search_space_results).std(axis=0),
                                         columns=method_results.columns.tolist())

    baselines_results_rank = pd.DataFrame(np.array(search_space_results_rank).mean(axis=0),
                                          columns=method_results.columns.tolist())

    baselines_results = pd.DataFrame(np.array(search_space_results).mean(axis=0),
                                     columns=method_results.columns.tolist())
