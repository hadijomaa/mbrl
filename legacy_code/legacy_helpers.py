import json
import os

import numpy as np


def get_hpo_results_directory(rootdir, search_space, method, hpo_seed):
    results_directory = os.path.join(rootdir, "legacy_results_v0")
    if method == "ABLR":
        results_directory = os.path.join(results_directory, "ABLR", f"BOSS_bench_{search_space}", "ablr", f"{hpo_seed}")
    elif method not in ["rs", "dgp", "gpy"]:
        results_directory = os.path.join(results_directory, search_space, method, f"{hpo_seed}")
    else:
        pass
    return results_directory


def get_hpo_results(results_directory, task, search_space, method, hpo_seed):
    if method not in ["rs", "gpy"]:
        task_result = 1 - np.array(
            json.load(open(os.path.join(results_directory, f"{task}.json"), "r"))["results"])
        task_result = task_result[5:]
    elif method == "gpy":
        task_result = 1 - np.array(
            json.load(open(os.path.join(results_directory, f"{method}.json"), "r"))["results"][search_space][task][
                f"test{hpo_seed}"])
        task_result = task_result[1:]
    elif method == "rs":
        task_result = 1 - np.array(
            json.load(open(os.path.join(results_directory, f"{method}.json"), "r"))[search_space][task][
                f"test{hpo_seed}"])
        task_result = task_result[1:]
    else:
        raise Exception
    return task_result
