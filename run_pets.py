from helpers import parsers
from runners import Tester
import random as rn

import numpy as np
import tensorflow as tf


if __name__ == "__main__":
    parser = parsers.get_runner_parser()
    parser = parsers.get_hp_parser(parser=parser)
    parser = parsers.get_tuner_parser(parser=parser)
    parser = parsers.get_pets_parser(parser=parser)
    args = parser.parse_args()

    np.random.seed(args.model_seed)
    rn.seed(args.model_seed)
    tf.random.set_seed(args.model_seed)
    args.reptile = False
    args.apply_lookahead = bool(args.apply_lookahead)
    runner = Tester(args)
    runner.perform_hpo(args.num_trials)

