from helpers import parsers
from runners import Runner
import random as rn

import numpy as np
import tensorflow as tf
np.random.seed(37)
rn.seed(1254)
tf.random.set_seed(96)

if __name__ == "__main__":
    parser = parsers.get_runner_parser()
    parser = parsers.get_hp_parser(parser=parser)
    parser = parsers.get_tuner_parser(parser=parser)

    args = parser.parse_args()
    runner = Runner(args)
    runner.compile_model()
    runner.fit()

