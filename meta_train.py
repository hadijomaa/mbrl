from helpers import parsers
from runners import MetaTrainer
import random as rn

import numpy as np
import tensorflow as tf
if __name__ == "__main__":
    parser = parsers.get_runner_parser()
    parser = parsers.get_hp_parser(parser=parser)
    parser = parsers.get_transformer_parser(parser=parser)

    args = parser.parse_args()

    np.random.seed(args.model_seed)
    rn.seed(args.model_seed)
    tf.random.set_seed(args.model_seed)
    args.reptile = False
    runner = MetaTrainer(args)
    runner.compile_model()
    runner.fit()

