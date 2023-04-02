from helpers import parsers
import os
import pandas as pd

if __name__ == "__main__":

    parser = parsers.get_runner_parser()
    parser = parsers.get_hp_parser(parser=parser)
    parser = parsers.get_tuner_parser(parser=parser)

    parser.add_argument('--min_cs_seed', type=int, default=0, help="Minimum seed of configuration seed")
    parser.add_argument('--max_cs_seed', type=int, default=500, help="Maximum seed of configuration seed")
    parser.add_argument('--training', type=int, default=0, help="training or validation split")

    args = parser.parse_args()
    args.reptile = bool(args.reptile)
    args.training = bool(args.training)

    rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    save_path = os.path.join(rootdir, args.save_path)

    results = pd.DataFrame()
    model_path = None
    evaluation_metric = "loss" if args.training else "val_loss"
    desc = "reptile" if args.is_reptile == 1 else "joint"
    for args.cs_seed in range(args.min_cs_seed, args.max_cs_seed):
        model_path = os.path.join(save_path, args.search_space, desc, f"seed-{args.cs_seed}")

        try:
            # read metrics
            experiment_result = pd.read_csv(os.path.join(rootdir, args.save_path, model_path, "metrics.csv"),
                                            index_col=0)
            # get best performing epoch
            best_epoch = experiment_result[evaluation_metric].idxmin()
            results = pd.concat([results, pd.DataFrame(experiment_result[args.evaluation_metric].loc[best_epoch],
                                                       columns=[f"seed-{args.cs_seed}"])], axis=1)
        except Exception as e:
            print(e, args.seed)

    results.to_csv(os.path.join(rootdir, f"{args.search_space}-{desc}.csv"))
