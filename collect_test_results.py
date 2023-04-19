import os

import pandas as pd

from helpers import parsers

if __name__ == "__main__":
    parser = parsers.get_pets_parser()
    parser = parsers.get_hp_parser(parser)
    args = parser.parse_args()
    args.apply_lookahead = bool(args.apply_lookahead)
    args.load_pretrained = bool(args.load_pretrained)

    log_path = os.path.join("./results", "pre-trained" if args.load_pretrained  else "random-initialization",
                            args.search_space, f"horizon-{args.horizon}",
                            f"trajectories-{args.num_random_trajectories}", f"particles-{args.num_particles}",
                            f"{'LookAhead' if args.apply_lookahead  else 'MPC'}", args.inference_optimizer,
                            f"{args.inference_learning_rate}", f"mpc-{args.mpc_seed}")

    for dataset_id in os.listdir(log_path):
        # construct full file path
        dataset_folder = os.path.join(log_path, dataset_id)
        for subfolders in os.listdir(dataset_folder):
            results_folder = os.path.join(dataset_folder, subfolders)
            try:
                pd.read_csv(os.path.join(results_folder, "results.csv"))
            except Exception as e:
                print(dataset_id)
