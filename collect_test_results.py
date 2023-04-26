import os

import pandas as pd

from helpers import parsers

template_script = "cat /work/ws/nemo/fr_hj1023-LookAhead-0/mbrl/scripts/sbatch/{search_space}/joint/{subfolder1}/cs_seed-{cs_seed}/test/horizon-{horizon}/trajectories-{trajectories}/particles-{particles}/{subfolder2}/mpc-{mpc}/{dataset_id}.txt"

if __name__ == "__main__":
    parser = parsers.get_pets_parser()
    parser = parsers.get_hp_parser(parser)
    args = parser.parse_args()
    args.apply_lookahead = bool(args.apply_lookahead)
    args.load_pretrained = bool(args.load_pretrained)

    log_path = os.path.join("./results", "pre-trained" if args.load_pretrained else "random-initialization",
                            args.search_space, f"horizon-{args.horizon}",
                            f"trajectories-{args.num_random_trajectories}", f"particles-{args.num_particles}",
                            f"{'LookAhead' if args.apply_lookahead else 'MPC'}", args.inference_optimizer,
                            f"lr-{args.inference_learning_rate}", f"mpc-{args.mpc_seed}")
    found = 0
    not_found = 0
    for dataset_id in os.listdir(log_path):
        # construct full file path
        dataset_folder = os.path.join(log_path, dataset_id)
        try:
            pd.read_csv(os.path.join(dataset_folder, "results.csv"))
            found += 1
        except Exception as e:
            not_found += 1
            subfolder1 = "pre-trained" if args.load_pretrained else "random-initialization"
            subfolder2 = "LookAhead" if args.apply_lookahead else "MPC"
            print("Rerunning missing")
            rerun_script = template_script.format(search_space=args.search_space,
                                                  subfolder1=subfolder1,
                                                  subfolder2=subfolder2,
                                                  cs_seed=args.cs_seed,
                                                  horizon=args.horizon,
                                                  trajectories=args.num_trajectories,
                                                  particles=args.num_particles,
                                                  mpc=args.mpc_seed,
                                                  dataset_id=dataset_id,
                                                  )
    print(
        f"found {found}; not found {not_found} , {args.num_particles}, {args.horizon}, {args.apply_lookahead}, {args.load_pretrained}, {args.num_random_trajectories}")
