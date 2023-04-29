import os
import subprocess

# for particles in 10 100 1000 ; do for horizon in 3 5 10 ; do for trajectories in 100 250 500 ; do for apply_lookahead in 0 1 ; do for load_pretrained in 1 0 ; do python collect_test_results.py --horizon $horizon --num_particles $particles --num_random_trajectories $trajectories --apply_lookahead $apply_lookahead --load_pretrained $load_pretrained;done;done;done;done;done;
import pandas as pd
from helpers import parsers

template_script = "{command} /work/ws/nemo/fr_hj1023-LookAhead-0/mbrl/scripts/sbatch/{search_space}/joint/{subfolder1}/cs_seed-45/test/horizon-{horizon}/trajectories-{trajectories}/particles-{particles}/{subfolder2}/mpc-{mpc}/{dataset_id}.sh"

if __name__ == "__main__":
    parser = parsers.get_pets_parser()
    parser = parsers.get_hp_parser(parser)
    parser.add_argument('--rerun', type=int, default=1, choices=[0, 1], help='re-run failed experiments')
    args = parser.parse_args()
    args.apply_lookahead = bool(args.apply_lookahead)
    args.rerun = bool(args.rerun)
    args.load_pretrained = bool(args.load_pretrained)

    rootdir = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(rootdir, "results", "pre-trained" if args.load_pretrained else "random-initialization",
                            args.search_space, f"horizon-{args.horizon}",
                            f"trajectories-{args.num_random_trajectories}", f"particles-{args.num_particles}",
                            f"{'LookAhead' if args.apply_lookahead else 'MPC'}", args.inference_optimizer,
                            f"lr-{args.inference_learning_rate}", f"mpc-{args.mpc_seed}")
    found = 0
    not_found = 0
    try:
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

                rerun_command = template_script.format(command="cat" if not args.rerun else "msub",
                                                       search_space=args.search_space,
                                                       subfolder1=subfolder1,
                                                       subfolder2=subfolder2,
                                                       horizon=args.horizon,
                                                       trajectories=args.num_random_trajectories,
                                                       particles=args.num_particles,
                                                       mpc=args.mpc_seed,
                                                       dataset_id=dataset_id,
                                                       )
                if args.rerun:
                    print(f"Rerunning missing using command: {rerun_command}")
                    subprocess.run(rerun_command.split(" "))
        print(
        f"found {found}; not found {not_found} , {args.num_particles}, {args.horizon}, {args.apply_lookahead}, {args.load_pretrained}, {args.num_random_trajectories}")
    except Exception as ex:
        subfolder1 = "pre-trained" if args.load_pretrained else "random-initialization"
        subfolder2 = "LookAhead" if args.apply_lookahead else "MPC"

        rerun_command = "for particles in {particles}; do for horizon in {horizon}; do for trajectories in {trajectories};  do for file in /work/ws/nemo/fr_hj1023-LookAhead-0/mbrl/scripts/sbatch/{search_space}/joint/{subfolder1}/cs_seed-45/test/horizon-$horizon/trajectories-$trajectories/particles-$particles/{subfolder2}/mpc-{mpc}/*; do {command} $file;done;done;done;done;done;done;"
        rerun_command = rerun_command.format(command="cat" if not args.rerun else "msub",
                                               search_space=args.search_space,
                                               subfolder1=subfolder1,
                                               subfolder2=subfolder2,
                                               horizon=args.horizon,
                                               trajectories=args.num_random_trajectories,
                                               particles=args.num_particles,
                                               mpc=args.mpc_seed
                                               )
        print(f"Rerunning missing using command: {rerun_command}")
        subprocess.run(rerun_command.split(" "))
