import os
import json

template = \
    """#!/bin/bash
#MSUB -l walltime=96:00:00
#MSUB -l nodes=1:ppn=4
#MSUB -l pmem=12gb
#MSUB -N {jobname}
#MSUB -e {output1}/{jobid1}-error.txt
#MSUB -o {output2}/{jobid2}-output.txt

envs=LookAhead

cs_seed={cs_seed}
reptile={reptile}
search_space={search_space}
dataset_id={dataset_id}
num_particles={num_particles}
num_random_trajectories={num_random_trajectories}
horizon={horizon}
mpc_seed={mpc_seed}
num_trials={num_trials}
apply_lookahead={apply_lookahead}
source /home/fr/fr_fr/fr_hj1023/miniconda3/bin/activate $envs 
   
cd /work/ws/nemo/fr_hj1023-LookAhead-0/mbrl || exit 
python run_pets.py --cs_seed $cs_seed --reptile $reptile --search_space $search_space --horizon $horizon --mpc_seed $mpc_seed --num_trials $num_trials --dataset_id $dataset_id --num_particles $num_particles --num_random_trajectories $num_random_trajectories --apply_lookahead $apply_lookahead

"""


if __name__ == "__main__":
    rootdir = os.path.dirname(os.path.realpath(__file__))

    from helpers.parsers import get_pets_parser, get_hp_parser, get_tuner_parser

    parser = get_pets_parser()
    parser = get_tuner_parser(parser)
    parser = get_hp_parser(parser)
    args = parser.parse_args()
    args.part = "test"

    # define sbatch folder
    sbatchfolder = os.path.join(rootdir, "scripts/sbatch")
    # define output folder
    sbatchfolderoutput = os.path.join(rootdir, "scripts/output")

    # make folders
    [os.makedirs(_, exist_ok=True) for _ in [sbatchfolder, sbatchfolderoutput]]

    model_path = os.path.join(args.search_space, f"{'reptile' if args.reptile == 1 else 'joint'}", "test",
                              f"horizon-{args.horizon}", f"trajectories-{args.num_random_trajectories}",
                              f"particles-{args.num_particles}",
                              f"{'LookAhead' if args.apply_lookahead == 1 else 'MPC'}", f"mpc-{args.mpc_seed}")
    file_dir = os.path.join(sbatchfolder, model_path)
    output_dir = os.path.join(sbatchfolderoutput, model_path)

    os.makedirs(file_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    job_name = model_path.replace("/", "-")
    job_id = "${MOAB_JOBID}"

    # read train/validation/test dataset ids
    with open(os.path.join(args.data_directory, "splits", f"{args.part}.json"), "r") as f:
        dataset_ids = json.load(f)[args.search_space]

    for dataset_id in dataset_ids:
        script = template.format(jobname=job_name,
                                 output1=output_dir,
                                 output2=output_dir,
                                 jobid1=job_id,
                                 jobid2=job_id,
                                 cs_seed=args.cs_seed,
                                 reptile=args.reptile,
                                 search_space=args.search_space,
                                 dataset_id=dataset_id,
                                 num_particles=args.num_particles,
                                 num_random_trajectories=args.num_random_trajectories,
                                 horizon=args.horizon,
                                 mpc_seed=args.mpc_seed,
                                 num_trials=args.num_trials,
                                 apply_lookahead=args.apply_lookahead
                                 )
        file_name = f"{dataset_id}.sh"
        file = open(os.path.join(file_dir, file_name), 'w')
        file.write(script + '\n')
        file.close()
