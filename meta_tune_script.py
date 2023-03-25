import os

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
source /home/fr/fr_fr/fr_hj1023/miniconda3/bin/activate $envs 

cd /work/ws/nemo/fr_hj1023-LookAhead-0/mbrl || exit 
python meta_tune.py --cs_seed $cs_seed --reptile $reptile

"""

if __name__ == "__main__":
    rootdir = os.path.dirname(os.path.realpath(__file__))

    from helpers.parsers import get_tuner_parser

    parser = get_tuner_parser()
    args = parser.parse_args()

    # define sbatch folder
    sbatchfolder = os.path.join(rootdir, "scripts/sbatch")
    # define output folder
    sbatchfolderoutput = os.path.join(rootdir, "scripts/output")

    # make folders
    [os.makedirs(_, exist_ok=True) for _ in [sbatchfolder, sbatchfolderoutput]]

    model_path = os.path.join(f"{'reptile' if args.reptile == 1 else 'joint'}", f"cs_seed-{args.cs_seed}")

    file_dir = os.path.join(sbatchfolder, model_path)
    output_dir = os.path.join(sbatchfolderoutput, model_path)

    os.makedirs(file_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    file_name = "meta_tune.sh"
    job_name = model_path.replace("/", "-")
    job_id = "${MOAB_JOBID}"
    file = open(os.path.join(file_dir, file_name), 'w')

    script = template.format(jobname=job_name,
                             output1=output_dir,
                             output2=output_dir,
                             jobid1=job_id,
                             jobid2=job_id,
                             cs_seed=args.cs_seed,
                             reptile=args.reptile)

    file.write(script + '\n')
    file.close()

# for cv in 1; do for seed in $(seq 1000 1999) ; do for toy in toy; do for freeze in freeze; do for reset_final_layer in reset; do for innerupdates in n_batch_updates;
# do msub /work/ws/nemo/fr_hj1023-UniTab-0/TabNet/scripts/sbatch/$toy/joint/contextunitab/$freeze/$reset_final_layer/$innerupdates/deterministic/seed-$seed/cv-$cv/train.sh; done;done;done;done;done; done


# for cv in 1; do for seed in $(seq 0 2000) ; do for toy in realworld; do for freeze in freeze; do for reset_final_layer in reset; do cat /work/ws/nemo/fr_hj1023-UniTab-0/TabNet/scripts/sbatch/$toy/joint/transformer/$freeze/$reset_final_layer/seed-$seed/cv-$cv/train.sh; done;done;done;done;done