#!/bin/bash

#SBATCH --job-name=opt_uniq
#SBATCH --array=0-19
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
# create a run from 0 to 20
for run_ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ; do
    for ds in D_s ; do
        optim_id="coordinate_ascent_eh-obj=${ds}-n_iter=50-n_samples=225-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
        optim_list[$i]="$optim_id"
        run_list[$i]="$run_"
        i=$i+1
    done
done




echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running optimiation: ${optim_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running run: ${run_list[$SLURM_ARRAY_TASK_ID]}"


RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME


. /om/user/ehoseini/.bash_profile
. /om/user/ehoseini/.bashrc
conda activate neural_nlp_2022


/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om2/user/ehoseini/sent_sampling/sampling/test_uniqueness_optimize_ANNSet1_on_UDset.py ${run_list[$SLURM_ARRAY_TASK_ID]}
