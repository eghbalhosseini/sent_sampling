#!/bin/bash

#SBATCH --job-name=opt_jsd
#SBATCH --array=0-3
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for sub_id in 0 1 2 3 ; do
        subject_list[$i]="$sub_id"
        i=$i+1
done


echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running subject list : ${subject_list[$SLURM_ARRAY_TASK_ID]}"

RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME

. /om/user/ehoseini/.bash_profile
. /om/user/ehoseini/.bashrc
conda activate neural_nlp_2022

/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om2/user/ehoseini/sent_sampling/shape_metric/compute_procrustes_for_brain_DeepJuice_norm.py ${subject_list[$SLURM_ARRAY_TASK_ID]}
