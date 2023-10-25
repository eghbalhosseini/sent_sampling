#!/bin/bash

#SBATCH --job-name=EX_PA
#SBATCH --array=0-399
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for dataset in  coca_preprocessed_all_clean_no_dup_100K_sample_1 coca_preprocessed_all_clean_no_dup_100K_sample_2 ; do
  for group_ids in `seq 0 1 199` ; do
      for model in gpt2-xl  ; do
        for stim_type in textNoPeriod ; do
          model_list[$i]="$model"
          dataset_list[$i]="$dataset"
          group_id_list[$i]=$group_ids
          stim_type_list[$i]="$stim_type"
          i=$i+1
      done
      done
      done
done

#  2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running dataset ${dataset_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running stim type ${stim_type_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running group ${group_id_list[$SLURM_ARRAY_TASK_ID]}"

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

. /om/weka/evlab/ehoseini/.bash_profile
. /om/weka/evlab/ehoseini/.bashrc
conda activate neural_nlp_2022

echo $(which python)

/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini//sent_sampling/extract_model_activations_parallel.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${dataset_list[$SLURM_ARRAY_TASK_ID]} ${stim_type_list[$SLURM_ARRAY_TASK_ID]} ${group_id_list[$SLURM_ARRAY_TASK_ID]}
