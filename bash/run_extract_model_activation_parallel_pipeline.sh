#!/bin/bash

#SBATCH --job-name=EX_PA
#SBATCH --array=0-14
#SBATCH --time=144:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
# create group_id that goes from 0 to 19
# 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
# roberta-base xlnet-large-cased bert-large-uncased-whole-word-masking \
  #          xlm-mlm-en-2048 gpt2-xl albert-xxlarge-v2 ctrl

for dataset in ud_sentencez_ds_max_100_edited ud_sentencez_ds_random_100_edited ; do
  for group_ids in  15 ; do
    for stim_type in textNoPeriod ; do
      for model in  roberta-base xlnet-large-cased bert-large-uncased-whole-word-masking \
          xlm-mlm-en-2048 gpt2-xl albert-xxlarge-v2 ctrl ; do
            model_list[$i]="$model"
            dataset_list[$i]="$dataset"
            stim_type_list[$i]="$stim_type"
            group_id_list[$i]=$group_ids
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

. /om/user/ehoseini/.bash_profile
. /om/user/ehoseini/.bashrc
conda activate neural_nlp_2022

/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/user/ehoseini/sent_sampling/extract_model_activations_parallel.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${dataset_list[$SLURM_ARRAY_TASK_ID]} ${stim_type_list[$SLURM_ARRAY_TASK_ID]} ${group_id_list[$SLURM_ARRAY_TASK_ID]}


