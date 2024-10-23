#!/bin/bash

#SBATCH --job-name=CM_PA
#SBATCH --array=0-6
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
models="roberta-base xlnet-large-cased bert-large-uncased-whole-word-masking xlm-mlm-en-2048 albert-xxlarge-v2 ctrl gpt2-xl"
layers="2 24 12 12 5 47 44" #
# for gpt2 I did 200 splits
#models="gpt2-xl"
#layers="44" #

#models="ctrl"
#layers="47" #


split=20

model_arr=($models)
layer_arr=($layers)
#split=100

len=${#layer_arr[@]}
# coca_preprocessed_all_clean_100K_sample_1
#roberta-base xlnet-large-cased bert-large-uncased-whole-word-masking \
#        xlm-mlm-en-2048 gpt2-xl albert-xxlarge-v2 ctrl
for dataset in timit_sentences  ; do
  for stim_type in textNoPeriod ; do
    for (( idx_model=0; idx_model<$len; idx_model++ )) ; do
      for average_mode in False ; do
        model_list[$i]="${model_arr[$idx_model]}"
        layer_list[$i]="${layer_arr[$idx_model]}"
        dataset_list[$i]="$dataset"
        average_list[$i]="$average_mode"
        stim_type_list[$i]="$stim_type"
        i=$i+1
      done
    done
  done
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running dataset ${dataset_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running average_mode ${average_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running stim type ${stim_type_list[$SLURM_ARRAY_TASK_ID]}"

module add openmind/singularity
RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

. /om/user/ehoseini/.bash_profile
. /om/user/ehoseini/.bashrc
conda activate neural_nlp_2022

/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om2/user/ehoseini/sent_sampling/sampling/combine_model_activations_parallel_per_layer.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${dataset_list[$SLURM_ARRAY_TASK_ID]} ${stim_type_list[$SLURM_ARRAY_TASK_ID]} ${average_list[$SLURM_ARRAY_TASK_ID]} ${split} ${layer_list[$SLURM_ARRAY_TASK_ID]}
