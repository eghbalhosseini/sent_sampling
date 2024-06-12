#!/bin/bash

#SBATCH --job-name=RM_PA
#SBATCH --array=0-34
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for dataset in coca_spok_filter_punct_10K_sample_1 coca_spok_filter_punct_10K_sample_2 coca_spok_filter_punct_10K_sample_3 coca_spok_filter_punct_10K_sample_4 coca_spok_filter_punct_10K_sample_5 ; do
  for model in roberta-base \
      xlnet-large-cased \
      gpt2-xl \
      bert-large-uncased-whole-word-masking \
      xlm-mlm-en-2048 \
      albert-xxlarge-v2 ; do
      for stim_type in "" ; do
          model_list[$i]="$model"
          dataset_list[$i]="$dataset"
          stim_type_list[$i]="$stim_type"
          i=$i+1
      done
  done
done

#  2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running dataset ${dataset_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running stim type ${stim_type_list[$SLURM_ARRAY_TASK_ID]}"


RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
DATA_DIR=/nese/mit/group/evlab/u/ehoseini/MyData/sent_sampling/
# first find the group data
model=${model_list[$SLURM_ARRAY_TASK_ID]}
dataset=${dataset_list[$SLURM_ARRAY_TASK_ID]}
stim_type=${stim_type_list[$SLURM_ARRAY_TASK_ID]}
look_up_pattern="${dataset}${stim_type}_${model}_layer_*_activation_group_*.pkl"
folder_to_look=${DATA_DIR}/${model}
for file in $(find $folder_to_look -name $look_up_pattern); do
    echo "deleting $file"
    rm $file
done
# second find the crunched data
look_up_pattern="${dataset}${stim_type}_${model}_layer_*_activation_*.pkl"
folder_to_look=${DATA_DIR}
# delete the files from find and print which files are deleted
for file in $(find $folder_to_look -name $look_up_pattern); do
  echo "deleting $file"
  rm $file
done

# third delete files from result caching
look_up_pattern="identifier=${model},stimuli_identifier=${dataset}${stim_type}_group_*"
activation_store='neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored'
folder_to_look=${RESULTCACHING_HOME}/${activation_store}

# Print folder to look
echo "looking in $folder_to_look"

# Print look up pattern
echo "looking for $look_up_pattern"

# Find and delete files in a single command
find "$folder_to_look" -name "$look_up_pattern" -exec echo "deleting {}" \; -exec rm {} \;

