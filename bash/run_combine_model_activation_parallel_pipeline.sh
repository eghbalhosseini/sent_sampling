#!/bin/bash

#SBATCH --job-name=CM_PA
#SBATCH --array=0-2
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for dataset in ud_sentencez_token_filter_v3 coca_spok_filter_punct_10K_sample_1 ; do
      for model in gpt2-xl-untrained ; do
              for average_mode in False True None ; do
                  model_list[$i]="$model"
                  dataset_list[$i]="$dataset"
                  average_list[$i]="$average_mode"
                  dataset_list[$i]="$dataset"
                  i=$i+1
              done
      done
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running dataset ${dataset_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running average_mode ${average_list[$SLURM_ARRAY_TASK_ID]}"

export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
export RESULTCACHING_HOME

singularity exec -B /om:/om,/om2:/om2 /om/user/`whoami`/simg_images/neural_nlp_master.simg /usr/local/bin/python /om/user/ehoseini/sent_sampling/combine_model_activations_parallel.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${dataset_list[$SLURM_ARRAY_TASK_ID]} ${average_list[$SLURM_ARRAY_TASK_ID]}

#for dataset in coca_spok_filter_punct_10K_sample_1 \
# coca_spok_filter_punct_10K_sample_2 \
# coca_spok_filter_punct_10K_sample_3 \
# coca_spok_filter_punct_10K_sample_4 \
# coca_spok_filter_punct_10K_sample_5 ; do
#      for model in gpt2-xl gpt2-large gpt2-medium gpt2 distilgpt2 openaigpt \
#          albert-xxlarge-v2 albert-xlarge-v2 \
#          t5-11b t5-3b t5-large \
#          xlnet-large-cased \
#          ctrl \
#          bert-large-uncased-whole-word-masking distilbert-base-uncased \
#          xlm-mlm-en-2048 \
#          transfo-xl-wt103 \
#          roberta-base; do
#              for average_mode in True False None ; do
#                  model_list[$i]="$model"
#                  dataset_list[$i]="$dataset"
#                  average_list[$i]="$average_mode"
#                  dataset_list[$i]="$dataset"
#                  i=$i+1
#              done
#      done
#done