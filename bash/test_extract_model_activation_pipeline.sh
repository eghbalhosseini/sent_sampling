#!/bin/bash

#SBATCH --job-name=ext_mdl_act
#SBATCH --array=0-40
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --mem=180G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for dataset in ud_sentences_token_filter_v3 ; do
      for model in transformer \
          bert-base-uncased bert-base-multilingual-cased bert-large-uncased bert-large-uncased-whole-word-masking \
          openaigpt gpt2 gpt2-medium gpt2-large gpt2-xl distilgpt2 \
          transfo-xl-wt103 \
          xlnet-base-cased xlnet-large-cased xlm-mlm-en-2048 xlm-mlm-enfr-1024 xlm-mlm-xnli15-1024 xlm-clm-enfr-1024 xlm-mlm-100-1280 \
          roberta-base roberta-large distilroberta-base \
          distilbert-base-uncased \
          ctrl \
          albert-base-v1 albert-base-v2 albert-large-v1 albert-large-v2 albert-xlarge-v1 albert-xlarge-v2 albert-xxlarge-v1 albert-xxlarge-v2 \
          t5-small t5-base t5-large t5-3b t5-11b \
          xlm-roberta-base xlm-roberta-large ; do
            model_list[$i]="$model"
            dataset_list[$i]="$dataset"
          i=$i+1
      done
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${dataset_list[$SLURM_ARRAY_TASK_ID]}"

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME
singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_nlp_master.simg python /om/user/ehoseini/sent_sampling/extract_model_activations.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${dataset_list[$SLURM_ARRAY_TASK_ID]}