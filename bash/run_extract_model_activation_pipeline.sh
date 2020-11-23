#!/bin/bash

#SBATCH --job-name=ext_mdl_act
#SBATCH --array=0-4
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=120G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for dataset in ud_sentences_token_filter_v3_sample ; do
      for model in bert-large-uncased-whole-word-masking \
        xlm-mlm-en-2048 \
        ctrl \
        albert-xxlarge-v2 \
        gpt2-xl ; do
          model_list[$i]="$model"
          dataset_list[$i]="$dataset"
          i=$i+1
      done
done


module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME
singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_nlp_master.simg python /om/user/ehoseini/sent_sampling/extract_model_activations.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${dataset_list[$SLURM_ARRAY_TASK_ID]}