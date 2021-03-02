#!/bin/bash

#SBATCH --job-name=ext_mdl_act
#SBATCH --array=0-9
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --mem=180G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for dataset in ud_sentencez_token_filter_v3 ; do
  for group_ids in 0 1 2 3 4 5 6 7 8 9 ; do
      for model in  distilgpt2 ; do
          model_list[$i]="$model"
          dataset_list[$i]="$dataset"
          group_id_list[$i]=$group_ids
          i=$i+1
      done
      done
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running dataset ${dataset_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running group ${group_id_list[$SLURM_ARRAY_TASK_ID]}"

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME
singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_nlp_master.simg /usr/local/bin/python /om/user/ehoseini/sent_sampling/extract_model_activations_parallel.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${dataset_list[$SLURM_ARRAY_TASK_ID]} ${group_id_list[$SLURM_ARRAY_TASK_ID]}


#transfo-xl-wt103 \
#        t5-3b \
#        xlnet-large-cased \
#        bert-large-uncased-whole-word-masking \
#        xlm-mlm-en-2048 \
#        gpt2-xl \
#        albert-xxlarge-v2 \
#        ctrl