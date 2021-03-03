#!/bin/bash

#SBATCH --job-name=EX_PA
#SBATCH --array=0-152%40
#SBATCH --time=144:00:00
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for dataset in ud_sentencez_token_filter_v3 ; do
  for group_ids in 0 1 2 3 4 5 6 7 8 ; do
      for model in gpt2-large gpt2-medium distilgpt2 openaigpt \
      albert-xxlarge-v1 albert-xlarge-v2 albert-xlarge-v1 albert-large-v2 albert-large-v1 \
      t5-11b t5-3b t5-large t5-base t5-small \
      xlnet-large-cased xlnet-base-cased \
      transfo-xl-wt103 ; do
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

export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
export RESULTCACHING_HOME

singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_nlp_master.simg /usr/local/bin/python /om/user/ehoseini/sent_sampling/extract_model_activations_parallel.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${dataset_list[$SLURM_ARRAY_TASK_ID]} ${group_id_list[$SLURM_ARRAY_TASK_ID]}


#transfo-xl-wt103 \
#        t5-3b \
#        xlnet-large-cased \
#        bert-large-uncased-whole-word-masking \
#        xlm-mlm-en-2048 \
#        gpt2-xl \
#        albert-xxlarge-v2 \
#        ctrl