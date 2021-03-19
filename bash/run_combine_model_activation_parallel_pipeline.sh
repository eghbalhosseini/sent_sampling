#!/bin/bash

#SBATCH --job-name=CM_PA
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for dataset in ud_sentencez_token_filter_v3 ; do
      for model in openaigpt gpt2 gpt2-xl ; do
          model_list[$i]="$model"
          dataset_list[$i]="$dataset"
          i=$i+1
      done
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running dataset ${dataset_list[$SLURM_ARRAY_TASK_ID]}"

export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
export RESULTCACHING_HOME

singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_nlp_master.simg /usr/local/bin/python /om/user/ehoseini/sent_sampling/combine_model_activations_parallel.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${dataset_list[$SLURM_ARRAY_TASK_ID]}


#transfo-xl-wt103 \
#        t5-3b \
#        xlnet-large-cased \
#        bert-large-uncased-whole-word-masking \
#        xlm-mlm-en-2048 \
#        gpt2-xl \
#        albert-xxlarge-v2 \
#        ctrl