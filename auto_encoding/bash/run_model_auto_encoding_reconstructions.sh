#!/bin/bash

#SBATCH --job-name=opt_eh
#SBATCH --array=0-111
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
#'roberta-base' 'xlnet-large-cased' 'bert-large-uncased-whole-word-masking' 'xlm-mlm-en-2048' \
#    'gpt2-xl'  'albert-xxlarge-v2' 'ctrl'
for model_name in 'roberta-base' 'xlnet-large-cased' 'bert-large-uncased-whole-word-masking' 'xlm-mlm-en-2048' \
'gpt2-xl'  'albert-xxlarge-v2' 'ctrl' ; do
  for bottleneck_size in 16 32 ; do
    for hidden_size in  128 256 ; do
      for alpha_r in '0' '0.00001' '0.001'  '0.1' ; do
        model_list[$i]="$model_name"
        bottleneck_list[$i]="$bottleneck_size"
        hidden_list[$i]="$hidden_size"
        alpha_r_list[$i]="$alpha_r"
        i=$i+1
        done
      done
    done
  done



RESULTCACHING_HOME=/om/user/${USER}/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/${USER}/st
export XDG_CACHE_HOME

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running bottleneck ${bottleneck_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running hidden ${hidden_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running alpha_r ${alpha_r_list[$SLURM_ARRAY_TASK_ID]}"

. /om/weka/evlab/ehoseini/.bash_profile
. /om/weka/evlab/ehoseini/.bashrc
conda activate neural_nlp_2022

/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini//sent_sampling/auto_encoding/train_model_autoencoder_with_reconstruction_loss.py \
--model_name ${model_list[$SLURM_ARRAY_TASK_ID]} --bottleneck_size ${bottleneck_list[$SLURM_ARRAY_TASK_ID]} --hidden_size ${hidden_list[$SLURM_ARRAY_TASK_ID]} --alpha_r ${alpha_r_list[$SLURM_ARRAY_TASK_ID]}
