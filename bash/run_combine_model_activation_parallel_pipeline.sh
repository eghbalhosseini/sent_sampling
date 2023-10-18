#!/bin/bash

#SBATCH --job-name=CM_PA
#SBATCH --array=0-6
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for dataset in neural_ctrl_stim ; do
      for stim_type in textNoPeriod ; do
      for model in  gpt2-xl xlnet-large-cased ; do
              for average_mode in False True None ; do
                  model_list[$i]="$model"
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
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

. /om/user/ehoseini/.bash_profile
. /om/user/ehoseini/.bashrc
conda activate neural_nlp_2022

/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/user/ehoseini/sent_sampling/combine_model_activations_parallel.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${dataset_list[$SLURM_ARRAY_TASK_ID]} ${stim_type_list[$SLURM_ARRAY_TASK_ID]} ${average_list[$SLURM_ARRAY_TASK_ID]}
