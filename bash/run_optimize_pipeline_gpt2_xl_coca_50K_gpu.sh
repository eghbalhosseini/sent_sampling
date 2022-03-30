#!/bin/bash

#SBATCH --job-name=COCA_50K
#SBATCH --array=0
#SBATCH --time=5-12:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:RTXA6000:1
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/${USER}/st/
RESULTCACHING_HOME=/om/user/${USER}/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/${USER}/st
export XDG_CACHE_HOME

#echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
#echo "Running extraction: ${extract_pool[$SLURM_ARRAY_TASK_ID]}"
#echo "Running optimiation: ${optim_pool[$SLURM_ARRAY_TASK_ID]}"

. ~/.bash_profile
conda activate neural_nlp_cuda
echo $(which python)
cd /om/user/ehoseini/sent_sampling/

python extract_and_optimize_on_gpu_gpt2_xl_coca_50K.py