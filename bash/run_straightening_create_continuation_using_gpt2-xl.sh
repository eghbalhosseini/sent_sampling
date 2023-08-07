#!/bin/bash

#SBATCH --job-name=continuation
#SBATCH --array=0-6
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for continuation  in 7 14 ; do
  for context in 3 6 9 ; do
      contination_list[$i]="$continuation"
      context_list[$i]="$context"
      i=$i+1
  done
done

i=0



echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running continuation: ${contination_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running context: ${context_list[$SLURM_ARRAY_TASK_ID]}"



XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

. /om/user/ehoseini/.bash_profile
conda activate neural_nlp_2022

/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/user/ehoseini/sent_sampling/straightening_create_continuation_using_gpt2-xl.py --continuation ${contination_list[$SLURM_ARRAY_TASK_ID]} --context ${context_list[$SLURM_ARRAY_TASK_ID]}
