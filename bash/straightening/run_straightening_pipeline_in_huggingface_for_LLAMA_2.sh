#!/bin/bash

#SBATCH --job-name=LLAMA
#SBATCH --array=0
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=high-capacity
#SBATCH --mem=256G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for model in  70B ; do
            model_list[$i]="$model"
            i=$i+1
done
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"

conda activate neural_nlp_2022

echo $(which python)

/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om2/user/ehoseini/sent_sampling/straightening/straightening_pipeline_in_huggingface_for_LLAMA_2.py --modelname ${model_list[$SLURM_ARRAY_TASK_ID]}


