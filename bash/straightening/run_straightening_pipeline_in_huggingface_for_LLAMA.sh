#!/bin/bash

#SBATCH --job-name=LLAMA
#SBATCH --array=0-3
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --mem=150G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for model in  LLAMA_7B LLAMA_13B LLAMA_30B LLAMA_65B ; do
            model_list[$i]="$model"
            i=$i+1
done
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"

conda activate neural_nlp_2022

/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om2/user/ehoseini/sent_sampling/straightening/straightening_pipeline_in_huggingface_for_LLAMA.py --modelname ${model_list[$SLURM_ARRAY_TASK_ID]}


