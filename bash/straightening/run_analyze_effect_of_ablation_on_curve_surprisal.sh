#!/bin/bash

#SBATCH --job-name=Ablate
#SBATCH --array=0-9
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu


i=0
for modelname in gpt2-xl ; do
    for ablation_type in  Attn_key Attn_all ; do
        for layer_to_ablate in 5 15 25 35 45 ; do
          model_list[$i]="$modelname"
          ablation_list[$i]="$ablation_type"
          layer_to_ablate_list[$i]="$layer_to_ablatet"
          i=$i+1
      done
    done
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running ablation ${ablation_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running layer_to_ablate ${layer_to_ablate_list[$SLURM_ARRAY_TASK_ID]}"

XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

. /om/user/ehoseini/.bash_profile
conda activate neural_nlp_2022

/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om2/user/ehoseini/sent_sampling/straightening/analyze_the_effect_of_attention_ablation_on_curvature_and_surprisal_gpt2-xl.py --model ${model_list[$SLURM_ARRAY_TASK_ID]} --ablation_type ${ablation_list[$SLURM_ARRAY_TASK_ID]} --layer_to_ablate ${layer_to_ablate_list[$SLURM_ARRAY_TASK_ID]}
