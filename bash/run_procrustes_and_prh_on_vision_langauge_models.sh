#!/bin/bash

#SBATCH --job-name=prh_proc
#SBATCH --array=0-19
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=high-capacity
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for vision_type in  in21k mae dinov2 clip clip_ft_in12k ; do
    for layer_method in prh last ; do
      for layer_method_k in 3 5 ; do
        vision_list[$i]="$vision_type"
        layer_list[$i]="$layer_method"
        layer_k_list[$i]="$layer_method_k"
        i=$i+1
      done
    done
done

#  2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${vision_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running layer ${layer_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running layer_k ${layer_k_list[$SLURM_ARRAY_TASK_ID]}"


. /om/weka/evlab/ehoseini/.bash_profile
. /om/weka/evlab/ehoseini/.bashrc
conda activate neural_nlp_2022

echo $(which python)

/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om2/user/ehoseini/sent_sampling/shape_metric/compute_procrustes_for_vit_models_platonic_Jul2025.py ${vision_list[$SLURM_ARRAY_TASK_ID]} ${layer_list[$SLURM_ARRAY_TASK_ID]} ${layer_k_list[$SLURM_ARRAY_TASK_ID]}
