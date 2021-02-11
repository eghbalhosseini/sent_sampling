#!/bin/bash

#SBATCH --job-name=ext_mdl_act
#SBATCH --array=0-4
#SBATCH --time=167:00:00
#SBATCH --ntasks=1
#SBATCH --mem=120G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for dataset in coca_spok_filter_punct_10K_sample_1 \
               coca_spok_filter_punct_10K_sample_2 \
               coca_spok_filter_punct_10K_sample_3 \
               coca_spok_filter_punct_10K_sample_4 \
               coca_spok_filter_punct_10K_sample_5 ; do
                for model in albert-xxlarge-v2 ; do
                      model_list[$i]="$model"
                      dataset_list[$i]="$dataset"
                  i=$i+1
                done
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running dataset ${dataset_list[$SLURM_ARRAY_TASK_ID]}"

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME
singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_nlp_master.simg python /om/user/ehoseini/sent_sampling/extract_model_activations.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${dataset_list[$SLURM_ARRAY_TASK_ID]}