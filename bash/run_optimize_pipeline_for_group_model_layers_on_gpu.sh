#!/bin/bash

#SBATCH --job-name=opt_eh
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint="pascal|turing|volta"
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for optim_method in coordinate_ascent_eh ; do
  for n_iter in 1000 ; do
    for N_s in  50 200 ; do
      for init in 1 ; do
        for opt in D_s ; do
        optim_id="${optim_method}-obj=${opt}-n_iter=${n_iter}-n_samples=${N_s}-n_init=${init}-run_gpu=True"
        optim_list[$i]="$optim_id"
        i=$i+1
        done
      done
    done
  done
done



module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/${USER}/st/
RESULTCACHING_HOME=/om/user/${USER}/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/${USER}/st
export XDG_CACHE_HOME

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running optimiation: ${optim_list[$SLURM_ARRAY_TASK_ID]}"

singularity exec --nv -B /om:/om /om/user/${USER}/simg_images/neural_nlp_master_cuda.simg python /om/user/ehoseini/sent_sampling/group_extract_and_optimize_low_dim_on_gpu.py ${optim_list[$SLURM_ARRAY_TASK_ID]}