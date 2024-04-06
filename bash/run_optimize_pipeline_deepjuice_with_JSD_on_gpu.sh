#!/bin/bash

#SBATCH --job-name=opt_juice
#SBATCH --array=0-11
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=high-capacity
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for optim_method in coordinate_ascent_eh ; do
  for n_iter in 500 ; do
    for N_s in  80 100 150 ; do
      for ds in D_s_jsd 2-D_s_jsd ; do
        for low_dim in False ; do
          for pca_var in 0.9 ;do
            for pca_type in sklearn ; do
              for set in original redux ; do
                for init in 1 ; do
                  optim_id="${optim_method}-obj=${ds}-n_iter=${n_iter}-n_samples=${N_s}-n_init=${init}-low_dim=${low_dim}-pca_var=${pca_var}-pca_type=${pca_type}-run_gpu=True"
                  optim_list[$i]="$optim_id"
                  extract_list[$i]="${set}"
                  i=$i+1
              done
            done
          done
        done
      done
    done
  done
done
done



echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running optimiation: ${optim_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running extract: ${extract_list[$SLURM_ARRAY_TASK_ID]}"

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

. /om/user/ehoseini/.bash_profile
. /om/user/ehoseini/.bashrc
conda activate neural_nlp_2022


/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om2/user/ehoseini/sent_sampling/sampling/optimize_models_from_DeepJuice_with_JSD.py ${extract_list[$SLURM_ARRAY_TASK_ID]} ${optim_list[$SLURM_ARRAY_TASK_ID]}
