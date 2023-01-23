#!/bin/bash

#SBATCH --job-name=opt_lowdim
#SBATCH --array=0-7
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for optim_method in coordinate_ascent_eh ; do
  for n_iter in 500 ; do
    for N_s in  100  125 ; do
      for ds in D_s 2-D_s ; do
        for low_dim in True False ; do
          for init in 1 ; do
            optim_id="${optim_method}-obj=${ds}-n_iter=${n_iter}-n_samples=${N_s}-n_init=${init}-low_dim=${low_dim}-run_gpu=True"
            optim_list[$i]="$optim_id"
            i=$i+1
          done
        done
      done
    done
  done
done

i=0
extract_name="activation"
bench_type="None"
extract_name=($extract_name)
bench_type=($bench_type)

#extractor_id = f'group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3_textPeriod-activation-bench=None-ave=False'

for set in best_performing_pereira_1 ; do
  for idx in 0 ; do
    for ave in False ; do
      for dataset in ud_sentencez_token_filter_v3_minus_ev_sentences ; do
        for text in textNoPeriod ; do
          extract_id="group=${set}-dataset=${dataset}_${text}-${extract_name[$idx]}-bench=${bench_type[$idx]}-ave=${ave}"
          extract_list[$i]="$extract_id"
          i=$i+1
        done
      done
    done
  done
done
run=0

for extract in ${extract_list[@]} ; do
  for optim in ${optim_list[@]} ; do
    extract_pool[$run]="$extract"
    optim_pool[$run]="$optim"
    run=$run+1
  done
done



echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running extraction: ${extract_pool[$SLURM_ARRAY_TASK_ID]}"
echo "Running optimiation: ${optim_pool[$SLURM_ARRAY_TASK_ID]}"

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

. /om/user/ehoseini/.bash_profile
. /om/user/ehoseini/.bashrc
conda activate neural_nlp_2022


/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/user/ehoseini/sent_sampling/extract_and_optimize_on_gpu.py ${extract_pool[$SLURM_ARRAY_TASK_ID]} ${optim_pool[$SLURM_ARRAY_TASK_ID]}
