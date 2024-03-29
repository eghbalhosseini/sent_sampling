#!/bin/bash

#SBATCH --job-name=opt_eh
#SBATCH --array=0-6
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for optim_method in coordinate_ascent_eh ; do
  for n_iter in 1000 ; do
    for N_s in  25 50 75 100 125 150 175 ; do
      for init in 1 ; do
        optim_id="${optim_method}-obj=D_s-n_iter=${n_iter}-n_samples=${N_s}-n_init=${init}-run_gpu=True"
        optim_list[$i]="$optim_id"
        i=$i+1
      done
    done
  done
done

i=0
extract_name="activation brain_resp_Pereira_exp1 brain_resp_Pereira_exp2"
bench_type="None Pereira2018-encoding-weights Pereira2018-encoding-weights"
extract_name=($extract_name)
bench_type=($bench_type)


for set in best_performing_pereira_1 ; do
  for idx in 0 ; do
    for ave in False ; do
    for dataset in ud_sentencez_token_filter_v3 ; do
      extract_id="group=${set}-dataset=${dataset}-${extract_name[$idx]}-bench=${bench_type[$idx]}-ave=${ave}"
      extract_list[$i]="$extract_id"
      i=$i+1
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



module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/${USER}/st/
RESULTCACHING_HOME=/om/user/${USER}/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/${USER}/st
export XDG_CACHE_HOME

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running extraction: ${extract_pool[$SLURM_ARRAY_TASK_ID]}"
echo "Running optimiation: ${optim_pool[$SLURM_ARRAY_TASK_ID]}"


singularity exec --nv -B /om:/om /om/user/${USER}/simg_images/neural_nlp_master_cuda.simg python /om/user/ehoseini/sent_sampling/extract_and_optimize_on_gpu.py ${extract_pool[$SLURM_ARRAY_TASK_ID]} ${optim_pool[$SLURM_ARRAY_TASK_ID]}