#!/bin/bash

#SBATCH --job-name=ext_opt
#SBATCH --array=0-5
#SBATCH --time=72:00:00
#SBATCH --mem=120G
#SBATCH -c 16
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for optim_method in coordinate_ascent ; do
  for n_iter in 1000 ; do
    for N_s in  300 ; do
      for init in 1 ; do
        optim_id="${optim_method}-obj=D_s-n_iter=${n_iter}-n_samples=${N_s}-n_init=${init}"
        optim_list[$i]="$optim_id"
        i=$i+1
      done
    done
  done
done

i=0
extract_type="activation brain_resp brain_resp"
bench_type="None Fedorenko2016v3-encoding-weights Pereira2018-encoding-weights"
extract_type=($extract_type)
bench_type=($bench_type)


for set in second_best_performing_pereira \
 best_performing_pereira ; do
  for idx in 0 1 2 ; do
    for ave in False ; do
    for dataset in ud_sentences_token_filter_v3 ; do
      extract_id="group=${set}-dataset=${dataset}-${extract_type[$idx]}-bench=${bench_type[$idx]}-ave=${ave}"
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
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running extraction: ${extract_pool[$SLURM_ARRAY_TASK_ID]}"
echo "Running optimiation: ${optim_pool[$SLURM_ARRAY_TASK_ID]}"


singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_nlp_master.simg python /om/user/ehoseini/sent_sampling/extract_and_optimize.py ${extract_pool[$SLURM_ARRAY_TASK_ID]} ${optim_pool[$SLURM_ARRAY_TASK_ID]}