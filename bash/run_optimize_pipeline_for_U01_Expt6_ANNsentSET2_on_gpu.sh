#!/bin/bash

#SBATCH --job-name=opt_eh
#SBATCH --array=0-111%20
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for optim_method in coordinate_ascent_eh ; do
  for n_iter in 1000 ; do
    for N_s in  50 75 ; do
      for init in 2 ; do
        for opt in D_s D_s_var ; do
        optim_id="${optim_method}-obj=${opt}-n_iter=${n_iter}-n_samples=${N_s}-n_init=${init}-run_gpu=True"
        optim_list[$i]="$optim_id"
        i=$i+1
        done
      done
    done
  done
done

i=0
extract_name="activation brain_resp_Pereira_exp1 brain_resp_Pereira_exp2"
bench_type="None Pereira2018-encoding-weights Pereira2018-encoding-weights"
extract_name=($extract_name)
bench_type=($bench_type)


for set in gpt2-xl_layer_compare_v1 \
 ctrl_layer_compare_v1 \
 bert-large-uncased-whole-word-masking_layer_compare_v1 \
 albert-xxlarge-v2_layer_compare_v1 \
 roberta-base_layer_compare_v1 \
 xlm-mlm-en-2048_layer_compare_v1 \
 xlnet-large-cased_layer_compare_v1 ; do
  for idx in 0 ; do
    for ave in False ; do
    for dataset in coca_spok_filter_punct_10K_sample_2 coca_spok_filter_punct_10K_sample_3 \
     coca_spok_filter_punct_10K_sample_4 coca_spok_filter_punct_10K_sample_5 ; do
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