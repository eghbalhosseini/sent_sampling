#!/bin/bash

#SBATCH --job-name=layer_analyze
#SBATCH --array=0-27
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for optim_method in coordinate_ascent_eh ; do
  for n_iter in 1000 ; do
    for N_s in  25 ; do
      for init in 1 ; do
        optim_id="${optim_method}-obj=D_s_var-n_iter=${n_iter}-n_samples=${N_s}-n_init=${init}-run_gpu=True"
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


for set in roberta-base bert-large-uncased-whole-word-masking xlm-mlm-en-2048 gpt2-xl albert-xxlarge-v2 ctrl xlnet-large-cased ; do
    for idx in 0 ; do
      for ave in False ; do
        for dataset in ud_sentencez_token_filter_v3 coca_spok_filter_punct_10K_sample_1 ; do
            extract_id="group=${set}_layers-dataset=${dataset}-${extract_name[$idx]}-bench=${bench_type[$idx]}-ave=${ave}"
            extract_list[$i]="$extract_id"
            i=$i+1
      done
    done
  done
done

i=0
for pca in fix equal_var ; do
      pca_list[$i]="$pca"
      i=$i+1
done

run=0
for extract in ${extract_list[@]} ; do
  for optim in ${optim_list[@]} ; do
    for pca in ${pca_list[@]} ; do
    #echo $extract
    extract_pool[$run]="$extract"
    optim_pool[$run]="$optim"
    pca_pool[$run]="$pca"
    run=$run+1
  done
  done
done


module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/${USER}/st/
RESULTCACHING_HOME=/om/user/${USER}/.result_caching
export RESULTCACHING_HOME

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running extraction: ${extract_pool[$SLURM_ARRAY_TASK_ID]}"
echo "Running optimiation: ${optim_pool[$SLURM_ARRAY_TASK_ID]}"
echo "Running pca type: ${pca_pool[$SLURM_ARRAY_TASK_ID]}"


singularity exec --nv -B /om:/om /om/user/${USER}/simg_images/neural_nlp_master_cuda.simg python /om/user/ehoseini/sent_sampling/analyze_model_layer_representations_gpu.py ${extract_pool[$SLURM_ARRAY_TASK_ID]} ${optim_pool[$SLURM_ARRAY_TASK_ID]} ${pca_pool[$SLURM_ARRAY_TASK_ID]}