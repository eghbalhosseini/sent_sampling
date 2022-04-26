#!/bin/bash

#SBATCH --job-name=ext_opt
#SBATCH --array=0-6
#SBATCH --time=144:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu


i=0
extract_name="activation"
bench_type="None"
extract_name=($extract_name)
bench_type=($bench_type)


for set in albert-xxlarge-v2 \
 ctrl \
 bert-large-uncased-whole-word-masking \
 roberta-base \
 xlnet-large-cased \
 gpt2-xl \
 xlm-mlm-en-2048 ; do
  for idx in 0 ; do
    for ave in False ; do
    for dataset in ud_sentences_U01_AnnSET1_ordered_for_RDM ; do
        extract_id="group=${set}_layers-dataset=${dataset}-${extract_name[$idx]}-bench=${bench_type[$idx]}-ave=${ave}"
        extract_list[$i]="$extract_id"
        i=$i+1
      done
    done
  done
done


run=0

for extract in ${extract_list[@]} ; do
    extract_pool[$run]="$extract"
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


singularity exec -B /om:/om /om/user/${USER}/simg_images/neural_nlp_master.simg python /om/user/ehoseini/sent_sampling/extract.py ${extract_pool[$SLURM_ARRAY_TASK_ID]}
