#!/bin/bash

#SBATCH --job-name=COCA_50K
#SBATCH -N 1
#SBATCH --time=5-12:00:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:RTXA6000:2
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --nodelist=node094
#SBATCH -p evlab
#SBATCH --mail-user=ehoseini@mit.edu

export SINGULARITY_CACHEDIR=/om/user/${USER}/st/
XDG_CACHE_HOME=/om/user/${USER}/st
export XDG_CACHE_HOME

. ~/.bash_profile
conda activate neural_nlp_cuda
echo $(which python)
cd /om/user/ehoseini/sent_sampling/

python optimize_gpt2_xl_on_COCA_50K_syl_2to4sec_on_gpu.py "group=gpt2-xl_layers-dataset=coca_spok_filter_punct_50K_sylb_2to4sec-activation-bench=None-ave=False" "coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=250-n_init=1-run_gpu=True"