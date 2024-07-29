#!/bin/bash

#SBATCH --job-name=opt_jsd
#SBATCH --array=0-7
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for multply in 5 10 ; do
  for theshold in 0.1 0.2 ; do
      for ds in D_s_jsd 2-D_s_jsd ; do
        optim_id="coordinate_ascent_eh-obj=${ds}-n_iter=50-n_samples=225-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True"
        optim_list[$i]="$optim_id"
        multiply_list[$i]="$multply"
        theshold_list[$i]="$theshold"
        i=$i+1
      done
  done
done




echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running optimiation: ${optim_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running multiply: ${multiply_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running theshold: ${theshold_list[$SLURM_ARRAY_TASK_ID]}"


RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME


. /om/user/ehoseini/.bash_profile
. /om/user/ehoseini/.bashrc
conda activate neural_nlp_2022


/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om2/user/ehoseini/sent_sampling/sampling/optimize_ANNSet1_on_UDset_js_divergence.py ${optim_list[$SLURM_ARRAY_TASK_ID]} ${multiply_list[$SLURM_ARRAY_TASK_ID]} ${theshold_list[$SLURM_ARRAY_TASK_ID]}
