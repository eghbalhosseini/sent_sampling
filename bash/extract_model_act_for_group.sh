#!/bin/bash
#
#SBATCH --job-name=EX_PA
#SBATCH --exclude node[017-018]
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1

GRAND_FILE=$1
OVERWRITE='false' # or 'true'
#

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  JID=$SLURM_ARRAY_TASK_ID    # Taking the task ID in a job array as an input parameter.
else
  JID=$2       # Taking the task ID as an input parameter.
fi
echo "${GRAND_FILE}"
echo $JID

while IFS=, read -r line_count model dataset stim_type splits group_id ; do
  #echo "line_count ${model}"
  if [ $JID == $line_count ]
    then
      echo "found the right match ${line_count}"

      run_model=$model
      run_dataset=$dataset
      run_stim_type=$stim_type
      run_splits=$splits
      run_group_id=$group_id
      do_run=true
      break
    else
      do_run=false
      #echo "didnt the right match"
  fi

done <"${GRAND_FILE}"
echo "model ${run_model}"
echo "dataset ${run_dataset}"
echo "stim_type ${run_stim_type}"
echo "splits ${run_splits}"
echo "group_id ${run_group_id}"


module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

. /om/weka/evlab/ehoseini/.bash_profile
. /om/weka/evlab/ehoseini/.bashrc
conda activate neural_nlp_2022

echo $(which python)
/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini/sent_sampling/extract_model_activations_parallel.py "${run_model}" "${run_dataset}" "${run_stim_type}" "${splits}" "${run_group_id}"

#/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini//sent_sampling/extract_model_activations_parallel.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${dataset_list[$SLURM_ARRAY_TASK_ID]} ${stim_type_list[$SLURM_ARRAY_TASK_ID]} ${splits} ${group_id_list[$SLURM_ARRAY_TASK_ID]}