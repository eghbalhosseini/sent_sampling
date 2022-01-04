#!/bin/bash
#
#SBATCH --job-name=EX_PA
#SBATCH --exclude node[017-018]
#SBATCH --time=168:00:00
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

while IFS=, read -r line_count model dataset group_id ; do
  #echo "line_count ${model}"
  if [ $JID == $line_count ]
    then
      echo "found the right match ${line_count}"

      run_model=$model
      run_dataset=$dataset
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
echo "group_id ${run_group_id}"
module add openmind/singularity
XDG_CACHE_HOME=/om/user/${USER}/st
export XDG_CACHE_HOME
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om2/user/`whoami`/.result_caching
export RESULTCACHING_HOME
#
. ~/.bash_profile
conda activate neural_nlp
echo $(which python)
python /om/user/ehoseini/sent_sampling/extract_model_activations_parallel.py "${run_model}" "${run_dataset}" "${run_group_id}"