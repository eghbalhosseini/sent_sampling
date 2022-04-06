#!/bin/bash
#
#SBATCH --job-name=txt2spch
#SBATCH --exclude node[017-018]
#SBATCH --time=1:00:00
#SBATCH --mem=64G
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

while IFS=$'\t' read -r line_count framename sentence save_file ; do
  #echo "line_count ${model}"
  if [ $JID == $line_count ]
    then
      echo "found the right match ${line_count}"

      run_framename=$framename
      run_sentence=$sentence
      run_save_file=$save_file
      do_run=true
      break
    else
      do_run=false
      #echo "didnt the right match"
  fi

done <"${GRAND_FILE}"
echo "id ${run_framename}"
echo "sentence: ${run_sentence}"
echo "save location: ${run_save_file}"
#
. /home/ehoseini/.bash_profile

conda activate neural_nlp_1
export HOME=/om/user/ehoseini/
echo $(which python)

tts --text "${run_sentence}" --model_name tts_models/en/ljspeech/vits  --out_path "${run_save_file}"
