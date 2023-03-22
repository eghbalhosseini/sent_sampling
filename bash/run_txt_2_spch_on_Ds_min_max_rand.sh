#!/bin/bash
DATA_DIR=/nese/mit/group/evlab/u/ehoseini/MyData/sent_sampling/analysis/
i=0
i=$(expr ${i} + 0)
LINE_COUNT=0

extract_id='group=best_performing_pereira_1-dataset=ud_sentencez_ds_max_100_edited_selected_textNoPeriod-activation-bench=None-ave=False'
GRAND_PIPE_FILE="${DATA_DIR}/PIPE_${extract_id}.txt"
TEXT_FILE="${DATA_DIR}/sentences_${extract_id}.txt"
rm -f $GRAND_PIPE_FILE
touch $GRAND_PIPE_FILE
model="tts_models/en/ljspeech/glow-tts"
# replace / in model with _
model_name=${model//\//_}
printf "%s\t%s\t%s\t%s\t%s\n" "row" "id" "sentence" "model" "file"  >> $GRAND_PIPE_FILE
while read string; do
      framename=$(printf '%02d' $i)
      possible_file="${DATA_DIR}/wav_${extract_id}/${model_name}_sentence_${framename}.wav"
      if [ -f "$possible_file" ]
      then
        true
      else
        LINE_COUNT=$(expr ${LINE_COUNT} + 1)
        printf "%d\t%s\t%s\t%s\t%s\n" "$LINE_COUNT" "$framename" "$string" "$model" "$possible_file"  >> $GRAND_PIPE_FILE
  fi
    i=$(expr ${i} + 1)
done < $TEXT_FILE

echo $LINE_COUNT
run_val=0
if [ "$LINE_COUNT" -gt "$run_val" ]; then
  echo "running  ${LINE_COUNT} "
   #nohup /cm/shared/admin/bin/submit-many-jobs 3000 950 600 350 txt_2_spch_coca_50K.sh $GRAND_PIPE_FILE
   nohup /cm/shared/admin/bin/submit-many-jobs 10 5 3 2 txt_2_spch.sh $GRAND_PIPE_FILE &
   #nohup /cm/shared/admin/bin/submit-many-jobs $LINE_COUNT  250 300 50 extract_model_act_for_group.sh $GRAND_PIPE_FILE
  else
    echo $LINE_COUNT
fi
