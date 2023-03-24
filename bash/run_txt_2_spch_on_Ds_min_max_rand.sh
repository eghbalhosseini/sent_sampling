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
model="tts_models/en/ek1/tacotron2"
# replace / in model with _

printf "%s\t%s\t%s\t%s\t%s\n" "row" "id" "sentence" "model" "file"  >> $GRAND_PIPE_FILE

for model in tts_models/bg/cv/vits tts_models/en/ek1/tacotron2  tts_models/en/ljspeech/tacotron2-DDC tts_models/en/ljspeech/tacotron2-DDC_ph \
 tts_models/en/ljspeech/glow-tts tts_models/en/ljspeech/speedy-speech tts_models/en/ljspeech/tacotron2-DCA tts_models/en/ljspeech/vits \
  tts_models/en/ljspeech/vits--neon tts_models/en/ljspeech/fast_pitch tts_models/en/ljspeech/overflow tts_models/en/ljspeech/neural_hmm tts_models/en/vctk/vits \
  tts_models/en/vctk/fast_pitch tts_models/en/sam/tacotron-DDC tts_models/en/blizzard2013/capacitron-t2-c50 tts_models/en/blizzard2013/capacitron-t2-c150_v2 ; do
  model_name=${model//\//_}
  while read string; do
      framename=$(printf '%02d' $i)
      possible_file="${DATA_DIR}/wav_${extract_id}/${model_name}_sentence_${framename}.wav"
      # if parent directory doesnt exist create it
      if [ ! -d "${DATA_DIR}/wav_${extract_id}" ]; then
        mkdir -p "${DATA_DIR}/wav_${extract_id}"
      fi
      if [ -f "$possible_file" ]
      then
        true
      else
        LINE_COUNT=$(expr ${LINE_COUNT} + 1)
        printf "%d\t%s\t%s\t%s\t%s\n" "$LINE_COUNT" "$framename" "$string" "$model" "$possible_file"  >> $GRAND_PIPE_FILE
  fi
    i=$(expr ${i} + 1)
done < $TEXT_FILE
done
echo $LINE_COUNT
run_val=0
if [ "$LINE_COUNT" -gt "$run_val" ]; then
  echo "running  ${LINE_COUNT} "
   #nohup /cm/shared/admin/bin/submit-many-jobs 3000 950 600 350 txt_2_spch_coca_50K.sh $GRAND_PIPE_FILE
#   nohup /cm/shared/admin/bin/submit-many-jobs 10 5 3 2 txt_2_spch.sh $GRAND_PIPE_FILE &
   #nohup /cm/shared/admin/bin/submit-many-jobs $LINE_COUNT  250 300 50 extract_model_act_for_group.sh $GRAND_PIPE_FILE
  else
    echo $LINE_COUNT
fi
