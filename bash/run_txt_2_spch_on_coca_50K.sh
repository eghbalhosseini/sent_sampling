#!/bin/bash
DATA_DIR=/om/user/ehoseini/MyData/sent_sampling/
i=0
i=$(expr ${i} + 0)
LINE_COUNT=0
GRAND_PIPE_FILE="${DATA_DIR}/Grand_coca_txt_2_spch_pipe_list.txt"
rm -f $GRAND_PIPE_FILE
touch $GRAND_PIPE_FILE
printf "%s\t%s\t%s\t%s\n" "row" "id" "sentence" "file"  >> $GRAND_PIPE_FILE
while read string; do
      framename=$(printf '%05d' $i)
      possible_file="/om/user/ehoseini/MyData/sent_sampling/coca_spok_filter_punct_50K/wav//coca_spok_filter_punct_50K_${framename}.wav"
      if [ -f "$possible_file" ]
      then
        true
      else
        LINE_COUNT=$(expr ${LINE_COUNT} + 1)
        printf "%d\t%s\t%s\t%s\n" "$LINE_COUNT" "$framename" "$string" "$possible_file"  >> $GRAND_PIPE_FILE
  fi
    i=$(expr ${i} + 1)
done < /om/user/ehoseini/MyData/sent_sampling/coca_spok_filter_punct_50K/text/coca_spok_filter_punct_50K_all.txt

echo $LINE_COUNT
run_val=0
if [ "$LINE_COUNT" -gt "$run_val" ]; then
  echo "running  ${LINE_COUNT} "
   #nohup /cm/shared/admin/bin/submit-many-jobs 3000 950 600 350 txt_2_spch_coca_50K.sh $GRAND_PIPE_FILE
   nohup /cm/shared/admin/bin/submit-many-jobs 10 5 3 2 txt_2_spch_coca_50K.sh $GRAND_PIPE_FILE
   #nohup /cm/shared/admin/bin/submit-many-jobs $LINE_COUNT  250 300 50 extract_model_act_for_group.sh $GRAND_PIPE_FILE
  else
    echo $LINE_COUNT
fi
