#!/bin/bash

DATA_DIR=/om/user/ehoseini/MyData/sent_sampling/
i=0
LINE_COUNT=0
GRAND_PIPE_FILE="${DATA_DIR}/Grand_extraction_pipe_list.csv"
rm -f $GRAND_PIPE_FILE
touch $GRAND_PIPE_FILE
for dataset in  coca_preprocessed_all_clean_100K_sample_1 ; do
  for group_ids in `seq 0 1 199` ; do
      for model in gpt2-xl ; do
          model_list[$i]="$model"
          dataset_list[$i]="$dataset"
          group_id_list[$i]=$group_ids
          i=$i+1

          look_up_pattern="${dataset}_${model}_*_group_${group_ids}*"
          folder_to_look=${DATA_DIR}/${model}
          printf "finding pattern ${look_up_pattern} \n"
          #lines=$(find $folder_to_look -name "${dataset}_${model}_*_group_${group_ids}*.pkl" | wc -l)
          lines=$(find $folder_to_look -name $look_up_pattern | wc -l)
          if [ $lines -eq 0 ]; then
              echo "${dataset}_${model}_group_${group_ids} dosent exists, adding it \n"
              LINE_COUNT=$(expr ${LINE_COUNT} + 1)
              printf "%d,%s,%s,%d\n" "$LINE_COUNT" "$model" "$dataset" "$group_ids" >> $GRAND_PIPE_FILE
          else
              printf "found  ${lines} files \n"
          fi
      done
      done
done

