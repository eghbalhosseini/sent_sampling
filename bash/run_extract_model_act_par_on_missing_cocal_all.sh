#!/bin/bash

DATA_DIR=/om/user/ehoseini/MyData/sent_sampling/
i=0
LINE_COUNT=0
GRAND_PIPE_FILE="${DATA_DIR}/Grand_extraction_pipe_list.csv"
rm -f $GRAND_PIPE_FILE
touch $GRAND_PIPE_FILE

#models="gpt2-xl xlnet-large-cased bert-large-uncased-whole-word-masking xlm-mlm-en-2048 albert-xxlarge-v2 ctrl"
#layers="49 25 25 13 13 49" # adding the embedding layer so its layer plus 1
models="gpt2-xl"
layers="49" # adding the embedding layer so its layer plus 1
#
model_arr=($models)
layer_arr=($layers)
#
for dataset in  coca_preprocessed_all_clean_100K_sample_1 ; do
  for group_ids in `seq 0 1 199` ; do
      for idx_model in seq 0 1 ${#layer_arr[@]} ; do
          model_list[$i]="${model_arr[$idx_model]}"
          layer_list[$i]="${layer_arr[$idx_model]}"
          dataset_list[$i]="$dataset"
          group_id_list[$i]=$group_ids
          i=$i+1
          look_up_pattern="${dataset}_${model}_*_group_${group_ids}*.pkl"
          folder_to_look=${DATA_DIR}/${model}
          #lines=$(find $folder_to_look -name "${dataset}_${model}_*_group_${group_ids}*.pkl" | wc -l)
          lines=$(find $folder_to_look -name $look_up_pattern | wc -l)
          if [ "$lines" = "${layer_arr[$idx_model]}" ]; then
              echo "${lines} vs ${layer_arr[$idx_model]}  - ${dataset}_${model}_group_${group_ids} dosent exists, adding it \n"
              LINE_COUNT=$(expr ${LINE_COUNT} + 1)
              printf "%d,%s,%s,%d\n" "$LINE_COUNT" "$model" "$dataset" "$group_ids" >> $GRAND_PIPE_FILE
          else
              pass
          fi
      done
      done
done

