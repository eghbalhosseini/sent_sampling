#!/bin/bash
DATA_DIR=/nese/mit/group/evlab/u/ehoseini/MyData/sent_sampling/
RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching/
i=0
LINE_COUNT=0
GRAND_PIPE_FILE="${DATA_DIR}/Grand_cleanup_pipe_list.csv"
rm -f $GRAND_PIPE_FILE
touch $GRAND_PIPE_FILE
models="albert-xxlarge-v2"
layers="12" # adding the embedding layer so its layer plus 1
#models="gpt2-xl"
#layers="49" # adding the embedding layer so its layer plus 1
#
model_arr=($models)
layer_arr=($layers)
splits=100
len=${#layer_arr[@]}
#ud_sentences_U01_SET1_paraphrase_albert-xxlarge-v2_layer_4_activation_group_2.pkl

#coca_preprocessed_all_clean_no_dup_100K_sample_1_textNoPeriod_gpt2-xl_layer_34_activation_group_113.pkl
printf "%s,%s,%s,%s,%s,%s\n" "row" "model" "dataset" "stim_type" "splits" "group_id"  >> $GRAND_PIPE_FILE
for dataset in  ud_sentences_U01_SET1_paraphrase ; do
  for (( idx_model=0; idx_model<$len; idx_model++ )) ; do
    for stim_type in "" ; do
      # make group_id go from 0 to splits -1
      # print idx of model
      for (( group_ids=0; group_ids<$splits; group_ids++ )) ; do
          model="${model_arr[$idx_model]}"
          model_list[$i]="${model_arr[$idx_model]}"
          layer_list[$i]="${layer_arr[$idx_model]}"
          dataset_list[$i]="$dataset"
          stim_type_list[$i]="$stim_type"
          group_id_list[$i]=$group_ids
          i=$i+1
          # find pattern that has the layer number
          #look_up_pattern="${dataset}_${stim_type}_${model}_layer_*_activation_group_${group_ids}.pkl"
          look_up_pattern="${dataset}${stim_type}_${model}_layer_*_activation_group_${group_ids}.pkl"
          #echo $look_up_pattern
          folder_to_look=${DATA_DIR}/${model}
          echo $look_up_pattern
          #lines=$(find $folder_to_look -name "${dataset}_${model}_*_group_${group_ids}*.pkl" | wc -l)
          lines=$(find $folder_to_look -name $look_up_pattern | wc -l)
          echo $lines
          if [ "$lines" != "${layer_arr[$idx_model]}" ]; then
              echo "${lines} vs ${layer_arr[$idx_model]}  - ${dataset}${stim_type}_${model}_group_${group_ids} dosent exists, adding it \n"
              LINE_COUNT=$(expr ${LINE_COUNT} + 1)
              printf "%d,%s,%s,%s,%s,%d\n" "$LINE_COUNT" "$model" "$dataset" "$stim_type" "$splits" "$group_ids" >> $GRAND_PIPE_FILE

          fi
        done
      done
    done
done


#echo $LINE_COUNT
#run_val=0
#if [ "$LINE_COUNT" -gt "$run_val" ]; then
#  echo "running  ${LINE_COUNT} jobs"
#  if [ "$LINE_COUNT" -lt 200 ] ; then
#    echo "less than 100 jobs:  ${LINE_COUNT} jobs"
#    nohup /cm/shared/admin/bin/submit-many-jobs $LINE_COUNT "$LINE_COUNT" "$LINE_COUNT" 0 extract_model_act_for_group.sh  $GRAND_PIPE_FILE
#  else
#     echo "more than 100 jobs:  ${LINE_COUNT} jobs"
#    nohup /cm/shared/admin/bin/submit-many-jobs $LINE_COUNT 180 200 20 extract_model_act_for_group.sh  $GRAND_PIPE_FILE
#  fi
#  else
#    echo $LINE_COUNT
#fi