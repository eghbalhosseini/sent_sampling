#!/bin/sh

i=0
for optim_method in coordinate_ascent ; do
  for n_iter in 500 1000 ; do
    for N_s in  200 ; do
      for init in 3 ; do
        optim_id="${optim_method}-obj=D_s-n_iter=${n_iter}-n_samples=${N_s}-n_init=${init}"
        optim_list[$i]="$optim_id"
        i=$i+1
      done
    done
  done
done

i=0
extract_list="network_act brain_resp"
bench_list="None Fedorenko2016v3-encoding-weights_v2"
extract_list=($extract_list)
bench_list=($bench_list)


for set in set_3 ; do
  for idx in 0 1 ; do
    for ave in False ; do
    for dataset in ud_sentences_filter ; do
      extract_id="group=${set}-dataset=${dataset}-${extract_list[$idx]}-bench=${bench_list[$idx]}-ave=${ave}"
      extract_list[$i]="$extract_id"
      i=$i+1
      done
    done
  done
done

run=0

for extract in ${extract_list[@]} ; do
  for optim in ${optim_list[@]} ; do
    extract_pool[$run]="$extract"
    optim_pool[$run]="$optim"
    run=$run+1
  done
done

sbatch --array=0-$(expr ${run} - 1) --mem 64G -p normal extract_and_optimize.sh ${extract_pool[@]} ${optim_pool[@]}