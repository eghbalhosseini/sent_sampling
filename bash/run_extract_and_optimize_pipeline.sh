#!/bin/sh

i=0
for optim_method in coordinate_ascent ; do
  for n_iter in 500 1000 ; do
    for N_s in  200 ; do
      for init in 3 ; do
        optim_id="${optim_method}-obj=D_s-n_iter=${n_iter}-n_samples=${N_s}-n_init=${init}"
        optim_list[$i]="$optim_id"
        echo "optim: ${optim_id}"
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
      echo "extract: ${extract_id}"
      i=$i+1
      done
    done
  done
done

i=0

for extract in ${extract_list[@]} ; do
  for optim in ${optim_list[@]} ; do
    echo "extract ${extract}"
    echo "optim ${optim}"
    extract_pool[$i]="$extract"
    optim_pool[$i]="$optim"
    i=$i+1
  done
done