#!/bin/bash
#
#SBATCH -n 1 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 2:00:00

EXTRACT_POOL=$1
OPTIM_POOL=$2

let FILE_LINE=$SLURM_ARRAY_TASK_ID
echo "Line ${FILE_LINE}"
extract=${EXTRACT_POOL[$FILE_LINE]}
optim=${OPTIM_POOL[$FILE_LINE]}
echo "extraction : ${extract}"
echo "optimization : ${optim}"