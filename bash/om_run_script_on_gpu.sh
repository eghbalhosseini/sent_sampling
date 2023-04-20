#!/bin/bash

#SBATCH --job-name=opt
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --gres=gpu:A100:2
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH -p evlab

PYTHON_SCR=$1
echo "running :${PYTHON_SCR}"
. /om/weka/evlab/ehoseini/.bashrc
conda activate neural_nlp_2022
echo $(which python)
python $PYTHON_SCR