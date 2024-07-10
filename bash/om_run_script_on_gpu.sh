#!/bin/bash
#SBATCH --job-name=opt
#SBATCH --time=36:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=high-capacity
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu

PYTHON_SCR=$1
echo "running :${PYTHON_SCR}"
. /om/weka/evlab/ehoseini/.bashrc
conda activate neural_nlp_2022
echo $(which python)
python $PYTHON_SCR



###SBATCH --gres=gpu:a100:2