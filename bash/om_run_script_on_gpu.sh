#!/bin/bash
#SBATCH --job-name=opt
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
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