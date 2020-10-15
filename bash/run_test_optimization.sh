#!/bin/bash

#SBATCH --job-name=sent_sampling
#SBATCH --array=0
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=120G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_cachingc
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_nlp.simg python /om/user/ehoseini/sent_sampling/test_optimization.py
