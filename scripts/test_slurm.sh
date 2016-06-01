#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -n 4
#SBATCH -p gpu_short

#Prepare python environment
export $PYTHONPATH=$HOME/pythonpackages/lib/python:$PYTHONPATH
module load python/2.7.9

#Go to project folder
cd $HOME/luna16/src


#Go!!!
THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1' python learn.py
