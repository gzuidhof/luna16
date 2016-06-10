#!/bin/bash
#SBATCH -t 1:00:00

#SBATCH -p gpu

#Prepare python environment
export PYTHONPATH=$HOME/pythonpackages/lib/python:$PYTHONPATH
module load python/2.7.9
module load cuda
module load cudnn

#Go to project folder
cd $HOME/luna16/src


#Go!!!

echo "starting python"
#export THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1' 
srun -u python learn.py
