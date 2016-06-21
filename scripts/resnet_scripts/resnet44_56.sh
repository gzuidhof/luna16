#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH -p gpu
#SBATCH -c 12

#Prepare python environment
export PYTHONPATH=$HOME/pythonpackages/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/pythonpackages/lib/python2.7/site-packages:$PYTHONPATH
module load python/2.7.9
module load cuda
module load cudnn

#Go to project folder
cd $HOME/luna16/src/deep

#Go!!!

echo "starting python"
export THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1'
srun -u python train.py ../../config/resnet44_56.ini
