#!/bin/bash
#SBATCH -n 4
#SBATCH -p gpu_short

#Prepare python environment
export $PYTHONPATH=$HOME/pythonpackages/lib/python:$PYTHONPATH
module load python/2.7.9

#Go to project folder
cd $HOME/luna16/src

#Go!
python learn.py
