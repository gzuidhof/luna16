#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -p normal
#SBATCH -c 12

#Prepare python environment
export PYTHONPATH=$HOME/pythonpackages/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/pythonpackages/lib/python2.7/site-packages:$PYTHONPATH
module load python/2.7.9

#Go to project folder
cd $HOME/luna16/src/data_processing

#Go!!!
echo "starting python"
srun -u python create_xy_xz_yz_CARTESIUS.py 7 candidates_v2.csv
