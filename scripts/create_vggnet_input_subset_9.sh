#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -p normal

#Prepare python environment
export PYTHONPATH=$HOME/pythonpackages/lib/python:$PYTHONPATH
module load python/2.7.9

#Go to project folder
cd $HOME/luna16/src/data_processing

#Go!!!
echo "starting python"
srun -u python create_xy_xz_yz_CARTESIUS.py 9 candidates_subset89.csv
