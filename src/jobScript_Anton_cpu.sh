#!/bin/bash
#SBATCH --time=0-13:00:00
#SBATCH --mem=10000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=antonwiehe@gmail.com
module load Python/3.5.2-foss-2016a
module load matplotlib/1.5.3-foss-2016a-Python-3.5.2
module load tensorflow/1.3.1-foss-2016a-Python-3.5.2
module load h5py/2.5.0-foss-2016a-Python-3.5.1-HDF5-1.8.16 
python -O ./aigar.py < trainNoGui.txt