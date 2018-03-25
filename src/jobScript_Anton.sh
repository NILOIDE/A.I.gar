#!/bin/bash
#SBATCH --time=1-12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=antonwiehe@gmail.com
module load Python/3.5.2-foss-2016a
module load matplotlib/1.5.3-foss-2016a-Python-3.5.2
module load tensorflow/1.5.0-foss-2016a-Python-3.5.2-CUDA-9.1.85
module load h5py/2.5.0-foss-2016a-Python-3.5.1-HDF5-1.8.16 
python -O ./aigar.py < trainNoGui.txt