#!/bin/bash
#SBATCH --time=2-23:00:00
#SBATCH --mem=20000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nilstoltanso@gmail.com
#SBATCH --cpus-per-task=2
#SBATCH --output=CNN_test_%j.out
export OMP_NUM_THREADS=1
module load matplotlib/2.1.2-foss-2018a-Python-3.6.4
module load TensorFlow/1.6.0-foss-2018a-Python-3.6.4
module load h5py/2.7.1-foss-2018a-Python-3.6.4
python3 -O ./aigar.py > out.txt<<EOF
0
0
1
CNN_REPR
True
1
MAX_TRAINING_STEPS
100000
0
0
EOF
