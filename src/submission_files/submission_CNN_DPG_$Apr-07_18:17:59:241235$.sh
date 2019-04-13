#!/bin/bash
#SBATCH --time=2-23:59:59
#SBATCH --mem=120000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nilstoltanso@gmail.com
#SBATCH --output=name%j.out
export OMP_NUM_THREADS=1
module load matplotlib/2.1.2-foss-2018a-Python-3.6.4
module load TensorFlow/1.6.0-foss-2018a-Python-3.6.4
module load h5py/2.7.1-foss-2018a-Python-3.6.4
python -O ./aigar.py <<EOF
0
1
ALGORITHM="DPG"&CNN_REPR=True&MAX_TRAINING_STEPS=300000
$Apr-07_18:17:59:241235$
0
0
EOF
