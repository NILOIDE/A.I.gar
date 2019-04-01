#!/bin/bash
#SBATCH --time=1-12:00:00
#SBATCH --mem=20000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nilstoltanso@gmail.com
#SBATCH --cpus-per-task=4
#SBATCH --output=test_32_4_%j.out
export OMP_NUM_THREADS=4
module load matplotlib/2.1.2-foss-2018a-Python-3.6.4
module load TensorFlow/1.6.0-foss-2018a-Python-3.6.4
module load h5py/2.7.1-foss-2018a-Python-3.6.4
python3 -O ./aigar.py > out_thread_32_4.txt<<EOF
0
0
1
EXP_REPLAY_ENABLED
False
1
NUM_COLLECTORS
32
1
MAX_TRAINING_STEPS
100000
0
0
EOF
