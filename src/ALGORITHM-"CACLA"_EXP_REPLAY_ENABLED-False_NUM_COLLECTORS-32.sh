#!/bin/bash
#SBATCH --time=0-23:00:00
#SBATCH --mem=20000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nilstoltanso@gmail.com
#SBATCH --output=ALGORITHM-"CACLA"_EXP_REPLAY_ENABLED-False_NUM_COLLECTORS-32_%j.out
module load matplotlib/2.1.2-foss-2018a-Python-3.6.4
module load TensorFlow/1.6.0-foss-2018a-Python-3.6.4
module load h5py/2.7.1-foss-2018a-Python-3.6.4
python -O ./aigar.py <<EOF
0
0
1
ALGORITHM
"CACLA"
1
EXP_REPLAY_ENABLED
False
1
NUM_COLLECTORS
32
0
0
EOF
