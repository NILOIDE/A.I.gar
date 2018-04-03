GPUS = 1

# Experience replay:
MEMORY_CAPACITY = 75000
MEMORIES_PER_UPDATE = 40 # Must be divisible by 4 atm due to experience replay
REPLAY_AFTER_X_STEPS = 0

# Q-learning
USE_POLICY_NETWORK = False
USE_TARGET = True # Otherwise td-error is used in value network
EXP_REPLAY_ENABLED = True
GRID_VIEW_ENABLED = True
TARGET_NETWORK_STEPS = 10000
TARGET_NETWORK_MAX_STEPS = 10000 # 2000 performs worse than 5000. 20000 was a bit better than 5000
DISCOUNT = 0.9 # 0.9 seems best so far. Better than 0.995 and 0.9999 . 0.5 and below performs much worse
# Higher discount seems to lead to much more stable learning, less variance

Exploration = True
EPSILON = 0.0 if Exploration else 0 # Exploration rate. 0 == No Exploration
# epsilon set to 0 performs best so far... (keep in mind that it declines from 1 to 0 throughout the non-gui training
EPSILON_DECREASE_RATE = 1
FRAME_SKIP_RATE = 9
GRID_SQUARES_PER_FOV = 12 #Grid size of 9 performs really good with 100 hidden neurons (90 better on average than 12 and 15)!
NUM_OF_GRIDS = 3

#ANN
NEURON_TYPE = "MLP" #"LSTM" lstm does not work yet
ALPHA = 0.00025 #Learning rate
OPTIMIZER = "Adam"
ACTIVATION_FUNC_HIDDEN = 'relu' #'tanh' seems to perform worse than sigmoid
ACTIVATION_FUNC_OUTPUT = 'linear'

#Layer neurons
STATE_REPR_LEN = GRID_SQUARES_PER_FOV * GRID_SQUARES_PER_FOV * NUM_OF_GRIDS + 2
HIDDEN_LAYER_1 = 100
HIDDEN_LAYER_2 = 100
HIDDEN_LAYER_3 = 100
# The rule for hidden neurons seems to be: the more the better 500 hidden neurons for all 3 layers performed best so far.
