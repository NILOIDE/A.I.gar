ALGORITHM = "CACLA" # "Q-learning" or "CACLA" or "DPG" so far

Default = False
VERY_DEBUG = False
VIEW_ENABLED = True
GUI_COLLECTOR_SET = {1}

GAME_NAME = "Agar.io"
# Game
PELLET_SPAWN = True
VIRUS_SPAWN = False
RESET_LIMIT = 20000
EXPORT_POINT_AVERAGING = 500
NUM_GREEDY_BOTS = 0
NUM_NN_BOTS = 1
NUM_RANDOM_BOTS = 0
ENABLE_GREEDY_SPLIT = False

# Experience replay:
EXP_REPLAY_ENABLED = False
PRIORITIZED_EXP_REPLAY_ENABLED = False if EXP_REPLAY_ENABLED else False
MEMORY_CAPACITY = 40000
MEMORY_BATCH_LEN = 32
MEMORY_ALPHA = 0.6
MEMORY_BETA = 0.4

# General Training:
ENABLE_TRAINING = True
ENABLE_TESTING = True
DUR_TRAIN_TEST_NUM = 5
TRAIN_PERCENT_TEST_INTERVAL = 5
FINAL_TEST_NUM = 10
FRAME_SKIP_RATE = 7
MAX_TRAINING_STEPS = 500000
CURRENT_STEP = 0
MAX_SIMULATION_STEPS = MAX_TRAINING_STEPS * (FRAME_SKIP_RATE + 1)
ENABLE_SPLIT = False
ENABLE_EJECT = False
# Q-Learning
ENABLE_QV = False # Enable discrete QV Learning instead of Q Learning
OPTIMIZER = "Adam" #SGD has much worse performance
NESTEROV = 0
AMSGRAD = False
GRADIENT_CLIP_NORM = 0
GRADIENT_CLIP = 0
DROPOUT = 0
MAXNORM = 3
BATCHNORM = False
# General RL:
MASS_AS_REWARD = False
DISCOUNT = 0.9
END_DISCOUNT = 0#0.85 # set to 0 to disable
DISCOUNT_INCREASE_FACTOR = (1 - END_DISCOUNT) ** (1 / MAX_TRAINING_STEPS) if MAX_TRAINING_STEPS != 0 else 0
# Noise and Exploration:
NOISE_TYPE = "Gaussian"  # "Gaussian" / "Orn-Uhl"
GAUSSIAN_NOISE = 1.0 # Initial noise
NOISE_AT_HALF_TRAINING = 0.02
NOISE_DECAY = NOISE_AT_HALF_TRAINING ** (1 / (MAX_TRAINING_STEPS / 2)) if MAX_TRAINING_STEPS != 0 else 0
ORN_UHL_THETA = 0.15
ORN_UHL_DT = 0.01
ORN_UHL_MU = 0
Exploration = True
EPSILON = 1 if Exploration else 0 # Exploration rate. 0 == No Exploration
EXPLORATION_STRATEGY = "e-Greedy" # "Boltzmann" or "e-Greedy"
TEMPERATURE = 7
TEMPERATURE_AT_END_TRAINING = 0.0025
TEMPERATURE_DECAY = TEMPERATURE_AT_END_TRAINING ** (1 / MAX_TRAINING_STEPS) if MAX_TRAINING_STEPS != 0 else 0
#Reward function:
REWARD_TERM = 0#-200000
REWARD_SCALE = 2
DEATH_TERM = -40
DEATH_FACTOR = 1.5

# State representation parameters:
if GAME_NAME == "Agar.io":
    MULTIPLE_BOTS_PRESENT = True if NUM_GREEDY_BOTS + NUM_NN_BOTS + NUM_RANDOM_BOTS> 1 else False
    NORMALIZE_GRID_BY_MAX_MASS = False
    PELLET_GRID = True
    SELF_GRID = ENABLE_SPLIT or VIRUS_SPAWN
    SELF_GRID_LF = ENABLE_SPLIT
    SELF_GRID_SLF = False
    WALL_GRID = MULTIPLE_BOTS_PRESENT
    VIRUS_GRID = VIRUS_SPAWN
    ENEMY_GRID = MULTIPLE_BOTS_PRESENT
    ENEMY_GRID_LF = ENABLE_SPLIT
    ENEMY_GRID_SLF = False
    SIZE_GRID = False
    ALL_PLAYER_GRID = False
    if ALL_PLAYER_GRID:
        SELF_GRID = False
        ENEMY_GRID = False
    USE_FOVSIZE = True
    USE_LAST_FOVSIZE = ENABLE_SPLIT
    USE_TOTALMASS = True
    USE_LAST_ACTION = ENABLE_SPLIT
    USE_SECOND_LAST_ACTION = False
    GRID_SQUARES_PER_FOV = 11 #11 is pretty good so far.
    NUM_OF_GRIDS = PELLET_GRID + SELF_GRID + WALL_GRID + VIRUS_GRID + ENEMY_GRID \
                   + SIZE_GRID + SELF_GRID_LF + SELF_GRID_SLF \
                   + ENEMY_GRID_LF + ENEMY_GRID_SLF + ALL_PLAYER_GRID
    EXTRA_INPUT = USE_FOVSIZE + USE_TOTALMASS + USE_LAST_ACTION * 4 + USE_SECOND_LAST_ACTION * 4 + USE_LAST_FOVSIZE
    STATE_REPR_LEN = GRID_SQUARES_PER_FOV * GRID_SQUARES_PER_FOV * NUM_OF_GRIDS + EXTRA_INPUT
else:
    EXTRA_INPUT = 0
    STATE_REPR_LEN = 0

INITIALIZER = "glorot_uniform" #"Default" or "glorot_uniform" or "glorot_normal"


# Q-learning
NEURON_TYPE = "MLP"
Q_LAYERS = (100,100)
ALPHA = 0.001 * (1 + DROPOUT * 9)
SQUARE_ACTIONS = True
NUM_ACTIONS = 25
ACTIVATION_FUNC_HIDDEN = 'relu'
ELU_ALPHA = 1 # TODO: only works for Q-learning so far. Test if it is useful, if so implement for others too
ACTIVATION_FUNC_OUTPUT = 'linear'
GRID_VIEW_ENABLED = True
TARGET_NETWORK_STEPS = 1500
USE_ACTION_AS_INPUT = False
TD = 0
Q_WEIGHT_DECAY    = 0#0.001 #0.001 L2 weight decay parameter. Set to 0 to disable


# Actor-critic:
ACTOR_IS = False # Enable importance sampling of priotized exp replay also for the actor
AC_ACTOR_TDE = 0 # How much to use the TDE of the Actor to prioritize experience in the replay buffer
AC_DELAY_ACTOR_TRAINING = 0
AC_ACTOR_TRAINING_START = AC_DELAY_ACTOR_TRAINING * MAX_TRAINING_STEPS
AC_NOISE_AT_HALF = 0.05
AC_NOISE_DECAY = AC_NOISE_AT_HALF ** (1 / (MAX_TRAINING_STEPS / 2)) if MAX_TRAINING_STEPS != 0 else 0
#ACTOR_CRITIC_TYPE = "CACLA" # "Standard"/"CACLA" / "DPG". Standard multiplies gradient by tdE, CACLA only updates once for positive tdE
SOFT_TARGET_UPDATES = True
POLICY_OUTPUT_ACTIVATION_FUNC = "sigmoid"
ACTOR_REPLAY_ENABLED = True
OPTIMIZER_POLICY = "Adam"
ACTIVATION_FUNC_HIDDEN_POLICY = "relu"

# CACLA:
CACLA_CRITIC_LAYERS         = (100,100)
CACLA_CRITIC_ALPHA          = 0.000075
CACLA_ACTOR_LAYERS          = (100,100)
CACLA_ACTOR_ALPHA           = 0.0005
CACLA_TAU                   = 0.02
CACLA_UPDATE_ON_NEGATIVE_TD = False
CACLA_CRITIC_WEIGHT_DECAY   = 0     #0.001 #0.001 L2 weight decay parameter. Set to 0 to disable
CACLA_OFF_POLICY_CORR       = 0
CACLA_OFF_POLICY_CORR_SIGN  = False

CACLA_VAR_ENABLED           = True
CACLA_VAR_START             = 1
CACLA_VAR_BETA              = 0.001

# Sampled Policy Gradient (SPG):
OCACLA_ENABLED               = True if ALGORITHM == "SPG" else False
OCACLA_EXPL_SAMPLES          = 5
OCACLA_ONLINE_SAMPLES        = 0
OCACLA_ONLINE_SAMPLING_NOISE = 0
OCACLA_MOVING_GAUSSIAN       = True
OCACLA_REPLACE_TRANSITIONS   = False
OCACLA_START_NOISE           = 1.0
OCACLA_NOISE                 = OCACLA_START_NOISE
OCACLA_END_NOISE             = 0.0004
OCACLA_NOISE_DECAY           = OCACLA_END_NOISE ** (OCACLA_START_NOISE / MAX_TRAINING_STEPS) if MAX_TRAINING_STEPS != 0 else 0

# Deterministic Policy Gradient (DPG):
DPG_TAU                    = 0.001 # How quickly the weights of the target networks are updated
DPG_CRITIC_LAYERS          = (100,100)
DPG_CRITIC_ALPHA           = 0.0005
DPG_CRITIC_FUNC            = "relu"
DPG_CRITIC_WEIGHT_DECAY    = 0 #0.001 L2 weight decay parameter. Set to 0 to disable
DPG_ACTOR_LAYERS           = (100,100)
DPG_ACTOR_ALPHA            = 0.0001
DPG_ACTOR_FUNC             = "relu"
DPG_Q_VAL_INCREASE         = 2
DPG_FEED_ACTION_IN_LAYER   = 1
DPG_USE_DPG_ACTOR_TRAINING = True
DPG_USE_TARGET_MODELS      = True
DPG_USE_CACLA              = False
DPG_CACLA_ALTERNATION      = 0 #fraction of training time in which cacla is used instead of dpg
DPG_CACLA_INV_ALTERNATION  = 0 #fraction of training time after which cacla is used instead of dpg
DPG_CACLA_STEPS            = DPG_CACLA_ALTERNATION * MAX_TRAINING_STEPS
DPG_DPG_STEPS              = DPG_CACLA_INV_ALTERNATION * MAX_TRAINING_STEPS
DPG_ACTOR_OPTIMIZER        = "Adam"
DPG_ACTOR_NESTEROV         = 0

# LSTM
ACTIVATION_FUNC_LSTM = "sigmoid"
UPDATE_LSTM_MOVE_NETWORK = 1
TRACE_MIN = 1 # The minimum amount of traces that are not trained on, as they have insufficient hidden state info
MEMORY_TRACE_LEN = 15 # The length of memory traces retrieved via exp replay

# CNN
CNN_REPR = False

# Handcraft representation
CNN_USE_L1 = True
CNN_L1 = (8, 4, 32)
CNN_INPUT_DIM_1 = 42

CNN_USE_L2 = True
CNN_L2 = (4, 2, 64)
CNN_INPUT_DIM_2 = 84

CNN_USE_L3 = False
CNN_L3 = (3, 1, 64)
CNN_INPUT_DIM_3 = 42

# Pixel representation
CNN_P_REPR = True if CNN_REPR else False
CNN_P_RGB = False if CNN_P_REPR else False
CNN_P_INCEPTION = False
CNN_LAST_GRID = False

# Multiprocessing:
NUM_COLLECTORS = 1
NUM_EXPS_BEFORE_TRAIN = 64#int(MEMORY_CAPACITY/40)
ENABLE_GPU = False
NUM_GPUS = 1 if ENABLE_GPU == True else 0
TESTING_WORKER_POOL = DUR_TRAIN_TEST_NUM
GATHER_EXP = True
ONE_BEHIND_TRAINING = False
