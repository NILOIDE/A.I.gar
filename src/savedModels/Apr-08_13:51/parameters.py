import math

# General Parameters
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 900
MAXBOTS = 1000
MAXHUMANPLAYERS = 3

# Simulation Parameters
FPS = 30
GAME_SPEED = 1  # 1sec/1sec
SPEED_MODIFIER = GAME_SPEED / FPS

# Field Parameters
HASH_BUCKET_SIZE = 20
SIZE_INCREASE_PER_PLAYER = 130
START_MASS = 10
START_RADIUS = math.sqrt(START_MASS / math.pi)
MAX_COLLECTIBLE_DENSITY = 0.015  # per unit area
MAX_VIRUS_DENSITY = 0.0002
VIRUS_BASE_SIZE = 100
VIRUS_BASE_RADIUS = math.sqrt(VIRUS_BASE_SIZE / math.pi)
VIRUS_EXPLOSION_BASE_MASS = 15
EJECTEDBLOB_BASE_MASS = 18
#EJECTEDBLOB_BASE_MOMENTUM = 6

# Cell Parameters
MAX_MASS_SINGLE_CELL = 22500
BASE_MERGE_TIME = 20
CELL_MOVE_SPEED = 90 * SPEED_MODIFIER #units/sec
#CELL_SPLIT_SPEED = 30 * SPEED_MODIFIER #units/sec
CELL_MASS_DECAY_RATE = 1 - (0.01 * SPEED_MODIFIER) #default: 1- (0.01 * SPEED_MODIFIER)

# State Representation Parameters
GRID_SIDE_LENGTH = 3

# Player Parameters:
#MOMENTUM_PROPORTION_TO_MASS = 0.003
#MOMENTUM_BASE = 6
