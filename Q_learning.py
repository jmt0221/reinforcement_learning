import numpy as np
import matplotlib.pyplot as plt
from grid_world import grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from mc_exploring_starts import max_dict
from td0_prediction import random_action

GAMMA = 0.9
ALPHA = 0.1
ACTIONS = ("U", "D", "L", "R")