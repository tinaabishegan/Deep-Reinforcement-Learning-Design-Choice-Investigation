# config.py

import torch
import torch.nn as nn

def init_weights_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.1)
        nn.init.constant_(m.bias, 0)

hyperparameters = {
    'CartPole-v1': {
        'BATCH_SIZE': 128,
        'LEARNING_RATE': 0.0005,
        'GAMMA': 0.98,
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.01,
        'EPSILON_DECAY': 0.99,
        'MEMORY_CAPACITY': 5000,
        'TARGET_UPDATE_EVERY': 50,  # Hard target update interval
        'NUM_EPISODES': 3000,
        'MAX_T': 10000,
        'SOLVED_SCORE': 475,
        'SOLVED_LENGTH': 100,
        'HIDDEN_LAYERS': [128, 64],  # Specific to CartPole
        'ACTIVATION_FN': nn.ReLU(),
        'INIT_FN': init_weights_normal,
        'SIGMA_INIT': 0.45,  # Initial noise scale
        'K_FINAL': 4.0,         # Final value of k
        'INF_REWARD': 0,     # Minimum possible reward
        'SUP_REWARD': 200,
        'PER_ALPHA': 0.5,
        'PER_BETA_START': 0.4,
        'PER_BETA_FRAMES': 15000,
        'SUB_GOAL_KEYS': ['theta', 'x'],  # Keys expected in sub_goal
    },
    'LunarLander-v3': {
        'BATCH_SIZE': 128,
        'LEARNING_RATE': 5e-4,
        'GAMMA': 0.99,
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.01,
        'EPSILON_DECAY': 0.993,
        'MEMORY_CAPACITY': int(5e4),
        'TARGET_UPDATE_EVERY': 2,  # Soft target updates
        'NUM_EPISODES': 4000,
        'MAX_T': 1000,
        'SOLVED_SCORE': 200,
        'SOLVED_LENGTH': 100,
        'HIDDEN_LAYERS': [256, 128],  # Specific to LunarLander
        'ACTIVATION_FN': nn.ReLU(),
        'INIT_FN': None,
        'TAU': 1e-3,
        'SIGMA_INIT': 0.4,                 # Increased initial noise for better exploration
        'K_FINAL': 2.75,                    # Lower K_FINAL to reduce overemphasis on noise reduction
        'INF_REWARD': -175,                # Minimum possible reward
        'SUP_REWARD': 275,                 # Adjusted to reflect typical reward range
        'PER_ALPHA': 0.5,
        'PER_BETA_START': 0.4,
        'PER_BETA_FRAMES': 15000,
        'SUB_GOAL_KEYS': ['y', 'y_dot', 'x', 'x_dot', 'theta', 'theta_dot'],
    }
}
