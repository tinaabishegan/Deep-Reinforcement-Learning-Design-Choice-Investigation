
# config.py
import torch 
import torch.nn as nn

def init_weights_normal(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
        torch.nn.init.constant_(m.bias, 0)

# config.py

hyperparameters = {
    'CartPole-v1': {
        'BATCH_SIZE': 64,
        'LEARNING_RATE': 0.002,
        'GAMMA': 0.995,
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.01,
        'EPSILON_DECAY': 0.997,
        'MEMORY_CAPACITY': 2000,
        'TARGET_UPDATE_EVERY': 200,  # Hard target update interval
        'NUM_EPISODES': 3000,
        'MAX_T': 5000,
        'SOLVED_SCORE': 475,
        'SOLVED_LENGTH': 100,
        'HIDDEN_LAYERS': [64, 32],  # Specific to CartPole
        'ACTIVATION_FN': nn.ReLU(),
        'INIT_FN': init_weights_normal,
        'USE_NOISY': True,  # Enable noisy networks
        'NOISE_SCALE': 0.45,  # Initial noise scale
        'PER_ALPHA': 0.7,
        'PER_BETA_START': 0.5,
        'PER_BETA_FRAMES': 5000,
        'SUB_GOAL_KEYS': ['theta', 'x'],  # Keys expected in sub_goal
    }
    ,
    'LunarLander-v3': {
        'BATCH_SIZE': 64,
        'LEARNING_RATE': 0.001,
        'GAMMA': 0.995,
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.01,
        'EPSILON_DECAY': 0.997,
        'MEMORY_CAPACITY': int(1e4),
        'TARGET_UPDATE_EVERY': 4,  # Soft target updates
        'NUM_EPISODES': 5000,
        'MAX_T': 1000,
        'SOLVED_SCORE': 200,
        'SOLVED_LENGTH': 100,
        'HIDDEN_LAYERS': [256, 128],  # Specific to LunarLander
        'ACTIVATION_FN': nn.ReLU(),
        'INIT_FN': None,
        'TAU': 1e-3,  # Soft update parameter
        # Noisy Network parameters
        'USE_NOISY': True,
        'SIGMA_INIT': 0.4,
        'PER_ALPHA': 0.7,
        'PER_BETA_START': 0.4,
        'PER_BETA_FRAMES': 20000,
        'SUB_GOAL_KEYS': ['y', 'y_dot', 'x', 'x_dot', 'theta', 'theta_dot'],
    }
}

