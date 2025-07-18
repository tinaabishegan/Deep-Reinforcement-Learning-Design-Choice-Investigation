
# config.py
import torch 
import torch.nn.functional as F

def init_weights_normal(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
        torch.nn.init.constant_(m.bias, 0)

# config.py

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
        'NUM_EPISODES': 2000,
        'MAX_T': 10000,
        'SOLVED_SCORE': 475,
        'SOLVED_LENGTH': 100,
        'HIDDEN_LAYERS': [128, 64],  # Specific to CartPole
        'ACTIVATION_FN': F.relu,
        'INIT_FN': init_weights_normal,
        'SUB_GOAL_KEYS': ['theta', 'x'],  # Keys expected in sub_goal
    },
    'LunarLander-v3': {
        'BATCH_SIZE': 128,
        'LEARNING_RATE': 1e-4,
        'GAMMA': 0.99,
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.01,
        'EPSILON_DECAY': 0.995,
        'MEMORY_CAPACITY': int(5e4),
        'TARGET_UPDATE_EVERY': 2,  # Soft target updates
        'NUM_EPISODES': 2000,
        'MAX_T': 1000,
        'SOLVED_SCORE': 200,
        'SOLVED_LENGTH': 100,
        'HIDDEN_LAYERS': [256, 128],  # Specific to LunarLander
        'ACTIVATION_FN': F.relu,
        'INIT_FN': None,
        'TAU': 1e-3,  # Soft update parameter
        'SUB_GOAL_KEYS': ['y', 'y_dot', 'x', 'x_dot', 'theta', 'theta_dot'],
    }
}

