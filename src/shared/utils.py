# shared/utils.py

import numpy as np

def manage_noise_scaling(episode, total_episodes, keep_high_duration, initial_std_dev, final_std_dev):
    """
    Manages noise scaling based on the current episode.

    Parameters:
    - episode (int): Current episode number.
    - total_episodes (int): Total number of episodes.
    - keep_high_duration (int): Number of episodes to keep high noise.
    - initial_std_dev (float): Starting standard deviation for Gaussian noise.
    - final_std_dev (float): Ending standard deviation for Gaussian noise.

    Returns:
    - float: Adjusted standard deviation.
    """
    decay_duration = total_episodes - keep_high_duration

    if episode < keep_high_duration:
        return initial_std_dev
    else:
        progress = (episode - keep_high_duration) / decay_duration
        return final_std_dev + (initial_std_dev - final_std_dev) * (1 - np.clip(progress, 0, 1))
