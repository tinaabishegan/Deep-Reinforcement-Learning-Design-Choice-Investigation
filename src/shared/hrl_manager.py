# shared/hrl_manager.py

import numpy as np
from collections import deque
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import json
import time  # Import for timing

# Import the shared manage_noise_scaling function
from shared.utils import manage_noise_scaling

class HRLManager:
    def __init__(self, high_level_controller, low_level_policies, environment_name, noise_wrappers=None, custom_reward_function=None, results_dir=None):
        self.high_level_controller = high_level_controller
        self.low_level_policies = low_level_policies
        self.environment_name = environment_name
        self.writer = SummaryWriter(log_dir=f"runs/{environment_name}")
        self.global_step = 0
        self.noise_wrappers = noise_wrappers
        self.custom_reward_function = custom_reward_function

        # Initialize noise parameters (these will be set from main.py)
        self.initial_std_dev = 0.2        # Starting standard deviation for Gaussian noise
        self.final_std_dev = 0.0001       # Ending standard deviation
        self.initial_failure_rate = 0.3   # Starting failure rate
        self.final_failure_rate = 0.000  # Ending failure rate
        self.initial_wind_force = 0.2     # Starting wind force
        self.final_wind_force = 0.0001    # Ending wind force

        self.validate_std_dev = 0.1
        self.validate_failure_rate = 0.15
        self.validate_wind_force = 0.1

        self.results_dir = results_dir
        if self.results_dir:
            os.makedirs(self.results_dir, exist_ok=True)
        self.episode_data = []

    def modify_state_with_sub_goal(self, state, sub_goal, sub_goal_keys):
        """
        Modify the state by concatenating sub-goal values in the order of sub_goal_keys.
        """
        if sub_goal:
            sub_goal_values = np.array([sub_goal[key] for key in sub_goal_keys])
            modified_state = np.concatenate((state, sub_goal_values))
        else:
            modified_state = state
        return modified_state

    def train(self, env, hyperparameters, mode='train'):
        num_episodes = hyperparameters['NUM_EPISODES']
        max_t = hyperparameters['MAX_T']
        scores = []
        scores_window = deque(maxlen=hyperparameters.get('SOLVED_LENGTH', 100))

        for i_episode in range(1, num_episodes + 1):
            episode_start_time = time.time()  # Start timing the episode

            if self.noise_wrappers:
                if mode == 'train':
                    # Define keep_high_duration based on environment
                    if self.environment_name == "CartPole-v1":
                        keep_high_duration = 25
                        total_episodes = 125
                        std_dev = manage_noise_scaling(i_episode, total_episodes, keep_high_duration, self.initial_std_dev, self.final_std_dev)
                        failure_rate = manage_noise_scaling(i_episode, total_episodes, keep_high_duration, self.initial_failure_rate, self.final_failure_rate)
                        wind_force = manage_noise_scaling(i_episode, total_episodes, keep_high_duration, self.initial_wind_force, self.final_wind_force)
                    else:
                        keep_high_duration = 150
                        total_episodes = 350
                        std_dev = manage_noise_scaling(i_episode, total_episodes, keep_high_duration, self.initial_std_dev, self.final_std_dev)
                        failure_rate = manage_noise_scaling(i_episode, total_episodes, keep_high_duration, self.initial_failure_rate, self.final_failure_rate)
                        wind_force = manage_noise_scaling(i_episode, total_episodes, keep_high_duration, self.initial_wind_force, self.final_wind_force)
                else:
                    # Validation mode: Use fixed final noise values
                    std_dev = self.validate_std_dev
                    failure_rate = self.validate_failure_rate
                    wind_force = self.validate_wind_force

                # Update noise parameters in the wrappers
                self.noise_wrappers['iid_gaussian_wrapper'].std_dev = std_dev
                self.noise_wrappers['engine_failure_wrapper'].failure_rate = failure_rate
                self.noise_wrappers['wind_wrapper'].wind_force = wind_force
            else:
                std_dev = failure_rate = wind_force = 0  # Default values when noise is off

            state, _ = env.reset()
            score = 0
            done = False
            t = 0
            while not done and t < max_t:
                # Select subtask using high-level controller
                subtask = self.high_level_controller.select_subtask(state)
                sub_goal = self.high_level_controller.get_sub_goal(subtask, state)
                sub_goal_keys = self.high_level_controller.get_sub_goal_keys(subtask)
                policy = self.low_level_policies[subtask]

                # Modify state with sub_goal
                modified_state = self.modify_state_with_sub_goal(state, sub_goal, sub_goal_keys)

                # Pass modified state to act
                action = policy.act(modified_state)

                next_state, reward, done, truncated, info = env.step(action)

                # Apply custom reward if provided
                if self.custom_reward_function:
                    if self.environment_name == 'CartPole-v1':
                        x, x_dot, theta, theta_dot = next_state
                        reward = self.custom_reward_function(env, x, x_dot, theta, theta_dot)

                # Modify next_state with sub_goal for storage and learning
                modified_next_state = self.modify_state_with_sub_goal(next_state, sub_goal, sub_goal_keys)

                if mode == 'train':
                    # Update low-level policy
                    loss = policy.step(modified_state, action, reward, modified_next_state, done)
                    # Update high-level controller
                    high_level_reward = reward  # You can design a specific reward for the high-level controller
                    self.high_level_controller.update(state, subtask, high_level_reward, done)
                else:
                    loss = None  # No learning in validate mode

                state = next_state
                score += reward
                t += 1
                self.global_step += 1

                # Logging loss to TensorBoard
                if loss is not None:
                    self.writer.add_scalar(f'Loss/Subtask{subtask}', loss, self.global_step)

            episode_time = time.time() - episode_start_time  # Calculate time for the episode

            # After the episode ends
            episode_info = {
                'episode': i_episode,
                'score': float(score),
                'noise_levels': {
                    'std_dev': std_dev,
                    'failure_rate': failure_rate,
                    'wind_force': wind_force
                },
                'loss': getattr(self.high_level_controller, 'last_loss', None),
                'time': episode_time  # Log time taken
            }
            # Collect losses from low-level policies
            episode_info['losses'] = {}
            for idx, policy in enumerate(self.low_level_policies):
                loss = getattr(policy, 'last_loss', None)
                episode_info['losses'][f'low_level_agent_{idx}'] = loss
            # Collect high-level loss if available
            if hasattr(self.high_level_controller, 'last_loss'):
                episode_info['losses']['high_level_controller'] = self.high_level_controller.last_loss
            # Append episode data
            self.episode_data.append(episode_info)

            # Save per-episode data to file
            with open(os.path.join(self.results_dir, 'training_log.json'), 'a') as f:
                f.write(json.dumps(episode_info) + '\n')

            scores_window.append(score)
            scores.append(score)

            # Update epsilon and log it
            if mode == 'train':
                for idx, policy in enumerate(self.low_level_policies):
                    policy.update_epsilon()
                    epsilon = getattr(policy, 'epsilon', None)
                    if epsilon is not None:
                        self.writer.add_scalar(f'Epsilon/Subtask{idx}', epsilon, i_episode)

                if hasattr(self.high_level_controller, 'update_epsilon'):
                    self.high_level_controller.update_epsilon()
                    high_level_epsilon = getattr(self.high_level_controller, 'epsilon', None)
                    if high_level_epsilon is not None:
                        self.writer.add_scalar('Epsilon/HighLevel', high_level_epsilon, i_episode)
            else:
                if hasattr(self.high_level_controller, 'epsilon'):
                    high_level_epsilon = self.high_level_controller.epsilon
                else:
                    high_level_epsilon = 0.0

            # Log episode score
            self.writer.add_scalar('Score', score, i_episode)

            # Print statements with average score and noise levels
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tNoise Level: std_dev={std_dev:.3f}, failure_rate={failure_rate:.3f}, wind_force={wind_force:.3f}', end="")
            if i_episode % 100 == 0:
                print()  # Move to a new line every 100 episodes

            # Early stopping if environment is solved
            if mode == 'train' and np.mean(scores_window) >= hyperparameters.get('SOLVED_SCORE', 200):
                print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
                # Save models
                if mode == 'train':
                    for idx, policy in enumerate(self.low_level_policies):
                        model_path = os.path.join(self.results_dir, f'low_level_agent_{idx}_checkpoint.pth')
                        if self.environment_name == "CartPole-v1":
                            torch.save(policy.eval_net.state_dict(), model_path)
                        else:
                            torch.save(policy.qnetwork_local.state_dict(), model_path)
                    if hasattr(self.high_level_controller, 'eval_net'):
                        model_path = os.path.join(self.results_dir, 'high_level_controller_checkpoint.pth')
                        torch.save(self.high_level_controller.eval_net.state_dict(), model_path)
                    elif hasattr(self.high_level_controller, 'qnetwork_local'):
                        model_path = os.path.join(self.results_dir, 'high_level_controller_checkpoint.pth')
                        torch.save(self.high_level_controller.qnetwork_local.state_dict(), model_path)
                break
        self.writer.close()
        return scores
