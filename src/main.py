# main.py

import argparse
import gymnasium as gym
import importlib
import torch
import numpy as np
from collections import deque
import os
import json
import time  # Import for timing

# Import the shared manage_noise_scaling function
from shared.utils import manage_noise_scaling

def custom_cartpole_reward(env, x, x_dot, theta, theta_dot):
    """
    Custom reward function for CartPole-v1.
    Balances the pole while penalizing instability.
    """
    base_env = env.unwrapped
    r1 = (base_env.x_threshold - abs(x)) / base_env.x_threshold - 0.5  # Penalize deviation from center
    r2 = (base_env.theta_threshold_radians - abs(theta)) / base_env.theta_threshold_radians - 0.5  # Penalize pole angle
    reward = r1 + r2
    return reward

def create_results_dir(use_hrl, high_arch, low_arch, env, noise, mode):
    """
    Creates a results directory based on the provided parameters.
    Includes high_arch only if use_hrl is 'yes'.
    """
    if use_hrl == 'yes':
        dir_name = f"./results/use_hrl_{use_hrl}_high_{high_arch}_low_{low_arch}_env_{env}_noise_{noise}_mode_{mode}"
    else:
        dir_name = f"./results/use_hrl_{use_hrl}_low_{low_arch}_env_{env}_noise_{noise}_mode_{mode}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_hrl", type=str, default='yes', choices=['yes', 'no'],
                        help="Use Hierarchical Reinforcement Learning (yes or no)")
    parser.add_argument("--high_arch", type=str, required=False, default="RuleBased",
                        help="High-level Architecture (e.g., DQN, RuleBased)")
    parser.add_argument("--low_arch", type=str, required=True,
                        help="Low-level Architecture (e.g., DQN, DoubleDQN, etc.)")
    parser.add_argument("--env", type=str, required=True,
                        help="Environment (e.g., CartPole-v1, LunarLander-v3)")
    parser.add_argument("--noise", type=str, default='off', choices=['on', 'off'],
                        help="Turn noise injection on or off")
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'validate'],
                        help="Mode: train or validate")
    args = parser.parse_args()

    environment_name = args.env
    use_hrl = args.use_hrl
    high_arch = args.high_arch
    low_arch = args.low_arch

    env = gym.make(environment_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    results_dir = create_results_dir(use_hrl, high_arch, low_arch, environment_name, args.noise, args.mode)

    # Noise wrappers
    if args.noise == 'on':
        # Initialize noise parameters
        initial_std_dev = 0.2        # Starting standard deviation for Gaussian noise
        final_std_dev = 0.00       # Ending standard deviation
        initial_failure_rate = 0.3   # Starting failure rate
        final_failure_rate = 0.00  # Ending failure rate
        initial_wind_force = 0.2     # Starting wind force
        final_wind_force = 0.00    # Ending wind force

        validate_std_dev = 0.1
        validate_failure_rate = 0.15
        validate_wind_force = 0.1

        # Import noise wrappers
        from environments.environment_wrappers import (
            IIDGaussianNoiseWrapper,
            WindSimulationWrapper,
            EngineFailureWrapper,
        )

        # Create noise wrappers with initial parameters
        wind_wrapper = WindSimulationWrapper(env, wind_force=initial_wind_force)
        iid_gaussian_wrapper = IIDGaussianNoiseWrapper(wind_wrapper, std_dev=initial_std_dev)
        engine_failure_wrapper = EngineFailureWrapper(iid_gaussian_wrapper, failure_rate=initial_failure_rate)

        # The final environment with noise wrappers applied
        env = engine_failure_wrapper

        # Store references to the wrappers for easy access
        noise_wrappers = {
            'wind_wrapper': wind_wrapper,
            'iid_gaussian_wrapper': iid_gaussian_wrapper,
            'engine_failure_wrapper': engine_failure_wrapper,
        }
    else:
        noise_wrappers = None
        # Initialize noise parameters
        initial_std_dev = 0.0        # Starting standard deviation for Gaussian noise
        final_std_dev = 0.0       # Ending standard deviation
        initial_failure_rate = 0.0   # Starting failure rate
        final_failure_rate = 0.0  # Ending failure rate
        initial_wind_force = 0.0     # Starting wind force
        final_wind_force = 0.0   # Ending wind force

        validate_std_dev = 0.1
        validate_failure_rate = 0.15
        validate_wind_force = 0.1

    if use_hrl == 'yes':
        # HRL code
        # Define number of subtasks
        if environment_name == "CartPole-v1":
            num_subtasks = 2
        elif environment_name == "LunarLander-v3":
            num_subtasks = 2  # Adjusted based on your sub-goal definitions
        else:
            raise ValueError(f"Unsupported environment: {environment_name}")

        # Dynamically import the agent class and hyperparameters for low-level policies
        low_arch_module_name = f"architectures.{low_arch}"
        agent_module_name = f"{low_arch_module_name}.dqn_agent_cartpole" \
            if environment_name == "CartPole-v1" else f"{low_arch_module_name}.dqn_agent_lunarlander"
        low_config_module_name = f"configs.config_{low_arch}_low"

        agent_module = importlib.import_module(agent_module_name)
        config_module = importlib.import_module(low_config_module_name)

        AgentClass = getattr(agent_module, "DQNAgentCartPole") \
            if environment_name == "CartPole-v1" else getattr(agent_module, "DQNAgentLunarLander")
        hyperparameters = config_module.hyperparameters[environment_name]

        high_config_module_name = f"configs.config_{high_arch}_high"
        high_config_module = importlib.import_module(high_config_module_name)
        high_hyperparameters = high_config_module.hyperparameters[environment_name]

        # Use the same AgentClass for high-level controller, with action_size = num_subtasks
        # Define a HighLevelController class that inherits from AgentClass
        high_agent_module_name = f"architectures.{high_arch}.dqn_agent_cartpole" \
            if environment_name == "CartPole-v1" else f"architectures.{high_arch}.dqn_agent_lunarlander"
        high_agent_module = importlib.import_module(high_agent_module_name)
        HighAgentClass = getattr(high_agent_module, "DQNAgentCartPole") \
            if environment_name == "CartPole-v1" else getattr(high_agent_module, "DQNAgentLunarLander")

        class HighLevelController(HighAgentClass):
            def __init__(self, state_size, num_subtasks, hyperparameters, environment_name):
                super().__init__(state_size, num_subtasks, hyperparameters)
                self.environment_name = environment_name
                self.last_loss = None

            def select_subtask(self, state):
                """
                Select a subtask based on the current state.
                """
                return self.act(state)

            def get_sub_goal(self, subtask, state):
                """
                Define sub-goals for each subtask based on control objectives.
                """
                if self.environment_name == "CartPole-v1":
                    if subtask == 0:  # Balance Control
                        return {
                            "theta": state[2],     # pole angle
                            "theta_dot": state[3]  # angular velocity
                        }
                    elif subtask == 1:  # Position Control
                        return {
                            "x": state[0],         # cart position
                            "x_dot": state[1]      # cart velocity
                        }
                elif self.environment_name == "LunarLander-v3":
                    if subtask == 0:  # Altitude Control
                        return {
                            "y": state[1],         # vertical position
                            "y_dot": state[3]      # vertical velocity
                        }
                    elif subtask == 1:  # Landing Alignment
                        return {
                            "x": state[0],         # horizontal position
                            "x_dot": state[2],     # horizontal velocity
                            "theta": state[4],     # lander angle
                            "theta_dot": state[5]  # angular velocity
                        }
                return {}

            def get_sub_goal_keys(self, subtask):
                """
                Return the keys of the sub-goal for the given subtask.
                """
                if self.environment_name == "CartPole-v1":
                    if subtask == 0:  # Balance Control
                        return ["theta", "theta_dot"]
                    elif subtask == 1:  # Position Control
                        return ["x", "x_dot"]
                elif self.environment_name == "LunarLander-v3":
                    if subtask == 0:  # Altitude Control
                        return ["y", "y_dot"]
                    elif subtask == 1:  # Landing Alignment
                        return ["x", "x_dot", "theta", "theta_dot"]
                return []

            def update(self, state, subtask, reward, done):
                """
                Update the high-level controller.
                """
                # subtask = sub_goal  # In this context, sub_goal corresponds to subtask index
                next_state = state  # Assuming state transition happens after this call
                self.last_loss = self.step(state, subtask, reward, next_state, done)

        if args.mode == 'validate':
            # Adjust hyperparameters for validation
            hyperparameters['NUM_EPISODES'] = 500  # Fixed to run 500 episodes
            hyperparameters['SOLVED_SCORE'] = 100000000     # Random number to never let it complete early
            high_hyperparameters['NUM_EPISODES'] = 500  # Fixed to run 500 episodes
            high_hyperparameters['SOLVED_SCORE'] = 100000000     # Random number to never let it complete early
            high_hyperparameters['EPSILON_END'] = 0.0
            hyperparameters['EPSILON_END'] = 0.0
            hyperparameters['EPSILON_START'] = 0.0
            high_hyperparameters['EPSILON_START'] = 0.0
            high_hyperparameters['MAX_T'] = 1000

        # Instantiate the high-level controller
        high_level_controller = HighLevelController(state_size, num_subtasks,
                                                    high_hyperparameters, environment_name)

        # Create low-level policies for each subtask
        low_level_policies = []
        for subtask in range(num_subtasks):
            sub_goal_keys = high_level_controller.get_sub_goal_keys(subtask)
            adjusted_state_size = state_size + len(sub_goal_keys)
            print(f"Adjusted state size for low-level agent {subtask}: {adjusted_state_size}")
            agent = AgentClass(adjusted_state_size, action_size, hyperparameters)
            low_level_policies.append(agent)

        # If in validate mode, load saved models
        if args.mode == 'validate':
            train_results_dir = create_results_dir(use_hrl, high_arch, low_arch, environment_name, args.noise, 'train')
            for idx, agent in enumerate(low_level_policies):
                model_path = os.path.join(train_results_dir, f'low_level_agent_{idx}_checkpoint.pth')
                if environment_name == "CartPole-v1":
                    agent.eval_net.load_state_dict(torch.load(model_path))
                else:
                    agent.qnetwork_local.load_state_dict(torch.load(model_path))
                agent.epsilon = 0.0  # Disable exploration
            if hasattr(high_level_controller, 'eval_net'):
                model_path = os.path.join(train_results_dir, 'high_level_controller_checkpoint.pth')
                high_level_controller.eval_net.load_state_dict(torch.load(model_path))
                high_level_controller.epsilon = 0.0  # Disable exploration
            elif hasattr(high_level_controller, 'qnetwork_local'):
                model_path = os.path.join(train_results_dir, 'high_level_controller_checkpoint.pth')
                high_level_controller.qnetwork_local.load_state_dict(torch.load(model_path))
                high_level_controller.epsilon = 0.0  # Disable exploration

        # Create HRL Manager
        from shared.hrl_manager import HRLManager

        hrl_manager = HRLManager(
            high_level_controller,
            low_level_policies,
            environment_name,
            noise_wrappers=noise_wrappers,
            custom_reward_function=custom_cartpole_reward if environment_name == 'CartPole-v1' else None,
            results_dir=results_dir
        )

        # Set initial and final noise parameters in HRLManager
        hrl_manager.initial_std_dev = initial_std_dev
        hrl_manager.final_std_dev = final_std_dev
        hrl_manager.initial_failure_rate = initial_failure_rate
        hrl_manager.final_failure_rate = final_failure_rate
        hrl_manager.initial_wind_force = initial_wind_force
        hrl_manager.final_wind_force = final_wind_force
        hrl_manager.validate_std_dev = validate_std_dev
        hrl_manager.validate_failure_rate = validate_failure_rate
        hrl_manager.validate_wind_force = validate_wind_force

        # Train or validate the HRL system
        scores = hrl_manager.train(env, hyperparameters, mode=args.mode)
    else:
        # Non-HRL code
        # Dynamically import the agent class and hyperparameters
        arch_module_name = f"architectures.{low_arch}"
        agent_module_name = f"{arch_module_name}.dqn_agent_cartpole" if environment_name == "CartPole-v1" else f"{arch_module_name}.dqn_agent_lunarlander"
        config_module_name = f"{arch_module_name}.config"

        agent_module = importlib.import_module(agent_module_name)
        config_module = importlib.import_module(config_module_name)

        AgentClass = getattr(agent_module, "DQNAgentCartPole") if environment_name == "CartPole-v1" else getattr(agent_module, "DQNAgentLunarLander")
        hyperparameters = config_module.hyperparameters[environment_name]

        # Initialize noise parameters
        initial_std_dev = 0.2        # Starting standard deviation for Gaussian noise
        final_std_dev = 0.00      # Ending standard deviation
        initial_failure_rate = 0.3   # Starting failure rate
        final_failure_rate = 0.00  # Ending failure rate
        initial_wind_force = 0.2     # Starting wind force
        final_wind_force = 0.00    # Ending wind force

        # If in validate mode, load saved model
        if args.mode == 'validate':
            hyperparameters['NUM_EPISODES'] = 500  # Fixed to run 500 episodes
            hyperparameters['SOLVED_SCORE'] = 100000000     # Random number to never let it complete early
            hyperparameters['EPSILON_END'] = 0.0
            hyperparameters['EPSILON_START'] = 0.0
            hyperparameters['MAX_T'] = 1000
            train_results_dir = create_results_dir(use_hrl, high_arch, low_arch, environment_name, args.noise, 'train')
            agent = AgentClass(state_size, action_size, hyperparameters)
            model_path = os.path.join(train_results_dir, f'{environment_name}_checkpoint.pth')
            if environment_name == "CartPole-v1":
                agent.eval_net.load_state_dict(torch.load(model_path))
            else:
                agent.qnetwork_local.load_state_dict(torch.load(model_path))
            agent.epsilon = 0.0  # Disable exploration
        else:
            agent = AgentClass(state_size, action_size, hyperparameters)

        # Custom reward function
        use_custom_reward = environment_name == "CartPole-v1"
        # use_custom_reward = False

        # Initialize tracking variables
        num_episodes = hyperparameters['NUM_EPISODES']
        max_t = hyperparameters['MAX_T']
        scores = []
        scores_window = deque(maxlen=hyperparameters.get('SOLVED_LENGTH', 100))

        # Training loop
        for i_episode in range(1, num_episodes + 1):
            episode_start_time = time.time()  # Start timing the episode
            if noise_wrappers:
                if args.mode == 'train':
                    # Adjust noise based on episode number
                    if environment_name == "CartPole-v1":
                        keep_high_duration = 25
                        total_episodes = 125
                        std_dev = manage_noise_scaling(i_episode, total_episodes, keep_high_duration, initial_std_dev, final_std_dev)
                        failure_rate = manage_noise_scaling(i_episode, total_episodes, keep_high_duration, initial_failure_rate, final_failure_rate)
                        wind_force = manage_noise_scaling(i_episode, total_episodes, keep_high_duration, initial_wind_force, final_wind_force)
                    else:
                        keep_high_duration = 150
                        total_episodes = 350
                        std_dev = manage_noise_scaling(i_episode, total_episodes, keep_high_duration, initial_std_dev, final_std_dev)
                        failure_rate = manage_noise_scaling(i_episode, total_episodes, keep_high_duration, initial_failure_rate, final_failure_rate)
                        wind_force = manage_noise_scaling(i_episode, total_episodes, keep_high_duration, initial_wind_force, final_wind_force)
                else:
                    std_dev = validate_std_dev
                    failure_rate = validate_failure_rate
                    wind_force = validate_wind_force


                # Update noise parameters in the wrappers
                noise_wrappers['iid_gaussian_wrapper'].std_dev = std_dev
                noise_wrappers['engine_failure_wrapper'].failure_rate = failure_rate
                noise_wrappers['wind_wrapper'].wind_force = wind_force
            else:
                std_dev = failure_rate = wind_force = 0  # Default values when noise is off

            state, _ = env.reset()
            score = 0
            done = False

            for t in range(max_t):
                action = agent.act(state)
                next_state, reward, done, truncated, _ = env.step(action)

                # Apply custom reward for CartPole
                if use_custom_reward:
                    x, x_dot, theta, theta_dot = next_state
                    reward = custom_cartpole_reward(env, x, x_dot, theta, theta_dot)

                if args.mode == 'train':
                    agent.step(state, action, reward, next_state, done or truncated)
                state = next_state
                score += reward

                if done or truncated:
                    break

            episode_time = time.time() - episode_start_time  # Calculate time for the episode

            # After the episode ends
            episode_data = {
                'episode': i_episode,
                'score': float(score),
                'noise_levels': {
                    'std_dev': std_dev,
                    'failure_rate': failure_rate,
                    'wind_force': wind_force
                },
                'loss': getattr(agent, 'last_loss', None),
                'time': episode_time  # Log time taken
            }
            # print(episode_data)

            # Save per-episode data
            with open(os.path.join(results_dir, 'training_log.json'), 'a') as f:
                f.write(json.dumps(episode_data) + '\n')

            scores_window.append(score)
            scores.append(score)

            if args.mode == 'train':
                # Update epsilon
                agent.update_epsilon()

            # Print statements with average score and noise levels
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tTime: {episode_time:.2f}s\tNoise Level: std_dev={std_dev:.3f}, failure_rate={failure_rate:.3f}, wind_force={wind_force:.3f}', end="")
            if i_episode % 100 == 0:
                print()  # Move to a new line every 100 episodes

            # Early stopping if environment is solved
            if args.mode == 'train' and np.mean(scores_window) >= hyperparameters.get('SOLVED_SCORE', 200):
                print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
                # Save model
                model_path = os.path.join(results_dir, f'{environment_name}_checkpoint.pth')
                if environment_name == "CartPole-v1":
                    torch.save(agent.eval_net.state_dict(), model_path)
                else:
                    torch.save(agent.qnetwork_local.state_dict(), model_path)
                break
        return scores

if __name__ == "__main__":
    main()
