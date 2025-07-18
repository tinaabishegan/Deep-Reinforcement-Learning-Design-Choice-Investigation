import gymnasium as gym
import numpy as np

# Observation Noise Wrappers

class IIDGaussianNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, std_dev=0.1):
        super(IIDGaussianNoiseWrapper, self).__init__(env)
        self.std_dev = std_dev

    def observation(self, observation):
        noise = np.random.normal(0, self.std_dev, size=observation.shape)
        return observation + noise


class SensorShutdownWrapper(gym.ObservationWrapper):
    def __init__(self, env, shutdown_prob=0.01):
        super(SensorShutdownWrapper, self).__init__(env)
        self.shutdown_prob = shutdown_prob

    def observation(self, observation):
        if np.random.rand() < self.shutdown_prob:
            return np.zeros_like(observation)
        return observation


class SensorCalibrationFailureWrapper(gym.ObservationWrapper):
    def __init__(self, env, scale_factor=1.2):
        super(SensorCalibrationFailureWrapper, self).__init__(env)
        self.scale_factor = scale_factor

    def observation(self, observation):
        return observation * self.scale_factor


class SensorDriftWrapper(gym.ObservationWrapper):
    def __init__(self, env, drift_rate=0.001):
        super(SensorDriftWrapper, self).__init__(env)
        self.drift = np.zeros(env.observation_space.shape)
        self.drift_rate = drift_rate

    def observation(self, observation):
        self.drift += self.drift_rate
        return observation + self.drift


# Action Noise Wrappers

class UniformActionNoiseWrapper(gym.ActionWrapper):
    def __init__(self, env, noise_level=0.05):
        super(UniformActionNoiseWrapper, self).__init__(env)
        self.noise_level = noise_level

    def action(self, action):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return action  # No direct noise for discrete actions
        noise = np.random.uniform(-self.noise_level, self.noise_level, size=self.env.action_space.shape)
        return np.clip(action + noise, self.env.action_space.low, self.env.action_space.high)


class WindSimulationWrapper(gym.Wrapper):
    def __init__(self, env, wind_force=0.1):
        super(WindSimulationWrapper, self).__init__(env)
        self.wind_force = wind_force

    def step(self, action):
        # Traverse the wrapper stack to find the base environment
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        
        # Apply wind force for LunarLander
        if hasattr(base_env, "lander"):
            base_env.lander.ApplyForceToCenter((self.wind_force, 0), True)
        
        # For CartPole, simulate wind by altering the x_dot state
        next_state, reward, done, truncated, info = self.env.step(action)
        if 'CartPole' in self.env.spec.id:
            next_state[1] += self.wind_force  # Apply to x_dot
        return next_state, reward, done, truncated, info


class EngineFailureWrapper(gym.ActionWrapper):
    def __init__(self, env, failure_rate=0.2):
        super(EngineFailureWrapper, self).__init__(env)
        self.failure_rate = failure_rate

    def action(self, action):
        if np.random.rand() < self.failure_rate:
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                possible_actions = list(range(self.env.action_space.n))
                possible_actions.remove(action)
                return np.random.choice(possible_actions)
            return np.zeros_like(action)  # Zero out for continuous
        return action


# Utility Function for Modular Application of Noise Wrappers

def apply_noise_wrappers(env, observation_wrappers=None, action_wrappers=None):
    """
    Apply observation and action noise wrappers to an environment.
    
    Args:
        env (gym.Env): The base environment.
        observation_wrappers (list): List of observation noise wrapper functions.
        action_wrappers (list): List of action noise wrapper functions.
    
    Returns:
        gym.Env: The environment with the specified noise wrappers applied.
    """
    if observation_wrappers:
        for wrapper in observation_wrappers:
            env = wrapper(env)
    if action_wrappers:
        for wrapper in action_wrappers:
            env = wrapper(env)
    return env
