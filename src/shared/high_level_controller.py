# shared/high_level_controller.py

class HighLevelController:
    def __init__(self, environment_name):
        self.environment_name = environment_name
        # Initialize any high-level policy parameters here if learning is required

    def select_subtask(self, state):
        """
        Select a subtask based on the current state.
        """
        if self.environment_name == "CartPole-v1":
            x, x_dot, theta, theta_dot = state
            if abs(theta) > 0.1:
                return 0  # Subtask 0: Balance Control
            else:
                return 1  # Subtask 1: Position Control
        elif self.environment_name == "LunarLander-v3":
            x, y, x_dot, y_dot, theta, theta_dot, leg1, leg2 = state
            if y > 0.1 and abs(y_dot) > 0.1:
                return 0  # Subtask 0: Altitude Control
            elif abs(x_dot) > 0.1:
                return 1  # Subtask 1: Landing Alignment
            else:
                return 2  # Subtask 2: (Define as needed)
        else:
            raise ValueError(f"Unsupported environment: {self.environment_name}")

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

    def update(self, state, sub_goal, reward, done):
        """
        Update the high-level controller (if learning is required).
        """
        # For this implementation, the high-level controller is rule-based and does not learn.
        pass
