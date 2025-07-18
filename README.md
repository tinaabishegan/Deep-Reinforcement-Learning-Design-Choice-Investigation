# Investigating the Impact of Architectural Modularity and Design Choices on the Robustness of Deep Reinforcement Learning in Noisy Environments

## Overview
This project explores how different Deep Q-Network (DQN) based architectures fare when trained and evaluated under noisy conditions. Real-world scenarios often involve uncertainty such as sensor errors, environmental disturbances (e.g., wind), and actuator failures, all of which can degrade the performance and stability of RL agents. By studying various architectural enhancements, including Double DQN, Prioritised Experience Replay (PER), Dueling Networks, Noisy Networks (NoisyNets), Novel Random Weight Auxiliary Networks (NROWAN), and hierarchical (modular) reinforcement learning setups, we aim to identify configurations that improve an agent’s robustness and efficiency in noisy and transient conditions.

## Key Contributions
1. **Combined Noise Injection**:  
   Agents are trained with multiple types of noise—environmental wind, Gaussian sensor noise, and action (engine) failure—present only during an initial training phase. The noise is then gradually annealed to zero, simulating real-world scenarios where noise may be intense but limited in duration.

2. **Comparative Architectural Evaluation**:  
   We systematically compare baseline DQN agents to advanced architectures (DoubleDQN-PER-Dueling, DoubleDQN-PER-Dueling-Noisy, DoubleDQN-PER-Dueling-NROWAN) as well as hierarchical (modular) combinations of these. The goal is to understand which configurations yield the best balance of robustness (ability to handle noise) and efficiency (speed of learning).

3. **Realistic Training Protocols**:  
   By limiting noise exposure to the early stage of training, we test whether architectures can “learn from limited chances” to deal with noise and retain those strategies after noise subsides. Post-training, agents are validated in a moderate noise environment to measure how well they recall and handle noisy conditions.

## Architectures Explored
- **DQN** [Baseline]: The standard Deep Q-Network architecture.
- **DoubleDQN-PER-Dueling**: Combines Double Q-learning, Prioritised Experience Replay, and Dueling Networks, aiming for improved stability, sample efficiency, and better state-action value representation.
- **DoubleDQN-PER-Dueling-Noisy**: Adds NoisyNets to promote directed exploration and adaptability in uncertain environments.
- **DoubleDQN-PER-Dueling-NROWAN**: Integrates NROWAN for enhanced stability and exploration in complex conditions.
- **Hierarchical (HRL) Architectures**: Employ a high-level controller that manages low-level DQN variants, inspired by modularity research. These hierarchical setups aim to improve robustness by decomposing tasks and leveraging multiple architectures simultaneously.

## Noise Types and Protocol
1. **Environmental Noise (Wind)**:  
   A horizontal force applied at the start of training (e.g., force=0.2) and annealed to zero over time.
   
2. **Sensor Noise (Gaussian)**:  
   Gaussian noise applied to state observations (e.g., std=0.2 initially) and then gradually reduced to zero.

3. **Action Noise (Engine Failure)**:  
   A certain probability (e.g., 30%) that chosen actions fail initially, decreasing to zero as training progresses.

Noise is fully present in the initial episodes (e.g., first 25 episodes for CartPole-v1, first 150 episodes for LunarLander-v3) and linearly decreases over a specified duration (e.g., up to episode 125 for CartPole-v1, 350 for LunarLander-v3). After noise returns to zero, training continues without noise, allowing us to assess the agent’s retention of noise-handling strategies.

## Setup and Requirements
- **Python version**: ≥3.8 recommended.
- **Dependencies**: Listed in `requirements.txt`.  
  Key libraries include:
  - `gymnasium` for environment simulation.
  - `torch` for neural network training.
  - `matplotlib`, `pandas`, `openpyxl`, `tensorboard` for analysis and logging.
  - `box2d-py` for LunarLander environment.
  
To install all dependencies:
```bash
pip install -r requirements.txt
```

## Directory Structure
- `main.py`: Entry point for running experiments.
- `runner.py`: Automates training and validation runs across architectures and environments.
- `shared/`: Shared utilities, including HRL manager, replay buffer, abstract policy, and noise scaling functions.
- `architectures/`: Contains different architectures (DQN, DoubleDQN, DoubleDQNPER, DoubleDQNPERDueling, DQNNoisy, DQNNROWAN, DoubleDQNPERDuelingNoisy, DoubleDQNPERDuelingNROWAN).
- `environments/`: Noise injection wrappers for gym environments.
- `configs/`: Configuration files for each architecture and mode (high-level or low-level).
- `results/`: Stores generated `training_log.json` files and trained model checkpoints.

## Running Experiments
1. **Single Run (Non-HRL)**:
   For example, to train a DQN agent in CartPole-v1 with noise on:
   ```bash
   python main.py --use_hrl no --low_arch DQN --env CartPole-v1 --noise on --mode train
   ```
   After training completes, validate the model:
   ```bash
   python main.py --use_hrl no --low_arch DQN --env CartPole-v1 --noise on --mode validate
   ```

2. **Single Run (HRL)**:
   For a hierarchical architecture, specify `--use_hrl yes` and provide the `--high_arch` parameter:
   ```bash
   python main.py --use_hrl yes --high_arch DoubleDQNPERDuelingNoisy --low_arch DQN --env CartPole-v1 --noise on --mode train
   ```
   Then validate:
   ```bash
   python main.py --use_hrl yes --high_arch DoubleDQNPERDuelingNoisy --low_arch DQN --env CartPole-v1 --noise on --mode validate
   ```

3. **Batch Runs**:
   Use `runner.py` to automate training and validation runs for multiple architectures and configurations:
   ```bash
   python runner.py
   ```

## Analysing Results
- The `results/` directory will contain multiple folders for each configuration (use_hrl, high_arch, low_arch, env, noise, mode).
- Each folder includes `training_log.json` with episode-wise data: score, noise levels, loss, and more.
- Running `runner.py` ends with a summarised `results_summary.xlsx` file that consolidates metrics across all runs.  
- Plots and quadrant analyses (not generated by default) can be created from the logs in `results_summary.xlsx`.

## Interpretation
- **Episodic Noise Exposure**: The project tests an agent’s ability to “learn once and remember,” as noise is only present early on and then removed. Models that retain noise-handling strategies will perform better during validation.
- **Architectural Complexity vs. Efficiency**: Some architectures learn robustness but at the cost of longer training times or architectural complexity. Others solve quickly but fail to retain noise resiliency.
- **Quadrant Analysis (Figures 5 and 10)**: A scatter plot correlating training efficiency (episodes to solve) with robustness (validation average reward) categorises architectures into four quadrants. The ideal architectures are both robust and efficient, but many trade-offs exist.

## Future Work
- **Adaptive Architectures**: Dynamically adjusting architectural complexity or exploration parameters based on noise patterns could yield robust and efficient learning.
- **Broader Algorithmic Variations**: Extending beyond DQN-based methods to actor-critic frameworks or distributional RL could offer new insights.
- **Realistic, Non-Stationary Noise**: Testing agents in more complex environments and diverse noise scenarios, potentially including curriculum learning or reward shaping, might lead to more generally applicable solutions.

## Acknowledgements
This work builds upon multiple RL and machine learning concepts:  
- Noisy Networks: Fortunato et al. [1]  
- Modularity in RL: Najarro & Risi [3]  
- Prioritised Experience Replay: Schaul et al. [9]  
- Dueling Networks: Wang et al. [10]

For detailed references, see the main report or the references section in the `main.py`.

## Contact
For questions or collaborations, please contact the repository maintainer.
