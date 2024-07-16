# Learning Forward Dynamics of a Robotic Arm Using Reinforcement Learning Algorithms

## Objective
This project involves developing advanced models for learning the forward dynamics of a robotic arm using reinforcement learning algorithms, specifically Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO). The objective is to accurately predict the future states of a 2-link robotic arm based on applied torques and current states.

## Project Structure
- **dqn/**: Contains source code for the DQN model.
  - `dqn_architecture.py`: Defines the Q-network architecture.
  - `dqn_train.py`: Script to train the DQN model.
  - `replay_buffer.py`: Implements the replay buffer for experience replay.
- **ppo/**: Contains source code for the PPO model.
  - `parallel_env.py`: Implements parallel environments to speed up training.
  - `ppo_network.py`: Script to train the PPO model using Stable-Baselines3.

## Setup Instructions
1. Clone the repository:
  ```sh
  git clone https://github.com/username/robotic-arm-dynamics-rl.git
  ```
2. Navigate to the project directory:
  ```sh
  cd dqn-ppo
  ```
