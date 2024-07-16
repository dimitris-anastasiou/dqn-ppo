# PPO Network

from stable_baselines3.ppo import PPO
import os
import time
from stable_baselines3.common.utils import set_random_seed


# Default parameters
timesteps = 500000
nenv = 8  # number of parallel environments. This can speed up training when you have good CPUs
seed = 8
batch_size = 2048

# Generate path of the directory to save the checkpoint
timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join('ppo_models', timestr)

# Set random seed
set_random_seed(seed)

# Create arm
arm = make_arm()

# Create parallel envs
vec_env = make_vec_env(arm=arm, nenv=nenv, seed=seed)

# Model Architecture
model = PPO("MlpPolicy", env=vec_env, verbose=1, tensorboard_log="./ppo_tensorboard/",
    n_steps=batch_size, seed=seed)

# Train the model
model.learn(total_timesteps=timesteps)

# Save the model
model_path = os.path.join(save_dir, "ppo_network.zip")
model.save(model_path)
print(f"Model saved to {model_path}")
