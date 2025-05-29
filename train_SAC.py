"""
In this script, Soft Actor Critic is implemented to train a RL Agent to perform the lift task in gym xarm
"""

import gymnasium as gym
import gym_xarm  

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# Create and wrap the environment
env = gym.make("gym_xarm/XarmLift-v0")

# Wrap for SB3 compatibility (only if needed)
#env = DummyVecEnv([lambda: env])

# Create the SAC model
#model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./xarm_sac_logs")
model = SAC("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=1000000)

# Save the model
model.save("sac_xarm_lift")

"""
STATISTICS for training attempts with Soft Actor Critic:
- 100k total_timesteps: insufficient for the success
- 300k total_timesteps: insufficient for the success
- 1M total_timesteps:
""" 