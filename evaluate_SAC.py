import gymnasium as gym
import gym_xarm  # make sure this is registered

from stable_baselines3 import SAC

model = SAC.load("sac_xarm_lift")

env = gym.make("gym_xarm/XarmLift-v0", render_mode="human")
obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
