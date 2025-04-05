import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation

# Load the trained agent
env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True)

env = ResizeObservation(env, (64, 64))
env = GrayscaleObservation(env, keep_dim=True)

def make_env():
    def _init():
        env = gym.make("CarRacing-v3", continuous=True, lap_complete_percent=0.95, domain_randomize=False, render_mode="rgb_array") # entrainement sur "rgb-array" car plus opti en terme de compute, le mode human display qqch
        env = ResizeObservation(env, (64, 64))
        env = GrayscaleObservation(env, keep_dim=True)
        return env
    return _init

env = DummyVecEnv([make_env()])
env = VecFrameStack(env, n_stack=2)
env = VecNormalize(env, norm_reward=True, norm_obs=False)

# load agent
model = RecurrentPPO.load("q2_final.zip")

# Enjoy trained agent
vec_env  = env
obs = vec_env.reset()
episode_over = False
rewards = []
while not episode_over:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, dones, info = vec_env.step(action)
    rewards.append(reward)
    vec_env.render("human")
    episode_over = dones

# Print the total reward
print(f"Total reward: {sum(rewards)}")

# Close the environment
env.close()

plt.plot(np.arange(len(rewards)), rewards)
plt.xlabel("Time step")
plt.ylabel("Reward")
plt.title("Reward over time")
plt.show()
