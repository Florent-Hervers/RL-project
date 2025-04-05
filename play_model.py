import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation

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
model = RecurrentPPO.load("trained_models/q2_final.zip")

# Source : https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html
obs = env.reset()
episode_over = False
rewards = []
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)

while not episode_over:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, reward, dones, info = env.step(action)
    rewards.append(reward)
    # Update the display with the chosen action
    env.render("human")
    episode_over = dones

# Print the total reward
print(f"Total reward: {sum(rewards)}")

# Close the environment
env.close()