import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation

def make_env(seed=None):
    def _init():
        env = gym.make("CarRacing-v3", continuous=True, lap_complete_percent=0.95, domain_randomize=False, render_mode="rgb_array")
        env = ResizeObservation(env, (64, 64))
        env = GrayscaleObservation(env, keep_dim=True)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init

# Chargement du modèle
model = RecurrentPPO.load("trained_models/q2_config6.zip")

# Paramètres d'évaluation
n_episodes = 10
all_rewards = []

for i in range(n_episodes):
    print(f"\n=== Episode {i+1}/{n_episodes} ===")
    env = DummyVecEnv([make_env(seed=i)])
    env = VecFrameStack(env, n_stack=1)
    env = VecNormalize(env, norm_reward=True, norm_obs=False)
    obs = env.reset()

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    episode_rewards = []

    done = False
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_rewards.append(reward)
        episode_starts = done
        # Commenter si pas besoin d'affichage
        env.render("human")
        done = done[0]

    total_reward = sum(episode_rewards)
    print(f"Reward: {total_reward}")
    all_rewards.append(total_reward)
    env.close()

mean_reward = np.mean(all_rewards)
std_reward = np.std(all_rewards)

with open("Results/eval_results_config6.txt", "w") as f:
    f.write("=== Results summary ===\n")
    for i, reward in enumerate(all_rewards):
        f.write(f"Track {i+1}: Mean reward = {reward.mean():.2f}\n")
    f.write("\nSummary over all tracks:\n")
    f.write(f"Mean reward: {mean_reward:.2f}\n")
    f.write(f"Reward std: {std_reward:.2f}\n")
    f.write(f"Min: {np.min(all_rewards):.2f}\n")
    f.write(f"Max: {np.max(all_rewards):.2f}\n")