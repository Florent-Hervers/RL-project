import time
import yaml
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser, BooleanOptionalAction
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation


parser = ArgumentParser()

parser.add_argument("--human", default=True, action=BooleanOptionalAction, help="If true, display the run on the screen.")
parser.add_argument("-c", "--config", type=int, required=True, help="The number of the config to evaluate")
parser.add_argument("--evaluate", default=False, action=BooleanOptionalAction, help="If true, rerun the model 10 times to evaluate the model performances")

args = parser.parse_args()

def make_env(seed=None):
    def _init():
        env = gym.make("CarRacing-v3", continuous=True, lap_complete_percent=0.95, domain_randomize=False, render_mode="rgb_array", max_episode_steps = 12000)
        env = ResizeObservation(env, (64, 64))
        env = GrayscaleObservation(env, keep_dim=True)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init

# Chargement du modèle
model = RecurrentPPO.load(f"trained_models/q2_config{args.config}.zip")

try:
    with open(f"configs/Q2/config{args.config}.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    raise Exception(f"The following exceptions occured during the opening of the config file: \n{e}")

# Paramètres d'évaluation
if args.evaluate:
    n_episodes = 10
    seed = 1
else:
    n_episodes = 1
    seed = int(time.time())
all_rewards = []

for i in range(n_episodes):
    if args.evaluate:
        print(f"\n=== Episode {i+1}/{n_episodes} ===")
    env = DummyVecEnv([make_env(seed=seed + i)])
    env = VecFrameStack(env, n_stack=config["transformation"].get("frame_stack", 1))
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
        if args.human:
            env.render("human")
        done = done[0]

    total_reward = sum(episode_rewards)
    print(f"Reward: {total_reward}")
    all_rewards.append(total_reward)
    env.close()

mean_reward = np.mean(all_rewards)
std_reward = np.std(all_rewards)

if args.evaluate:
    with open(f"Results/eval_results_config{args.config}.txt", "w") as f:
        f.write("=== Results summary ===\n")
        for i, reward in enumerate(all_rewards):
            f.write(f"Track {i+1}: Mean reward = {reward.mean():.2f}\n")
        f.write("\nSummary over all tracks:\n")
        f.write(f"Mean reward: {mean_reward:.2f}\n")
        f.write(f"Reward std: {std_reward:.2f}\n")
        f.write(f"Min: {np.min(all_rewards):.2f}\n")
        f.write(f"Max: {np.max(all_rewards):.2f}\n")