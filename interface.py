import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from load_agent import load_best_agent
import torch

# load agent
env, model, observation = load_best_agent('Q2', seed=42)

# Run the agent in the environment
lstm_states = None
episode_over = False
rewards = []
while not episode_over:
    action, lstm_states = model.predict(observation, state=lstm_states, deterministic=True)
    observation, reward, done, info = env.step(action)
    rewards.append(reward)
    episode_over = done

# Print the total reward
print(f"Total reward: {sum(rewards)}")

# Close the environment
env.close()

plt.plot(np.arange(len(rewards)), rewards)
plt.xlabel("Time step")
plt.ylabel("Reward")
plt.title("Reward over time")
plt.show()

# load agent
env, model, observation, device = load_best_agent('Q3', seed=42)

# Run the agent in the environment
episode_over = False
rewards = []
while not episode_over:
    action, _, _, _ = model.get_action_and_value(observation, action=None)
    observation, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
    observation = torch.tensor(observation, dtype=torch.float32, device=device)
    rewards.append(reward)
    episode_over = terminated[0] or truncated[0]

# Print the total reward
print(f"Total reward: {sum(rewards)}")

# Close the environment
env.close()

plt.plot(np.arange(len(rewards)), rewards)
plt.xlabel("Time step")
plt.ylabel("Reward")
plt.title("Reward over time")
plt.show()
