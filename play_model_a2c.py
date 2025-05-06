import math
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import yaml
import time
from torch.distributions.normal import Normal
from argparse import ArgumentParser, BooleanOptionalAction
from tqdm import trange

parser = ArgumentParser()

parser.add_argument("--human", default=False, action=BooleanOptionalAction, help="If true, display the run on the screen.")
parser.add_argument("-c", "--config", type=int, required=True, help="The number of the config to evaluate")
parser.add_argument("--evaluate", default=False, action=BooleanOptionalAction, help="If true, rerun the model 10 times to evaluate the model performances")

args = parser.parse_args()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_env(config, seed=None):
    def thunk():
        env = gym.make(
            "CarRacing-v3",
            render_mode="human" if args.human else "rgb_array",
            lap_complete_percent=0.95,
            domain_randomize=False,
            continuous=True,
            max_episode_steps=12000,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ResizeObservation(env, (config["OBSERVATION_SIZE"], config["OBSERVATION_SIZE"]))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, config["NB_FRAMES"])
        if seed is not None:
            env.reset(seed=seed)
        return env
    return thunk

if args.config <= 7:
    class Agent(nn.Module):
        def __init__(self, envs, nb_frames, image_size, config=None):
            super(Agent, self).__init__()
            self.image_size = image_size
            self.nb_frames = nb_frames

            # Actor network
            self.actor_network = self.build_network()
            self.actor_mean = layer_init(
                nn.Linear(512, np.prod(envs.single_action_space.shape)),
                std=0.01
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

            # Critic network
            self.critic_network = self.build_network()
            self.critic = layer_init(nn.Linear(512, 1), std=1)

        def build_network(self):
            stride = [4, 2, 1]
            kernel_size = [8, 4, 3]
            input_channels = [self.nb_frames, 32, 64]
            output_channels = [32, 64, 64]
            image_size = self.image_size

            layers = []
            for i in range(len(stride)):
                layers.append(layer_init(nn.Conv2d(input_channels[i], output_channels[i], kernel_size[i], stride=stride[i])))
                layers.append(nn.ReLU())
                image_size = math.floor(((image_size - kernel_size[i]) / stride[i]) + 1)

            layers.append(nn.Flatten())
            layers.append(layer_init(nn.Linear(output_channels[-1] * image_size * image_size, 512)))
            layers.append(nn.ReLU())
            
            return nn.Sequential(*layers)

        def get_value(self, x):
            hidden = self.critic_network(x / 255.0)
            return self.critic(hidden)

        def get_action_and_value(self, x, action=None):
            actor_hidden = self.actor_network(x / 255.0)
            action_mean = self.actor_mean(actor_hidden)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            
            critic_hidden = self.critic_network(x / 255.0)
            value = self.critic(critic_hidden)

            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

else:
    class Agent(nn.Module):
        def __init__(self, envs, nb_frames, image_size, config=None):
            super(Agent, self).__init__()
            self.image_size = image_size
            self.nb_frames = nb_frames

            # Actor network
        # Actor: CNN → LSTM → Linear
            self.actor_cnn, self.actor_lstm, _ = self.build_network(use_lstm=True, config=config)
            self.actor_linear = layer_init(nn.Linear(config["LSTM_HIDDEN_SIZE"], np.prod(envs.single_action_space.shape)), std=0.01)
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

            # Critic: CNN → Linear
            self.critic_cnn, _, self.critic_linear = self.build_network(use_lstm=False)
            self.critic = layer_init(nn.Linear(512, 1), std=1)


        def build_network(self, use_lstm=False, config=None):
            stride = [4, 2, 1]
            kernel_size = [8, 4, 3]
            input_channels = [self.nb_frames, 32, 64]
            output_channels = [32, 64, 64]
            image_size = self.image_size

            cnn_layers = []
            for i in range(len(stride)):
                cnn_layers.append(layer_init(
                    nn.Conv2d(input_channels[i], output_channels[i], kernel_size[i], stride=stride[i])
                ))
                cnn_layers.append(nn.Tanh())
                image_size = math.floor(((image_size - kernel_size[i]) / stride[i]) + 1)

            cnn_layers.append(nn.Flatten())
            cnn = nn.Sequential(*cnn_layers)

            # Linear input size = 64 * image_size^2
            linear_input_size = output_channels[-1] * image_size * image_size

            if use_lstm:
                lstm = nn.LSTM(input_size=linear_input_size, hidden_size=config["LSTM_HIDDEN_SIZE"], batch_first=True, num_layers=config["LSTM_LAYERS"])
                linear = None  # handled after LSTM
            else:
                lstm = None
                linear = nn.Sequential(
                    layer_init(nn.Linear(linear_input_size, 512)),
                    nn.Tanh()
                )

            return cnn, lstm, linear


        def get_value(self, x):
            x = x / 255.0
            cnn_out = self.critic_cnn(x)              
            linear_out = self.critic_linear(cnn_out)   
            value = self.critic(linear_out)            
            return value
        
        def get_action_and_value(self, x, action=None):
            x = x / 255.0

            # Actor pipeline: CNN → LSTM → Linear
            cnn_out = self.actor_cnn(x)               # (B, flat_dim)
            lstm_in = cnn_out.unsqueeze(1)            # (B, 1, flat_dim)
            lstm_out, _ = self.actor_lstm(lstm_in)    # (B, 1, 512)
            actor_hidden = lstm_out.squeeze(1)        # (B, 512)
            action_mean = self.actor_linear(actor_hidden)

            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()

            # Critic pipeline: CNN → Linear → Value
            critic_hidden = self.critic_linear(self.critic_cnn(x))
            value = self.critic(critic_hidden)

            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value
        
try:
    with open(f"configs/Q3/config{args.config}.yaml", "r") as f:
        raw_config = yaml.safe_load(f)
        config = {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in raw_config.items()}
except Exception as e:
    raise Exception(f"The following exceptions occured during the opening of the config file: \n{e}")

random.seed(config["SEED"])
np.random.seed(config["SEED"])
torch.manual_seed(config["SEED"])
torch.backends.cudnn.deterministic = True

if args.evaluate:
    n_episodes = 10
    seed = 1
else:
    n_episodes = 1
    seed = int(time.time())

all_rewards = []

for i in trange(n_episodes, desc="Episodes"):
        env = gym.vector.SyncVectorEnv([make_env(config, seed=seed+i)])

        device = torch.device("cuda" if torch.cuda.is_available() and config["CUDA"] else "cpu")

        agent = Agent(env, config["NB_FRAMES"], config["OBSERVATION_SIZE"], config).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=config["LEARNING_RATE"], eps=1e-5)

        checkpoint = torch.load(f"trained_models/a2c/a2c_config{args.config}.pt", map_location=device)
        agent.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device)

        done = False
        episode_reward = 0.0

        while not done:
            action, _, _, _ = agent.get_action_and_value(obs, action=None)
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
            episode_reward += reward[0]
            done = terminated[0] or truncated[0]

        all_rewards.append(episode_reward)
        env.close()

mean_reward = np.mean(all_rewards)
std_reward = np.std(all_rewards)

if args.evaluate:
    with open(f"Results/Q3/eval_results_config{args.config}.txt", "w") as f:
        f.write("=== Results summary ===\n")
        for i, reward in enumerate(all_rewards):
            f.write(f"Track {i+1}: Mean reward = {reward.mean():.2f}\n")
        f.write("\nSummary over all tracks:\n")
        f.write(f"Mean reward: {mean_reward:.2f}\n")
        f.write(f"Reward std: {std_reward:.2f}\n")
        f.write(f"Min: {np.min(all_rewards):.2f}\n")
        f.write(f"Max: {np.max(all_rewards):.2f}\n")