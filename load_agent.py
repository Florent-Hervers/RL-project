# Place your imports below
import gymnasium as gym
import yaml
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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

def load_best_agent(question, seed=None):
    """
    Load the best agent from the specified path.

    This function is designed to load your pre-trained agent model so that it can be used to
    interact with the environment. Follow these steps to implement the function:

    1) Choose the right library for model loading:
       - Depending on whether you used PyTorch, TensorFlow, or another framework to train your model,
         import the corresponding library (e.g., `import torch` or `import tensorflow as tf`).

    2) Specify the correct file path:
       - Define the path where your trained model is saved.
       - Ensure the path is correct and accessible from your script.

    3) Load the model:
       - Use the appropriate loading function from your library.
         For example, with PyTorch you might use:
           ```python
           model = torch.load('path/to/your_model.pth')
           ```

    4) Ensure the model is callable:
       - The loaded model should be usable like a function. When you call:
           ```python
           action = model(observation)
           ```
         it should output an action based on the input observation.

    Returns:
        model: The loaded model. It must be callable so that when you pass an observation to it,
               it returns the corresponding action.

    Example usage:
        >>> model = load_best_agent()
        >>> observation = get_current_observation()  # Your method to fetch the current observation.
        >>> action = model(observation)
    """
    assert question in ['Q2', 'Q3'], "Invalid question. Choose from ['Q2', 'Q3']."

    if question == 'Q2':
        # Load the model for Q2
        def make_env(seed=None):
            def _init():
                env = gym.make("CarRacing-v3", continuous=True, lap_complete_percent=0.95, domain_randomize=False, render_mode="human", max_episode_steps = 12000)
                env = ResizeObservation(env, (64, 64))
                env = GrayscaleObservation(env, keep_dim=True)
                if seed is not None:
                    env.reset(seed=seed)
                return env
            return _init
        
        model = RecurrentPPO.load(f"trained_models/Q2/q2_config13.zip")

        try:
            with open(f"configs/Q2/config13.yaml", "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"The following exceptions occured during the opening of the config file: \n{e}")
        
        env = DummyVecEnv([make_env(seed=seed)])
        env = VecFrameStack(env, n_stack=config["transformation"].get("frame_stack", 1))

        obs = env.reset()
      
        return env, model, obs
    elif question == 'Q3':
        # Load the model for Q3
        def make_env(config, seed=None):
            def thunk():
                env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True, max_episode_steps=12000)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                env = gym.wrappers.ResizeObservation(env, (config["OBSERVATION_SIZE"], config["OBSERVATION_SIZE"]))
                env = gym.wrappers.GrayscaleObservation(env)
                env = gym.wrappers.FrameStackObservation(env, config["NB_FRAMES"])
                if seed is not None:
                    env.reset(seed=seed)
                return env
            return thunk
        
        try:
            with open(f"configs/Q3/config12.yaml", "r") as f:
                raw_config = yaml.safe_load(f)
                config = {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in raw_config.items()}
        except Exception as e:
            raise Exception(f"The following exceptions occured during the opening of the config file: \n{e}")
        
        env = gym.vector.SyncVectorEnv([make_env(config, seed=seed)])

        device = torch.device("cuda" if torch.cuda.is_available() and config["CUDA"] else "cpu")

        agent = Agent(env, config["NB_FRAMES"], config["OBSERVATION_SIZE"], config).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=config["LEARNING_RATE"], eps=1e-5)

        checkpoint = torch.load(f"trained_models/Q3/a2c_config12.pt", map_location=device)
        agent.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device)

        return env, agent, obs, device
