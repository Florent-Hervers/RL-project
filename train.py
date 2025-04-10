import gymnasium as gym
from sb3_contrib import RecurrentPPO  # PPO récurrent (LSTM)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback
import yaml
from argparse import ArgumentParser, BooleanOptionalAction
import os

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-c", "--config", required=True, type=str, help="Filename of the configuration to use (without the .yaml extension)")
    parser.add_argument("--config_path", default="configs/Q2", type=str, help="Relative path to the config directory.")
    parser.add_argument("--wandb", "-w", required=True, type=str, help="Name of the run in wandb")
    parser.add_argument("-o", "--output", default="trained_models/q2_final", type=str, help="Relative path + filename of the trained model")
    parser.add_argument("--save_model", required=True, action=BooleanOptionalAction, help="This flag can be set to false using --no-save_model. If True, save the model in the file defined by the output argument.")
    
    args = parser.parse_args()

    try:
        with open(os.path.join(args.config_path, args.config + ".yaml"), "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"The following exceptions occured during the opening of the config file: {e.args}")

    hyperparams = config["hyperparams"]
    policy_kwargs = config["policy_kwargs"]
    transformation = config["transformation"]
    total_timesteps = config["total_timesteps"]

    resize_obs_shape = tuple(transformation.get("resize_observation", None))
    grayscale_params = transformation.get("grayscale_observation", None)
    n_envs = transformation.get("n_envs", 1)
    frame_stack = transformation.get("frame_stack", 1)
    normalize_params = transformation.get("normalize", None)

    # Si learning_rate est défini comme un dictionnaire dans le YAML,
    # on crée une fonction de planification linéaire
    if isinstance(hyperparams.get("learning_rate"), dict):
        initial_lr = hyperparams["learning_rate"].get("initial", 1e-4)
        def linear_schedule(progress_remaining):
            return progress_remaining * initial_lr
        hyperparams["learning_rate"] = linear_schedule

    if "activation_fn" in policy_kwargs:
        activation_str = policy_kwargs["activation_fn"]
        policy_kwargs["activation_fn"] = getattr(nn, activation_str)

    hyperparams["policy_kwargs"] = policy_kwargs

    run = wandb.init(
        entity="Rl2025-project",
        project="RL Project",
        name= args.wandb,
        config=hyperparams,
        sync_tensorboard=True,  
        monitor_gym=True,       
    )

    def make_env():
        def _init():
            env = gym.make(
                "CarRacing-v3",
                continuous=True,
                lap_complete_percent=0.95,
                domain_randomize=False,
                render_mode="rgb_array"
            )
            if resize_obs_shape is not None:
                env = ResizeObservation(env, resize_obs_shape)
            if grayscale_params is not None:
                env = GrayscaleObservation(env, **grayscale_params)
            env = Monitor(env)
            return env
        return _init

    print("Loading CarRacing-v3 environment")
    env = DummyVecEnv([make_env() for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=frame_stack)
    if normalize_params is not None:
        env = VecNormalize(env, **normalize_params)

    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        verbose=1,
        stats_window_size=1,
        tensorboard_log=f"runs/{run.id}",
        **hyperparams
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=WandbCallback(
            verbose=2,
        ),
    )

    wandb.finish()

    if args.save_model:
        try:
            model.save(args.output)
        except Exception as e:
            model.save("backup")
            raise Exception(f"The following error occurs when saving the model: {e.args}")