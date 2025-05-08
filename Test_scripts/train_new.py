import gymnasium as gym
import numpy as np
import cv2
import imageio
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback
import yaml
from argparse import ArgumentParser, BooleanOptionalAction
import os

def save_video_or_gif(frames, output_path, is_gif=False):
    if is_gif:
        # Enregistrement en tant que GIF
        with imageio.get_writer(output_path, mode='I', duration=0.05) as writer:
            for frame in frames:
                writer.append_data(frame)
    else:
        # Enregistrement en tant que vidéo
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec vidéo
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()

class CustomWandbCallback(WandbCallback):
    def __init__(self, save_every_n_steps, log_video=True, **kwargs):
        super().__init__(**kwargs)
        self.save_every_n_steps = save_every_n_steps
        self.step_counter = 0
        self.frames = []
        self.log_video = log_video

    def _on_step(self) -> bool:
        self.step_counter += 1
        # Capture the frame from the environment
        if self.log_video:
            obs = self.training_env.render(mode='rgb_array')  # Get the current frame from the environment
            self.frames.append(obs)

            # Save video every n steps
            if self.step_counter % self.save_every_n_steps == 0 and len(self.frames) > 0:
                video_path = f"gif/car_racing_video_step_{self.step_counter}.mp4"
                save_video_or_gif(self.frames, video_path, is_gif=False)  # Save as video
                wandb.log({"video": wandb.Video(video_path)})  # Log video to wandb
                self.frames.clear()  # Clear frames after saving

        return True

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-c", "--config", required=True, type=str, help="Filename of the configuration to use (without the .yaml extension)")
    parser.add_argument("--config_path", default="configs/Q2", type=str, help="Relative path to the config directory.")
    parser.add_argument("--wandb", "-w", required=True, type=str, help="Name of the run in wandb")
    parser.add_argument("-o", "--output", default="trained_models/q2_final", type=str, help="Relative path + filename of the trained model")
    parser.add_argument("--save_model", required=True, action=BooleanOptionalAction, help="This flag can be set to false using --no-save_model. If True, save the model in the file defined by the output argument.")
    parser.add_argument("--save_video", action=BooleanOptionalAction, help="This flag saves a video or GIF of the car's actions")

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

    custom_callback = CustomWandbCallback(save_every_n_steps=hyperparams["n_steps"], log_video=args.save_video)


    model.learn(
        total_timesteps=total_timesteps,
        callback=custom_callback,
    )

    wandb.finish()

    if args.save_model:
        try:
            model.save(args.output)
        except Exception as e:
            model.save("backup")
            raise Exception(f"The following error occurs when saving the model: {e.args}")