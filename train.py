import gymnasium as gym
from sb3_contrib import RecurrentPPO  # PPO récurrent (LSTM)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback
import yaml

# Référence pour le dictionnaire d'hyperparamètre et des transformations:
# https://huggingface.co/sb3/ppo_lstm-CarRacing-v0

"""
DANS LE DICT IL Y A ENCORE

https://rl-baselines3-zoo.readthedocs.io/en/master/guide/config.html
https://stable-baselines3.readthedocs.io/en/master/common/logger.html => episodic len / episodic rew

('learning_rate', 'lin_1e-4') => decrease linéaire du lr je suppose (cfr TP5)
('normalize', "{'norm_obs': False, 'norm_reward': True}") => Normalization des reward mais pas des observations
('normalize_kwargs', {'norm_obs': False, 'norm_reward': False})] => à investiguer, c'est chelou car par cohérent avec le champ précédent
"""

# Charger la configuration YAML depuis "config.yaml"
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extraire les hyperparamètres et la configuration de la policy
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

# Convertir la chaîne de caractère de activation_fn en fonction réelle
if "activation_fn" in policy_kwargs:
    activation_str = policy_kwargs["activation_fn"]
    policy_kwargs["activation_fn"] = getattr(nn, activation_str)

# Ajouter policy_kwargs dans hyperparams pour le passer au modèle
hyperparams["policy_kwargs"] = policy_kwargs

# Initialisation de wandb
run = wandb.init(
    entity="Rl2025-project",
    project="RL Project",
    name="First_full_test_with_yaml_config",
    config=hyperparams,
    sync_tensorboard=True,  # auto-upload des métriques tensorboard de sb3
    monitor_gym=True,       # auto-upload des vidéos de l'agent
    # save_code=True,       # optionnel
)

# Fonction de création d'environnement
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

print("Doing PPOLSTM")
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
        gradient_save_freq=100,
        verbose=2,
    ),
)

wandb.finish()
# Sauvegarde du modèle final
model.save("q2_final")