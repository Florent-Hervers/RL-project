# RL Project
Code repository for the project of Reinforcement Learning course at ULiege.

## Training PPO-LSTM for Question 2 (External Library Algorithm)

To train a PPO-LSTM model using one of our predefined configurations, simply run the following command in a terminal:

```bash
python train.py -c CONFIG_NAME --wandb RUN_NAME --save_model
```

### Required Arguments

- `-c`, `--config` (str): **Required**. Filename of the configuration to use (**without** the `.yaml` extension).
- `--wandb`, `-w` (str): **Required**. Name of the run in Weights & Biases (wandb).
- `--save_model` / `--no-save_model`: **Required**. Flag indicating whether to save the model. Use `--no-save_model` to disable saving.

### Optional Arguments

- `--config_path` (str): Relative path to the config directory. Default is `configs/Q2`.
- `-o`, `--output` (str): Relative path + filename of the trained model. Default is `trained_models/q2_final`.

### Example

```bash
python train.py -c my_config --wandb my_experiment --save_model
```

## Training A2C using our jupyter notebook for Question 3
