# RL Project
Code repository for the project of Reinforcement Learning course at ULiege.

## Environement setup
As requested in the statement, the python environement can be set up using the following commands:
```bash
conda create --name car_racing_env python=3.10
conda activate car_racing_env
```
Once the conda environement using python 3.10 is set up, the required dependencies can be installed using
```bash
pip install -r requirements.txt
```

## Running a run to compare PPO-LSTM and A2C
As requested, it is also possible to run `interface.py` to load the best model from Question 2 and evaluate its performance on a single track. Then, the best model from Question 3 is run on the same track. This directly compares the two best models from the two questions on the same track. Simply run the following command in the terminal:

```bash
python interface.py
```

# Extra script definitions
The following section describe the script used to train, see the model in action and evaluate the trained models. All trained models are available on the [github](https://github.com/Florent-Hervers/RL-project). To not overload the submission archive only the best models were provided: the config number that yield those models are:
- Question 2 (PPO-LSTM model): 13
- Question 3 (A2C model): 12

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
python train.py -c config13 --wandb "Train best model" --save_model
```

### Configurations descriptions:
- The three first configs were test configs used to test the implementation and learn how the model behave due to changes in hyperparameters.
- Configuration 4 and 5 are the one use to evaluate the max_episode_length hyperparameter value impact. Configuration 4 use 3000 steps while configuration 5 use 12000 steps. To make the comparaison fair, all other hyperparameters weren't changed.
- For the N_STEP hyperparameter: the config 5 and 7 are the configuration with a N_STEP of 258 and the config 12 and 13 are the one with the same hyperparameters execpt the N_STEP set to 2048. Config 8 and 9 also illustrate the difference in N_STEP but as both models are very bad, there is nothing to conclude as the models are very very bad.
- For the N_ENV hyperparameter: the configs 6, 11, 12 and 13 use the same set of hyperparmeters but with a N_ENVS set respectively to 1, 4, 8 and 12.
- For the GAMMA hyperparameter: the configs 14 (GAMMA = 0.9975) and 15 (GAMMA = 0.98) tried to modify the model based on the configuration 13.
- For the FRAME_STACK hyperparameter: the configs 16, 17 and 18 tried to add more FRAME_STACK using the same hyperparameters than for config 13. Config 8 also tried to add frame stacks from config 6 but as the model resulting from config 8 is garbage, we can't deduce anything from it.

## Evaluating PPO-LSTM for Question 2
Our results in `Results/Q2` have been computed by running the *play_model.py* file. We decided to run the model on 10 different tracks to correctly assess its global performance. Simply run the following command in the terminal:

```bash
python play_model.py -c 13 --evaluate --human
```

### Required Arguments

- `-c`, `--config` (int): **Required**. The number of the config to evaluate (should be between 1 and 18). If the config 13 was selectet, the model should be in the `trained_models/Q2/q2_config13.zip` file.

### Optional Arguments

- `--human` (Boolean): If true, display the run on the screen.
- `--evaluate` (Boolean): If true, rerun the model 10 times to evaluate the model performances. The summary of the results on the evaluations tracks for every configuration can be found in the folder `Results/Q2`. Use `--no-evaluate` to run the model on a single random map.

## Training A2C using our jupyter notebook for Question 3

For Question 3, we chose to implement our solution in a Jupyter notebook. To re-run the experiment, simply navigate to the dedicated cell containing the hyperparameters and modify them accordingly. The parameters used for every models are described in the `configs/Q3` folder.

## Evaluating A2C for Question 3
Our results in `Results/Q3` have been computed by running the *play_model_a2c.py* file. We decided to run the model on 10 different tracks to correctly assess its global performance. Simply run the following command in the terminal:

```bash
python play_model_a2c.py -c 12 --evaluate --human
```

### Required Arguments

- `-c`, `--config` (int): **Required**. The number of the config to evaluate (should be between 1 and 14). If the config 12 was selectet, the model should be in the `trained_models/Q2/a2c_config12.pt` file.

### Optional Arguments

- `--human` (Boolean): If true, display the run on the screen.
- `--evaluate` (Boolean): If true, rerun the model 10 times to evaluate the model performances. The summary of the results on the evaluations tracks for every configuration can be found in the folder `Results/Q2`. Use `--no-evaluate` to run the model on a single random map
