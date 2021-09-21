# Training script for self-play using Stable baselines3
# Based on: https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_selfplay.py

# This script is used to:
# - Self-play training between agents
# - The agents are initialized with a policy
# - The policy of the opponent is being selected to be the latest model if exists if not then a random policy (Sampling from the action space)
# - The training is starting to train the first agent for several epochs then the second agent
# - The model is being saved in the local directory

# Train -> Evaluate -> calculate win-rate -> save (history_<num-round>_<reward/points/winrate>_m_<value>_s_<num-step>)

################################################################
# Hirarichey of that script in the whole project in point of view of wandb:
# Behavioral-Learning-Thesis:Self-Play:2D:evorobotpy2:predprey:1v1

################################################################
# In self-play:
# We have several major aspects:
# 1. How to sample the agents?  (1st player)    -> Here: latest
# 2. How to sample the opponents?   (2nd player)    -> Here: latest
# 3. How to train both of them? (Schema of the training)    -> Here: alternating
# 4. How to rank/evaluate the agents? How this agent is valuable for the training?  (e.g. points)   -> Here: None
################################################################

# TODO: Add flexible testing for specific models using argparse

import os
from datetime import datetime
import numpy as np

import torch
import gym_predprey

import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import PPO
from stable_baselines3.common import logger
from shutil import copyfile # keep track of generations
from callbacks import *

from wandb.integration.sb3 import WandbCallback
import wandb

from gym.envs.registration import register

from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPredEnv
from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPreyEnv
import random

# from bach_utils.archive import Archive
from archive import ArchiveSB3 as Archive

OBS = "full"
ACT = "vel"
ENV = "SelfPlay1v1-Pred_Prey-v0"
WORKERS = 1#3
ALGO = "PPO"
PRED_ALGO = "PPO"
PREY_ALGO = "PPO"

SEED_VALUE = 3
NUM_EVAL_EPISODES = 1#10
LOG_DIR = None
# PRED_TRAINING_EPISODES = 25  # in iterations
# PREY_TRAINING_EPISODES = 25  # in iterations
NUM_TIMESTEPS = int(25e3)#int(1e9)
EVAL_FREQ = int(5e3)#int(5e3) #in steps
NUM_ROUNDS = 5#50
SAVE_FREQ = int(5e3)#5000 # in steps -> if you want only to save at the end of training round -> NUM_TIMESTEPS
FINAL_SAVE_FREQ = 3 # in rounds
EVAL_METRIC = "winrate"

EVAL_OPPONENT_SELECTION = "random"
OPPONENT_SELECTION = "random"
NUM_SAMPLED_OPPONENT_PER_ROUND = 5
SAMPLE_AFTER_ROLLOUT = False    # This is made to choose opponent after reset or not

env_config = {"Obs": OBS, "Act": ACT, "Env": ENV, "Hierarchy":"2D:evorobotpy2:predprey:1v1"}

training_config = { "pred_algorithm": PRED_ALGO,
                    "prey_algorithm": PREY_ALGO,
                    "num_rounds": NUM_ROUNDS,
                    "save_freq": SAVE_FREQ,
                    "num_timesteps": NUM_TIMESTEPS,
                    # "pred_TRAINING_EPISODEs":PRED_TRAINING_EPISODES,
                    "num_eval_episodes": NUM_EVAL_EPISODES,
                    "num_workers": WORKERS,
                    "seed": SEED_VALUE,
                    "eval_freq": EVAL_FREQ,
                    "framework": "stable_baselines3",
                    "agent_selection": "latest",
                    "opponent_selection": OPPONENT_SELECTION,
                    "eval_opponent_selection": EVAL_OPPONENT_SELECTION,
                    "training_schema": "alternating",
                    "ranking": "none",
                    "final_save_freq": FINAL_SAVE_FREQ,
                    "eval_metric": EVAL_METRIC,
                    "num_sampled_per_round": NUM_SAMPLED_OPPONENT_PER_ROUND,
                    "sample_after_rollout": SAMPLE_AFTER_ROLLOUT,
                    }


pred_algorithm_config = {   "policy": "MlpPolicy",
                            "clip_range": 0.2,
                            "ent_coef": 0.0,
                            "lr": 3e-4,
                            "batch_size":64,
                            "gamma":0.99
                        }


prey_algorithm_config = {   "policy": "MlpPolicy",
                            "clip_range": 0.2,
                            "ent_coef": 0.0,
                            "lr": 3e-4,
                            "batch_size":64,
                            "gamma":0.99
                        }


wandb_experiment_config = dict(env_config, **training_config)

wandb_experiment_config["pred_algorithm_config"] = pred_algorithm_config

wandb_experiment_config["prey_algorithm_config"] = prey_algorithm_config




# Source: https://github.com/rlturkiye/flying-cavalry/blob/main/rllib/main.py
def make_deterministic(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # see https://github.com/pytorch/pytorch/issues/47672
    cuda_version = torch.version.cuda
    if cuda_version is not None and float(torch.version.cuda) >= 10.2:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'
    else:
        torch.set_deterministic(True)  # Not all Operations support this.
    # This is only for Convolution no problem
    torch.backends.cudnn.deterministic = True


def eval(log_dir):
    # train selfplay agent
    logger.configure(folder=log_dir)
    make_deterministic(SEED_VALUE)

    # --------------------------------------- Pred -------------------------------------------------------
    pred_env_eval = SelfPlayPredEnv(log_dir=log_dir, algorithm_class=PPO, archive=None)#, opponent_selection=OPPONENT_SELECTION) #SelfPlayPredEnv()
    pred_env_eval._name = "Evaluation"
    pred_opponent_sample_path = os.path.join(log_dir, "prey")
    pred_model = PPO(pred_algorithm_config["policy"], pred_env_eval, 
                     clip_range=pred_algorithm_config["clip_range"], ent_coef=pred_algorithm_config["ent_coef"],
                     learning_rate=pred_algorithm_config["lr"], batch_size=pred_algorithm_config["batch_size"],
                     gamma=pred_algorithm_config["gamma"], verbose=2,
                     tensorboard_log=os.path.join(log_dir,"pred"))
    # Here the EvalSaveCallback is used the archive to save the model and sample the opponent for evaluation
    pred_evalsave_callback = EvalSaveCallback(eval_env=pred_env_eval,
                                              log_path=os.path.join(log_dir, "pred"),
                                              eval_freq=EVAL_FREQ,
                                              n_eval_episodes=NUM_EVAL_EPISODES,
                                              deterministic=True,
                                              save_path=os.path.join(LOG_DIR, "pred"),
                                              eval_metric=EVAL_METRIC,
                                              eval_opponent_selection=EVAL_OPPONENT_SELECTION,
                                              eval_sample_path=pred_opponent_sample_path,
                                              save_freq=SAVE_FREQ,
                                              archive={"self":None, "opponent":None},
                                              agent_name="pred",
                                              num_rounds=NUM_ROUNDS)
    pred_evalsave_callback.OS = True

    # --------------------------------------- Prey -------------------------------------------------------
    prey_env_eval = SelfPlayPreyEnv(log_dir=log_dir, algorithm_class=PPO, archive=None)#, opponent_selection=OPPONENT_SELECTION) #SelfPlayPreyEnv()
    prey_env_eval._name = "Evaluation"
    prey_opponent_sample_path = os.path.join(log_dir, "pred")
    prey_model = PPO(prey_algorithm_config["policy"], prey_env_eval, 
                     clip_range=prey_algorithm_config["clip_range"], ent_coef=prey_algorithm_config["ent_coef"],
                     learning_rate=prey_algorithm_config["lr"], batch_size=prey_algorithm_config["batch_size"],
                     gamma=prey_algorithm_config["gamma"], verbose=2,
                     tensorboard_log=os.path.join(log_dir,"prey"))
    prey_evalsave_callback = EvalSaveCallback(eval_env=prey_env_eval,
                                              log_path=os.path.join(log_dir, "prey"),
                                              eval_freq=EVAL_FREQ,
                                              n_eval_episodes=NUM_EVAL_EPISODES,
                                              deterministic=True,
                                              save_path=os.path.join(LOG_DIR, "prey"),
                                              eval_metric=EVAL_METRIC,
                                              eval_opponent_selection=EVAL_OPPONENT_SELECTION,
                                              eval_sample_path=prey_opponent_sample_path,
                                              save_freq=SAVE_FREQ,
                                              archive={"self":None, "opponent":None},
                                              agent_name="prey",
                                              num_rounds=NUM_ROUNDS)
    prey_evalsave_callback.OS = True
    # ----------------------------------------------------------------------------------------------------
    pred_wandb_callback = WandbCallback()
    prey_wandb_callback = WandbCallback()

    for i in range(NUM_ROUNDS):
        print(f"Round: {i} -> HeatMap Evaluation for current round version of pred vs prey")
        pred_evalsave_callback.compute_eval_matrix_aggregate(prefix="history_", round_num=i, n_eval_rep=NUM_EVAL_EPISODES, algorithm_class=PPO, opponents_path=os.path.join(LOG_DIR, "prey"), agents_path=os.path.join(LOG_DIR, "pred"))
        print(f"Round: {i} -> HeatMap Evaluation for current round version of prey vs pred")
        prey_evalsave_callback.compute_eval_matrix_aggregate(prefix="history_", round_num=i, n_eval_rep=NUM_EVAL_EPISODES, algorithm_class=PPO, opponents_path=os.path.join(LOG_DIR, "pred"), agents_path=os.path.join(LOG_DIR, "prey"))

    wandb.log({f"pred/mid_eval/heatmap"'': wandb.plots.HeatMap([i for i in range(NUM_ROUNDS)], [j for j in range(NUM_ROUNDS)], pred_evalsave_callback.evaluation_matrix, show_text=True)})
    wandb.log({f"prey/mid_eval/heatmap"'': wandb.plots.HeatMap([i for i in range(NUM_ROUNDS)], [i for i in range(NUM_ROUNDS)], prey_evalsave_callback.evaluation_matrix, show_text=True)})

    # print("HeatMap Evaluation for preds vs preys")
    # pred_evalsave_callback.compute_eval_matrix(prefix="history_", num_rounds=NUM_ROUNDS, n_eval_rep=NUM_EVAL_EPISODES, algorithm_class=PPO, opponents_path=os.path.join(LOG_DIR, "prey"), agents_path=os.path.join(LOG_DIR, "pred"))
    # wandb.log({f"pred/mid_eval/heatmap"'': wandb.plots.HeatMap([i for i in range(NUM_ROUNDS)], [j for j in range(NUM_ROUNDS)], pred_evalsave_callback.evaluation_matrix, show_text=True)})

    # print("HeatMap Evaluation for preys vs preds")
    # prey_evalsave_callback.compute_eval_matrix(prefix="history_", num_rounds=NUM_ROUNDS, n_eval_rep=NUM_EVAL_EPISODES, algorithm_class=PPO, opponents_path=os.path.join(LOG_DIR, "pred"), agents_path=os.path.join(LOG_DIR, "prey"))
    # wandb.log({f"prey/mid_eval/heatmap"'': wandb.plots.HeatMap([i for i in range(NUM_ROUNDS)], [i for i in range(NUM_ROUNDS)], prey_evalsave_callback.evaluation_matrix, show_text=True)})

    pred_env_eval.close()
    prey_env_eval.close()

if __name__=="__main__":

    # Source: https://github.com/rlturkiye/flying-cavalry/blob/main/rllib/main.py
    if torch.cuda.is_available():
        print("## CUDA available")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("## CUDA not available")
    
    experiment_id = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    prefix = "test-" # ""
    # LOG_DIR = os.path.dirname(os.path.abspath(__file__)) + f'/selfplay-results/{prefix}save-' + ENV + '-' + ALGO + '-' + OBS + '-' + ACT + '-' + experiment_id
    LOG_DIR = "/home/hany606/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-results/test-save-SelfPlay1v1-Pred_Prey-v0-PPO-full-vel-09.21.2021_01.45.43/"
    wandb.tensorboard.patch(root_logdir=LOG_DIR)
    wandb.init(project="Behavioral-Learning-Thesis",
               group="self-play",
               config=wandb_experiment_config,
               sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
               monitor_gym=True,  # auto-upload the videos of agents playing the game
               save_code=True,  # optional
    )

    wandb.run.name = wandb.run.name + f"-v4.0-server-random-heatmap"#random" #f"-run-{experiment_id}"
    wandb.run.save()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR + '/')
    if not os.path.exists(os.path.join(LOG_DIR, "pred")):
        os.makedirs(os.path.join(LOG_DIR, "pred") + '/')
    if not os.path.exists(os.path.join(LOG_DIR, "prey")):
        os.makedirs(os.path.join(LOG_DIR, "prey") + '/')

    eval(LOG_DIR)
