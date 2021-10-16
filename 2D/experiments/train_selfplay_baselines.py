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


import os
from datetime import datetime
import numpy as np
import argparse
import random
from shutil import copyfile # keep track of generations

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import PPO
from stable_baselines3.common import logger

# from bach_utils.archive import Archive
from archive import ArchiveSB3 as Archive

import gym_predprey
import gym
from gym.envs.registration import register
from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPredEnv
from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPreyEnv

from callbacks import *

from wandb.integration.sb3 import WandbCallback
import wandb

from bach_utils.json_parser import ExperimentParser as experiment


OBS = "full"
ACT = "vel"
ENV = "SelfPlay1v1-Pred_Prey-v0"
WORKERS = 1#3
ALGO = "PPO"
PRED_ALGO = "PPO"
PREY_ALGO = "PPO"

SEED_VALUE = 3
NUM_EVAL_EPISODES = 5#1#10
LOG_DIR = None
# PRED_TRAINING_EPISODES = 25  # in iterations
# PREY_TRAINING_EPISODES = 25  # in iterations
NUM_TIMESTEPS = int(25e3)#int(1e9)
NUM_ROUNDS = 1#50
EVAL_FREQ = NUM_TIMESTEPS#int(1e3)#int(5e3)#int(5e3) #in steps
SAVE_FREQ = NUM_TIMESTEPS#int(5e3)#5000 # in steps -> if you want only to save at the end of training round -> NUM_TIMESTEPS
FINAL_SAVE_FREQ = 3 # in rounds
EVAL_METRIC = "winrate"

EVAL_OPPONENT_SELECTION = "random"
OPPONENT_SELECTION = "random"
NUM_SAMPLED_OPPONENT_PER_ROUND = 50#2
SAMPLE_AFTER_ROLLOUT = True #False    # This is made to choose opponent after reset or not

PRED_NUM_EPOCHS = 10#25
PRED_LEARNING_RATE = 3e-4#0.0003

PREY_NUM_EPOCHS = 3
PREY_LEARNING_RATE = 1e-4#0.0001

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
                            "lr": PRED_LEARNING_RATE,
                            "batch_size":64,
                            "gamma":0.99,
                            "n_epochs": PRED_NUM_EPOCHS,
                        }


prey_algorithm_config = {   "policy": "MlpPolicy",
                            "clip_range": 0.2,
                            "ent_coef": 0.0,
                            "lr": PREY_LEARNING_RATE,
                            "batch_size":64,
                            "gamma":0.99,
                            "n_epochs":PREY_NUM_EPOCHS,
                        }


wandb_experiment_config = dict(env_config, **training_config)

wandb_experiment_config["pred_algorithm_config"] = pred_algorithm_config

wandb_experiment_config["prey_algorithm_config"] = prey_algorithm_config



def init_argparse():
    parser = argparse.ArgumentParser(description='Self-play experiment training script')
    parser.add_argument('--exp', type=str, help='The experiemnt file path and name which the experiment should be loaded', metavar='')
    # args = parser.parse_args()
    return parser



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

def train(log_dir, experiment_data, agents_data, evaluation_data):
    # train selfplay agent
    logger.configure(folder=log_dir)
    make_deterministic(SEED_VALUE)

    pred_archive = Archive(sorting_keys=[EVAL_OPPONENT_SELECTION, OPPONENT_SELECTION],
                           sorting=True,
                           moving_least_freq_flag=False,
                           save_path=os.path.join(LOG_DIR, "pred")
                          )
    prey_archive = Archive(sorting_keys=[EVAL_OPPONENT_SELECTION, OPPONENT_SELECTION],
                           sorting=True,
                           moving_least_freq_flag=False,
                           save_path=os.path.join(LOG_DIR, "prey")
                          )

    # --------------------------------------- Pred -------------------------------------------------------
    # pred_env = create_env("SelfPlay1v1-Pred-v0", os.path.join(log_dir, "pred", "videos"), config={"log_dir": log_dir, "algorithm_class": PPO}) #SelfPlayPredEnv()
    # Here SelfPlayPredEnv will use the archive only for load the opponent nothing more -> Pass the opponent archive
    pred_env = SelfPlayPredEnv(log_dir=log_dir, algorithm_class=PPO, archive=prey_archive)#, opponent_selection=OPPONENT_SELECTION) #SelfPlayPredEnv()
    pred_env_eval = SelfPlayPredEnv(log_dir=log_dir, algorithm_class=PPO, archive=prey_archive)#, opponent_selection=OPPONENT_SELECTION) #SelfPlayPredEnv()
    pred_env._name = "Training"
    pred_env_eval._name = "Evaluation"
    pred_opponent_sample_path = os.path.join(log_dir, "prey")
    # pred_env = create_env(SelfPlayPredEnv, log_dir=log_dir, algorithm_class=PPO, opponent_selection=OPPONENT_SELECTION)
    # pred_env.seed(SEED_VALUE)
    pred_model = PPO(pred_algorithm_config["policy"], pred_env, 
                     clip_range=pred_algorithm_config["clip_range"], ent_coef=pred_algorithm_config["ent_coef"],
                     learning_rate=pred_algorithm_config["lr"], batch_size=pred_algorithm_config["batch_size"],
                     gamma=pred_algorithm_config["gamma"], verbose=2,
                     tensorboard_log=os.path.join(log_dir,"pred"),
                     n_epochs=pred_algorithm_config["n_epochs"])
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
                                              archive={"self":pred_archive, "opponent":prey_archive},
                                              agent_name="pred",
                                              num_rounds=NUM_ROUNDS)
    # Here the TrainingOpponentSelectionCallback is used the archive to sample the opponent for training
    # The name here pred_oppoenent -> the opponent of the predator
    pred_opponent_selection_callback = TrainingOpponentSelectionCallback(sample_path=pred_opponent_sample_path,
                                                                 env=pred_env, 
                                                                 opponent_selection=OPPONENT_SELECTION,
                                                                 sample_after_rollout=SAMPLE_AFTER_ROLLOUT,
                                                                 num_sampled_per_round=NUM_SAMPLED_OPPONENT_PER_ROUND,
                                                                 archive=prey_archive)


    # --------------------------------------- Prey -------------------------------------------------------
    # prey_env = create_env("SelfPlay1v1-Prey-v0", os.path.join(log_dir, "prey", "videos"), config={"log_dir": log_dir, "algorithm_class": PPO}) #SelfPlayPreyEnv()
    prey_env = SelfPlayPreyEnv(log_dir=log_dir, algorithm_class=PPO, archive=pred_archive)#, opponent_selection=OPPONENT_SELECTION) #SelfPlayPreyEnv()
    prey_env_eval = SelfPlayPreyEnv(log_dir=log_dir, algorithm_class=PPO, archive=pred_archive)#, opponent_selection=OPPONENT_SELECTION) #SelfPlayPreyEnv()
    prey_env._name = "Training"
    prey_env_eval._name = "Evaluation"
    prey_opponent_sample_path = os.path.join(log_dir, "pred")
    # prey_env = create_env(SelfPlayPreyEnv, log_dir=log_dir, algorithm_class=PPO, opponent_selection=OPPONENT_SELECTION)
    # prey_env.seed(SEED_VALUE)
    prey_model = PPO(prey_algorithm_config["policy"], prey_env, 
                     clip_range=prey_algorithm_config["clip_range"], ent_coef=prey_algorithm_config["ent_coef"],
                     learning_rate=prey_algorithm_config["lr"], batch_size=prey_algorithm_config["batch_size"],
                     gamma=prey_algorithm_config["gamma"], verbose=2,
                     tensorboard_log=os.path.join(log_dir,"prey"),
                     n_epochs=prey_algorithm_config["n_epochs"])
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
                                              archive={"self":prey_archive, "opponent":pred_archive},
                                              agent_name="prey",
                                              num_rounds=NUM_ROUNDS)
    prey_opponent_selection_callback = TrainingOpponentSelectionCallback(sample_path=prey_opponent_sample_path,
                                                                 env=prey_env, 
                                                                 opponent_selection=OPPONENT_SELECTION,
                                                                 sample_after_rollout=SAMPLE_AFTER_ROLLOUT,
                                                                 num_sampled_per_round=NUM_SAMPLED_OPPONENT_PER_ROUND,
                                                                 archive=pred_archive)

    # ----------------------------------------------------------------------------------------------------
    pred_wandb_callback = WandbCallback()
    prey_wandb_callback = WandbCallback()
    # --------------------------------------------- Training ---------------------------------------------
    # Here alternate training
    for round_num in range(NUM_ROUNDS):
        pred_evalsave_callback.set_name_prefix(f"history_{round_num}")
        prey_evalsave_callback.set_name_prefix(f"history_{round_num}")
        # pred_checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=os.path.join(LOG_DIR, "pred"),
        #                                             name_prefix=f"history_{round_num}")

        # prey_checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=os.path.join(LOG_DIR, "prey"),
        #                                               name_prefix=f"history_{round_num}")

        print(f"------------------- Pred {round_num}--------------------")
        pred_model.learn(total_timesteps=NUM_TIMESTEPS, 
                         callback=[pred_opponent_selection_callback, 
                                   pred_evalsave_callback,
                                   pred_wandb_callback], 
                         reset_num_timesteps=False)
        print(f"------------------- Prey {round_num}--------------------")
        prey_model.learn(total_timesteps=NUM_TIMESTEPS, 
                         callback=[prey_opponent_selection_callback, 
                                   prey_evalsave_callback, 
                                   prey_wandb_callback], 
                         reset_num_timesteps=False)        
        print(f"Round: {round_num} -> HeatMap Evaluation for current round version of pred vs prey")
        pred_evalsave_callback.compute_eval_matrix_aggregate(prefix="history_", round_num=round_num, n_eval_rep=NUM_EVAL_EPISODES, algorithm_class=PPO, opponents_path=os.path.join(LOG_DIR, "prey"), agents_path=os.path.join(LOG_DIR, "pred"))
        print(f"Round: {round_num} -> HeatMap Evaluation for current round version of prey vs pred")
        prey_evalsave_callback.compute_eval_matrix_aggregate(prefix="history_", round_num=round_num, n_eval_rep=NUM_EVAL_EPISODES, algorithm_class=PPO, opponents_path=os.path.join(LOG_DIR, "pred"), agents_path=os.path.join(LOG_DIR, "prey"))



        if(round_num%FINAL_SAVE_FREQ == 0):
            # TODO: Change it to save the best model till now, not the latest (How to define the best model)
            pred_model.save(os.path.join(LOG_DIR, "pred", "final_model"))
            prey_model.save(os.path.join(LOG_DIR, "prey", "final_model"))

            np.save(os.path.join(LOG_DIR, "pred", "evaluation_matrix"), pred_evalsave_callback.evaluation_matrix)
            np.save(os.path.join(LOG_DIR, "prey", "evaluation_matrix"), prey_evalsave_callback.evaluation_matrix)

    wandb.log({f"pred/mid_eval/heatmap"'': wandb.plots.HeatMap([i for i in range(NUM_ROUNDS)], [i for i in range(NUM_ROUNDS)], pred_evalsave_callback.evaluation_matrix, show_text=True)})
    wandb.log({f"prey/mid_eval/heatmap"'': wandb.plots.HeatMap([i for i in range(NUM_ROUNDS)], [i for i in range(NUM_ROUNDS)], prey_evalsave_callback.evaluation_matrix.T, show_text=True)}) # .T in order to make the x-axis predators and y-axis are preys


    pred_evalsave_callback._save_model_core()
    prey_evalsave_callback._save_model_core()

    pred_model.save(os.path.join(LOG_DIR, "pred", "final_model"))
    prey_model.save(os.path.join(LOG_DIR, "prey", "final_model"))

    np.save(os.path.join(LOG_DIR, "pred", "evaluation_matrix"), pred_evalsave_callback.evaluation_matrix)
    np.save(os.path.join(LOG_DIR, "prey", "evaluation_matrix"), prey_evalsave_callback.evaluation_matrix)



    print("Post Evaluation for Pred:")
    pred_evalsave_callback.post_eval(opponents_path=os.path.join(LOG_DIR, "prey"))
    print("Post Evaluation for Prey:")
    prey_evalsave_callback.post_eval(opponents_path=os.path.join(LOG_DIR, "pred"))

    # print("HeatMap Evaluation for current round version of pred against previous of prey")
    # pred_evalsave_callback.compute_eval_matrix(prefix="history_", num_rounds=NUM_ROUNDS, n_eval_rep=NUM_EVAL_EPISODES, algorithm_class=PPO)
    # wandb.log({f"pred/mid_eval/heatmap"'': wandb.plots.HeatMap([i for i in range(NUM_ROUNDS)], [j for j in range(NUM_ROUNDS)], pred_evalsave_callback.evaluation_matrix, show_text=True)})
   
    # print("HeatMap Evaluation for current round version of prey against previous of pred")
    # prey_evalsave_callback.compute_eval_matrix(prefix="history_", num_rounds=NUM_ROUNDS, n_eval_rep=NUM_EVAL_EPISODES, algorithm_class=PPO)
    # wandb.log({f"prey/mid_eval/heatmap"'': wandb.plots.HeatMap([i for i in range(NUM_ROUNDS)], [i for i in range(NUM_ROUNDS)], prey_evalsave_callback.evaluation_matrix, show_text=True)})

    pred_env.close()
    pred_env_eval.close()
    prey_env.close()
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
    LOG_DIR = os.path.dirname(os.path.abspath(__file__)) + f'/selfplay-results/{prefix}save-' + ENV + '-' + ALGO + '-' + OBS + '-' + ACT + '-' + experiment_id
    wandb.tensorboard.patch(root_logdir=LOG_DIR)
    wandb.init(
            #    project="Behavioral-Learning",
            #    group="self-play-2D",
            #    entity='iu_mas',
               project="Behavioral-Learning-Thesis",
               group="self-play",
               config=wandb_experiment_config,
               sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
               monitor_gym=True,  # auto-upload the videos of agents playing the game
               save_code=True,  # optional
               notes="Debugging why the graphs are not appearing",
    )

    wandb.run.name = wandb.run.name + "-debug-v4"#f"-v3.1-rep-nitro-random" #f"-run-{experiment_id}"
    wandb.run.save()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR + '/')
    if not os.path.exists(os.path.join(LOG_DIR, "pred")):
        os.makedirs(os.path.join(LOG_DIR, "pred") + '/')
    if not os.path.exists(os.path.join(LOG_DIR, "prey")):
        os.makedirs(os.path.join(LOG_DIR, "prey") + '/')

    train(LOG_DIR)
