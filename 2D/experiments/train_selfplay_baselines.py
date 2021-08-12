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

# TODO: Fix steps
# TODO: Add final evaluation
# TODO: Add flexible testing for specific models using argparse
# TODO: Test random opponent selection during training and let's see, there should be a lot of loading model prints

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
from stable_baselines3.common.callbacks import CheckpointCallback

from wandb.integration.sb3 import WandbCallback
import wandb

from gym.envs.registration import register

from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPredEnv
from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPreyEnv


OBS = "full"
ACT = "vel"
ENV = "SelfPlay1v1-Pred_Prey-v0"
WORKERS = 1#3
ALGO = "PPO"
PRED_ALGO = "PPO"
PREY_ALGO = "PPO"

SEED_VALUE = 3
NUM_EVAL_EPISODES = 10
LOG_DIR = None
# PRED_TRAINING_EPISODES = 25  # in iterations
# PREY_TRAINING_EPISODES = 25  # in iterations
NUM_TIMESTEPS = int(25e3)#int(1e9)
EVAL_FREQ = int(5e3) #in steps
NUM_ROUNDS = 2#50
SAVE_FREQ = 5000 # in steps
FINAL_SAVE_FREQ = 3 # in rounds
EVAL_METRIC = "winrate"

EVAL_OPPONENT_SELECTION = "random"
OPPONENT_SELECTION = "random" 

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


class make_env:
    def __init__(self, env_id, config=None):
        self.env_id = env_id
        self.config = config
    def make(self):
        env = None
        if(self.config is not None):
            env = gym.make(self.env_id, **self.config)
        else:
            env = gym.make(self.env_id)
        env = Monitor(env)  # record stats such as returns
        return env

def create_env_notused(env_id, dir, config=None):
    env = make_env(env_id, config)
    env = DummyVecEnv([lambda: env])#DummyVecEnvSelfPlay([lambda: env])
    # env = DummyVecEnv([env.make])
    # env = VecVideoRecorder(env, dir,
    #     record_video_trigger=lambda x: x % 2000 == 0, video_length=500)
    return env

def create_env(*args, **kwargs):
    env = args[0](**kwargs)
    env = DummyVecEnvSelfPlay([lambda: env]) #DummyVecEnv([lambda: env])#
    # env = DummyVecEnv([env.make])
    # env = VecVideoRecorder(env, dir,
    #     record_video_trigger=lambda x: x % 2000 == 0, video_length=500)
    return env

def train(log_dir):
    # train selfplay agent
    logger.configure(folder=log_dir)
    make_deterministic(SEED_VALUE)
    
    # --------------------------------------- Pred -------------------------------------------------------
    

    # pred_env = create_env("SelfPlay1v1-Pred-v0", os.path.join(log_dir, "pred", "videos"), config={"log_dir": log_dir, "algorithm_class": PPO}) #SelfPlayPredEnv()
    pred_env = SelfPlayPredEnv(log_dir=log_dir, algorithm_class=PPO, opponent_selection=OPPONENT_SELECTION) #SelfPlayPredEnv()
    # pred_env = create_env(SelfPlayPredEnv, log_dir=log_dir, algorithm_class=PPO, opponent_selection=OPPONENT_SELECTION)
    # pred_env.seed(SEED_VALUE)
    pred_model = PPO(pred_algorithm_config["policy"], pred_env, 
                     clip_range=pred_algorithm_config["clip_range"], ent_coef=pred_algorithm_config["ent_coef"],
                     learning_rate=pred_algorithm_config["lr"], batch_size=pred_algorithm_config["batch_size"],
                     gamma=pred_algorithm_config["gamma"], verbose=2,
                     tensorboard_log=os.path.join(log_dir,"pred"))
    pred_evalsave_callback = EvalSaveCallback(eval_env=pred_env,
                                              log_path=os.path.join(log_dir, "pred"),
                                              eval_freq=EVAL_FREQ,
                                              n_eval_episodes=NUM_EVAL_EPISODES,
                                              deterministic=True,
                                              save_path=os.path.join(LOG_DIR, "pred"),
                                              eval_metric=EVAL_METRIC,
                                              eval_opponent_selection=EVAL_OPPONENT_SELECTION,
                                              eval_sample_path=os.path.join(log_dir, "prey"),
                                              save_freq=SAVE_FREQ)


    # --------------------------------------- Prey -------------------------------------------------------
    # prey_env = create_env("SelfPlay1v1-Prey-v0", os.path.join(log_dir, "prey", "videos"), config={"log_dir": log_dir, "algorithm_class": PPO}) #SelfPlayPreyEnv()
    prey_env = SelfPlayPreyEnv(log_dir=log_dir, algorithm_class=PPO, opponent_selection=OPPONENT_SELECTION) #SelfPlayPreyEnv()
    # prey_env = create_env(SelfPlayPreyEnv, log_dir=log_dir, algorithm_class=PPO, opponent_selection=OPPONENT_SELECTION)
    # prey_env.seed(SEED_VALUE)
    prey_model = PPO(prey_algorithm_config["policy"], prey_env, 
                     clip_range=prey_algorithm_config["clip_range"], ent_coef=prey_algorithm_config["ent_coef"],
                     learning_rate=prey_algorithm_config["lr"], batch_size=prey_algorithm_config["batch_size"],
                     gamma=prey_algorithm_config["gamma"], verbose=2,
                     tensorboard_log=os.path.join(log_dir,"prey"))
    prey_evalsave_callback = EvalSaveCallback(eval_env=prey_env,
                                              log_path=os.path.join(log_dir, "prey"),
                                              eval_freq=EVAL_FREQ,
                                              n_eval_episodes=NUM_EVAL_EPISODES,
                                              deterministic=True,
                                              save_path=os.path.join(LOG_DIR, "prey"),
                                              eval_metric=EVAL_METRIC,
                                              eval_opponent_selection=EVAL_OPPONENT_SELECTION,
                                              eval_sample_path=os.path.join(log_dir, "pred"),
                                              save_freq=SAVE_FREQ)

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
        pred_model.learn(total_timesteps=NUM_TIMESTEPS, callback=[pred_evalsave_callback, pred_wandb_callback], reset_num_timesteps=False)
        print(f"------------------- Prey {round_num}--------------------")
        prey_model.learn(total_timesteps=NUM_TIMESTEPS, callback=[prey_evalsave_callback, prey_wandb_callback], reset_num_timesteps=False)
    
        if(round_num%FINAL_SAVE_FREQ == 0):
            # TODO: Change it to save the best model for now, not the latest (How to define the best model)
            pred_model.save(os.path.join(LOG_DIR, "pred", "final_model"))
            prey_model.save(os.path.join(LOG_DIR, "prey", "final_model"))

    pred_model.save(os.path.join(LOG_DIR, "pred", "final_model"))
    prey_model.save(os.path.join(LOG_DIR, "prey", "final_model"))

    print("Post Evaluation for Pred:")
    pred_evalsave_callback.post_eval(agent_name="pred", opponents_path=os.path.join(LOG_DIR, "prey"))
    print("Post Evaluation for Prey:")
    prey_evalsave_callback.post_eval(agent_name="prey", opponents_path=os.path.join(LOG_DIR, "pred"))

    pred_env.close()
    prey_env.close()

if __name__=="__main__":

    # Source: https://github.com/rlturkiye/flying-cavalry/blob/main/rllib/main.py
    if torch.cuda.is_available():
        print("## CUDA available")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("## CUDA not available")
    
    experiment_id = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    LOG_DIR = os.path.dirname(os.path.abspath(__file__)) + '/selfplay-results/save-' + ENV + '-' + ALGO + '-' + OBS + '-' + ACT + '-' + experiment_id
    wandb.tensorboard.patch(root_logdir=LOG_DIR)
    wandb.init(project="Behavioral-Learning-Thesis",
               group="self-play",
               config=wandb_experiment_config,
               sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
               monitor_gym=True,  # auto-upload the videos of agents playing the game
               save_code=True,  # optional
    )

    wandb.run.name = wandb.run.name + "-test"#f"-v3.1-rep-nitro-random" #f"-run-{experiment_id}"
    wandb.run.save()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR + '/')
    if not os.path.exists(os.path.join(LOG_DIR, "pred")):
        os.makedirs(os.path.join(LOG_DIR, "pred") + '/')
    if not os.path.exists(os.path.join(LOG_DIR, "prey")):
        os.makedirs(os.path.join(LOG_DIR, "prey") + '/')

    train(LOG_DIR)
