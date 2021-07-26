# Training script for self-play using Stable baselines3
# Based on: https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_selfplay.py

# This script is used to:
# - Self-play training between agents
# - The agents are initialized with a policy
# - The policy of the opponent is being selected to be the latest model if exists if not then a random policy (Sampling from the action space)
# - The training is starting to train the first agent for several epochs then the second agent
# - The model is being saved in the local directory

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

import torch
import gym_predprey

import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import PPO
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import EvalCallback
from shutil import copyfile # keep track of generations



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
NUM_EVAL_EPISODES = 5
LOG_DIR = None
# PRED_TRAINING_EPOCHS = 25  # in iterations
# PREY_TRAINING_EPOCHS = 25  # in iterations
NUM_TIMESTEPS = int(25e3)#int(1e9)
EVAL_FREQ = int(1e3)
NUM_ROUNDS = 50
SAVE_FREQ = 5000 # in steps


env_config = {"Obs": OBS, "Act": ACT, "Env": ENV, "Group":"2D:evorobotpy2:predprey:1v1"}

training_config = { "pred_algorithm": PRED_ALGO,
                    "prey_algorithm": PREY_ALGO,
                    "num_rounds": NUM_ROUNDS,
                    "save_freq": SAVE_FREQ,
                    "num_timesteps": NUM_TIMESTEPS,
                    # "pred_training_epochs":PRED_TRAINING_EPOCHS,
                    "num_eval_episodes": NUM_EVAL_EPISODES,
                    "num_workers": WORKERS,
                    "seed": SEED_VALUE,
                    "eval_freq": EVAL_FREQ,
                    "framework": "stable_baselines3",
                    "agent_selection": "latest",
                    "opponent_selection": "latest",
                    "training_schema": "alternating",
                    "ranking": "none",
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

def create_env(env_id, dir, config=None):
    env = make_env(env_id, config)
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
    pred_env = SelfPlayPredEnv(log_dir=log_dir, algorithm_class=PPO) #SelfPlayPredEnv()
    # pred_env.seed(SEED_VALUE)
    pred_model = PPO(pred_algorithm_config["policy"], pred_env, 
                     clip_range=pred_algorithm_config["clip_range"], ent_coef=pred_algorithm_config["ent_coef"],
                     learning_rate=pred_algorithm_config["lr"], batch_size=pred_algorithm_config["batch_size"],
                     gamma=pred_algorithm_config["gamma"], verbose=2,
                    tensorboard_log=log_dir)
    pred_eval_callback = EvalCallback(pred_env,
                                     log_path=os.path.join(log_dir, "pred"),
                                     eval_freq=EVAL_FREQ,
                                     n_eval_episodes=NUM_EVAL_EPISODES,
                                     deterministic=True)


    # --------------------------------------- Prey -------------------------------------------------------
    # prey_env = create_env("SelfPlay1v1-Prey-v0", os.path.join(log_dir, "prey", "videos"), config={"log_dir": log_dir, "algorithm_class": PPO}) #SelfPlayPreyEnv()
    prey_env = SelfPlayPreyEnv(log_dir=log_dir, algorithm_class=PPO) #SelfPlayPreyEnv()
    # prey_env.seed(SEED_VALUE)
    prey_model = PPO(prey_algorithm_config["policy"], prey_env, 
                     clip_range=prey_algorithm_config["clip_range"], ent_coef=prey_algorithm_config["ent_coef"],
                     learning_rate=prey_algorithm_config["lr"], batch_size=prey_algorithm_config["batch_size"],
                     gamma=prey_algorithm_config["gamma"], verbose=2,
                     tensorboard_log=log_dir)                     
    prey_eval_callback = EvalCallback(prey_env,
                                      log_path=os.path.join(log_dir, "prey"),
                                      eval_freq=EVAL_FREQ,
                                      n_eval_episodes=NUM_EVAL_EPISODES,
                                      deterministic=True)

    # ----------------------------------------------------------------------------------------------------

    # TODO: There is a problem here in reporting the results, results from the pred and the prey will be together
    # --------------------------------------------- Training ---------------------------------------------
    # Here alternate training
    for round_num in range(NUM_ROUNDS):
        pred_checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=os.path.join(LOG_DIR, "pred"),
                                                    name_prefix=f"history_{round_num}")

        prey_checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=os.path.join(LOG_DIR, "prey"),
                                                    name_prefix=f"history_{round_num}")

        print(f"------------------- Pred {round_num+1}--------------------")
        # pred_model.learn(n_epochs=PRED_TRAINING_EPOCHS, callback=[pred_eval_callback, pred_checkpoint_callback, WandbCallback()])
        pred_model.learn(total_timesteps=NUM_TIMESTEPS, callback=[pred_eval_callback, pred_checkpoint_callback, WandbCallback()])
        print(f"------------------- Prey {round_num+1}--------------------")
        # prey_model.learn(n_epochs=PREY_TRAINING_EPOCHS, callback=[prey_eval_callback, prey_checkpoint_callback, WandbCallback()])
        prey_model.learn(total_timesteps=NUM_TIMESTEPS, callback=[prey_eval_callback, prey_checkpoint_callback, WandbCallback()])
    
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
    
    wandb.init(project="Behavioral-Learning-Thesis",
               group="self-play",
               config=wandb_experiment_config,
               sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
               monitor_gym=True,  # auto-upload the videos of agents playing the game
               save_code=True,  # optional
    )

    wandb.run.name = wandb.run.name + f"-test" #f"-{experiment_id}"
    wandb.run.save()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR + '/')
    if not os.path.exists(os.path.join(LOG_DIR, "pred")):
        os.makedirs(os.path.join(LOG_DIR, "pred") + '/')
    if not os.path.exists(os.path.join(LOG_DIR, "prey")):
        os.makedirs(os.path.join(LOG_DIR, "prey") + '/')

    train(LOG_DIR)
