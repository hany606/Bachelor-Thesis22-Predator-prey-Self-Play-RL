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


# Note: this script is made only for now for pred and prey (1v1) setting

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
from stable_baselines3.common import callbacks, logger

# from bach_utils.archive import Archive
from archive import ArchiveSB3 as Archive

import gym_predprey
import gym
from gym.envs.registration import register
# Import all the used environments as they are going to be used from globals()[<env_name>]
from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPredEnv
from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPreyEnv

from callbacks import *

from wandb.integration.sb3 import WandbCallback
import wandb

from bach_utils.json_parser import ExperimentParser
from shared import *
from copy import deepcopy
import bach_utils.os as utos
from bach_utils.shared import *

class SelfPlayExp:
    def __init__(self):
        self.args = None
        self.experiment_filename = None
        self.experiment_configs = None
        self.agents_configs = None
        self.evaluation_configs = None
        self.testing_configs = None
        self.seed_value = None
        self.log_dir = None

    def _check_cuda(self):
        check_cuda()

    def make_deterministic(self, seed_value=None, cuda_check=True):
        seed = self.seed_value if seed_value is None else seed_value
        make_deterministic(seed, cuda_check=cuda_check)

    def _init_argparse(self, description, help):
        parser = argparse.ArgumentParser(description=description)
        # TODO: Force different seed from argparse if it exists against the one in the json file
        parser.add_argument('--exp', type=str, help=help, metavar='')
        self.args = parser.parse_args()

    def _load_configs(self, filename):        
        self.experiment_filename = self.args.exp if filename is None else filename
        self.experiment_configs, self.agents_configs, self.evaluation_configs, self.testing_configs, _ = ExperimentParser.load(self.experiment_filename)
        self.seed_value = self.experiment_configs["seed_value"] if self.seed_value is None else self.seed_value

    def log_configs(self):
        # TODO: use prettyprint
        # import pprint 
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(....)

        print("--------------- Logging configs ---------------")
        print(f"Experiment configs: {self.experiment_configs}")
        print(f"Agents configs: {self.agents_configs}")
        print(f"Evaluation configs: {self.evaluation_configs}")
        print(f"Testing config: {self.testing_configs}")
        print("-----------------------------------------------")

    def _generate_log_dir(self, dir_postfix):
        self.experiment_id = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        prefix = self.experiment_configs["experiment_log_prefix"] # ""
        env_name = self.experiment_configs["env"]
        # log_dir = os.path.dirname(os.path.abspath(__file__)) + f'/selfplay-results/{prefix}save-' + ENV + '-' + ALGO + '-' + OBS + '-' + ACT + '-' + experiment_id
        self.log_main_dir =  f'{prefix}save-'+ env_name + '-' + self.experiment_id
        log_dir = os.path.dirname(os.path.abspath(__file__)) + f'/selfplay-results-{dir_postfix}/{self.log_main_dir}'
        return log_dir

    def _init_log_files(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir + '/')
        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            if not os.path.exists(os.path.join(self.log_dir, agent_name)):
                os.makedirs(os.path.join(self.log_dir, agent_name) + '/')
                
    def _init_wandb(self):
        wandb_experiment_config = {"experiment": self.experiment_configs,
                                   "agents"    : self.agents_configs,
                                   "evaluation": self.evaluation_configs,
                                   "testing": self.testing_configs,
                                   "log_dir": self.log_dir,
                                   "experiment_id": self.experiment_id
                                   }
        wandb.tensorboard.patch(root_logdir=self.log_dir)
        wandb.init(
                project=self.experiment_configs["wandb_project"],
                group=self.experiment_configs["wandb_group"],
                entity= None if self.experiment_configs["wandb_entity"] == "None" else self.experiment_configs["wandb_entity"],
                config=wandb_experiment_config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=True,  # auto-upload the videos of agents playing the game
                save_code=True,  # optional
                notes=self.experiment_configs["wandb_notes"],
        )

        experiment_name = self.experiment_configs["experiment_name"]
        wandb.run.name = f"[Seed: {self.experiment_configs.get('seed_value', None)}] " + wandb.run.name + experiment_name + "-" + self.experiment_id
        wandb.run.save()
        wandb.save(self.experiment_filename)
        wandb.save("SelfPlayExp.py")
        wandb.save("callbacks.py")
        if(self.log_dir is not None):
            wandb.save(self.log_dir)


    def _init_exp(self, experiment_filename, logdir, wandb):
        if(experiment_filename is None):
            self._init_argparse()
        print(f"Parse from json file in {self.args.exp}" if experiment_filename is None else f"----- Loading experiment from: {experiment_filename}")
        self._load_configs(experiment_filename)
        
        if(logdir):
            self.log_dir = self._generate_log_dir()
            print(f"----- Initialize loggers")
            self._init_log_files()
            logger.configure(folder=self.log_dir)

        if(wandb):
            print(f"----- Initialize wandb")
            self._init_wandb()

        # They were moved down to be logged in wandb log
        print(f"----- Experiment logs are being stored in: {self.log_dir}")
        self.log_configs()

        self.make_deterministic()
    
    def create_env(self, key, name, algorithm_class=PPO, opponent_archive=None, seed_value=None, sample_after_reset=False, sampling_parameters=None, ret_seed=False):
        # Key is just used to get the environment class_name for intialization of the environment
        if(seed_value == "random"):
            seed_value = datetime.now().microsecond//1000
        else:
            seed_value = self.seed_value if seed_value is None else seed_value
        agent_configs = self.agents_configs[key]
        agent_name = agent_configs["name"]
        env_class_name = agent_configs["env_class"]
        # print(f"Create Env: {env_class_name}, Algorithm: {algorithm_class}, seed: {seed_value}")
        # Here e.g. SelfPlayPredEnv will use the archive only for load the opponent nothing more -> Pass the opponent archive
        env = globals()[env_class_name](algorithm_class=algorithm_class, archive=opponent_archive, seed_val=seed_value, sample_after_reset=sample_after_reset, sampling_parameters=sampling_parameters, gui=None)#, opponent_selection=OPPONENT_SELECTION) #SelfPlayPredEnv()
        env._name = name+f"-({agent_name})"
        if(not ret_seed):
            return env
        return env, seed_value

    def _init_env(self):
        raise NotImplementedError("Initialization for environment is not implemented")

    def _init_models(self):
        raise NotImplementedError("Initialization for models is not implemented")