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
from bach_utils.logger import init_logger
import logging

cli_logger = init_logger()

import os
from datetime import datetime
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common import logger

from bach_utils.list import reinit_seeder

import gym_predprey
import gym
from gym.envs.registration import register
# Import all the used environments as they are going to be used from globals()[<env_name>]
from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPredEnv
from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPreyEnv
from gym_pz_predprey.envs.SelfPlayPZPredPrey import SelfPlayPZPredEnv
from gym_pz_predprey.envs.SelfPlayPZPredPrey import SelfPlayPZPreyEnv
from gym_predprey_drones.envs.SelfPlayPredPreyDrones1v1 import SelfPlayPredDroneEnv
from gym_predprey_drones.envs.SelfPlayPredPreyDrones1v1 import SelfPlayPreyDroneEnv

from callbacks import *

import wandb

from bach_utils.json_parser import ExperimentParser
from bach_utils.shared import check_cuda

import pprint 
pp = pprint.PrettyPrinter(indent=4)


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
        # TODO:
        self.cli_log_vals_dict = {"debug": logging.DEBUG,
                                  "info": logging.INFO,
                                  "warn":logging.WARN,
                                  "error":logging.ERROR}
        self.cli_log_level = logging.DEBUG

    def _check_cuda(self):
        check_cuda()

    def make_deterministic(self, seed_value=None, cuda_check=True):
        seed = self.seed_value if seed_value is None else seed_value
        make_deterministic(seed, cuda_check=cuda_check)

    def _init_argparse(self, description, help):
        parser = argparse.ArgumentParser(description=description)
        # TODO: Force different seed from argparse if it exists against the one in the json file
        parser.add_argument('--exp', type=str, help=help, metavar='')
        parser.add_argument('--seed', type=int, help=help, default=-1)
        parser.add_argument('--prefix', type=str, help=help, default="")
        parser.add_argument('--notes', type=str, help=help, default="")
        parser.add_argument('--samplerseed', type=int, help=help, default=-1)
        parser.add_argument('--rendersleep', type=float, help=help, default=-1)
        parser.add_argument('--threaded', dest='threaded', action='store_true')
        parser.add_argument('--no-threaded', dest='threaded', action='store_false')
        parser.add_argument('--cli-log', type=str, help=help, choices=list(self.cli_log_vals_dict.keys()), default="debug")
        parser.set_defaults(threaded=False)
        self.args = parser.parse_args()

        self.cli_log_level = self.cli_log_vals_dict[self.args.cli_log]

    def _load_configs(self, filename):        
        self.experiment_filename = self.args.exp if filename is None else filename
        self.experiment_configs, self.agents_configs, self.evaluation_configs, self.testing_configs, self.merged_config = ExperimentParser.load(self.experiment_filename)
        self.experiment_configs["seed_value"] = self.experiment_configs["seed_value"] if self.args.seed == -1 else self.args.seed
        self.merged_config["experiment"]["seed_value"] = self.merged_config["experiment"]["seed_value"] if self.args.seed == -1 else self.args.seed
        self.seed_value = self.experiment_configs["seed_value"] if self.seed_value is None else self.seed_value
        

        # Sampler seed
        if self.experiment_configs.get("random_sampler_seed_value") is not None:
            self.experiment_configs["random_sampler_seed_value"] = self.experiment_configs["random_sampler_seed_value"] if self.args.samplerseed == -1 else self.args.samplerseed
            os.environ["SELFPLAY_SAMPLING_SEED"] = str(self.experiment_configs["random_sampler_seed_value"])
            reinit_seeder()

    def log_configs(self):
        # TODO: use prettyprint
        # import pprint 
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(....)

        self.clilog.info("--------------- Logging configs ---------------")
        self.clilog.info(f"Experiment configs:")
        pp.pprint(self.experiment_configs)
        self.clilog.info("================================")
        self.clilog.info(f"Agents configs:")
        pp.pprint(self.agents_configs)
        self.clilog.info("================================")
        self.clilog.info(f"Evaluation configs:")
        pp.pprint(self.agents_configs)
        self.clilog.info("================================")
        self.clilog.info(f"Testing config:")
        pp.pprint(self.testing_configs)
        self.clilog.info("-----------------------------------------------")

    def _generate_log_dir(self, dir_postfix):
        self.experiment_id = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        prefix = self.experiment_configs["experiment_log_prefix"] # ""
        env_name = self.experiment_configs["env"]
        # log_dir = os.path.dirname(os.path.abspath(__file__)) + f'/selfplay-results/{prefix}save-' + ENV + '-' + ALGO + '-' + OBS + '-' + ACT + '-' + experiment_id
        self.log_env_dir_name = self.experiment_configs["log_env_dir_name"]
        self.log_main_dir =  f'{prefix}save-'+ env_name + '-' + self.experiment_id
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'selfplay-results-{dir_postfix}', self.log_env_dir_name , self.log_main_dir)
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
                notes=self.experiment_configs["wandb_notes"]+f"\n{self.args.notes}",
                # allow_val_change=True,
        )

        experiment_name = self.experiment_configs["experiment_name"]
        wandb.run.name = self.args.prefix + f"[Seed: {self.experiment_configs.get('seed_value', None)}] " + wandb.run.name + experiment_name + "-" + self.experiment_id
        wandb.run.save()
        wandb.save(self.experiment_filename)
        wandb.save(os.path.join(self.log_dir, "merged_config.json"))
        wandb.save("SelfPlayExp.py")
        wandb.save("callbacks.py")
        if(self.log_dir is not None):
            wandb.save(self.log_dir)


    def _init_exp(self, experiment_filename, logdir, wandb, load_config_flag=True):
        if(experiment_filename is None):
            self._init_argparse()
        
        self.clilog = cli_logger
        self.clilog.setLevel(self.cli_log_level)

        self.clilog.info(f"Parse from json file in {self.args.exp}" if experiment_filename is None else f"----- Loading experiment from: {experiment_filename}")
        self._load_configs(experiment_filename)
        
        if(logdir):
            self.log_dir = self._generate_log_dir()
            self.clilog.info(f"----- Initialize loggers")
            self._init_log_files()
            logger.configure(folder=self.log_dir)

        if(self.log_dir is not None):
            ExperimentParser.save_combined(os.path.join(self.log_dir, "merged_config.json"), self.merged_config)

        if(wandb):
            self.clilog.info(f"----- Initialize wandb")
            self._init_wandb()

        # They were moved down to be logged in wandb log
        self.clilog.info(f"----- Experiment logs are being stored in: {self.log_dir}")
        self.log_configs()
        
        self.THREADED = self.args.threaded
        if(self.THREADED):
            self.clilog.error(f"THREADED seems not working correctly")
            self.clilog.info(f"**** Experiment is THREADED ****")

        self.make_deterministic()
    
    def create_env(self, key, name, algorithm_class=PPO, opponent_archive=None, seed_value=None, sample_after_reset=False, sampling_parameters=None, ret_seed=False, gui=False):
        # TODO: add support for gui and reward_type as in drones environment
        # Key is just used to get the environment class_name for intialization of the environment
        if(seed_value == "random"):
            seed_value = datetime.now().microsecond//1000
        else:
            seed_value = self.seed_value if seed_value is None else seed_value
        agent_configs = self.agents_configs[key]
        agent_name = agent_configs["name"]
        env_class_name = agent_configs["env_class"]
        # if(isinstance(algorithm_class, PPO)):
        self.clilog.info(f"Create Env: {env_class_name}, Algorithm: {algorithm_class}, seed: {seed_value}")
        # Here e.g. SelfPlayPredEnv will use the archive only for load the opponent nothing more -> Pass the opponent archive
        reward_type = agent_configs.get("reward_type", None)
        params = dict(algorithm_class=algorithm_class, archive=opponent_archive, seed_val=seed_value, sample_after_reset=sample_after_reset, sampling_parameters=sampling_parameters, gui=gui, reward_type=reward_type)
        # env = None
        # if(reward_type is not None):
        #     params["reward_type"] = reward_type
        env = globals()[env_class_name](**params)#, opponent_selection=OPPONENT_SELECTION) #SelfPlayPredEnv()
        env._name = name+f"-({agent_name})"
        if(not ret_seed):
            return env
        return env, seed_value

    def _init_env(self):
        raise NotImplementedError("Initialization for environment is not implemented")

    def _init_models(self):
        raise NotImplementedError("Initialization for models is not implemented")