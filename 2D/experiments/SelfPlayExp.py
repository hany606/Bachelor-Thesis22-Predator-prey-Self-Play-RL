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

from bach_utils.json_parser import ExperimentParser


# TODO: Should I do different classes?
# Here the class is for the whole experiment (train, evaluation(heatmaps, plots, ...etc), test (rendering))
class SelfPlayExp:
    def __init__(self):
        self.args = None
        self.envs = None
        self.eval_envs = None
        self.evalsave_callbacks = None
        self.experiment_configs = None
        self.agents_configs = None
        self.experiment_filename = None
        self.evaluation_configs = None
        self.logdir = None
        self.archives = None
        self.models = None
        self.opponent_selection_callbacks = None
        self.wandb_callbacks = None

    
    def _init_argparse_training(self):
        parser = argparse.ArgumentParser(description='Self-play experiment training script')
        parser.add_argument('--exp', type=str, help='The experiemnt configuration file path and name which the experiment should be loaded', metavar='')
        self.args = parser.parse_args()
        
    def _load_training_configs(self, filename):        
        self.experiment_filename = self.args.exp if filename is None else filename
        self.experiment_configs, self.agents_configs, self.evaluation_configs, self.testing_configs = ExperimentParser.load(self.experiment_filename)

    def log_configs(self):
        print(f"Experiment configs: {self.experiment_configs}")
        print(f"Agents configs: {self.agents_configs}")
        print(f"Evaluation configs: {self.evaluation_configs}")

    def _check_cuda(self):
        # Source: https://github.com/rlturkiye/flying-cavalry/blob/main/rllib/main.py
        if torch.cuda.is_available():
            print("## CUDA available")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            return 1
        else:
            print("## CUDA not available")
            return 0
    
    # Source: https://github.com/rlturkiye/flying-cavalry/blob/main/rllib/main.py
    def make_deterministic(self):
        seed = self.experiment_configs["seed_value"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        cuda_flag = self._check_cuda()
        if(cuda_flag):
            # see https://github.com/pytorch/pytorch/issues/47672
            cuda_version = torch.version.cuda
            if cuda_version is not None and float(torch.version.cuda) >= 10.2:
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'
            else:
                torch.set_deterministic(True)  # Not all Operations support this.
            # This is only for Convolution no problem
            torch.backends.cudnn.deterministic = True

    def _generate_log_dir(self):
        experiment_id = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        prefix = self.experiment_configs["experiment_log_prefix"] # ""
        env_name = self.experiment_configs["env"]
        # log_dir = os.path.dirname(os.path.abspath(__file__)) + f'/selfplay-results/{prefix}save-' + ENV + '-' + ALGO + '-' + OBS + '-' + ACT + '-' + experiment_id
        log_dir = os.path.dirname(os.path.abspath(__file__)) + f'/selfplay-results/{prefix}save-' + env_name + '-' + experiment_id
        return log_dir
        
    def _init_wandb(self):
        wandb_experiment_config = {"experiment": self.experiment_configs,
                                   "agents"    : self.agents_configs,
                                   "evaluation": self.evaluation_configs
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
        wandb.run.name = wandb.run.name + experiment_name
        wandb.run.save()
        wandb.save(self.experiment_filename)
        wandb.save("SelfPlayExp.py")
        wandb.save("callbacks.py")

    def _init_log_files(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir + '/')
        if not os.path.exists(os.path.join(self.log_dir, "pred")):
            os.makedirs(os.path.join(self.log_dir, "pred") + '/')
        if not os.path.exists(os.path.join(self.log_dir, "prey")):
            os.makedirs(os.path.join(self.log_dir, "prey") + '/')

    def _init_archives(self):
        self.archives = {}

        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            eval_opponent_selection = agent_configs["eval_opponent_selection"]
            opponent_selection = agent_configs["opponent_selection"]
            self.archives[agent_name] = Archive(sorting_keys=[eval_opponent_selection, opponent_selection],
                                                sorting=True,
                                                moving_least_freq_flag=False,
                                                save_path=os.path.join(self.log_dir, agent_name)
                                               )

    def _init_envs(self):
        self.envs = {}
        self.eval_envs = {}

        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            opponent_name = agent_configs["opponent_name"]
            opponent_archive = self.archives[opponent_name]
            env_class_name = agent_configs["env_class"]
            seed_val = self.experiment_configs["seed_value"]
            
            # pred_env = create_env("SelfPlay1v1-Pred-v0", os.path.join(log_dir, "pred", "videos"), config={"log_dir": log_dir, "algorithm_class": PPO}) #SelfPlayPredEnv()
            # pred_env = create_env(SelfPlayPredEnv, log_dir=log_dir, algorithm_class=PPO, opponent_selection=OPPONENT_SELECTION)
            # pred_env.seed(SEED_VALUE)

            # Here SelfPlayPredEnv will use the archive only for load the opponent nothing more -> Pass the opponent archive
            self.envs[agent_name] = globals()[env_class_name](log_dir=self.log_dir, algorithm_class=PPO, archive=opponent_archive, seed_val=seed_val)#, opponent_selection=OPPONENT_SELECTION) #SelfPlayPredEnv()
            self.envs[agent_name]._name = "Training"

            self.eval_envs[agent_name] = globals()[env_class_name](log_dir=self.log_dir, algorithm_class=PPO, archive=opponent_archive, seed_val=seed_val)#, opponent_selection=OPPONENT_SELECTION) #SelfPlayPredEnv()
            self.eval_envs[agent_name]._name = "Evaluation"

    def _init_models(self):
        self.models = {}

        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]

            self.models[agent_name] = PPO(  agent_configs["policy"], 
                                            self.envs[agent_name],
                                            clip_range=agent_configs["clip_range"], 
                                            ent_coef=agent_configs["ent_coef"],
                                            learning_rate=agent_configs["lr"], 
                                            batch_size=agent_configs["batch_size"],
                                            gamma=agent_configs["gamma"], 
                                            verbose=2,
                                            tensorboard_log=os.path.join(self.log_dir,agent_name),
                                            n_epochs=agent_configs["n_epochs"]
                                        )
    
    def _init_callbacks(self):
        self.opponent_selection_callbacks = {}
        self.evalsave_callbacks = {}
        self.wandb_callbacks = {}

        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            opponent_name = agent_configs["opponent_name"]

            opponent_sample_path = os.path.join(self.log_dir, opponent_name)
            agent_path = os.path.join(self.log_dir, agent_name)
            

            # Here the EvalSaveCallback is used the archive to save the model and sample the opponent for evaluation
            self.evalsave_callbacks[agent_name] = EvalSaveCallback(eval_env=self.eval_envs[agent_name],
                                                    log_path=agent_path,
                                                    eval_freq=int(agent_configs["eval_freq"]),
                                                    n_eval_episodes=agent_configs["num_eval_episodes"],
                                                    deterministic=True,
                                                    save_path=agent_path,
                                                    eval_metric=agent_configs["eval_metric"],
                                                    eval_opponent_selection=agent_configs["eval_opponent_selection"],
                                                    eval_sample_path=opponent_sample_path,
                                                    save_freq=int(agent_configs["save_freq"]),
                                                    archive={"self":self.archives[agent_name], "opponent":self.archives[opponent_name]},
                                                    agent_name=agent_name,
                                                    num_rounds=self.experiment_configs["num_rounds"])

            # Here the TrainingOpponentSelectionCallback is used the archive to sample the opponent for training
            # The name here pred_oppoenent -> the opponent of the predator
            self.opponent_selection_callbacks[agent_name] = TrainingOpponentSelectionCallback(sample_path=opponent_sample_path,
                                                                        env=self.envs[agent_name], 
                                                                        opponent_selection=agent_configs["opponent_selection"],
                                                                        sample_after_rollout=agent_configs["sample_after_rollout"],
                                                                        num_sampled_per_round=agent_configs["num_sampled_opponent_per_round"],
                                                                        archive=self.archives[opponent_name])
            self.wandb_callbacks[agent_name] = WandbCallback()

    def _init_training(self, experiment_filename):
        if(experiment_filename is None):
            self._init_argparse_training()
        print(f"----- Loading experiment from: {experiment_filename}")
        self._load_training_configs(experiment_filename)
                
        self.log_dir = self._generate_log_dir()
        print(f"----- Experiment logs are being stored in: {self.log_dir}")
        print(f"----- Initialize loggers, wandb")
        self._init_wandb()
        self._init_log_files()
        # They were moved down to be logged in wandb log
        self.log_configs()

        logger.configure(folder=self.log_dir)

        self.make_deterministic()

        print(f"----- Initialize archives, envs, models, callbacks")
        # Create the archives to store the models in the cache
        self._init_archives()
        # Create training and evaluation environments
        self._init_envs()
        # Create models for training based on RL algorithm
        self._init_models()
        # Create callbacks for evaluation and opponent selection
        self._init_callbacks()

    def _create_agents_names_list(self):
        # ids of the agents should be sequentially from 0->n (integers)
        agents_order = self.experiment_configs["agents_order"]
        agents_names_list = [None for i in range(len(agents_order.keys()))]
        for k,v in agents_order.items():
            agents_names_list[int(k)] = v
        return agents_names_list

    def _change_archives(self, agent_name, archive):
        self.archives[agent_name].change_archive_core(archive)

    def train(self, experiment_filename=None):
        self._init_training(experiment_filename=experiment_filename)
        num_rounds = self.experiment_configs["num_rounds"]
        population_size = self.experiment_configs["population_size"]    # population here has a shared archive
        agents_names_list = self._create_agents_names_list()
        self.old_archives = {}
        self.new_archives = {}
        # --------------------------------------------- Training Rounds ---------------------------------------------
        # Here alternate training
        for round_num in range(num_rounds):
            # wandb.log({f"round_num": round_num}, step=)
            # --------------------------------------------- Starting of the round ---------------------------------------------
            # Copy the archives before the training to the old_archives to be loaded before the training as opponents  
            for i,agent_name in enumerate(agents_names_list):
                self.evalsave_callbacks[agent_name].set_name_prefix(f"history_{round_num}")
                self.old_archives[agent_name] = deepcopy(self.archives[agent_name])

            # Train all agents then evaluate
            # --------------------------------------------- Training agent by agent ---------------------------------------------
            # In each loop, the agent is training and the opponent is not training (Single RL agent configuration)
            for agent_idx, agent_name in enumerate(agents_names_list):

                opponent_name = self.agents_configs[agent_name]["opponent_name"]
                # Agent will train on the previous version of the archive of the opponent agent before this round
                if(self.experiment_configs.get("parallel_alternate_training", True)):
                    self.archives[opponent_name].change_archive_core(self.old_archives[opponent_name])

                for population_num in range(population_size):
                    print(f"------------------- Train {agent_name}, round: {round_num},  population: {population_num}--------------------")
                    self.models[agent_name].learn(total_timesteps=int(self.agents_configs[agent_name]["num_timesteps"]), 
                                                  callback=[
                                                            self.opponent_selection_callbacks[agent_name], 
                                                            self.evalsave_callbacks[agent_name],
                                                            self.wandb_callbacks[agent_name]
                                                            ], 
                                                reset_num_timesteps=False)
                self.new_archives[agent_name] = deepcopy(self.archives[agent_name]) # Save the resulted archive for each agent to be stored after the training process for all the agents
                # wandb.log({f"round_num": round_num})

            if(self.experiment_configs.get("parallel_alternate_training", True)):    
                for agent_name in agents_names_list:
                    self.archives[agent_name].change_archive_core(self.new_archives[agent_name])
            print(f"------------------- Evaluation (Heatmap) --------------------")
            # --------------------------------------------- Evaluating agent by agent ---------------------------------------------            
            for j,agent_name in enumerate(agents_names_list):
                agent_config = self.agents_configs[agent_name]
                opponent_name = agent_config["opponent_name"]
                num_heatmap_eval_episodes = agent_config["num_heatmap_eval_episodes"]
                final_save_freq = agent_config["final_save_freq"]
                heatmap_log_freq = agent_config["heatmap_log_freq"]

                print(f"Round: {round_num} -> HeatMap Evaluation for current round version of {agent_name} vs {opponent_name}")
                self.evalsave_callbacks[agent_name].compute_eval_matrix_aggregate(prefix="history_", round_num=round_num, n_eval_rep=num_heatmap_eval_episodes, algorithm_class=PPO, opponents_path=os.path.join(self.log_dir, opponent_name), agents_path=os.path.join(self.log_dir, agent_name))
                
                if(round_num%heatmap_log_freq == 0 or round_num==num_rounds-1):
                    # Log intermediate results for the heatmap
                    evaluation_matrix = self.evalsave_callbacks[agent_name].evaluation_matrix
                    evaluation_matrix = evaluation_matrix if(j%2 == 0) else evaluation_matrix.T # .T in order to make the x-axis predators and y-axis are preys
                    if(round_num==num_rounds-1):
                        wandb.log({f"{agent_name}/heatmap"'': wandb.plots.HeatMap([i for i in range(num_rounds)], [i for i in range(num_rounds)], evaluation_matrix, show_text=True)})
                    wandb.log({f"{agent_name}/mid_eval/heatmap"'': wandb.plots.HeatMap([i for i in range(num_rounds)], [i for i in range(num_rounds)], evaluation_matrix, show_text=False)})

                if(round_num%final_save_freq == 0 or round_num==num_rounds-1):
                    # TODO: Change it to save the best model till now, not the latest (How to define the best model)
                    self.models[agent_name].save(os.path.join(self.log_dir, agent_name, "final_model"))
                    np.save(os.path.join(self.log_dir, agent_name, "evaluation_matrix"), self.evalsave_callbacks[agent_name].evaluation_matrix)
            # --------------------------------------------- End of the round ---------------------------------------------

        
        for j,agent_name in enumerate(agents_names_list):
            print(f"Post Evaluation for {agent_name}:")
            self.evalsave_callbacks[agent_name].post_eval(opponents_path=os.path.join(self.log_dir, self.agents_configs[agent_name]["opponent_name"]))

            self.envs[agent_name].close()
            self.eval_envs[agent_name].close()

    def _validate_argparse(self):
        if(self.args.exp is not None):
            return True
        if(self.args.pred_path is None or self.args.prey_path):
            if(self.args.exp is None):
                raise ValueError("exp or (pred-path and prey-path) must be defined")
            if(self.args.path is None):
                raise ValueError("path or (pred-path and prey-path) must be defined")

        if(',' not in self.args.mode):
            raise ValueError("Mode should be in the following form <pred_mode>, <prey_mode>")
        self.testing_modes = self.args.mode.lower().strip().split(',')

        if(sum([i in ["gen", "all", "random", "limit_s", "limit", "limit_e"] for i in self.testing_modes]) != len(self.testing_modes)):
            raise ValueError("Modes should be one of the following (Gen, All, Random, Limit)")

        self.testing_target_agents_configs = [self.args.pred, self.args.prey]
        self.testing_target_agents_indices = [None for i in range(len(self.testing_target_agents_configs))]
        for i in range(len(self.testing_target_agents_configs)):
            if(self.testing_modes[i] == "limit"):
                # if there is no comma, it means that it will start from 0 and finishes with that limit
                if(',' not in self.testing_target_agents_configs[i]):
                    self.testing_target_agents_indices[i] = [0, int(self.testing_target_agents_configs[i])]
                # if there is a comma, put the limit
                else:
                    idxs = self.args.mode.split(',')
                    self.testing_target_agents_indices[i] = [int(idx) for idx in idxs]

            # if the limit is that the start of the tested agents is that index and the end till the end
            elif(self.testing_modes[i] == "limit_s"):
                self.testing_target_agents_indices[i] = [int(self.testing_target_agents_configs[i]), -1]
            
            # if the limit is that the end of the tested agents is that index (including that index)
            elif(self.testing_modes[i] == "limit_e"):
                self.testing_target_agents_indices[i] = [0, int(self.testing_target_agents_configs[i])+1]

            elif(self.testing_modes[i] == "gen"):
                self.testing_taarget_agent_indices[i] = [int(self.testing_target_agents_configs[i]), int(self.testing_target_agents_configs[i])+1]

            elif(self.testing_modes[i] == "all"):
                self.testing_taarget_agent_indices[i] = [0, -1]
            
            elif(self.testing_modes[i] == "random"):
                self.testing_taarget_agent_indices[i] = [None, None]

    def _init_argparse_testing(self):
        parser = argparse.ArgumentParser(description='Self-play experiment testing script')
        parser.add_argument('--exp', type=str, default=None, help='The experiemnt configuration file path and name which the experiment should be loaded', metavar='')
        parser.add_argument('--mode', type=str, default=None, help='The mode for the evaluation (<pred>,<prey>) (Gen, All, Random, Limit: from specific generation to another)', metavar='') #(Gen vs All), (Gen vs Gen), (All vs All), (Gen vs Random), (Random vs Gen), (Random vs Random)', metavar='')
        parser.add_argument('--pred-path', type=str, default=None, help='Path for predator files', metavar='')
        parser.add_argument('--prey-path', type=str, default=None, help='Path for prey files', metavar='')
        parser.add_argument('--path', type=str, default=None, help='Path for predator and prey files', metavar='')
        parser.add_argument('--pred', type=str, default=None, help='targets versions for predator', metavar='')
        parser.add_argument('--prey', type=str, default=None, help='targets versions for prey', metavar='')

        self.args = parser.parse_args()
        self._validate_argparse()


    def _load_testing_configs(self, filename):
        self.testing_filename = self.args.exp if filename is None else filename
        if(self.testing_filename is not None):
            self.experiment_configs, self.agents_configs, self.evaluation_configs, self.testing_configs = ExperimentParser.load(self.experiment_filename)
        # if at the end it is None (filename is none, args.exp is none), then the user should have input the paths
        else:
            self.testing_configs = {
                                    "pred-path":self.pred_path,
                                    "prey-path":self.prey_path,
                                    }

    def _init_testing(self, testing_filename):
        if(testing_filename is None):
            self._init_argparse_evaluation_render()
        
        print(f"----- Loading experiment from: {testing_filename}")
        self._load_training_configs(testing_filename)

        self._init_testing_configs()

    def test(self, testing_filename=None):
        self._init_testing(test_filename=testing_filename)
