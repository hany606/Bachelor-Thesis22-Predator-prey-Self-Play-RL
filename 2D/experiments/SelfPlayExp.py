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


class SelfPlayExp:
    def __init__(self):
        pass


    def _init_argparse_training(self):
        parser = argparse.ArgumentParser(description='Self-play experiment training script')
        parser.add_argument('--exp', type=str, help='The experiemnt file path and name which the experiment should be loaded', metavar='')
        self.args = parser.parse_args()
        
    def _load_training_configs(self, filename=None):        
        self.experiment_filename = self.args.exp if filename is None else filename
        self.experiment_configs, self.agents_configs, self.evaluation_configs = ExperimentParser.load(self.experiment_filename)

    # Source: https://github.com/rlturkiye/flying-cavalry/blob/main/rllib/main.py
    def make_deterministic(self):
        seed = self.experiment_configs["seed_value"]
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

    def _check_cuda(self):
        # Source: https://github.com/rlturkiye/flying-cavalry/blob/main/rllib/main.py
        if torch.cuda.is_available():
            print("## CUDA available")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("## CUDA not available")

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

        self.envs_classes = {"pred":SelfPlayPredEnv, "prey":SelfPlayPreyEnv}

        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            opponent_name = agent_configs["opponent_name"]
            opponent_archive = self.archives[opponent_name]
            env_class_name = agent_configs["env_class"]
            
            # pred_env = create_env("SelfPlay1v1-Pred-v0", os.path.join(log_dir, "pred", "videos"), config={"log_dir": log_dir, "algorithm_class": PPO}) #SelfPlayPredEnv()
            # pred_env = create_env(SelfPlayPredEnv, log_dir=log_dir, algorithm_class=PPO, opponent_selection=OPPONENT_SELECTION)
            # pred_env.seed(SEED_VALUE)

            # Here SelfPlayPredEnv will use the archive only for load the opponent nothing more -> Pass the opponent archive
            env = globals()[env_class_name](log_dir=self.log_dir, algorithm_class=PPO, archive=opponent_archive)#, opponent_selection=OPPONENT_SELECTION) #SelfPlayPredEnv()
            env._name = "Training"
            self.envs[agent_name] = env

            eval_env = globals()[env_class_name](log_dir=self.log_dir, algorithm_class=PPO, archive=opponent_archive)#, opponent_selection=OPPONENT_SELECTION) #SelfPlayPredEnv()
            eval_env._name = "Evaluation"
            self.eval_envs[agent_name] = eval_env

    def _init_models(self):
        self.models = {}

        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            agent_env = self.envs[agent_name]

            model = PPO(agent_configs["policy"], 
                        agent_env,
                        clip_range=agent_configs["clip_range"], 
                        ent_coef=agent_configs["ent_coef"],
                        learning_rate=agent_configs["lr"], 
                        batch_size=agent_configs["batch_size"],
                        gamma=agent_configs["gamma"], 
                        verbose=2,
                        tensorboard_log=os.path.join(self.log_dir,agent_name),
                        n_epochs=agent_configs["n_epochs"]
                       )
            self.models[agent_name] = model
    
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
            

            env = self.envs[agent_name]
            eval_env = self.eval_envs[agent_name]

            eval_freq = agent_configs["eval_freq"]
            save_freq = agent_configs["save_freq"]

            num_eval_episodes = agent_configs["num_eval_episodes"]
            num_rounds = self.experiment_configs["num_rounds"]
            num_sampled_opponent_per_round = agent_configs["num_sampled_opponent_per_round"]


            opponent_selection = agent_configs["opponent_selection"]
            sample_after_rollout = agent_configs["sample_after_rollout"]

            eval_metric = agent_configs["eval_metric"]
            eval_opponent_selection = agent_configs["eval_opponent_selection"]

            agent_archive = self.archives[agent_name]
            opponent_archive = self.archives[opponent_name]



            # Here the EvalSaveCallback is used the archive to save the model and sample the opponent for evaluation
            evalsave_callback = EvalSaveCallback(eval_env=eval_env,
                                                    log_path=agent_path,
                                                    eval_freq=eval_freq,
                                                    n_eval_episodes=num_eval_episodes,
                                                    deterministic=True,
                                                    save_path=agent_path,
                                                    eval_metric=eval_metric,
                                                    eval_opponent_selection=eval_opponent_selection,
                                                    eval_sample_path=opponent_sample_path,
                                                    save_freq=save_freq,
                                                    archive={"self":agent_archive, "opponent":opponent_archive},
                                                    agent_name=agent_name,
                                                    num_rounds=num_rounds)
            self.evalsave_callbacks[agent_name] = evalsave_callback

            # Here the TrainingOpponentSelectionCallback is used the archive to sample the opponent for training
            # The name here pred_oppoenent -> the opponent of the predator
            opponent_selection_callback = TrainingOpponentSelectionCallback(sample_path=opponent_sample_path,
                                                                        env=env, 
                                                                        opponent_selection=opponent_selection,
                                                                        sample_after_rollout=sample_after_rollout,
                                                                        num_sampled_per_round=num_sampled_opponent_per_round,
                                                                        archive=opponent_archive)
            self.opponent_selection_callbacks[agent_name] = opponent_selection_callback

            wandb_callback = WandbCallback()
            self.wandb_callbacks[agent_name] = wandb_callback

    def _init_training(self, experiment_filename=None):
        if(experiment_filename is None):
            self._init_argparse_training()
        print(f"----- Loading experiment from: {experiment_filename}")
        self._load_training_configs(experiment_filename)
        
        self._check_cuda()
        
        self.log_dir = self._generate_log_dir()
        print(f"----- Experiment logs are being stored in: {self.log_dir}")
        print(f"----- Initialize loggers, wandb")
        self._init_wandb()
        self._init_log_files()
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
        agents_names_list = []
        for _,v in agents_order.items():
            agents_names_list.append(v)
        return agents_names_list

    def _change_archives(self, agent_name, archive):
        self.archives[agent_name].change_archive_core(archive)

    def train(self):
        self._init_training()
        num_rounds = self.experiment_configs["num_rounds"]
        population_size = self.experiment_configs["population_size"]    # population here has a shared archive
        agents_names_list = self._create_agents_names_list()
        self.old_archives = {}
        self.new_archives = {}
        # --------------------------------------------- Training Rounds ---------------------------------------------
        # Here alternate training
        for round_num in range(num_rounds):
            # --------------------------------------------- Starting of the round ---------------------------------------------
            # Copy the archives before the training to the old_archives to be loaded before the training as opponents  
            for agent_name in agents_names_list:
                self.old_archives[agent_name] = deepcopy(self.archives[agent_name])

            # Train all agents then evaluate
            # --------------------------------------------- Training agent by agent ---------------------------------------------
            # In each loop, the agent is training and the opponent is not training (Single RL agent configuration)
            for agent_name in agents_names_list:
                opponent_name = self.agents_configs[agent_name]["opponent_name"]
                # Agent will train on the previous version of the archive of the opponent agent before this round
                self.archives[opponent_name].change_archive_core(self.old_archives[opponent_name])
                num_timesteps = self.agents_configs[agent_name]["num_timesteps"]

                self.evalsave_callbacks[agent_name].set_name_prefix(f"history_{round_num}")
                for population_num in range(population_size):
                    print(f"------------------- Train {agent_name}, round: {round_num},  population: {population_num}--------------------")
                    self.models[agent_name].learn(total_timesteps=num_timesteps, 
                                                callback=[
                                                            self.opponent_selection_callbacks[agent_name], 
                                                            self.evalsave_callbacks[agent_name],
                                                            self.wandb_callbacks[agent_name]
                                                            ], 
                                                reset_num_timesteps=False)
                self.new_archives[agent_name] = deepcopy(self.archives[agent_name]) # Save the resulted archive for each agent to be stored after the training process for all the agents

            for agent_name in agents_names_list:
                self.archives[agent_name].change_archive_core(self.new_archives[agent_name])
            # --------------------------------------------- Evaluating agent by agent ---------------------------------------------            
            for agent_name in agents_names_list:
                agent_config = self.agents_configs[agent_name]
                opponent_name = agent_config["opponent_name"]
                num_eval_episodes = agent_config["num_eval_episodes"]
                final_save_freq = agent_config["final_save_freq"]

                print(f"Round: {round_num} -> HeatMap Evaluation for current round version of {agent_name} vs {opponent_name}")
                self.evalsave_callbacks[agent_name].compute_eval_matrix_aggregate(prefix="history_", round_num=round_num, n_eval_rep=num_eval_episodes, algorithm_class=PPO, opponents_path=os.path.join(self.log_dir, opponent_name), agents_path=os.path.join(self.log_dir, agent_name))

                if(round_num%final_save_freq == 0):
                    # TODO: Change it to save the best model till now, not the latest (How to define the best model)
                    self.models[agent_name].save(os.path.join(self.log_dir, agent_name, "final_model"))
                    np.save(os.path.join(self.log_dir, agent_name, "evaluation_matrix"), self.evalsave_callbacks[agent_name].evaluation_matrix)
            # --------------------------------------------- End of the round ---------------------------------------------

        
        for i,agent_name in enumerate(agents_names_list):
            evalsave_callback = self.evalsave_callbacks[agent_name]
            opponent_name = self.agents_configs[agent_name]["opponent_name"]

            evaluation_matrix = self.evalsave_callbacks[agent_name].evaluation_matrix
            evaluation_matrix = evaluation_matrix if(i%2 == 0) else evaluation_matrix.T # .T in order to make the x-axis predators and y-axis are preys
            
            wandb.log({f"{agent_name}/mid_eval/heatmap"'': wandb.plots.HeatMap([i for i in range(num_rounds)], [i for i in range(num_rounds)], evaluation_matrix, show_text=True)})


            evalsave_callback._save_model_core()

            self.models[agent_name].save(os.path.join(self.log_dir, agent_name, "final_model"))

            np.save(os.path.join(self.log_dir, agent_name, "evaluation_matrix"), evalsave_callback.evaluation_matrix)



            print(f"Post Evaluation for {agent_name}:")
            evalsave_callback.post_eval(opponents_path=os.path.join(self.log_dir, opponent_name))

            self.envs[agent_name].close()
            self.eval_envs[agent_name].close()