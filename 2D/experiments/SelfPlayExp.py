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


# This is a modified PPO to tackle problem related of loading from different version of pickle than it was saved with
class PPOMod(PPO):
    def __init__(self, *args, **kwargs):
        super(PPOMod, self).__init__(*args, **kwargs)

    # To fix issue while loading when loading from different versions of pickle and python from the server and the local machine
    # https://stackoverflow.com/questions/63329657/python-3-7-error-unsupported-pickle-protocol-5
    @staticmethod
    def load(model_path, env):
        custom_objects = {
            "lr_schedule": lambda x: .003,
            "clip_range": lambda x: .02
        }
        return PPO.load(model_path, env, custom_objects=custom_objects)

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
        parser.add_argument('--exp', type=str, help=help, metavar='')
        self.args = parser.parse_args()

    def _load_configs(self, filename):        
        self.experiment_filename = self.args.exp if filename is None else filename
        self.experiment_configs, self.agents_configs, self.evaluation_configs, self.testing_configs = ExperimentParser.load(self.experiment_filename)
        self.seed_value = self.experiment_configs["seed_value"] if self.seed_value is None else self.seed_value

    def log_configs(self):
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
        log_dir = os.path.dirname(os.path.abspath(__file__)) + f'/selfplay-results-{dir_postfix}/{prefix}save-' + env_name + '-' + self.experiment_id
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
        wandb.run.name = wandb.run.name + experiment_name + "-" + self.experiment_id
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
    
    def create_env(self, key, name, algorithm_class=PPO, opponent_archive=None, seed_value=None):
        seed_value = self.seed_value if seed_value is None else seed_value
        agent_configs = self.agents_configs[key]
        agent_name = agent_configs["name"]
        env_class_name = agent_configs["env_class"]
        # Here e.g. SelfPlayPredEnv will use the archive only for load the opponent nothing more -> Pass the opponent archive
        env = globals()[env_class_name](algorithm_class=algorithm_class, archive=opponent_archive, seed_val=seed_value)#, opponent_selection=OPPONENT_SELECTION) #SelfPlayPredEnv()
        env._name = name+f"-({agent_name})"
        return env

    def _init_env(self):
        raise NotImplementedError("Initialization for environment is not implemented")

    def _init_models(self):
        raise NotImplementedError("Initialization for models is not implemented")

# TODO: Should I do different classes?
# Here the class is for the whole experiment (train, evaluation(heatmaps, plots, ...etc), test (rendering))
class SelfPlayTraining(SelfPlayExp):
    def __init__(self, seed_value=None):
        super(SelfPlayTraining, self).__init__()
        self.envs = None
        self.eval_envs = None
        self.evalsave_callbacks = None
        self.archives = None
        self.models = None
        self.opponent_selection_callbacks = None
        self.wandb_callbacks = None
        self.seed_value = seed_value

    def _init_argparse(self):
        super(SelfPlayTraining, self)._init_argparse(description='Self-play experiment training script', help='The experiemnt configuration file path and name which the experiment should be loaded')
    
    def _generate_log_dir(self):
        return super(SelfPlayTraining, self)._generate_log_dir(dir_postfix="train")

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

            self.envs[agent_name] = super(SelfPlayTraining, self).create_env(key=k, name="Training", opponent_archive=opponent_archive)
            self.eval_envs[agent_name] = super(SelfPlayTraining, self).create_env(key=k, name="Evaluation", opponent_archive=opponent_archive)

    def _init_models(self):
        self.models = {}
        population_size = self.experiment_configs["population_size"]    # population here has a shared archive
        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            self.models[agent_name] = []

            for population_num in range(population_size):
                self.models[agent_name].append( PPO(agent_configs["policy"], 
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
                                             )
    
    def _init_callbacks(self):
        self.opponent_selection_callbacks = {}
        self.evalsave_callbacks = {}
        self.wandb_callbacks = {}
        population_size = self.experiment_configs["population_size"]    # population here has a shared archive
        # self.evalsave_callbacks_master_idx = population_size - 1

        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            opponent_name = agent_configs["opponent_name"]

            opponent_sample_path = os.path.join(self.log_dir, opponent_name)
            agent_path = os.path.join(self.log_dir, agent_name)
            
            self.evalsave_callbacks[agent_name] = []
            for population_num in range(population_size):
                # enable_evaluation_matrix = (population_num == self.evalsave_callbacks_master_idx)
                enable_evaluation_matrix = True
                # Here the EvalSaveCallback is used the archive to save the model and sample the opponent for evaluation
                self.evalsave_callbacks[agent_name].append(
                                                            EvalSaveCallback(eval_env=self.eval_envs[agent_name],
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
                                                                            num_rounds=self.experiment_configs["num_rounds"],
                                                                            seed_value=self.seed_value,
                                                                            enable_evaluation_matrix=enable_evaluation_matrix,
                                                                            randomly_reseed_sampling=agent_configs.get("randomly_reseed_sampling", False))
                                                            )
                self.evalsave_callbacks[agent_name][-1].population_idx = population_num

            # Here the TrainingOpponentSelectionCallback is used the archive to sample the opponent for training
            # The name here pred_oppoenent -> the opponent of the predator
            # TODO: extend maybe we can have different opponent selection criteria for each population! Hmmm interesting (I wanna see the results)!
            self.opponent_selection_callbacks[agent_name] = TrainingOpponentSelectionCallback(sample_path=opponent_sample_path,
                                                                        env=self.envs[agent_name], 
                                                                        opponent_selection=agent_configs["opponent_selection"],
                                                                        sample_after_rollout=agent_configs["sample_after_rollout"],
                                                                        num_sampled_per_round=agent_configs["num_sampled_opponent_per_round"],
                                                                        archive=self.archives[opponent_name],
                                                                        randomly_reseed_sampling=agent_configs.get("randomly_reseed_sampling", False))
            self.wandb_callbacks[agent_name] = WandbCallback()

    def _init_training(self, experiment_filename):
        super(SelfPlayTraining, self)._init_exp(experiment_filename, True, True)
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
            wandb.log({f"progress (round_num)": round_num})
            # --------------------------------------------- Starting of the round ---------------------------------------------
            # Copy the archives before the training to the old_archives to be loaded before the training as opponents  
            for i,agent_name in enumerate(agents_names_list):
                for population_num in range(population_size):
                    self.evalsave_callbacks[agent_name][population_num].set_name_prefix(f"history_{round_num}")
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
                    print(f"Model mem id: {self.models[agent_name][population_num]}")
                    # Here the model for a different population is trained as they are in parallel (Not sequentioal for the population)
                    # However, different population contributes here in the same archive, and select opponent from the same archive as all agents
                    self.models[agent_name][population_num].learn(  total_timesteps=int(self.agents_configs[agent_name]["num_timesteps"]), 
                                                                    callback=[
                                                                                self.opponent_selection_callbacks[agent_name], 
                                                                                self.evalsave_callbacks[agent_name][population_num],
                                                                                self.wandb_callbacks[agent_name]
                                                                             ], 
                                                                    reset_num_timesteps=False)
                self.new_archives[agent_name] = deepcopy(self.archives[agent_name]) # Save the resulted archive for each agent to be stored after the training process for all the agents
                # wandb.log({f"round_num": round_num})

            if(self.experiment_configs.get("parallel_alternate_training", True)):    
                for agent_name in agents_names_list:
                    self.archives[agent_name].change_archive_core(self.new_archives[agent_name])
            # print(f"------------------- Evaluation (Heatmap) --------------------")
            # --------------------------------------------- Evaluating agent by agent ---------------------------------------------            
            for j,agent_name in enumerate(agents_names_list):
                agent_config = self.agents_configs[agent_name]
                opponent_name = agent_config["opponent_name"]
                num_heatmap_eval_episodes = agent_config["num_heatmap_eval_episodes"]
                final_save_freq = agent_config["final_save_freq"]
                heatmap_log_freq = agent_config["heatmap_log_freq"]
                aggregate_eval_matrix = agent_config["aggregate_eval_matrix"]
                
                if(aggregate_eval_matrix):
                    print("--------------------------------------------------------------")
                    print(f"Round: {round_num} -> Aggregate HeatMap Evaluation for current round version of {agent_name} vs {opponent_name}")
                    # It does not matter which population will compute the evaluation matrix as all of them shares the same archive
                    # self.evalsave_callbacks[agent_name][self.evalsave_callbacks_master_idx].compute_eval_matrix_aggregate(prefix="history_", round_num=round_num, n_eval_rep=num_heatmap_eval_episodes, algorithm_class=PPO, opponents_path=os.path.join(self.log_dir, opponent_name), agents_path=os.path.join(self.log_dir, agent_name))
        
                    # Now, we will calculate the evaluation matrix (Tournment table) for each population and then take the average at the end
                    for population_num in range(population_size):
                        self.evalsave_callbacks[agent_name][population_num].compute_eval_matrix_aggregate(prefix="history_", round_num=round_num, n_eval_rep=num_heatmap_eval_episodes, algorithm_class=PPO, opponents_path=os.path.join(self.log_dir, opponent_name), agents_path=os.path.join(self.log_dir, agent_name))
                        
                
                # Log intermediate results for the heatmap
                if(aggregate_eval_matrix and (round_num%heatmap_log_freq == 0 or round_num==num_rounds-1)): # The logging frequency or the last round
                    evaluation_matrices = []
                    for population_num in range(population_size):
                        evaluation_matrix = self.evalsave_callbacks[agent_name][population_num].evaluation_matrix
                        evaluation_matrix = evaluation_matrix if(j%2 == 0) else evaluation_matrix.T # .T in order to make the x-axis predators and y-axis are preys
                        evaluation_matrices.append(evaluation_matrix)
                    mean_evaluation_matrix = np.mean(evaluation_matrices, axis=0)
                    std_evaluation_matrix = np.std(evaluation_matrices, axis=0)

                    if(round_num==num_rounds-1):
                        wandb.log({f"{agent_name}/heatmap"'': wandb.plots.HeatMap([i for i in range(num_rounds)], [i for i in range(num_rounds)], mean_evaluation_matrix, show_text=True)})
                        wandb.log({f"{agent_name}/std_heatmap"'': wandb.plots.HeatMap([i for i in range(num_rounds)], [i for i in range(num_rounds)], std_evaluation_matrix, show_text=True)})

                    wandb.log({f"{agent_name}/mid_eval/heatmap"'': wandb.plots.HeatMap([i for i in range(num_rounds)], [i for i in range(num_rounds)], mean_evaluation_matrix, show_text=False)})
                    wandb.log({f"{agent_name}/mid_eval/std_heatmap"'': wandb.plots.HeatMap([i for i in range(num_rounds)], [i for i in range(num_rounds)], std_evaluation_matrix, show_text=False)})
                    np.save(os.path.join(self.log_dir, agent_name, "evaluation_matrix"), mean_evaluation_matrix)
                    wandb.save(os.path.join(self.log_dir, agent_name, "evaluation_matrix")+".npy")
                
                if(round_num%final_save_freq == 0 or round_num==num_rounds-1):
                    print(f"------------------- Models saving freq --------------------")
                    # TODO: Change it to save the best model till now, not the latest (How to define the best model)
                    for population_num in range(population_size):
                        self.models[agent_name][population_num].save(os.path.join(self.log_dir, agent_name, f"final_model_pop{population_num}"))
                    # To keep the consistency of the old script generations
                    self.models[agent_name][-1].save(os.path.join(self.log_dir, agent_name, "final_model"))
            # --------------------------------------------- End of the round ---------------------------------------------

        
        for j,agent_name in enumerate(agents_names_list):
            agent_config = self.agents_configs[agent_name]
            aggregate_eval_matrix = agent_config["aggregate_eval_matrix"]

            if(not aggregate_eval_matrix):
                opponent_name = agent_config["opponent_name"]
                num_heatmap_eval_episodes = agent_config["num_heatmap_eval_episodes"]
                eval_matrix_testing_freq = agent_config["eval_matrix_testing_freq"]

                evaluation_matrices = []
                for population_num in range(population_size):
                    print(f"Full evaluation matrix for {agent_name} (population: {population_num})")
                    axis = self.evalsave_callbacks[agent_name][population_num].compute_eval_matrix(prefix="history_", num_rounds=num_rounds, n_eval_rep=num_heatmap_eval_episodes, algorithm_class=PPO, opponents_path=os.path.join(self.log_dir, opponent_name), agents_path=os.path.join(self.log_dir, agent_name), freq=eval_matrix_testing_freq, population_size=population_size)
                    evaluation_matrix = self.evalsave_callbacks[agent_name][population_num].evaluation_matrix
                    evaluation_matrix = evaluation_matrix if(j%2 == 0) else evaluation_matrix.T # .T in order to make the x-axis predators and y-axis are preys
                    evaluation_matrices.append(evaluation_matrix)
                mean_evaluation_matrix = np.mean(evaluation_matrices, axis=0)
                std_evaluation_matrix = np.std(evaluation_matrices, axis=0)
                # One with text and other without (I kept the name in wandb just not to be a problem with previous experiments)
                wandb.log({f"{agent_name}/heatmap"'': wandb.plots.HeatMap(axis[0], axis[1], mean_evaluation_matrix, show_text=True)})
                wandb.log({f"{agent_name}/mid_eval/heatmap"'': wandb.plots.HeatMap(axis[0], axis[1], mean_evaluation_matrix, show_text=False)})
                wandb.log({f"{agent_name}/std_heatmap"'': wandb.plots.HeatMap(axis[0], axis[1], std_evaluation_matrix, show_text=True)})

                np.save(os.path.join(self.log_dir, agent_name, "evaluation_matrix"), mean_evaluation_matrix)
                wandb.save(os.path.join(self.log_dir, agent_name, "evaluation_matrix")+".npy")
            
            # TODO: should I do it for all the agents or not or just one?
            # Now it is made for all the agents and takes the mean and the standard deviation
            post_eval_list = []
            for population_num in range(population_size):
                print("-----------------------------------------------------------------------")
                print(f"Post Evaluation for {agent_name} (population: {population_num})")
                print("-----------------------------------------------------------------------")
                eval_return_list = self.evalsave_callbacks[agent_name][population_num].post_eval(opponents_path=os.path.join(self.log_dir, self.agents_configs[agent_name]["opponent_name"]), population_size=population_size)
                post_eval_list.append(eval_return_list)
            mean_post_eval = np.mean(post_eval_list, axis=0)
            std_post_eval = np.std(post_eval_list, axis=0)
            data = [[x, y] for (x, y) in zip([i for i in range(len(mean_post_eval))], mean_post_eval)]
            table = wandb.Table(data=data, columns = ["opponent idx", "win-rate"])
            std_data = [[x, y] for (x, y) in zip([i for i in range(len(std_post_eval))], std_post_eval)]
            std_table = wandb.Table(data=std_data, columns = ["opponent idx", "win-rate"])
            wandb.log({f"{agent_name}/post_eval/table": wandb.plot.line(table, "opponent idx", "win-rate", title=f"Post evaluation {agent_name}")})
            wandb.log({f"{agent_name}/post_eval/std_table": wandb.plot.line(std_table, "opponent idx", "win-rate", title=f"Std Post evaluation {agent_name}")})

            self.envs[agent_name].close()
            self.eval_envs[agent_name].close()
            
class SelfPlayTesting(SelfPlayExp):
    def __init__(self, seed_value=None, render_sleep_time=0.01):
        super(SelfPlayTesting, self).__init__()
        self.seed_value = seed_value
        self.load_prefix = "history_"
        self.render = True
        self.deterministic = True
        self.warn = True
        self.render_sleep_time = render_sleep_time

    def _init_argparse(self):
        super(SelfPlayTesting, self)._init_argparse(description='Self-play experiment testing script', help='The experiemnt configuration file path and name which the experiment should be loaded')

    def _generate_log_dir(self):
        super(SelfPlayTesting, self)._generate_log_dir(dir_postfix="test")

    def _load_testing_conditions(self, path):
        self.testing_conditions = {}
        self.testing_modes = {}

        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            testing_config = self.testing_configs[agent_name]
            agent_testing_path = os.path.join(path, agent_name) if(testing_config["path"] is None) else testing_config["path"]
            mode = testing_config["mode"]

            self.testing_conditions[agent_name] = {"path": agent_testing_path}
            self.testing_modes[agent_name] = mode
            num_rounds = self.experiment_configs["num_rounds"]

            if(mode == "limit"):
                self.testing_conditions[agent_name]["limits"] = [0, testing_config["gens"], testing_config["freq"]]
            # if the limit is that the start of the tested agents is that index and the end till the end
            elif(mode == "limit_s"):
                self.testing_conditions[agent_name]["limits"] = [testing_config["gens"], num_rounds-1, testing_config["freq"]]
            
            # if the limit is that the end of the tested agents is that index (including that index: in the for loop we will +1)
            elif(mode == "limit_e"):
                self.testing_conditions[agent_name]["limits"] = [0, testing_config["gens"], testing_config["freq"]]

            elif(mode == "gen"):
                self.testing_conditions[agent_name]["limits"] = [testing_config["gens"], testing_config["gens"], testing_config["freq"]]

            elif(mode == "all"):
                self.testing_conditions[agent_name]["limits"] = [0, num_rounds-1, testing_config["freq"]]
            
            elif(mode == "random"):
                self.testing_conditions[agent_name]["limits"] = [None, None, testing_config["freq"]]

            elif(mode == "round"):  # The round of pred vs round of prey
                print(num_rounds)
                self.testing_conditions[agent_name]["limits"] = [0, num_rounds-1, testing_config["freq"]]
            print(self.testing_conditions[agent_name]["limits"])
    def _init_envs(self):
        self.envs = {}

        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            # env = globals()["SelfPlayPredEnv"](algorithm_class=PPOMod, archive=None, seed_val=3)
            env = super(SelfPlayTesting, self).create_env(key=k, name="Testing", opponent_archive=None, algorithm_class=PPOMod)
            # if not isinstance(env, VecEnv):
            #     env = DummyVecEnv([lambda: env])

            # if not isinstance(env, DummyVecEnvSelfPlay):
            #     env.__class__ = DummyVecEnvSelfPlay   # This works fine, the other solution is commented
            
            self.envs[agent_name] = env
    
    def _init_models(self):
        self.models = {}

        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]

            self.models[agent_name] = PPOMod
            # (agent_configs["policy"], 
            #                                 self.envs[agent_name],
            #                                 clip_range=agent_configs["clip_range"], 
            #                                 ent_coef=agent_configs["ent_coef"],
            #                                 learning_rate=agent_configs["lr"], 
            #                                 batch_size=agent_configs["batch_size"],
            #                                 gamma=agent_configs["gamma"], 
            #                                 verbose=2,
            #                                 # tensorboard_log=os.path.join(self.log_dir,agent_name),
            #                                 n_epochs=agent_configs["n_epochs"]
            #                             )     


    def _init_testing(self, experiment_filename, logdir, wandb):
        super(SelfPlayTesting, self)._init_exp(experiment_filename, logdir, wandb)
        print(f"----- Load testing conditions")
        self._load_testing_conditions(experiment_filename)
        # print(f"----- Initialize environments")
        # self._init_envs()
        # print(f"----- Initialize models")
        # self._init_models()

    def render_callback(self, ret):
        # if(ret == 1):
        #     return -1
        return ret

    def _test_round_by_round(self, key, n_eval_episodes):
        agent_configs = self.agents_configs[key]
        agent_name = agent_configs["name"]
        opponent_name = agent_configs["opponent_name"]
        # TODO: debug why if we did not do this (redefine the env again) it does not work properly for the rendering
        # self.envs[agent_name] = super(SelfPlayTesting, self).create_env(key=key, name="Testing", opponent_archive=None, algorithm_class=PPOMod)
        for round_num in range(0, self.experiment_configs["num_rounds"], self.testing_conditions[agent_name]["limits"][2]):
            print("----------------------------------------")
            self.make_deterministic(cuda_check=False)   # This was added as we observed that previous rounds affect the other rounds
            startswith_keyword = f"{self.load_prefix}{round_num}_"
            # 1. fetch the agent
            agent_latest = utos.get_latest(self.testing_conditions[agent_name]["path"], startswith=startswith_keyword)
            if(len(agent_latest) == 0): # the experiment might have not be completed yet
                continue
            sampled_agent = os.path.join(self.testing_conditions[agent_name]["path"], agent_latest[0])  # Join it with the agent path
            # 2. load to the model
            # TODO: debug why if we did not do this (redefine the env again) it does not work properly for the rendering
            env = super(SelfPlayTesting, self).create_env(key=key, name="Testing", opponent_archive=None, algorithm_class=PPOMod)
            agent_model = PPOMod.load(sampled_agent, env)
            # 3. fetch the opponent
            opponent_latest = utos.get_latest(self.testing_conditions[opponent_name]["path"], startswith=startswith_keyword)
            if(len(opponent_latest) == 0):
                continue
            sampled_opponent = os.path.join(self.testing_conditions[opponent_name]["path"], opponent_latest[0])  # Join it with the agent path
            # 4. load the opponent to self.envs._load_opponent
            sampled_opponents = [sampled_opponent]
            mean_reward, std_reward, win_rate, std_win_rate, render_ret = evaluate_policy_simple(
                                                                                                    agent_model,
                                                                                                    env,
                                                                                                    n_eval_episodes=n_eval_episodes,
                                                                                                    render=self.render,
                                                                                                    deterministic=self.deterministic,
                                                                                                    return_episode_rewards=False,
                                                                                                    warn=self.warn,
                                                                                                    callback=None,
                                                                                                    sampled_opponents=sampled_opponents,
                                                                                                    render_extra_info=f"{round_num} vs {round_num}",
                                                                                                    render_callback=self.render_callback,
                                                                                                    sleep_time=self.render_sleep_time, #0.1,
                                                                                                )

            print(f"{round_num} vs {round_num} -> win rate: {100 * win_rate:.2f}% +/- {std_win_rate:.2f}\trewards: {mean_reward:.2f} +/- {std_reward:.2f}")
            env.close()
    def _test_different_rounds(self, key, n_eval_episodes):
        # TODO: for random
        agent_configs = self.agents_configs[key]
        agent_name = agent_configs["name"]
        opponent_name = agent_configs["opponent_name"]
        for i in range(self.testing_conditions[agent_name]["limits"][0], self.testing_conditions[agent_name]["limits"][1]+1, self.testing_conditions[agent_name]["limits"][2]):
            agent_startswith_keyword = f"{self.load_prefix}{i}_"
            # 1. fetch the agent
            agent_latest = utos.get_latest(self.testing_conditions[agent_name]["path"], startswith=agent_startswith_keyword)
            if(len(agent_latest) == 0): # the experiment might have not be completed yet
                continue
            sampled_agent = os.path.join(self.testing_conditions[agent_name]["path"], agent_latest[0])  # Join it with the agent path
            # 2. load to the model
            env = super(SelfPlayTesting, self).create_env(key=key, name="Testing", opponent_archive=None, algorithm_class=PPOMod)
            agent_model = PPOMod.load(sampled_agent, env)
            for j in range(self.testing_conditions[opponent_name]["limits"][0], self.testing_conditions[opponent_name]["limits"][1]+1, self.testing_conditions[opponent_name]["limits"][2]):
                print("----------------------------------------")
                self.make_deterministic(cuda_check=False)   # This was added as we observed that previous rounds affect the other rounds

                opponent_startswith_keyword = f"{self.load_prefix}{j}_"
                # 3. fetch the opponent
                opponent_latest = utos.get_latest(self.testing_conditions[opponent_name]["path"], startswith=opponent_startswith_keyword)
                if(len(opponent_latest) == 0):
                    continue
                sampled_opponent = os.path.join(self.testing_conditions[opponent_name]["path"], opponent_latest[0])  # Join it with the agent path
                # 4. load the opponent to self.envs._load_opponent
                sampled_opponents = [sampled_opponent]
                mean_reward, std_reward, win_rate, std_win_rate, render_ret = evaluate_policy_simple(
                                                                                                        agent_model,
                                                                                                        env,
                                                                                                        n_eval_episodes=n_eval_episodes,
                                                                                                        render=self.render,
                                                                                                        deterministic=self.deterministic,
                                                                                                        return_episode_rewards=False,
                                                                                                        callback=None,
                                                                                                        warn=self.warn,
                                                                                                        sampled_opponents=sampled_opponents,
                                                                                                        render_extra_info=f"{i} vs {j}",
                                                                                                        render_callback=self.render_callback,
                                                                                                        sleep_time=self.render_sleep_time, #0.1,
                                                                                                    )
                print(f"{i} vs {j} -> win rate: {win_rate}")
        
    def test(self, experiment_filename=None, logdir=False, wandb=False, n_eval_episodes=1):
        self._init_testing(experiment_filename=experiment_filename, logdir=logdir, wandb=wandb)
        already_evaluated_agents = []
        # In order to extend it multipe agents, we can make it as a recursive function (list:[models....,, None]) and pass the next element in the list, the termination criteria if the argument is None
        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            agent_opponent_joint = sorted([agent_name, agent_configs["opponent_name"]])
            if(agent_opponent_joint in already_evaluated_agents):
                continue

            if(self.testing_modes[agent_name] == "round"):
                self._test_round_by_round(k, n_eval_episodes)
                # break
            else:
                self._test_different_rounds(k, n_eval_episodes)
            already_evaluated_agents.append(agent_opponent_joint)


        # 1. fetch the agent
        # 2. load to the model to self.models
        # 3. fetch the opponent
        # 4. load the opponent to self.envs._load_opponent
        # Inside the for loop check if the name exists or not (the experiment might have not be completed yet)




# Old code compatible with argparse
# def _load_argparse_configs(self):
#     if(self.args.path is not None):
        
#     self.testing_target_agents_configs = {self.args.pred, self.args.prey}
#     self.testing_target_agents_indices = {}
#     for i in range(len(self.testing_target_agents_configs)):
#         if(self.testing_modes[i] == "limit"):
#             # if there is no comma, it means that it will start from 0 and finishes with that limit
#             if(',' not in self.testing_target_agents_configs[i]):
#                 self.testing_target_agents_indices[i] = [0, int(self.testing_target_agents_configs[i])]
#             # if there is a comma, put the limit
#             else:
#                 idxs = self.args.mode.split(',')
#                 self.testing_target_agents_indices[i] = [int(idx) for idx in idxs]

#         # if the limit is that the start of the tested agents is that index and the end till the end
#         elif(self.testing_modes[i] == "limit_s"):
#             self.testing_target_agents_indices[i] = [int(self.testing_target_agents_configs[i]), -1]
        
#         # if the limit is that the end of the tested agents is that index (including that index)
#         elif(self.testing_modes[i] == "limit_e"):
#             self.testing_target_agents_indices[i] = [0, int(self.testing_target_agents_configs[i])+1]

#         elif(self.testing_modes[i] == "gen"):
#             self.testing_taarget_agent_indices[i] = [int(self.testing_target_agents_configs[i]), int(self.testing_target_agents_configs[i])+1]

#         elif(self.testing_modes[i] == "all"):
#             self.testing_taarget_agent_indices[i] = [0, -1]
        
#         elif(self.testing_modes[i] == "random"):
#             self.testing_taarget_agent_indices[i] = [None, None]

# def _validate_argparse(self):
#     # if the experiment parameter is specified, do not parse anything more, just parse from that file
#     if(self.args.exp is not None):
#         return True

#     # if not, then validate the parameters
#     # if any of them is None (not set) then check others and raise Errors
#     if(self.args.pred_path is None or self.args.prey_path):
#         if(self.args.exp is None):
#             raise ValueError("exp or (pred-path and prey-path) must be defined")
#         if(self.args.path is None):
#             raise ValueError("path or (pred-path and prey-path) must be defined")
#         # if only one path is defined in --path parameter
#         else:
#             self.testing_paths = [self.args.path for _ in range(2)]
#     # if both of them are set
#     else:
#         self.testing_paths = [self.args.pred_path, self.args.prey_path]

#     if(',' not in self.args.mode):
#         raise ValueError("Mode should be in the following form <pred_mode>, <prey_mode>")
#     self.testing_modes = self.args.mode.lower().strip().split(',')

#     if(sum([i in ["gen", "all", "random", "limit_s", "limit", "limit_e"] for i in self.testing_modes]) != len(self.testing_modes)):
#         raise ValueError("Modes should be one of the following (Gen, All, Random, Limit)")
#     # if everything is fine, then load the data from argparse
#     self._load_argparse_configs()

# def _init_argparse_testing(self):
#     parser = argparse.ArgumentParser(description='Self-play experiment testing script')
#     parser.add_argument('--exp', type=str, default=None, help='The experiemnt configuration file path and name which the experiment should be loaded', metavar='')
#     parser.add_argument('--mode', type=str, default=None, help='The mode for the evaluation (<pred>,<prey>) (Gen, All, Random, Limit: from specific generation to another)', metavar='') #(Gen vs All), (Gen vs Gen), (All vs All), (Gen vs Random), (Random vs Gen), (Random vs Random)', metavar='')
#     parser.add_argument('--agent0-path', type=str, default=None, help='Path for predator files', metavar='')
#     parser.add_argument('--agent1-path', type=str, default=None, help='Path for prey files', metavar='')
#     parser.add_argument('--path', type=str, default=None, help='Path for predator and prey files', metavar='')
#     parser.add_argument('--agent0', type=str, default=None, help='targets versions for predator', metavar='')
#     parser.add_argument('--agent1', type=str, default=None, help='targets versions for prey', metavar='')

#     self.args = parser.parse_args()
#     self._validate_argparse()

# def _load_testing_configs(self, filename):
#     self.testing_filename = self.args.exp if filename is None else filename
#     if(self.testing_filename is not None):
#         self.experiment_configs, self.agents_configs, self.evaluation_configs, self.testing_configs = ExperimentParser.load(self.experiment_filename)
#     # if at the end it is None (filename is none, args.exp is none), then the user should have input the paths
#     # TODO: do it good with agents_configs
#     else:
#         self.testing_configs = {
#                                 "pred_path": self.testing_paths[0],
#                                 "prey_path": self.testing_paths[1],
#                                 "pred_gens": self.testing_target_agents_indices[0],
#                                 "prey_gens": self.testing_target_agents_indices[1]
#                                 }

# def _init_testing(self, testing_filename):
#     if(testing_filename is None):
#         self._init_argparse_testing()
    
#     print(f"----- Loading experiment from: {testing_filename}")
#     self._load_testing_configs(testing_filename)

#     self._init_testing_configs()

# def test(self, testing_filename=None):
#     self._init_testing(test_filename=testing_filename)