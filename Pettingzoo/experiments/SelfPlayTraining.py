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
import numpy as np

from stable_baselines3 import PPO

# from bach_utils.archive import Archive
from archive import ArchiveSB3 as Archive

from callbacks import *

from wandb.integration.sb3 import WandbCallback
import wandb

import gym_predprey
import gym
from gym.envs.registration import register
from gym_pz_predprey.envs.SelfPlayPZPredPrey import SelfPlayPZPredEnv
from gym_pz_predprey.envs.SelfPlayPZPredPrey import SelfPlayPZPreyEnv


from shared import *
from copy import deepcopy
from bach_utils.shared import *
from SelfPlayExp import SelfPlayExp

# TODO: save the experimenet file in the training folder

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
        self.deterministic = False

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
            sampling_parameters = {
                                    "opponent_selection":agent_configs["opponent_selection"],
                                    "sample_path":os.path.join(self.log_dir, opponent_name),
                                    "randomly_reseed_sampling": agent_configs.get("randomly_reseed_sampling", False)
                                  }

            self.envs[agent_name] = super(SelfPlayTraining, self).create_env(key=k, name="Training", opponent_archive=opponent_archive, sample_after_reset=agent_configs["sample_after_reset"], sampling_parameters=sampling_parameters)
            self.eval_envs[agent_name] = super(SelfPlayTraining, self).create_env(key=k, name="Evaluation", opponent_archive=opponent_archive, sample_after_reset=False, sampling_parameters=None)

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
                                                    n_epochs=agent_configs["n_epochs"],
                                                    n_steps=agent_configs.get("n_steps", 2048)
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
                                                            EvalSaveCallback(
                                                                            eval_env=self.eval_envs[agent_name],
                                                                            log_path=agent_path,
                                                                            eval_freq=int(agent_configs["eval_freq"]),
                                                                            n_eval_episodes=agent_configs["num_eval_episodes"],
                                                                            deterministic=self.deterministic,
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
            self.opponent_selection_callbacks[agent_name] = TrainingOpponentSelectionCallback(
                                                                                                sample_path=opponent_sample_path,
                                                                                                env=self.envs[agent_name], 
                                                                                                opponent_selection=agent_configs["opponent_selection"],
                                                                                                sample_after_rollout=agent_configs["sample_after_rollout"],
                                                                                                sample_after_reset=agent_configs["sample_after_reset"], # agent_configs.get("sample_after_reset", False)
                                                                                                num_sampled_per_round=agent_configs["num_sampled_opponent_per_round"],
                                                                                                archive=self.archives[opponent_name],
                                                                                                randomly_reseed_sampling=agent_configs.get("randomly_reseed_sampling", False)
                                                                                             )
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