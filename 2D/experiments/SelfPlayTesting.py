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

# Note: player1 -> predator -> agent, player2 -> prey -> opponent

# TODO: make this script extendable with NvM competitive games

import os

from stable_baselines3 import PPO


from callbacks import *

from shared import normalize_reward, evaluate_policy_simple
import bach_utils.os as utos
from bach_utils.shared import *
from SelfPlayExp import SelfPlayExp
from bach_utils.heatmapvis import *

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

class SelfPlayTesting(SelfPlayExp):
    def __init__(self, seed_value=None, render_sleep_time=0.01):
        super(SelfPlayTesting, self).__init__()
        self.seed_value = seed_value
        self.load_prefix = "history_"
        self.deterministic = True
        self.warn = True
        self.render = None # it is being set by the configuration file
        self.crosstest_flag = None
        self.render_sleep_time = render_sleep_time

    def _init_testing(self, experiment_filename, logdir, wandb):
        super(SelfPlayTesting, self)._init_exp(experiment_filename, logdir, wandb)
        self.render = self.testing_configs.get("render", True)
        self.crosstest_flag = self.testing_configs.get("crosstest", False)
        print(f"----- Load testing conditions")
        self._load_testing_conditions(experiment_filename)
        # print(f"----- Initialize environments")
        # self._init_envs()
        # print(f"----- Initialize models")
        # self._init_models()

    # Overloading functions for testing
    def _init_argparse(self):
        super(SelfPlayTesting, self)._init_argparse(description='Self-play experiment testing script', help='The experiemnt configuration file path and name which the experiment should be loaded')

    def _generate_log_dir(self):
        super(SelfPlayTesting, self)._generate_log_dir(dir_postfix="test")
    # --------------------------------------------------------------------------------

    def _load_testing_conditions(self, path):
        self.testing_conditions = {}
        self.testing_modes = {}

        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            testing_config = self.testing_configs[agent_name]
            agent_testing_path = os.path.join(path, agent_name) if(testing_config["path"] is None) else testing_config["path"]
            agent_testing_path = os.path.join(agent_testing_path, self.testing_configs[agent_name]["dirname"])

            mode = testing_config["mode"]

            self.testing_conditions[agent_name] = {"path": agent_testing_path}
            self.testing_modes[agent_name] = mode
            num_rounds = self.experiment_configs["num_rounds"]

            if(mode == "limit"):
                # Use parameters to limit the history of rounds for the agent
                self.testing_conditions[agent_name]["limits"] = [0, testing_config["gens"], testing_config["freq"]]
            # if the limit is that the start of the tested agents is that index and the end till the end
            elif(mode == "limit_s"):
                # Use gen parameter to limit the history of rounds for the agent
                self.testing_conditions[agent_name]["limits"] = [testing_config["gens"], num_rounds-1, testing_config["freq"]]
            
            # if the limit is that the end of the tested agents is that index (including that index: in the for loop we will +1)
            elif(mode == "limit_e"):
                # Use gen parameter to limit the history of rounds for the agent
                self.testing_conditions[agent_name]["limits"] = [0, testing_config["gens"], testing_config["freq"]]

            elif(mode == "gen"):
                # Specific generation (round)
                self.testing_conditions[agent_name]["limits"] = [testing_config["gens"], testing_config["gens"], testing_config["freq"]]

            elif(mode == "all"):
                # Test all the available rounds for the agent (if we have n rounds then we have n*m evaluation rounds such that m is the number of rounds specified in the parameter for the other agent)
                self.testing_conditions[agent_name]["limits"] = [0, num_rounds-1, testing_config["freq"]]
            
            elif(mode == "random"):
                # Random generations/rounds
                self.testing_conditions[agent_name]["limits"] = [None, None, testing_config["freq"]]

            elif(mode == "round"):  # The round of pred vs round of prey
                # print(num_rounds)
                # Test round by round (all the available rounds) (if we have n rounds then we have n rounds of evaluation)
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
    
    # TODO: create _init_archives() but after the archive is integrated with the sampling and the indexing
    def _init_archives(self):
        raise NotImplementedError("_init_archives() not implemented")
    
    # Useless now as there is a problem and we have to recreate the model again with each evaluation
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

    def render_callback(self, ret):
        # if(ret == 1):
        #     return -1
        return ret

    def _run_one_evaluation(self, agent_conifgs_key,
                                  sampled_agent,
                                  sampled_opponents,
                                  n_eval_episodes,
                                  render_extra_info,
                                  env=None,
                                  agent_model=None,
                                  seed_value=None,
                                  return_episode_rewards=False):
        print("----------------------------------------")
        print(render_extra_info)
        self.make_deterministic(cuda_check=False)   # This was added as we observed that previous rounds affect the other rounds
        # TODO: debug why if we did not do this (redefine the env again) it does not work properly for the rendering
        # Create environment for each evaluation
        if(env is None and agent_model is None):
            env = super(SelfPlayTesting, self).create_env(key=agent_conifgs_key, name="Testing", opponent_archive=None, algorithm_class=PPOMod, seed_value=seed_value)
            agent_model = PPOMod.load(sampled_agent, env)
        mean_reward, std_reward, win_rate, std_win_rate, render_ret = evaluate_policy_simple(
                                                                                                agent_model,
                                                                                                env,
                                                                                                n_eval_episodes=n_eval_episodes,
                                                                                                render=self.render,
                                                                                                deterministic=self.deterministic,
                                                                                                return_episode_rewards=return_episode_rewards,
                                                                                                warn=self.warn,
                                                                                                callback=None,
                                                                                                sampled_opponents=sampled_opponents,
                                                                                                render_extra_info=render_extra_info,
                                                                                                render_callback=self.render_callback,
                                                                                                sleep_time=self.render_sleep_time, #0.1,
                                                                                            )

        print(f"{render_extra_info} -> win rate: {100 * win_rate:.2f}% +/- {std_win_rate:.2f}\trewards: {mean_reward:.2f} +/- {std_reward:.2f}")
        env.close()
        return mean_reward, std_reward, win_rate, std_win_rate, render_ret

    # Test and evaluate round of agent with the same round of the opponent (round no. (x) predator vs round no. (x) prey)
    def _test_round_by_round(self, key, n_eval_episodes):
        agent_configs = self.agents_configs[key]
        agent_name = agent_configs["name"]
        opponent_name = agent_configs["opponent_name"]
        # TODO: debug why if we did not do this (redefine the env again) it does not work properly for the rendering
        # self.envs[agent_name] = super(SelfPlayTesting, self).create_env(key=key, name="Testing", opponent_archive=None, algorithm_class=PPOMod)
        for round_num in range(0, self.experiment_configs["num_rounds"], self.testing_conditions[agent_name]["limits"][2]):
            startswith_keyword = f"{self.load_prefix}{round_num}_"
            # 1. fetch the agent
            agent_latest = utos.get_latest(self.testing_conditions[agent_name]["path"], startswith=startswith_keyword)
            if(len(agent_latest) == 0): # the experiment might have not be completed yet
                continue
            sampled_agent = os.path.join(self.testing_conditions[agent_name]["path"], agent_latest[0])  # Join it with the agent path
            # 3. fetch the opponent
            opponent_latest = utos.get_latest(self.testing_conditions[opponent_name]["path"], startswith=startswith_keyword)
            if(len(opponent_latest) == 0):
                continue
            sampled_opponent = os.path.join(self.testing_conditions[opponent_name]["path"], opponent_latest[0])  # Join it with the agent path
            sampled_opponents = [sampled_opponent]
            # 4. Run the evaluation
            self._run_one_evaluation(key, sampled_agent, sampled_opponents, n_eval_episodes, f"{round_num} vs {round_num}")

    def _test_different_rounds(self, key, n_eval_episodes):
        # TODO: for random
        agent_configs = self.agents_configs[key]
        agent_name = agent_configs["name"]
        opponent_name = agent_configs["opponent_name"]
        # The extra for loop
        # TODO: integrate with _test_round_by_round by creating a smaller function that takes two extra parameters (Round numbers of agent1, agent2) and return the evaluation
        for i in range(self.testing_conditions[agent_name]["limits"][0], self.testing_conditions[agent_name]["limits"][1]+1, self.testing_conditions[agent_name]["limits"][2]):
            agent_startswith_keyword = f"{self.load_prefix}{i}_"
            # 1. fetch the agent
            agent_latest = utos.get_latest(self.testing_conditions[agent_name]["path"], startswith=agent_startswith_keyword)
            if(len(agent_latest) == 0): # the experiment might have not be completed yet
                continue
            sampled_agent = os.path.join(self.testing_conditions[agent_name]["path"], agent_latest[0])  # Join it with the agent path
            for j in range(self.testing_conditions[opponent_name]["limits"][0], self.testing_conditions[opponent_name]["limits"][1]+1, self.testing_conditions[opponent_name]["limits"][2]):
                opponent_startswith_keyword = f"{self.load_prefix}{j}_"
                # 3. fetch the opponent
                opponent_latest = utos.get_latest(self.testing_conditions[opponent_name]["path"], startswith=opponent_startswith_keyword)
                if(len(opponent_latest) == 0):
                    continue
                sampled_opponent = os.path.join(self.testing_conditions[opponent_name]["path"], opponent_latest[0])  # Join it with the agent path
                # 4. load the opponent to self.envs._load_opponent
                sampled_opponents = [sampled_opponent]
                self._run_one_evaluation(key, sampled_agent, sampled_opponents, n_eval_episodes, f"{i} vs {j}")
            
    # TODO: add seed feature to run multiple seeds
    def _compute_single_round_gain_score(self, round_nums, n_eval_episodes, seed=3):
        gain_evaluation_models = self.testing_configs["gain_evaluation_models"]
        # +2 different models
        # 1. fetch model    1 from type 1(pred) from model type 1
        # 2. fetch model    2 from type 2(prey) from model type 1
        # 3. fetch model    3 from type 1(pred) from model type 2
        # 4. fetch model    4 from type 2(prey) from model type 2
        # ....

        agents_list = {k:[] for k in self.agents_configs.keys()}
        for model_idx, model_path in enumerate(gain_evaluation_models):
            round_num = round_nums[model_idx]
            agent_startswith_keyword = f"{self.load_prefix}{round_num}_"
            # agent_type: pred/prey (for now)
            for agent_type in self.agents_configs.keys():
                # 1. fetch the agent
                path = os.path.join(model_path, agent_type)
                agent_latest = utos.get_latest(path, startswith=agent_startswith_keyword)
                if(len(agent_latest) == 0): # the experiment might have not be completed yet
                    continue
                sampled_agent = os.path.join(path, agent_latest[0])  # Join it with the agent path
                agents_list[agent_type].append((sampled_agent, model_idx, round_num))

        #         agent vs opponent
        # * Evaluate m1 vs m2 (predators1 against prey1)
        # * Evaluate m1 vs m4 (predators1 against prey2)
        # * Evaluate m2 vs m1 (prey1 against predators1)
        # * Evaluate m2 vs m3 (prey1 against predators2)
        gain_list = []
        # TODO: something seems not correct here: check it later
        # Professor Nolfi: Gain = performance(predators1 against prey1) - performance(predators1 against prey2) + performance(prey1 against predators1) - performance(prey1 against predators2)
        # performance(prey[i] against predator[i]) =  - performance(predator[i] against prey[i]) 
        # Gain = performance(predators1 against prey1) - performance(predators1 against prey2) - performance(predators1 against prey1) + performance(predators2 against prey1)
        # Max(Gain) = 0.5 - 0 + 0.5 - 0 = 1 -> (0.5: normalized value for 0 reward, 0: normalized value for -1000 reward)
        # Min(Gain) = 0 - 0.5 + 0 - 0.5 = -1 -> (0.5: normalized value for 0 reward (the predator caught the prey directly), 0: normalized value for -1010 reward (the predator could not catch the prey at all))
        
        # When to say that model 1 is better: (predator and prey of model 1 are stronger than model 2)
        # Gain = 0.2 - 0.5(pred1 wins) + 0.2 - 0(prey1 wins)  -> such that 0.2 kinda mid value -> value around zero towards the negative more

        # When to say that model 2 is better: (predator and prey of model 2 are stronger than model 1)
        # Gain = 0.2 - 0(prey2 wins) + 0.2 - 0.5(pred2 wins) ->


        # Only for 1v1
        # allowed_pairs = [(1, 0, "pred", 0, "prey"), (-1, 0, "pred", 1, "prey"), (+1, 0, "prey", 0, "pred"), (-1, 1, "prey", 0, "pred")]
        # for (factor, agent_idx, agent_type, opponent_idx, opponent_type) in allowed_pairs:
        allowed_pairs = [(1, 0,0), (-1, 0,1), (-1, 0,0), (+1, 1,0)]
        for (factor, agent_idx, opponent_idx) in allowed_pairs:
            agent_type = "pred"
            opponent_type = "prey"
            sampled_agent, agent_idx_ret, round_num1 = agents_list[agent_type][agent_idx]
            sampled_opponents, opponent_idx_ret, round_num2 = agents_list[opponent_type][opponent_idx]
            sampled_opponents = [sampled_opponents] 
            assert agent_idx == agent_idx_ret
            assert opponent_idx == opponent_idx_ret
            print("###########") 
            print(f"Pair: {(agent_idx, opponent_idx)}")
            mean_reward, std_reward, win_rate, std_win_rate, render_ret = self._run_one_evaluation(agent_type, sampled_agent, sampled_opponents, n_eval_episodes, f"{agent_type}({agent_idx}):{round_num1} vs {opponent_type}({opponent_idx}):{round_num2}", seed_value=seed)
            score = normalize_reward(mean_reward, mn=-1010, mx=0)
            print(f"Score (Normalized reward): {score}")
            gain = factor*score
            print(f"Gain: {gain}")
            gain_list.append(gain)
        return sum(gain_list)

        # This works fine if for each agent from predator (agent type 1) of all the models specified  is against prey of all the models specified
        # TODO: make it work, this is a generalization somehow
        # # TODO: Just for now
        # allowed_pairs = [(1, 0,0), (-1, 0,1), (1, 0,0), (-1, 1,0)]
        # allowed_pairs_wo_factors = [i[1:] for i in allowed_pairs]
        # # TODO: Some how refactor this part
        # allowed_pairs_factors = {i[1]:{i[2]:i[0]} for i in allowed_pairs}

        # pair_idx = 0
        # For each agent_type
        # for agent_type in self.agents_configs.keys():
        #   For each opponent type
        #     for opponent_type in self.agents_configs.keys():
        # If the type is the same then it cannot be
        #         if(agent_type == opponent_type):
        #             continue
        #       For each agent in the agent_list
        #         for agent in agents_list[agent_type]:
        #             sampled_agent, agent_idx, round_num1 = agent
        #           For each agent in the agent list as opponent
        #             for opponent in agents_list[opponent_type]:
        #                 sampled_opponents, opponent_idx, round_num2 = opponent
                        
        #                 if((agent_idx, opponent_idx) not in allowed_pairs_wo_factors):
        #                     continue

        #                 # TODO: Added it back
        #                 # Enhance this a little bit
        #                 # agent_opponent_joint = sorted([sampled_agent+str(agent_idx), sampled_opponents+str(opponent_idx)])
        #                 # if(agent_opponent_joint in already_evaluated_agents):
        #                 #     continue   

        #                 sampled_opponents = [sampled_opponents] 
        #                 print("###########") 
        #                 print(f"Pair: {(agent_idx, opponent_idx)}")
        #                 # TODO: refactor this part  
        #                 mean_reward, std_reward, win_rate, std_win_rate, render_ret = self._run_one_evaluation(agent_type, sampled_agent, sampled_opponents, n_eval_episodes, f"{agent_type}({agent_idx}):{round_num1} vs {opponent_type}({opponent_idx}):{round_num2}")
        #                 score = normalize_reward(mean_reward)

        #                 # TODO: added it back
        #                 # if it is from the same model then the factor is +1, otherwise if -1
        #                 # factor = 1 if(agent_idx == opponent_idx) else -1

        #                 factor = allowed_pairs_factors[pair_idx][0]
        #                 gain_list.append(factor*score)
        #                 pair_idx += 1
        #                 # TODO: Added it back
        #                 # already_evaluated_agents.append(agent_opponent_joint)
        #     # TODO: just for now
        #     break

        # return gain_list

    def _compute_gain_score(self, n_eval_episodes, n_seeds):
        print("###############################################")
        print(" ------------- Compute Gain Score -------------")
        num_rounds = self.experiment_configs["num_rounds"]
        round_axis = [i for i in range(0, num_rounds, self.testing_configs["gain_score_freq"])]
        # Enforce doing evaluation for the last generation
        if(round_axis[-1] != num_rounds-1):
            round_axis.append(num_rounds-1)

        # Initiate the gain_score matrix
        gain_matrix = np.zeros([len(round_axis) for _ in self.agents_configs.keys()])
        # TODO: Now there is a for loop for each agent type, later we want to extend it for NvM with recursive or looping through the keys
        for ei, i in enumerate(round_axis): #, freq):
            for ej, j in enumerate(round_axis): #, freq):
                print("--------------------------------------------")
                print(f"Compute Gain score round: {i} vs {j}")
                gain_scores = []
                for seed_idx in range(n_seeds):
                    print(f"Seed iteration: {seed_idx}")
                    gain_score = self._compute_single_round_gain_score([i, j], n_eval_episodes=n_eval_episodes, seed="random")
                    gain_scores.append(gain_score)
                gain_matrix[ei, ej] = np.mean(gain_scores)
        print("####################################################")
        print(f"Gain score {np.mean(gain_matrix):.4f} +/- {np.std(gain_matrix):.4f}")
        print("####################################################")
        # TODO: later also visualize the axis of the rounds
        HeatMapVisualizer.visPlotly(gain_matrix, xrange=round_axis, yrange=round_axis)

    # ------------------- TODO --------------------------------
    # TODO
    def _get_best_agent(self, agent_num_rounds, opponent_num_rounds, search_radius, agent_path, opponent_path):
        # TODO: Use the metric that is saved with the name of the model to get the best model
        best_rewards = None
        best_agent = None
        for agent_idx in range(agent_num_rounds-search_radius, agent_num_rounds):
            agent = None #TODO: get the latest predator with ith round
            freq = 3     # TODO: parse it from the config file
            opponents_rounds_idx = [i for i in range(0, opponent_num_rounds, freq)]  # TODO: think to make it as Prof. Nolfi said
            # for opponent_idx in opponents_selected_rounds:
            opponents = None    # TODO: get the opponents using opponents_rounds_idx
            # self._run_one_evaluation(agent_conifgs_key, sampled_agent, sampled_opponents, n_eval_episodes, render_extra_info, env=None, agent_model=None, seed_value=None):
            reward = None # TODO: evaluate
            best_rewards.append([agent_idx, reward])
        best_rewards = np.array(best_rewards)
        best_agent_idx = np.argmax(best_rewards, axis=0)[0]
        best_agent = None # Get the best agent using best_agent_idx
        return best_agent

    def _compute_performance(self, agent, opponent, negative_score_flag=False):
        def normalize_performance(performance, negative_score_flag):
            max_val = None # TODO: get from the config file
            min_val = None # TODO: get from the config file
            if(negative_score_flag):
                return max_val - abs(performance) / (max_val - min_val)
            else:
                return performance / max_val
        reward = None # TODO: get the performance reward
        return normalize_performance(reward, negative_score_flag)

    def crosstest(self, n_eval_episodes, n_seeds):
        num_rounds1 = None      # TODO
        num_rounds2 = None      # TODO
        search_radius = None    # TODO
        
        best_agent1     = self._get_best_agent() # TODO
        best_opponent1  = self._get_best_agent() # TODO

        best_agent2     = self._get_best_agent() # TODO
        best_opponent2  = self._get_best_agent() # TODO

        # agent1 predator -> performance is related to the reward 
        perf_agent1_opponent2 = self._compute_performance(self, best_agent1, best_opponent2, negative_score_flag=True)
        perf_agent1_opponent1 = self._compute_performance(self, best_agent1, best_opponent1, negative_score_flag=True)
        perf_opponent1_agent2 = self._compute_performance(self, best_opponent1, best_agent2, negative_score_flag=False)
        perf_opponent1_agent1 = self._compute_performance(self, best_opponent1, best_agent1, negative_score_flag=False)

        perf_agent = perf_agent1_opponent2 - perf_agent1_opponent1
        perf_opponent = perf_opponent1_agent2 - perf_opponent1_agent1
        
        gain = perf_agent + perf_opponent

        if(perf_agent > 0):
            print(f"Configuration 1 is better {1} to generate predators (agent)") # TODO: print the path of the condition 1 parent directory
        else:
            print(f"Configuration 2 is better {2} to generate predators (agent)") # TODO: print the path of the condition 1 parent directory

        if(perf_opponent > 0):
            print(f"Configuration 1 is better {1} to generate preys") # TODO: print the path of the condition 1 parent directory
        else:
            print(f"Configuration 2 is better {2} to generate preys") # TODO: print the path of the condition 1 parent directory

        if(gain > 0):
            print(f"Configuration 1 is better {1}") # TODO: print the path of the condition 1 parent directory
            return 1
        else:
            print(f"Configuration 2 is better {2}") # TODO: print the path of the condition 1 parent directory
            return 2
            

    def test(self, experiment_filename=None, logdir=False, wandb=False, n_eval_episodes=1):
        self._init_testing(experiment_filename=experiment_filename, logdir=logdir, wandb=wandb)
        n_eval_episodes_configs = self.testing_configs.get("repetition", None)
        n_eval_episodes = n_eval_episodes_configs if n_eval_episodes_configs is not None else n_eval_episodes


        if(self.crosstest_flag):
            n_seeds = self.testing_configs.get("n_seeds", 1)
            # self._compute_gain_score(n_eval_episodes, n_seeds)
        else:
            already_evaluated_agents = []
            # In order to extend it multipe agents, we can make it as a recursive function (list:[models....,, None]) and pass the next element in the list, the termination criteria if the argument is None
            for k in self.agents_configs.keys():
                agent_configs = self.agents_configs[k]
                agent_name = agent_configs["name"]
                agent_opponent_joint = sorted([agent_name, agent_configs["opponent_name"]])
                if(agent_opponent_joint in already_evaluated_agents):
                    continue

                if(self.testing_modes[agent_name] == "round"):
                    # Do not care about all other parameters just evaluate using agents from the same rounds against each other
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