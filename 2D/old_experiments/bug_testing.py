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
        self.deterministic = False
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
                                  render=None,
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
                                                                                                render=self.render if render is None else render,
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


    def get_latest_agent_path(self, idx, path, population_idx):
        agent_startswith_keyword = f"{self.load_prefix}{idx}_"
        agent_latest = utos.get_latest(path, startswith=agent_startswith_keyword, population_idx=population_idx)
        ret = True
        if(len(agent_latest) == 0):
            ret = False
        latest_agent = os.path.join(path, agent_latest[0])  # Join it with the agent path
        return ret, latest_agent

    def _compute_performance(self, agent, opponent, key, n_eval_episodes=1, n_seeds=1, negative_score_flag=False, render=False, render_extra_info=None):
        def normalize_performance(min_val, max_val, performance, negative_score_flag):
            if(negative_score_flag):
                performance = min(0, performance) # to bound it in case the agent caught it directly (never happens) -> performance = +10 
                # -1010 ->  0, 0 -> 1
                return (max_val - abs(performance)) / max_val
            else:
                performance = max(0, performance) # to bound it in case the agent caught it directly (never happens) -> performance = -10 
                # 1010 ->  1, 0 -> 0
                return performance / max_val
        # for i in range(n_seeds):
            # random_seed = datetime.now().microsecond//1000
            # random.seed(random_seed)
        mean_reward, std_reward, win_rate, std_win_rate, render_ret = self._run_one_evaluation(key, agent, [opponent], n_eval_episodes, render=render, render_extra_info=f"{agent} vs {opponent}" if render_extra_info is None else render_extra_info)
        reward = np.mean(mean_reward) # get the performance reward
        limits = self.testing_configs.get("crosstest_rewards_limits")
        normalized_reward = normalize_performance(*limits, reward, negative_score_flag)
        print(f"Nomralized: {normalized_reward}, {reward}")
        return normalized_reward

    def _get_best_agent(self, num_rounds, search_radius, paths, key, num_population, min_gamma_val=0.05, n_eval_episodes=1, n_seeds=1, render=False, negative_score_flag=False):#, negative_reward_flag=False):
        agent_num_rounds, opponent_num_rounds = num_rounds[:]
        agent_path, opponent_path = paths[:]
        print(f"## Getting the best model for {key}")
        # TODO: Use the metric that is saved with the name of the model to get the best model
        best_rewards = []
        freq = self.testing_configs.get("crosstest_freq")
        # Create the list of opponenets that will 
        opponents_rounds_idx = [i for i in range(0, opponent_num_rounds, freq)]  # TODO: think to make it as Prof. Nolfi said
        if(not opponent_num_rounds-1 in opponents_rounds_idx):
            opponents_rounds_idx.append(opponent_num_rounds-1)

        gamma = min_gamma_val**(1/len(opponents_rounds_idx))

        # Test each agent in each population against an evaluation set (opponenets from all the populations)
        for agent_population_idx in range(num_population):
            for agent_idx in range(agent_num_rounds-search_radius-1, agent_num_rounds):
                # TODO: refactor the code and make the following 4 lines as a function
                ret, agent = self.get_latest_agent_path(agent_idx, agent_path, agent_population_idx)
                if(not ret):
                    continue
                rewards = []
                for opponent_population_idx in range(num_population):
                    print(f"POP: {agent_population_idx}, {opponent_population_idx}")
                    for i, opponent_idx in enumerate(opponents_rounds_idx):
                        ret, sampled_opponent = self.get_latest_agent_path(opponent_idx, opponent_path, opponent_population_idx)
                        if(not ret):
                            continue
                        # mean_reward, std_reward, win_rate, std_win_rate, render_ret = self._run_one_evaluation(key, agent, [sampled_opponent], n_eval_episodes, render=False, render_extra_info=f"{agent_idx}({agent_population_idx}) vs {opponent_idx} ({opponent_population_idx})")
                        mean_reward = self._compute_performance(agent, sampled_opponent, key, n_eval_episodes, n_seeds, negative_score_flag, render, render_extra_info=f"{agent_idx}({agent_population_idx}) vs {opponent_idx} ({opponent_population_idx})")
                        weight = (gamma**(len(opponents_rounds_idx)-i))
                        # if(negative_reward_flag):
                        #     discount = (gamma**(i))
                        weighted_reward = weight * mean_reward
                        print(f"Weight: {weight}\tPerformance: {mean_reward}\tWeighted Performance: {weighted_reward}")
                        rewards.append(weighted_reward)
                mean_reward = np.mean(np.array(rewards))
                best_rewards.append([agent_population_idx, agent_idx, mean_reward])
        best_rewards = np.array(best_rewards)
        print(best_rewards)
        best_agent_idx = np.argmax(best_rewards[:,2])
        # Get the best agent using best_agent_idx
        agent_idx = int(best_rewards[best_agent_idx,1]) #agent_num_rounds-search_radius-1 + (best_agent_idx%)
        agent_population_idx = int(best_rewards[best_agent_idx,0])
        print(f"Best agent: idx {agent_idx}, population {agent_population_idx}")
        startswith_keyword = f"{self.load_prefix}{agent_idx}_"
        agent_latest = utos.get_latest(agent_path, startswith=startswith_keyword, population_idx=agent_population_idx)
        best_agent = os.path.join(agent_path, agent_latest[0])  # Join it with the agent path
        return best_agent


    # TODO: this code need to be parallel
    # TODO: need to speed up the code by not running one evaluation and then delete the environment
    def crosstest(self, n_eval_episodes, n_seeds):
        print(f"---------------- Running Crosstest ----------------")
        # For now both the approaches have the same number of rounds
        # TODO: save the configuration file with the experiment, so we can parse the training configs with it and the number of round for each approach,
        # TODO: make crosstest as a seperate or a big element in testing 
        # TODO: change opponent with adversery word (maybe)
        # TODO: refactor and make it better and make it for more than 1v1
        num_rounds = self.testing_configs.get("crosstest_num_rounds")
        num_rounds1, num_rounds2 = num_rounds[0], num_rounds[1]
        search_radius = self.testing_configs.get("crosstest_search_radius")
        print(f"Num. rounds: {num_rounds1}, {num_rounds2}")

        # def _get_best_agent(self, agent_num_rounds, opponent_num_rounds, search_radius, agent_path, opponent_path):
        approaches_path = self.testing_configs.get("crosstest_approaches_path")
        approach1_path, approach2_path = approaches_path[0], approaches_path[1]
        print(f"Paths:\n{approach1_path}\n{approach2_path}")

        names = [self.agents_configs[k]["name"] for k in self.agents_configs.keys()]
        agent_name, opponent_name = names[0], names[1]
        print(f"names: {agent_name}, {opponent_name}")

        
        agent1_path = os.path.join(approach1_path, agent_name)
        opponent1_path = os.path.join(approach1_path, opponent_name)
        agent2_path = os.path.join(approach2_path, agent_name)
        opponent2_path = os.path.join(approach2_path, opponent_name)

        print(f"Agent1 path: {agent1_path}")
        print(f"Opponenet1 path: {opponent1_path}")
        print(f"Agent2 path: {agent2_path}")
        print(f"Opponenet2 path: {opponent2_path}")

        num_population1, num_population2 = self.testing_configs.get("crosstest_populations")
        print(f"Num. populations: {num_population1}, {num_population2}")

        # O(num_pop * search_radius * num_pop * (opponent_num_rounds//freq) * 4 * n_eval * n_seeds) 
        # 5*10*5*10*4 *1*1 = 10000 (num_pop = 5, search_radius = 10, freq = 5, opponent_num_rounds = 50) -> very expensive operations
        # 10k * 5sec / (3600) = 13.8888889 hours !!!!
        best_agent1     = self._get_best_agent([num_rounds1, num_rounds1], search_radius, [agent1_path, opponent1_path], agent_name, num_population1, n_eval_episodes=1, negative_score_flag=True, n_seeds=n_seeds)
        print(f"Best agent1: {best_agent1}")
        best_opponent1  = self._get_best_agent([num_rounds1, num_rounds1], search_radius, [opponent1_path, agent1_path], opponent_name, num_population1, n_eval_episodes=1, negative_score_flag=False, n_seeds=n_seeds)
        print(f"Best opponent1: {best_opponent1}")
        best_agent2     = self._get_best_agent([num_rounds2, num_rounds2], search_radius, [agent2_path, opponent2_path], agent_name, num_population2, n_eval_episodes=1, negative_score_flag=True, n_seeds=n_seeds)
        print(f"Best agent2: {best_agent2}")
        best_opponent2  = self._get_best_agent([num_rounds2, num_rounds2], search_radius, [opponent2_path, agent2_path], opponent_name, num_population2, n_eval_episodes=1, negative_score_flag=False, n_seeds=n_seeds)
        print(f"Best opponent2: {best_opponent2}")

        print("###############################################################")
        print(f"# Best agent1: {best_agent1}")
        print(f"# Best opponent1: {best_opponent1}")
        print(f"# Best agent2: {best_agent2}")
        print(f"# Best opponent2: {best_opponent2}")
        print("###############################################################")

        # Multi-population -> 0.0
        #############################################################33
        #	Best agent1: /home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-results/save-SelfPlay1v1-Pred_Prey-v0-12.21.2021_02.40.22/pred/history_46_winrate_m_0.4_s_1250000_p_1
        #	Best opponent1: /home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-results/save-SelfPlay1v1-Pred_Prey-v0-12.21.2021_02.40.22/prey/history_44_winrate_m_1.0_s_1175000_p_0
        #	Best agent2: /home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-results/save-SelfPlay1v1-Pred_Prey-v0-12.21.2021_02.39.50/pred/history_49_winrate_m_0.8_s_1325000_p_0
        #	Best opponent2: /home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-results/save-SelfPlay1v1-Pred_Prey-v0-12.21.2021_02.39.50/prey/history_45_winrate_m_1.0_s_1200000_p_0
        #############################################################33

        # Single-populations -> 0.0
        ###############################################################
        # Best agent1: /home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-results/save-SelfPlay1v1-Pred_Prey-v0-12.21.2021_02.40.22/pred/history_46_winrate_m_0.4_s_1250000_p_0
        # Best opponent1: /home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-results/save-SelfPlay1v1-Pred_Prey-v0-12.21.2021_02.40.22/prey/history_44_winrate_m_1.0_s_1175000_p_0
        # Best agent2: /home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-results/save-SelfPlay1v1-Pred_Prey-v0-12.21.2021_02.39.50/pred/history_49_winrate_m_0.8_s_1325000_p_0
        # Best opponent2: /home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-results/save-SelfPlay1v1-Pred_Prey-v0-12.21.2021_02.39.50/prey/history_45_winrate_m_1.0_s_1200000_p_0
        ###############################################################

        # Single-population but increased search radius and decrease the freq & increased the repetitions for evaluation -> gain: 0.32112211221122117
        ###############################################################
        # Best agent1: /home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-results/save-SelfPlay1v1-Pred_Prey-v0-12.21.2021_02.40.22/pred/history_47_winrate_m_0.4_s_1275000_p_0
        # Best opponent1: /home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-results/save-SelfPlay1v1-Pred_Prey-v0-12.21.2021_02.40.22/prey/history_39_winrate_m_1.0_s_1050000_p_0
        # Best agent2: /home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-results/save-SelfPlay1v1-Pred_Prey-v0-12.21.2021_02.39.50/pred/history_49_winrate_m_0.8_s_1325000_p_0
        # Best opponent2: /home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-results/save-SelfPlay1v1-Pred_Prey-v0-12.21.2021_02.39.50/prey/history_45_winrate_m_1.0_s_1200000_p_0
        ###############################################################





        # agent1 predator -> performance is related to the reward 
        print(f"################# Agent1 vs Opponent2 #################")
        perf_agent1_opponent2 = self._compute_performance(best_agent1, best_opponent2, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=True)
        print(f"################# Agent1 vs Opponent1 #################")
        perf_agent1_opponent1 = self._compute_performance(best_agent1, best_opponent1, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=True)

        print(f"################# Agent2 vs Opponent2 #################")
        perf_agent2_opponent2 = self._compute_performance(best_agent2, best_opponent2, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=True)
        print(f"################# Agent2 vs Opponent1 #################")
        perf_agent2_opponent1 = self._compute_performance(best_agent2, best_opponent1, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=True)
        
        print(f"################# Opponent1 vs Agent2 #################")
        perf_opponent1_agent2 = self._compute_performance(best_opponent1, best_agent2, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=True)
        print(f"################# Opponent1 vs Agent1 #################")
        perf_opponent1_agent1 = self._compute_performance(best_opponent1, best_agent1, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=True)

        print(f"################# Opponent1 vs Agent2 #################")
        perf_opponent2_agent2 = self._compute_performance(best_opponent2, best_agent2, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=True)
        print(f"################# Opponent2 vs Agent1 #################")
        perf_opponent2_agent1 = self._compute_performance(best_opponent2, best_agent1, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=True)


        perf_agent = perf_agent1_opponent2 - perf_agent1_opponent1 + perf_agent2_opponent2 - perf_agent2_opponent1
        perf_opponent = perf_opponent1_agent2 - perf_opponent1_agent1 + perf_opponent2_agent2 - perf_opponent2_agent1
        
        gain = perf_agent + perf_opponent
        print("-----------------------------------------------------------------")
        print(f"perf_agent: {perf_agent}\tperf_opponent: {perf_opponent}\tgain: {gain}")

        # perf_agent: 0.32112211221122117	perf_opponent: 0.9633663366336633	gain: 1.2844884488448844
        eps = 1e-3
        if(perf_agent > 0):
            print(f"Configuration 1 is better {1} to generate predators (path: {approach1_path})")
        elif(-eps <= perf_agent <= eps):
            print(f"Configuration 1 & 2 are close to each other to generate predators")
        else:
            print(f"Configuration 2 is better {2} to generate predators (path: {approach2_path})")

        if(perf_opponent > 0):
            print(f"Configuration 1 is better {1} to generate preys (path: {approach1_path})")
        elif(-eps <= perf_opponent <= eps):
            print(f"Configuration 1 & 2 are close to each other to generate prey")
        else:
            print(f"Configuration 2 is better {2} to generate preys (path: {approach2_path})")

        if(gain > 0):
            print(f"Configuration 1 is better {1} (path: {approach1_path})")
            return 1
        elif(-eps <= gain <= eps):
            print(f"Configuration 1 & 2 are close to each other")
        else:
            print(f"Configuration 2 is better {2} (path: {approach2_path})")
            return 2
            

    def test(self, experiment_filename=None, logdir=False, wandb=False, n_eval_episodes=1):
        self._init_testing(experiment_filename=experiment_filename, logdir=logdir, wandb=wandb)
        n_eval_episodes_configs = self.testing_configs.get("repetition", None)
        n_eval_episodes = n_eval_episodes_configs if n_eval_episodes_configs is not None else n_eval_episodes

        if(self.crosstest_flag):
            n_seeds = self.testing_configs.get("n_seeds", 1)
            self.crosstest(n_eval_episodes, n_seeds)
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
    
    def _bug_compute_performance(self, agent, opponent, key, n_eval_episodes=1, n_seeds=1, negative_score_flag=False, render=False, render_extra_info=None):
        # for i in range(n_seeds):
            # random_seed = datetime.now().microsecond//1000
            # random.seed(random_seed)

        mean_reward, std_reward, win_rate, std_win_rate, render_ret = self._run_one_evaluation(key, agent, [opponent], n_eval_episodes, render=render, render_extra_info=f"{agent} vs {opponent}" if render_extra_info is None else render_extra_info)
        return mean_reward

    def _bug_compute_performance2(self, agent, opponent, key, n_eval_episodes=1, n_seeds=1, negative_score_flag=False, render=False, render_extra_info=None, agent_model=None, env=None):
        # for i in range(n_seeds):
            # random_seed = datetime.now().microsecond//1000
            # random.seed(random_seed)

        mean_reward, std_reward, win_rate, std_win_rate, render_ret = self._run_one_evaluation(key, agent, [opponent], n_eval_episodes, agent_model=agent_model, env=env, render=render, render_extra_info=f"{agent} vs {opponent}" if render_extra_info is None else render_extra_info)
        return mean_reward

    def bug(self):
        agent = "/home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-crosstest-models/standard1-01.11.2022_21.57.03/pred/history_46_lastreward_m_-230.2_s_1251328_p_0"
        opponent = "/home/hany606/University/Thesis/Drones-PEG-Bachelor-Thesis-2022/2D/experiments/selfplay-crosstest-models/standard1-01.11.2022_21.57.03/prey/history_47_lastreward_m_802.6_s_1277952_p_0"
        agent_name = "pred"
        opponent_name = "prey"
        self.agents_configs = {}
        self.agents_configs["pred"] = {"name": "pred", "env_class":"SelfPlayPredEnv"}
        self.agents_configs["prey"] = {"name": "prey", "env_class":"SelfPlayPreyEnv"}
        self.seed_value = 3
        self.render_sleep_time = 0.0001

        perf_agent_opponent = self._bug_compute_performance(agent, opponent, agent_name, n_eval_episodes=50, n_seeds=1, negative_score_flag=True, render=False)
        perf_opponent_agent = self._bug_compute_performance(opponent, agent, opponent_name, n_eval_episodes=50, n_seeds=1, negative_score_flag=False, render=False)

        # for i in range(3):
        #     perf_agent_opponent = self._bug_compute_performance(agent, opponent, agent_name, n_eval_episodes=1, n_seeds=1, negative_score_flag=True, render=True)
        # for i in range(3):
        #     perf_opponent_agent = self._bug_compute_performance(opponent, agent, opponent_name, n_eval_episodes=1, n_seeds=1, negative_score_flag=False, render=True)

        # agent_env = super(SelfPlayTesting, self).create_env(key=agent_name, name="Testing", opponent_archive=None, algorithm_class=PPOMod, seed_value=self.seed_value)
        # agent_model = PPOMod.load(agent, agent_env)


        # opponent_env = super(SelfPlayTesting, self).create_env(key=opponent_name, name="Testing", opponent_archive=None, algorithm_class=PPOMod, seed_value=self.seed_value)
        # opponent_model = PPOMod.load(opponent, opponent_env)

        # for i in range(10):
        #     perf_agent_opponent = self._bug_compute_performance2(None, opponent, None, agent_model=agent_model, env=agent_env, n_eval_episodes=1, n_seeds=1, negative_score_flag=True, render=True)
        # for i in range(10):
        #     perf_opponent_agent = self._bug_compute_performance2(None, agent, None, agent_model=opponent_model, env=opponent_env, n_eval_episodes=1, n_seeds=1, negative_score_flag=False, render=False)


        # perf_agent_opponent = self._bug_compute_performance(agent, opponent, agent_name, n_eval_episodes=3, n_seeds=1, negative_score_flag=True, render=False)
        # perf_opponent_agent = self._bug_compute_performance(opponent, agent, opponent_name, n_eval_episodes=3, n_seeds=1, negative_score_flag=False, render=False)


if __name__ == "__main__":
    testing = SelfPlayTesting()

    testing.bug()