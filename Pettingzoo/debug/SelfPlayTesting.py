# Testing script for self-play using Stable baselines3

# Note: this script is made only for now for pred and prey (1v1) setting

# Note: player1 -> predator -> agent, player2 -> prey -> opponent

# TODO: make this script extendable with NvM competitive games

# Done: Seed fixing -> enable n_seeds -> enable random reseeding inside the env if the seed is passed with None -> done in _create_env -> just specify "random"
# TODO: test prey model with fixed predator


# Fix that the evaluation is not asymetic (Fixed with appropriate seeding to the env )

# TODO: make the cmd args force on json parameters


# TODO: enable testing only takes from the json of the experiment specified from the path, if not exist use the configuration in its json (to be compatible with the old codes)

from SelfPlayExp import SelfPlayExp # Import it at the begining of the file to correctly init the logger

import os

from stable_baselines3 import PPO
from stable_baselines3 import SAC

from callbacks import *

from shared import evaluate_policy_simple
import bach_utils.os as utos
from bach_utils.shared import *
from bach_utils.heatmapvis import *
from bach_utils.json_parser import ExperimentParser
import random


# This is a modified PPO to tackle problem related of loading from different version of pickle than it was saved with
class PPOMod(PPO):
    def __init__(self, *args, **kwargs):
        super(PPOMod, self).__init__(*args, **kwargs)

    # To fix issue while loading when loading from different versions of pickle and python from the server and the local machine
    # https://stackoverflow.com/questions/63329657/python-3-7-error-unsupported-pickle-protocol-5
    # @staticmethod
    # def load(model_path, env):
    #     custom_objects = {
    #         "lr_schedule": lambda x: .003,
    #         "clip_range": lambda x: .2
    #     }
    #     return PPO.load(model_path, env, custom_objects=custom_objects)

class SelfPlayTesting(SelfPlayExp):
    def __init__(self, seed_value=None, render_sleep_time=0.001):
        super(SelfPlayTesting, self).__init__()
        self.seed_value = seed_value
        self.load_prefix = "history_"
        self.deterministic = True  # This flag is used wrongly, it is for deterministic flag in the callback evaluation not the determinism of the experiemnt
        self.warn = True
        self.render = None # it is being set by the configuration file
        self.crosstest_flag = None
        self.render_sleep_time = render_sleep_time

    # This is made in case that the agents have different RL policies (inner optimization)
    def _import_original_configs(self):
        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            testing_config = self.testing_configs[agent_name]
            agent_config_file_path = os.path.join(testing_config["path"], "experiment_config.json")
            # if file exists then merge the configuration of that agent
            if(os.path.isfile(agent_config_file_path)):
                self.clilog.info(f"Parse from json file in {agent_config_file_path}")
                _experiment_configs, _agents_configs, _evaluation_configs, _testing_configs, _merged_config = ExperimentParser.load(agent_config_file_path)
                agent_original_config = _agents_configs[k]
                self.agents_configs[k] = agent_original_config


    def _init_testing(self, experiment_filename, logdir, wandb):
        super(SelfPlayTesting, self)._init_exp(experiment_filename, logdir, wandb)

        # Only the agents_configs are overwritten
        self._import_original_configs()

        self.render = self.testing_configs.get("render", True)
        self.crosstest_flag = self.testing_configs.get("crosstest", False)
        self.clilog.info(f"----- Load testing conditions")
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
            self.clilog.debug(self.testing_conditions[agent_name]["limits"])

    def _get_opponent_algorithm_class(self, agent_configs):
        algorithm_class = None
        opponent_algorithm_class_cfg = agent_configs.get("opponent_rl_algorithm", agent_configs["rl_algorithm"])
        if(opponent_algorithm_class_cfg == "PPO"):
            algorithm_class = PPOMod
        elif(opponent_algorithm_class_cfg == "SAC"):
            algorithm_class = SAC
        return algorithm_class

    def _init_envs(self):
        self.envs = {}

        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            # env = globals()["SelfPlayPredEnv"](algorithm_class=PPOMod, archive=None, seed_val=3)
            algorithm_class = self._get_opponent_algorithm_class(agent_configs)
            env = super(SelfPlayTesting, self).create_env(key=k, name="Testing", opponent_archive=None, algorithm_class=algorithm_class, gui=True)
            # if not isinstance(env, VecEnv):
            #     env = DummyVecEnv([lambda: env])

            # if not isinstance(env, DummyVecEnvSelfPlay):
            #     env.__class__ = DummyVecEnvSelfPlay   # This works fine, the other solution is commented
            
            self.envs[agent_name] = env
    
    # TODO: create _init_archives() but after the archive is integrated with the sampling and the indexing
    def _init_archives(self):
        raise NotImplementedError("_init_archives() not implemented")
    
    # Useless now as there is a problem and we have to recreate the model again with each evaluation
    #[Deprecated]
    def _init_models(self):
        self.models = {}

        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            algorithm_class = None
            if(agent_configs["rl_algorithm"] == "PPO"):
                algorithm_class = PPOMod
            elif(agent_configs["rl_algorithm"] == "SAC"):
                algorithm_class = SAC

            self.models[agent_name] = algorithm_class
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
        self.clilog.info("----------------------------------------")
        self.clilog.info(render_extra_info)
        self.make_deterministic(cuda_check=False)   # This was added as we observed that previous rounds affect the other rounds
        # TODO: debug why if we did not do this (redefine the env again) it does not work properly for the rendering
        # Create environment for each evaluation
        if(env is None and agent_model is None):
            agent_configs = self.agents_configs[agent_conifgs_key]
            # print(f"Create Env: {self.agents_configs[agent_conifgs_key]['env_class']}, Algorithm: {PPOMod}, seed: {seed_value}")
            opponent_algorithm_class = self._get_opponent_algorithm_class(agent_configs)
            env, seed_value = super(SelfPlayTesting, self).create_env(key=agent_conifgs_key, name="Testing", opponent_archive=None, algorithm_class=opponent_algorithm_class, seed_value=seed_value, ret_seed=True, gui=True)
            # print(f"Sampled agent loading {sampled_agent}")
            algorithm_class = None
            if(agent_configs["rl_algorithm"] == "PPO"):
                algorithm_class = PPOMod
            elif(agent_configs["rl_algorithm"] == "SAC"):
                algorithm_class = SAC
            # print("Debug")
            # PPO.load -> calls env.seed for some reason with wrong seed value (I think the random seed value used during the training is stored and restored with the model)
            self.clilog.debug(f"loading agent model: {sampled_agent}, {algorithm_class}, {env}")
            agent_model = algorithm_class.load(sampled_agent, env)
            # print("======")
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
                                                                                                # seed_value=seed_value
                                                                                            )
        mean_reward_, std_reward_, win_rate_, std_win_rate_ = mean_reward, std_reward, win_rate, std_win_rate
        if(return_episode_rewards):
            # return episodes_reward, episodes_length, win_rates, std_win_rate, render_ret
            mean_reward_  = np.mean(mean_reward)
            std_reward_   = np.std(mean_reward) 
            win_rate_     = np.mean(win_rate)
        self.clilog.info(f"{render_extra_info} -> win rate: {100 * win_rate_:.2f}% +/- {std_win_rate_:.2f}\trewards: {mean_reward_:.2f} +/- {std_reward_:.2f}")
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
        episodes_reward, episodes_length, win_rates, std_win_rate, render_ret = self._run_one_evaluation(key, agent, [opponent], n_eval_episodes, render=render, render_extra_info=f"{agent} vs {opponent}" if render_extra_info is None else render_extra_info, return_episode_rewards=True)
        length = np.mean(episodes_length)
        limits = [0,1000] # maximum number of steps
        normalized_length = normalize_performance(*limits, length, negative_score_flag)
        self.clilog.debug(f"Nomralized: {normalized_length}, {length}")
        return normalized_length

        mean_reward, std_reward, win_rate, std_win_rate, render_ret = self._run_one_evaluation(key, agent, [opponent], n_eval_episodes, render=render, render_extra_info=f"{agent} vs {opponent}" if render_extra_info is None else render_extra_info)
        reward = np.mean(mean_reward) # get the performance reward
        limits = self.testing_configs.get("crosstest_rewards_limits")
        normalized_reward = normalize_performance(*limits, reward, negative_score_flag)
        print(f"Nomralized: {normalized_reward}, {reward}")
        return normalized_reward

    def _get_best_agent(self, num_rounds, search_radius, paths, key, num_population, min_gamma_val=0.05, n_eval_episodes=1, n_seeds=1, render=False, negative_score_flag=False):#, negative_reward_flag=False):
        agent_num_rounds, opponent_num_rounds = num_rounds[:]
        agent_path, opponent_path = paths[:]
        self.clilog.info("##################################################################")
        self.clilog.info(f"## Getting the best model for {key}")
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
                for opponent_population_idx_ in range(1):
                    # print(f"POP: {agent_population_idx}, {opponent_population_idx}")
                    opponent_population_indices = [random.randint(0,num_population-1) for _ in range(len(opponents_rounds_idx))]
                    for i, opponent_idx in enumerate(opponents_rounds_idx):
                        opponent_population_idx = opponent_population_indices[i]#random.randint(0,num_population-1)
                        ret, sampled_opponent = self.get_latest_agent_path(opponent_idx, opponent_path, opponent_population_idx)
                        if(not ret):
                            continue
                        # mean_reward, std_reward, win_rate, std_win_rate, render_ret = self._run_one_evaluation(key, agent, [sampled_opponent], n_eval_episodes, render=False, render_extra_info=f"{agent_idx}({agent_population_idx}) vs {opponent_idx} ({opponent_population_idx})")
                        mean_reward = self._compute_performance(agent, sampled_opponent, key, n_eval_episodes, n_seeds, negative_score_flag, render, render_extra_info=f"{agent_idx}({agent_population_idx}) vs {opponent_idx} ({opponent_population_idx})")
                        weight = 1#(gamma**(len(opponents_rounds_idx)-i))
                        # if(negative_reward_flag):
                        #     discount = (gamma**(i))
                        weighted_reward = weight * mean_reward
                        self.clilog.debug(f"Weight: {weight}\tPerformance: {mean_reward}\tWeighted Performance: {weighted_reward}")
                        rewards.append(weighted_reward)
                mean_reward = np.mean(np.array(rewards))
                best_rewards.append([agent_population_idx, agent_idx, mean_reward])
        best_rewards = np.array(best_rewards)
        self.clilog.debug(best_rewards)
        best_agent_idx = np.argmax(best_rewards[:,2])
        # Get the best agent using best_agent_idx
        agent_idx = int(best_rewards[best_agent_idx,1]) #agent_num_rounds-search_radius-1 + (best_agent_idx%)
        agent_population_idx = int(best_rewards[best_agent_idx,0])
        self.clilog.info(f"Best agent: idx {agent_idx}, population {agent_population_idx}")
        startswith_keyword = f"{self.load_prefix}{agent_idx}_"
        agent_latest = utos.get_latest(agent_path, startswith=startswith_keyword, population_idx=agent_population_idx)
        best_agent = os.path.join(agent_path, agent_latest[0])  # Join it with the agent path
        return best_agent


    # TODO: this code need to be parallel
    # TODO: need to speed up the code by not running one evaluation and then delete the environment
    def crosstest(self, n_eval_episodes, n_seeds):
        self.clilog.info(f"---------------- Running Crosstest ----------------")
        # For now both the approaches have the same number of rounds
        # TODO: save the configuration file with the experiment, so we can parse the training configs with it and the number of round for each approach,
        # TODO: make crosstest as a seperate or a big element in testing 
        # TODO: change opponent with adversery word (maybe)
        # TODO: refactor and make it better and make it for more than 1v1
        num_rounds = self.testing_configs.get("crosstest_num_rounds")
        num_rounds1, num_rounds2 = num_rounds[0], num_rounds[1]
        search_radius = self.testing_configs.get("crosstest_search_radius")
        self.clilog.info(f"Num. rounds: {num_rounds1}, {num_rounds2}")

        # def _get_best_agent(self, agent_num_rounds, opponent_num_rounds, search_radius, agent_path, opponent_path):
        approaches_path = self.testing_configs.get("crosstest_approaches_path")
        approach1_path, approach2_path = approaches_path[0], approaches_path[1]
        self.clilog.info(f"Paths:\n{approach1_path}\n{approach2_path}")

        names = [self.agents_configs[k]["name"] for k in self.agents_configs.keys()]
        agent_name, opponent_name = names[0], names[1]
        self.clilog.info(f"names: {agent_name}, {opponent_name}")

        
        agent1_path = os.path.join(approach1_path, agent_name)
        opponent1_path = os.path.join(approach1_path, opponent_name)
        agent2_path = os.path.join(approach2_path, agent_name)
        opponent2_path = os.path.join(approach2_path, opponent_name)

        self.clilog.info(f"Agent1 path: {agent1_path}")
        self.clilog.info(f"Opponenet1 path: {opponent1_path}")
        self.clilog.info(f"Agent2 path: {agent2_path}")
        self.clilog.info(f"Opponenet2 path: {opponent2_path}")

        num_population1, num_population2 = self.testing_configs.get("crosstest_populations")
        self.clilog.info(f"Num. populations: {num_population1}, {num_population2}")

        # O(num_pop * search_radius * num_pop * (opponent_num_rounds//freq) * 4 * n_eval * n_seeds) 
        # 5*10*5*10*4 *1*1 = 10000 (num_pop = 5, search_radius = 10, freq = 5, opponent_num_rounds = 50) -> very expensive operations
        # 10k * 5sec / (3600) = 13.8888889 hours !!!!
        # TODO: add threading here: https://realpython.com/intro-to-python-threading/
        n_eval_episodes_best_agent = self.testing_configs.get("n_eval_episodes_best_agent", 1)
        best_agent1     = self._get_best_agent([num_rounds1, num_rounds1], search_radius, [agent1_path, opponent1_path], agent_name, num_population1, n_eval_episodes=n_eval_episodes_best_agent, negative_score_flag=True, n_seeds=n_seeds)
        self.clilog.info(f"Best agent1: {best_agent1}")
        best_opponent1  = self._get_best_agent([num_rounds1, num_rounds1], search_radius, [opponent1_path, agent1_path], opponent_name, num_population1, n_eval_episodes=n_eval_episodes_best_agent, negative_score_flag=False, n_seeds=n_seeds)
        self.clilog.info(f"Best opponent1: {best_opponent1}")
        best_agent2     = self._get_best_agent([num_rounds2, num_rounds2], search_radius, [agent2_path, opponent2_path], agent_name, num_population2, n_eval_episodes=n_eval_episodes_best_agent, negative_score_flag=True, n_seeds=n_seeds)
        self.clilog.info(f"Best agent2: {best_agent2}")
        best_opponent2  = self._get_best_agent([num_rounds2, num_rounds2], search_radius, [opponent2_path, agent2_path], opponent_name, num_population2, n_eval_episodes=n_eval_episodes_best_agent, negative_score_flag=False, n_seeds=n_seeds)
        self.clilog.info(f"Best opponent2: {best_opponent2}")

        self.clilog.info("###############################################################")
        self.clilog.info(f"# Best agent1: {best_agent1}")
        self.clilog.info(f"# Best opponent1: {best_opponent1}")
        self.clilog.info(f"# Best agent2: {best_agent2}")
        self.clilog.info(f"# Best opponent2: {best_opponent2}")
        self.clilog.info("###############################################################")

        # agent1 predator -> performance is related to the reward 
        render = self.testing_configs.get("render")
        self.clilog.info(f"################# Agent1 vs Opponent2 #################")
        perf_agent1_opponent2 = self._compute_performance(best_agent1, best_opponent2, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=render)
        self.clilog.info(f"################# Agent1 vs Opponent1 #################")
        perf_agent1_opponent1 = self._compute_performance(best_agent1, best_opponent1, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=render)

        self.clilog.info(f"################# Agent2 vs Opponent2 #################")
        perf_agent2_opponent2 = self._compute_performance(best_agent2, best_opponent2, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=render)
        self.clilog.info(f"################# Agent2 vs Opponent1 #################")
        perf_agent2_opponent1 = self._compute_performance(best_agent2, best_opponent1, agent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=True, render=render)

        self.clilog.info(f"################# Opponent1 vs Agent2 #################")
        perf_opponent1_agent2 = self._compute_performance(best_opponent1, best_agent2, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=render)
        self.clilog.info(f"################# Opponent1 vs Agent1 #################")
        perf_opponent1_agent1 = self._compute_performance(best_opponent1, best_agent1, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=render)

        self.clilog.info(f"################# Opponent1 vs Agent2 #################")
        perf_opponent2_agent2 = self._compute_performance(best_opponent2, best_agent2, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=render)
        self.clilog.info(f"################# Opponent2 vs Agent1 #################")
        perf_opponent2_agent1 = self._compute_performance(best_opponent2, best_agent1, opponent_name, n_eval_episodes=n_eval_episodes, n_seeds=n_seeds, negative_score_flag=False, render=render)


        # Performance against agent1
        perf_agent1 = perf_agent1_opponent2 - perf_agent1_opponent1 
        perf_agent2 = perf_agent2_opponent2 - perf_agent2_opponent1
        perf_opponent1 = perf_opponent1_agent2 - perf_opponent1_agent1 
        perf_opponent2 = perf_opponent2_agent2 - perf_opponent2_agent1
        
        gain1 = perf_agent1 + perf_opponent1
        gain2 = perf_agent2 + perf_opponent2

        self.clilog.info("-----------------------------------------------------------------")
        self.clilog.info(f"perf_agent1: {perf_agent1}\tperf_opponent1: {perf_opponent1}\tgain1: {gain1}")
        self.clilog.info(f"perf_agent2: {perf_agent2}\tperf_opponent2: {perf_opponent2}\tgain2: {gain2}")
        self.clilog.info(f"perf_agent: {perf_agent1+perf_agent2}\tperf_opponent: {perf_opponent1+perf_opponent2}\tgain(sum): {gain1+gain2}")


        gain = [gain1, gain2, gain1+gain2]
        perf_agent = [perf_agent1, perf_agent2, perf_agent1+perf_agent2]
        perf_opponent = [perf_opponent1, perf_opponent2, perf_opponent1+perf_opponent2]
        for i in range(3):
            self.clilog.info(f" ----- Part {i+1} ----- ")
            eps = 1e-3
            if(perf_agent[i] > 0):
                self.clilog.info(f"Configuration 1 is better {1} to generate preys (path: {approach1_path})")
            elif(-eps <= perf_agent[i] <= eps):
                self.clilog.info(f"Configuration 1 & 2 are close to each other to generate preys")
            else:
                self.clilog.info(f"Configuration 2 is better {2} to generate preys (path: {approach2_path})")

            if(perf_opponent[i] > 0):
                self.clilog.info(f"Configuration 1 is better {1} to generate predators (path: {approach1_path})")
            elif(-eps <= perf_opponent[i] <= eps):
                self.clilog.info(f"Configuration 1 & 2 are close to each other to generate predators")
            else:
                self.clilog.info(f"Configuration 2 is better {2} to generate predators (path: {approach2_path})")

            if(gain[i] > 0):
                self.clilog.info(f"Configuration 1 is better {1} (path: {approach1_path})")
                # return 1
            elif(-eps <= gain[i] <= eps):
                self.clilog.info(f"Configuration 1 & 2 are close to each other")
            else:
                self.clilog.info(f"Configuration 2 is better {2} (path: {approach2_path})")
                # return 2
            

    def test(self, experiment_filename=None, logdir=False, wandb=False, n_eval_episodes=1):
        self._init_testing(experiment_filename=experiment_filename, logdir=logdir, wandb=wandb)
        self.render_sleep_time = self.render_sleep_time if self.args.rendersleep <= 0 else self.args.rendersleep

        # print(self.agents_configs)
        # exit()
        n_eval_episodes_configs = self.testing_configs.get("repetition", None)
        n_eval_episodes = n_eval_episodes_configs if n_eval_episodes_configs is not None else n_eval_episodes

        if(self.crosstest_flag):
            n_seeds = self.testing_configs.get("n_seeds", 1)
            self.crosstest(n_eval_episodes, n_seeds)
            # self._compute_gain_score(n_eval_episodes, n_seeds)
        else:
            already_evaluated_agents = []
            # In order to extend it multipe agents, we can make it as a recursive function (list:[models....,, None]) and pass the next element in the list, the termination criteria if the argument is None
            self.clilog.debug(self.testing_modes)
            # exit()
            keys = self.agents_configs.keys()
            keys = ["pred", "prey"]
            for k in keys:
                agent_configs = self.agents_configs[k]
                agent_name = agent_configs["name"]
                agent_opponent_joint = sorted([agent_name, agent_configs["opponent_name"]])

                # if(self.testing_modes[agent_name] == "round"):
                if("round" in self.testing_modes.values()):
                    if(agent_opponent_joint in already_evaluated_agents):
                        continue

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
