# Testing script for self-play using Stable baselines3

# Note: this script is made only for now for pred and prey (1v1) setting

# Note: player1 -> predator -> agent, player2 -> prey -> opponent

# TODO: make this script extendable with NvM competitive games

# Done: Seed fixing -> enable n_seeds -> enable random reseeding inside the env if the seed is passed with None -> done in _create_env -> just specify "random"
# TODO: test prey model with fixed predator


# Fix that the evaluation is not asymetic (Fixed with appropriate seeding to the env )

# TODO: make the cmd args force on json parameters


# TODO: enable testing only takes from the json of the experiment specified from the path, if not exist use the configuration in its json (to be compatible with the old codes)

from operator import le
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
from os import walk

from datetime import datetime


MAX_NUM_STEPS = 1000

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
        self._env_name = "Testing"


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
        self.best_flag = self.testing_configs.get("best", False)
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
            env = super(SelfPlayTesting, self).create_env(key=k, name=self._env_name, opponent_archive=None, algorithm_class=algorithm_class, gui=True)
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
                                  return_episode_rewards=False,
                                  eval_seed=None):
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
                                                                                                seed_value=eval_seed,
                                                                                                trajectory_heatmap=self.testing_configs.get("trajectory_heatmap", False)
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
        # def normalize_performance(min_val, max_val, performance, negative_score_flag):
        #     if(negative_score_flag):
        #         performance = min(0, performance) # to bound it in case the agent caught it directly (never happens) -> performance = +10 
        #         # -1010 ->  0, 0 -> 1
        #         return (max_val - abs(performance)) / max_val
        #     else:
        #         performance = max(0, performance) # to bound it in case the agent caught it directly (never happens) -> performance = -10 
        #         # 1010 ->  1, 0 -> 0
        #         return performance / max_val
        lengths = []
        for i in range(n_seeds):
            random_seed = None if(n_seeds == 1) else datetime.now().microsecond//1000
            # random.seed(random_seed)
            episodes_reward, episodes_length, win_rates, std_win_rate, render_ret = self._run_one_evaluation(key, agent, [opponent], n_eval_episodes, render=render, render_extra_info=f"{agent} vs {opponent}" if render_extra_info is None else render_extra_info, return_episode_rewards=True, eval_seed=random_seed)
            length = np.mean(episodes_length)
            lengths.append(length)
        length = np.mean(lengths)
        limits = [0,1000] # maximum number of steps
        normalized_length = length
        if(negative_score_flag):
            normalized_length = MAX_NUM_STEPS - length
        normalized_length /= MAX_NUM_STEPS #normalize_performance(*limits, length, negative_score_flag)
        self.clilog.debug(f"Nomralized: {normalized_length}, {length}")
        return normalized_length

    def crosstest(self, n_eval_episodes, n_seeds):
        self.deterministic = False
        self._env_name = "Evaluation"

        self.clilog.info(f"---------------- Running Crosstest ----------------")
        # Get the experiments of the approaches
        # crosstest_methods_path: dict -> Method is the key of the dictionary and the value is list of experiments trained with the same method
        methods_experiments_parent_path = self.testing_configs.get("crosstest_methods_parent_path")
        methods_experiments_path = self.testing_configs.get("crosstest_methods_path")
        
        methods_experiments_best_agents = {} # {method1: {exp1: {agent1, agent2}, exp2: ... }, method2: ... }
        filtered_methods_experiments_path = {}
        # Get the best agent names per each method per each experiment per each agent 
        # Per each method
        for m in list(methods_experiments_path.keys()):
            self.clilog.info(f"=============== Method: {m} ===============")
            method_parent_path = methods_experiments_parent_path[m]
            self.clilog.info(f"Parent path: {method_parent_path}")
            methods_experiments_best_agents[m] = {}
            filtered_methods_experiments_path[m] = []
            if(len(methods_experiments_path[m]) == 0):
                self.clilog.warn(f"For method {m}, the script will automatically get all the experiments inside this method parent directory {method_parent_path}")
                methods_experiments_path[m] = next(os.walk(method_parent_path))[1]
            # Per each experiment
            for exp in methods_experiments_path[m]:
                self.clilog.info(f"Exp Path: {exp}")
                # Load the experiment
                experiment_path = os.path.join(method_parent_path, exp, "experiment_config.json")
                # Parse the experiment configuration with evaluation configuration
                _evaluation_configs = None
                try:
                    _, _, _evaluation_configs, _, _ = ExperimentParser.load(experiment_path)
                except Exception as e:
                    self.clilog.error("This experiment is not finished and does not have experiment configuration as a result of finish the experiment")
                    self.clilog.error(e)
                    continue
                self.clilog.debug(_evaluation_configs)
                filtered_methods_experiments_path[m].append(exp)
                methods_experiments_best_agents[m][exp] = {}
                for k in self.agents_configs.keys():
                    agent_name = self.agents_configs[k]["name"]
                    methods_experiments_best_agents[m][exp][agent_name] = os.path.join(method_parent_path, exp, agent_name, _evaluation_configs[agent_name]['best_agent_name'])
                    self.clilog.info(f"Best {agent_name}: {_evaluation_configs[agent_name]['best_agent_name']}\nPath: {methods_experiments_best_agents[m][exp][agent_name]}")
            self.clilog.info(f"========================================================================================")

        method_names = list(filtered_methods_experiments_path.keys())

        agents_names = [self.agents_configs[k]["name"] for k in self.agents_configs.keys()]
        best_agents_method = {}
        # Get best agents for each method 
        for m in method_names:
            self.clilog.info(f"=============== Method: {m} ===============")
            best_agents_method[m] = {i: methods_experiments_best_agents[m][filtered_methods_experiments_path[m][0]][i] for i in agents_names}
            for i in range(len(filtered_methods_experiments_path[m])-1):
                break
                self.clilog.info(f"Exp {i} vs {i+1}")
                other_agents_method = methods_experiments_best_agents[m][filtered_methods_experiments_path[m][i+1]]
                best_agents_idx, _, _ = self._crosstest(best_agents_method[m], other_agents_method, agents_names, n_eval_episodes, n_seeds, None, None, False)
                for agent_name in agents_names:
                    eps = 0.5
                    if( 
                        best_agents_idx[agent_name] == 2 or \
                        (best_agents_idx[agent_name] == 0 and np.random.rand() > eps)   # Both methods are similar to each other, so we make it randomly to choose which to continue with as best
                     ):
                        best_agents_method[m][agent_name] = other_agents_method[agent_name]
                    # if the first method is better, it will stick with the best
            self.clilog.info(f"========================================================================================")
            self.clilog.info(f"Best agents for method {m}: {best_agents_method[m]}")
            # input("Input to break")
            self.clilog.info(f"========================================================================================")

        # self.clilog.info("###############################################################")
        # self.clilog.info(f"# Best agent1: {best_agent1}")
        # self.clilog.info(f"# Best opponent1: {best_opponent1}")
        # self.clilog.info(f"# Best agent2: {best_agent2}")
        # self.clilog.info(f"# Best opponent2: {best_opponent2}")
        # self.clilog.info("###############################################################")

        # # agent1 predator -> performance is related to the reward 
        render = self.testing_configs.get("render")
        num_methods = len(method_names)
        crosstest_mat = np.zeros((num_methods, num_methods))
        crosstest_mat2 = np.zeros((num_methods, num_methods))
        for i in range(num_methods):
            method1 = best_agents_method[method_names[i]]
            for j in range(i, num_methods):
            # for j in range(i+1, num_methods):
                method2 = best_agents_method[method_names[j]]
                best_agents_idx, best_method_idx, scores = self._crosstest(method1, method2, agents_names, n_eval_episodes, n_seeds, method_names[i], method_names[j], render)
                
                gain_score = scores[2][2]
                crosstest_mat[i,j] = gain_score
                crosstest_mat2[i,j] = scores[2][3]

        self.clilog.critical(method_names)
        self.clilog.critical(f"Mat1: (g1+g2)\n{crosstest_mat}")
        self.clilog.critical(f"Mat2: (g1-g2)\n{crosstest_mat2}")
        np.save(f"{self.testing_configs.get('crosstest_save_name', 'crosstest_res')}1.npy", crosstest_mat)
        np.save(f"{self.testing_configs.get('crosstest_save_name', 'crosstest_res')}2.npy", crosstest_mat2)
        # best_method = {i: best_agents_method[method_names[0]][i] for i in agents_names}
        # for i in range(num_methods-1):
        #     other_method = best_agents_method[method_names[i+1]]
        #     best_agents_idx, best_method_idx = self._crosstest(best_method[m], other_method, agents_names, n_eval_episodes, n_seeds, None, None, render)
        #     # input("Input to break")
        #     # crosstest_mat[]
        #     eps = 0.5
        #     if(best_method_idx)
        #     elif(best_agents_idx[agent_name] == 2):
        #         best_agents_method[m][agent_name] = other_agents_method[agent_name]
        #     if(best_method_idx == 1):
        #         self.clilog.info(f"Method 1 ({method_names}) is better")
        #     self.clilog.info(f"========================================================================================")
    
    def _crosstest(self, 
                    method1_agents, method2_agents, 
                    agent_names, 
                    n_eval_episodes, n_seeds, 
                    approach1_path=None, approach2_path=None, 
                    render=False):
        best_agents_idx = []
        # For simplicity, we will assume that we are evaluating only two agents
        agent_name, opponent_name = agent_names
        best_agent1, best_opponent1 = [method1_agents[a] for a in agent_names]
        best_agent2, best_opponent2 = [method2_agents[a] for a in agent_names]

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

        self.clilog.info(f"################# Opponent2 vs Agent2 #################")
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


        gain = [gain1, gain2, gain1+gain2, gain1-gain2]
        perf_agent = [perf_agent1, perf_agent2, perf_agent1+perf_agent2]
        perf_opponent = [perf_opponent1, perf_opponent2, perf_opponent1+perf_opponent2]
        best_agents_idx = {}
        best_method_idx = None
        # for i in range(3):
        i = 2
        self.clilog.critical(f"{approach1_path} vs {approach2_path}")
        self.clilog.critical(f" ----- Part {i+1} ----- ")
        eps = 1e-3
        if(perf_agent[i] > 0):
            self.clilog.critical(f"Configuration 1 is better {1} to generate preys (path: {approach1_path})")
            best_agents_idx[agent_name] = 1
        elif(-eps <= perf_agent[i] <= eps):
            self.clilog.critical(f"Configuration 1 & 2 are close to each other to generate preys ({approach1_path}, {approach2_path})")
            best_agents_idx[agent_name] = 0
        else:
            self.clilog.critical(f"Configuration 2 is better {2} to generate preys (path: {approach2_path})")
            best_agents_idx[agent_name] = 2


        if(perf_opponent[i] > 0):
            self.clilog.critical(f"Configuration 1 is better {1} to generate predators (path: {approach1_path})")
            best_agents_idx[opponent_name] = 1
        elif(-eps <= perf_opponent[i] <= eps):
            self.clilog.critical(f"Configuration 1 & 2 are close to each other to generate predators ({approach1_path}, {approach2_path})")
            best_agents_idx[opponent_name] = 0
        else:
            self.clilog.critical(f"Configuration 2 is better {2} to generate predators (path: {approach2_path})")
            best_agents_idx[opponent_name] = 2

        if(gain[i] > 0):
            self.clilog.critical(f"Configuration 1 is better {1} (path: {approach1_path})")
            best_method_idx = 1
        elif(-eps <= gain[i] <= eps):
            self.clilog.critical(f"Configuration 1 & 2 are close to each other ({approach1_path}, {approach2_path})")
            best_method_idx = 0
        else:
            self.clilog.critical(f"Configuration 2 is better {2} (path: {approach2_path})")
            best_method_idx = 2

        return best_agents_idx, best_method_idx, [perf_agent, perf_opponent, gain]

    def best_testing(self, path, n_eval_episodes, n_seeds):
        agents = []
        agent_names = {}
        for k in self.agents_configs.keys():
            agent_configs = self.agents_configs[k]
            agent_name = agent_configs["name"]
            testing_config = self.testing_configs[agent_name]
            testing_path = os.path.join(path, agent_name) if(testing_config["path"] is None) else testing_config["path"]
            # # Load the experiment
            experiment_path = os.path.join(testing_path, "experiment_config.json")
            # Parse the experiment configuration with evaluation configuration
            _evaluation_configs = None
            try:
                _, _, _evaluation_configs, _, _ = ExperimentParser.load(experiment_path)
            except Exception as e:
                self.clilog.error("This experiment is not finished and does not have experiment configuration as a result of finish the experiment")
                self.clilog.error(e)
                continue
            # self.clilog.debug(_evaluation_configs)
            best_agent_name = os.path.join(testing_path, agent_name, _evaluation_configs[agent_name]['best_agent_name'])
            self.clilog.info(f"Best {agent_name}: {_evaluation_configs[agent_name]['best_agent_name']}\nPath: {best_agent_name}")
            agents.append(best_agent_name)
            agent_names[best_agent_name] = agent_name
        for a in agents:
            for b in (set(agents) - {a}):
                sampled_opponents = [b]
                sampled_agent = a
                key = agent_names[a]
                self.clilog.critical("------------------------------------------------")
                self.clilog.info(f"Env: {key}")
                self.clilog.info(f"{a} vs {b}")
                self._run_one_evaluation(key, sampled_agent, sampled_opponents, n_eval_episodes, f"{a} vs {b}")
                self.clilog.critical("------------------------------------------------")
    
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
        elif(self.best_flag):
            n_seeds = self.testing_configs.get("n_seeds", 1)
            self.best_testing(experiment_filename, n_eval_episodes, n_seeds)
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
