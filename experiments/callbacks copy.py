from bach_utils.logger import get_logger
clilog = get_logger()

# Generation and round are being used as the same meaning
from random import sample
from stable_baselines3.common.callbacks import EvalCallback, EventCallback
import os
import bach_utils.sorting as utsrt
import bach_utils.os as utos
import bach_utils.list as utlst
import bach_utils.sampling as utsmpl

import numpy as np

from stable_baselines3.common.vec_env import sync_envs_normalization
import wandb
from shared import *
from bach_utils.shared import *

OS = False #True   # This flag just for testing now in order not to break the compatibility and the working of the code

# Select opponent for training
class TrainingOpponentSelectionCallback(EventCallback):
    def __init__(self, *args, **kwargs):
        self.sample_path = kwargs.pop("sample_path")
        self.startswith_keyword = "history"
        self.env = kwargs.pop("env")
        self.opponent_selection = kwargs.pop("opponent_selection")
        self.sample_after_rollout = kwargs.pop("sample_after_rollout")
        self.sample_after_reset = kwargs.pop("sample_after_reset")
        self.num_sampled_per_round = kwargs.pop("num_sampled_per_round")
        self.archive = kwargs.pop("archive")
        self.randomly_reseed_sampling = kwargs.pop("randomly_reseed_sampling")
        self.OS = OS    # Global flag

        new_kwargs = {}
        for k in kwargs.keys():
            if(k == "archive"):
                continue
            new_kwargs[k] = kwargs[k]



        self.sampled_per_round = []
        self.sampled_idx = 0

        super(TrainingOpponentSelectionCallback, self).__init__(*args, **new_kwargs)

    # Only once the training starts of the round
    def _on_training_start(self):
        self.env.reset_counter = 0
        # if(not self.sampled_per_round):
        clilog.info("training started")
        if(not (self.sample_after_rollout or self.sample_after_reset)):
            clilog.debug("Sample opponets for the training round at the start")
            # Sample opponents for the round
            if(not self.OS):
                # print("Not OS")
                archive = self.archive.get_sorted(self.opponent_selection) # this return [sorted_names, sorted_policies]
                models_names = archive[0]
                self.sampled_per_round = utsmpl.sample_opponents(models_names, self.num_sampled_per_round, selection=self.opponent_selection, sorted=True, randomly_reseed=self.randomly_reseed_sampling)
            if(self.OS):
                self.sampled_per_round = utsmpl.sample_opponents_os(self.sample_path, self.startswith_keyword, self.num_sampled_per_round, selection=self.opponent_selection, randomly_reseed=self.randomly_reseed_sampling)
            # If it is specified to have only one sample per round, then load it only once at the training start (now) and do not change it (not load it again)
            if(self.num_sampled_per_round == 1):
                clilog.debug("Set the opponent only once as num_sampled_per_round=1")
                self.env.set_target_opponent_policy_name(self.sampled_per_round[0])
        super(TrainingOpponentSelectionCallback, self)._on_training_start()

    # def _on_training_end(self):
    #     self.sampled_per_round = []
    #     super(TrainingOpponentSelectionCallback, self)._on_training_start()

    # With every rollout
    def _on_rollout_start(self):
        # print("Start of Rollout")
        # If sample_after_reset is false -> Use the sampling method with the rollout
        #       if it is true -> the oppoenent will be sampled from the environment itself
        if(not self.sample_after_reset):
            # If sample_after_rollout is true -> sample different opponent after each rollout 
            if(self.sample_after_rollout):
                clilog.debug("Sample opponents again with the start rollout")
                opponent = None
                if(not self.OS):
                    archive = self.archive.get_sorted(self.opponent_selection)
                    models_names = archive[0]
                    opponent = utsmpl.sample_opponents(models_names, 1, selection=self.opponent_selection, sorted=True, randomly_reseed=self.randomly_reseed_sampling)[0]
                if(self.OS):
                    opponent = utsmpl.sample_opponents_os(self.sample_path, self.startswith_keyword, 1, selection=self.opponent_selection, randomly_reseed=self.randomly_reseed_sampling)[0]
                clilog.debug("Change sampled agent")
                self.env.set_target_opponent_policy_name(opponent)
            # If sample_after_rollout is fales -> Do not sample anymore and just use the the current bag of samples as a circular buffer
            else:
                if(self.num_sampled_per_round > 1): # just condition not for resetting the same agent multiple times if the sampled agent is always the same
                    clilog.debug("Change sampled agent")
                    self.env.set_target_opponent_policy_name(self.sampled_per_round[self.sampled_idx % self.num_sampled_per_round]) # as a circular buffer
                    self.sampled_idx += 1

        super(TrainingOpponentSelectionCallback, self)._on_rollout_start()


# Based on: https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_selfplay.py
# Based on: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py
class EvalSaveCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        self.save_path = kwargs.pop("save_path")
        self.eval_metric = kwargs.pop("eval_metric")
        self.eval_opponent_selection = kwargs.pop("eval_opponent_selection")
        self.eval_sample_path = kwargs.pop("eval_sample_path")
        self.save_freq = kwargs.pop("save_freq")
        archive = kwargs.pop("archive")
        self.archive = archive["self"]  # pass it by reference
        self.opponent_archive = archive["opponent"]  # pass it by reference
        self.agent_name = kwargs.pop("agent_name")
        self.num_rounds = kwargs.pop("num_rounds")
        self.seed_value = kwargs.pop("seed_value")
        self.enable_evaluation_matrix = kwargs.pop("enable_evaluation_matrix")
        self.randomly_reseed_sampling = kwargs.pop("randomly_reseed_sampling")
        self.eval_matrix_method = kwargs.pop("eval_matrix_method")
        self.name_prefix = None
        self.startswith_keyword = "history"
        self.OS = OS
        self.population_idx = 0
        # self.OS = True

        self.win_rate = None
        self.best_mean_reward = None
        self.last_mean_reward = None
        self.checkpoint_num = 0
        self.max_checkpoint_num = -1
        self.last_save_timestep = 0



        new_kwargs = {}
        for k in kwargs.keys():
            if(k == "archive"):
                continue
            new_kwargs[k] = kwargs[k]

        super(EvalSaveCallback, self).__init__(*args, **new_kwargs)
        self.eval_env = kwargs.get("eval_env")
        # Used with shared.evaluate_policy()
        # if not isinstance(self.eval_env, DummyVecEnvSelfPlay):
        #     self.eval_env.__class__ = DummyVecEnvSelfPlay   # This works fine, the other solution is commented



        # self.opponent_archive = self.eval_env.archive
        # This is to prevent storing more memory while the evaluation matrix will not be computed for this evaluation callback (optimization for the memory)
        if(self.enable_evaluation_matrix):
            self.evaluation_matrix = np.zeros((self.num_rounds, self.num_rounds))
        # Just for debugging, TODO: remove it
        # for i in range(self.num_rounds):
        #     for j in range(self.num_rounds):
        #         self.evaluation_matrix[i,j] = -1

        # if isinstance(self.eval_env, DummyVecEnv):
        #     eval_env = DummyVecEnvMod([lambda: eval_env])

    def set_name_prefix(self, name_prefix):
        self.name_prefix = name_prefix


    def _evaluate(self, model, n_eval_episodes, deterministic, sampled_opponents, return_episode_rewards=True, make_deterministic_flag=True):
        # Sync training and eval env if there is VecNormalize
        sync_envs_normalization(self.training_env, self.eval_env)
        # This is made in order to prevent making different generatations evaluations affect the others
        # if(make_deterministic_flag):
        #     make_deterministic(seed_value=self.seed_value, cuda_check=False)

        # Deterministic is true through the evaluation
        deterministic = True 
        # Reset success rate buffer
        self._is_success_buffer = []
        return evaluate_policy_simple(
                                model,
                                self.eval_env,
                                n_eval_episodes=n_eval_episodes,
                                render=self.render,
                                deterministic=deterministic,
                                return_episode_rewards=return_episode_rewards,
                                warn=self.warn,
                                callback=self._log_success_callback,
                                sampled_opponents=sampled_opponents,
                                seed_value=self.seed_value,
                              )

    def _evaluate_policy_core(self, logger_prefix, n_eval_episodes, deterministic, sampled_opponents, override=False) -> bool:
        episode_rewards, episode_lengths, win_rates, std_win_rate, _ = self._evaluate(self.model, n_eval_episodes, deterministic, sampled_opponents)

        if self.log_path is not None:
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)
            self.evaluations_length.append(episode_lengths)

            kwargs = {}
            # Save success log if present
            if len(self._is_success_buffer) > 0:
                self.evaluations_successes.append(self._is_success_buffer)
                kwargs = dict(successes=self.evaluations_successes)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
                **kwargs,
            )
        win_rate = np.mean(win_rates)
        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        self.mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        self.last_mean_reward = mean_reward
        self.win_rate, std_win_rate = win_rate, std_win_rate # ratio [0,1]

        if self.verbose > 0:
            clilog.info(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            clilog.info(f"Episode length: {self.mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        # Add to current Logger
        self.logger.record(f"{logger_prefix}/mean_reward", float(mean_reward))
        self.logger.record(f"{logger_prefix}/mean_ep_length", self.mean_ep_length)
        self.logger.record(f"{logger_prefix}/record_timesteps", self.num_timesteps)
        if self.verbose > 0:
            clilog.debug(f"{win_rate}")
            clilog.info(f"Win rate: {100 * win_rate:.2f}% +/- {std_win_rate:.2f}")
        self.logger.record(f"{logger_prefix}/win_rate", win_rate)

        if len(self._is_success_buffer) > 0:
            success_rate = np.mean(self._is_success_buffer)
            if self.verbose > 0:
                clilog.debug(f"Success rate: {100 * success_rate:.2f}%")
            self.logger.record(f"{logger_prefix}/success_rate", success_rate)

        # Dump log so the evaluation results are printed with the correct timestep
        # self.logger.record(f"{logger_prefix}/time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)
        
        if mean_reward > self.best_mean_reward:
            if self.verbose > 0:
                clilog.debug("New best mean reward!")
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            self.best_mean_reward = mean_reward
            # Trigger callback if needed
            if self.callback is not None:
                return self._on_event()
        return True

    # In order to keep the oringinal callback functions, this is just a wrapper for the parameterized _evaluate_policy() -> _evaluate_policy_core()
    def _evaluate_policy(self, force_evaluation=False) -> bool:
        if (force_evaluation or (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0)):
            sampled_opponents = None
            clilog.debug("Sample models for evaluation")
            if(not self.OS):
                # print(self.opponent_archive.archive_dict.keys())
                archive = self.opponent_archive.get_sorted(self.eval_opponent_selection)
                models_names = archive[0]
                # print(len(models_names))
                sampled_opponents = utsmpl.sample_opponents(models_names, self.n_eval_episodes, selection=self.eval_opponent_selection, sorted=True, randomly_reseed=self.randomly_reseed_sampling)
            if(self.OS):
                sampled_opponents = utsmpl.sample_opponents_os(self.eval_sample_path, self.startswith_keyword, self.n_eval_episodes, selection=self.eval_opponent_selection, randomly_reseed=self.randomly_reseed_sampling)
            return self._evaluate_policy_core(logger_prefix="eval", 
                                            n_eval_episodes=self.n_eval_episodes,
                                            deterministic=self.deterministic,
                                            sampled_opponents=sampled_opponents)
        return True

    def _save_model_core(self):
        metric_value = None
        if(self.eval_metric == "length"):
            metric_value = self.mean_ep_length
        elif(self.eval_metric == "bestreward"):
            metric_value = self.best_mean_reward 
        elif(self.eval_metric == "lastreward"):
            metric_value = self.last_mean_reward
        elif(self.eval_metric == "winrate"):
            metric_value = self.win_rate
        # history_<num-round>_<reward/points/winrate>_m_<value>_s_<num-step>
        name = f"{self.name_prefix}_{self.eval_metric}_m_{metric_value}_s_{self.num_timesteps}_c_{self.checkpoint_num}_p_{self.population_idx}"#_c_{self.checkpoint_num}"
        self.checkpoint_num += 1
        path = os.path.join(self.save_path, name)
        self.model.save(path)
        self.last_save_timestep = self.num_timesteps
        if(not self.OS):
            self.archive.add(name, self.model) # Add the model to the archive
        if self.verbose > 0:
            clilog.debug(f"Saving model checkpoint to {path}")
        return name

    def _save_model(self, force_saving=False):
        if (force_saving or (self.save_freq > 0 and self.n_calls % self.save_freq == 0)):
            name = self._save_model_core()
            return name
        return None

    def _on_step(self) -> bool:
        # 1. evaluate the policy
        result = self._evaluate_policy()#super(EvalSaveCallback, self)._on_step()
        # 2. Save the results
        name = self._save_model()
        return result

    # This doesn't work -> make save_freq=NUM_TIMESTEPS and eval_freq=NUM_TIMESTEPS will work like this
    # 22.12.2021: Seems that it is working fine! ?
    # The models are not just stored with 25000 steps factors as it is taking more steps a little bit
    def _on_training_end(self) -> None:
        clilog.info("-------- Training End --------")
        self.max_checkpoint_num = max(self.max_checkpoint_num, self.checkpoint_num)
        # if(self.save_freq == 0 and self.eval_freq == 0):
        if(self.last_save_timestep != self.num_timesteps):
            self.eval_freq = self.n_calls   # There is a problem when I do not set it, thus, I have made this setting (The plots are not being reported in wandb)
            clilog.debug("Evaluating the model according to the metric and save it")
            result = self._evaluate_policy(force_evaluation=True)
            name = self._save_model(force_saving=True)
            # self.eval_freq = 0   # There is a problem when this line is not
        self.checkpoint_num = 0
        self.last_save_timestep = self.num_timesteps
        super(EvalSaveCallback, self)._on_training_end()

    def _get_score(self, model, n_eval_rep, deterministic, opponents, eval_matrix_method, make_deterministic_flag=True):
        episodes_rewards_ret, episode_lengths_if_eval_method, win_rates_ret, _, _ = self._evaluate( model,
                                                                                                    n_eval_episodes=n_eval_rep,
                                                                                                    deterministic=deterministic,
                                                                                                    sampled_opponents=opponents,
                                                                                                    return_episode_rewards=True if eval_matrix_method == "length" else False,
                                                                                                    make_deterministic_flag=make_deterministic_flag
                                                                                                    )
        # win_rate = np.mean(win_rates_ret)
        # win_rates.append(win_rate)
        score = None
        clilog.debug(eval_matrix_method)
        if(eval_matrix_method == "reward"):
            score = np.mean(episodes_rewards_ret)
        elif(eval_matrix_method == "win_rate"):
            score = np.mean(win_rates_ret)
        elif(eval_matrix_method == "length"):
            # episode_lengths_means = np.mean(episode_lengths_if_eval_method)
            score = np.mean(episode_lengths_if_eval_method)
        return score
    # TODO: Add a feature that it will use the correct sorted from the archive if the metric for the archive is steps!
    # TODO: It would be better to update the aggregate evaluation and then use it within here
    # [Deprecated]
    def compute_eval_matrix_aggregate(self, prefix, round_num, opponents_path=None, agents_path=None, n_eval_rep=5, deterministic=None, algorithm_class=None):
        models_names = None
        deterministic = self.deterministic if deterministic is None else deterministic  # https://stackoverflow.com/questions/66455636/what-does-deterministic-true-in-stable-baselines3-library-means
        if(self.OS and (opponents_path is None or agents_path is None)):
            raise ValueError("Wrong value for opponent/agent path")

        if(not self.OS):
            opponent_archive = self.opponent_archive.get_sorted("random")
            opponent_models_names = opponent_archive[0] # 0 index to get just the names
            archive = self.archive.get_sorted("random")
            models_names = archive[0]

        else:
            self.eval_env.set_attr("OS", True)

        # 1. evaluate for the current round of agent 1 against all previous (including the current round) of agent 2
        i = round_num
        startswith_keyword = f"{prefix}{i}_"
        agent_model = None
        sampled_agent = None
        # Get 1st agent
        if(not self.OS):
            # Get the agents that has models' names starts with "history_<i>_"
            sampled_agent_startswith = utlst.get_startswith(models_names, startswith=startswith_keyword)
            # Get the latest agent in this round/generation
            sampled_agent = utlst.get_latest(sampled_agent_startswith)[0]
            # Load the model
            agent_model = self.archive.load(name=sampled_agent, env=self.eval_env, algorithm_class=algorithm_class)
        # The same but for OS stored
        else:
            # sampled_agent_startswith = utos.get_startswith(self.save_path, startswith=startswith_keyword)
            sampled_agent = os.path.join(agents_path, utos.get_latest(self.save_path, startswith=startswith_keyword)[0])  # Join it with the agent path
            agent_model = algorithm_class.load(sampled_agent, env=self.eval_env)

        # TODO: Make it in one loop (Easy)
        for j in range(round_num+1):
            clilog.info("------------------------------")
            clilog.info(f"Round: {i} vs {j}")
            opponent_startswith_keyword = f"{prefix}{j}_"
            sampled_opponent = None
            # Get 2nd agent
            if(not self.OS):
                # if(len(models_names) == 0): Not possible as we are evaluating after training
                sampled_opponent_startswith = utlst.get_startswith(opponent_models_names, startswith=opponent_startswith_keyword)
                sampled_opponent = utlst.get_latest(sampled_opponent_startswith)[0]

            else:
                # sampled_opponent_startswith = utos.get_startswith(self.eval_sample_path, startswith=opponent_startswith_keyword)
                sampled_opponent = os.path.join(opponents_path, utos.get_latest(self.eval_sample_path, startswith=opponent_startswith_keyword)[0])
            
            clilog.info("---------------")
            clilog.info(f"Model {sampled_agent} vs {sampled_opponent}")
            # Run evaluation n_eval_rep for each opponent
            eval_model_list = [sampled_opponent]
            # The current model vs the iterated model from the opponent (last opponent in each generation/round)
            # TODO: it is possible to change this agent_model to self.model as they are the same in this loop
            _, _, win_rates, _, _ = self._evaluate(agent_model, n_eval_episodes=n_eval_rep,
                                            deterministic=deterministic,
                                            sampled_opponents=eval_model_list)
            # Save the result to a matrix (nxm) -> n -agent, m -opponents -> Index by round number
            # Add this matrix to __init__
            # It will be redundent to have 2 matrices but it is fine
            win_rate = np.mean(win_rates)
            self.evaluation_matrix[i,j] = win_rate
            clilog.info(f"win rate: {win_rate}")

        # 2. evaluate for the previous rounds of agent 1 against the current round of agent 2
        j = round_num
        opponent_startswith_keyword = f"{prefix}{j}_"
        sampled_opponent = None
        # Get 2nd agent
        if(not self.OS):
            # if(len(models_names) == 0): Not possible as we are evaluating after training
            sampled_opponent_startswith = utlst.get_startswith(opponent_models_names, startswith=opponent_startswith_keyword)
            sampled_opponent = utlst.get_latest(sampled_opponent_startswith)[0]

        else:
            # sampled_opponent_startswith = utos.get_startswith(self.eval_sample_path, startswith=opponent_startswith_keyword)
            sampled_opponent = os.path.join(opponents_path, utos.get_latest(self.eval_sample_path, startswith=opponent_startswith_keyword)[0])

        eval_model_list = [sampled_opponent]

        for i in range(round_num):
            clilog.info("------------------------------")
            clilog.info(f"Round: {i} vs {j}")
            startswith_keyword = f"{prefix}{i}_"
            agent_model = None
            
            # Get 1st agent
            if(not self.OS):
                # Get the agents that has models' names starts with "history_<i>_"
                sampled_agent_startswith = utlst.get_startswith(models_names, startswith=startswith_keyword)
                # Get the latest agent in this round/generation
                sampled_agent = utlst.get_latest(sampled_agent_startswith)[0]
                # Load the model
                agent_model = self.archive.load(name=sampled_agent, env=self.eval_env, algorithm_class=algorithm_class)
            # The same but for OS stored
            else:
                # sampled_agent_startswith = utos.get_startswith(self.save_path, startswith=startswith_keyword)
                sampled_agent = os.path.join(agents_path, utos.get_latest(self.save_path, startswith=startswith_keyword)[0])  # Join it with the agent path
                agent_model = algorithm_class.load(sampled_agent, env=self.eval_env)
            clilog.info("---------------")
            clilog.info(f"Model {sampled_agent} vs {sampled_opponent}")
            # Run evaluation n_eval_rep for each opponent
            # The current model vs the iterated model from the opponent (last opponent in each generation/round)
            _, _, win_rates, _, _ = self._evaluate(agent_model, n_eval_episodes=n_eval_rep,
                                            deterministic=deterministic,
                                            sampled_opponents=eval_model_list)
            # Save the result to a matrix (nxm) -> n -agent, m -opponents -> Index by round number
            win_rate = np.mean(win_rates)
            self.evaluation_matrix[i,j] = win_rate
            clilog.info(f"win rate: {win_rate}")
            

    # TODO: It would be better to update the aggregate evaluation and then use it within here as a sub function
    # Evaluate the whole matrix
    def compute_eval_matrix(self, prefix, num_rounds, opponents_path=None, agents_path=None, n_eval_rep=5, deterministic=None, algorithm_class=None, freq=1, population_size=1, negative_indicator=False):
        models_names = None
        deterministic = self.deterministic if deterministic is None else deterministic  # https://stackoverflow.com/questions/66455636/what-does-deterministic-true-in-stable-baselines3-library-means
        if(self.OS and (opponents_path is None or agents_path is None)):
            raise ValueError("Wrong value for opponent/agent path")

        # self.evaluation_matrix.append([])
        # Evaluate the model and save it (+ Regular evaluation)
        # self._save_model()
        # Evaluate the current model vs all the previous opponents
        # Get the list of all the previous opponents

        if(not self.OS):
            opponent_archive = self.opponent_archive.get_sorted("random")
            opponent_models_names = opponent_archive[0] # 0 index to get just the names
            archive = self.archive.get_sorted("random")
            models_names = archive[0]

        else:
            self.eval_env.set_attr("OS", True)

        # dim = num_rounds//freq+1
        agent_axis = [i for i in range(0, num_rounds, freq)]
        ret_agent_axis = []
        agent_names = []
        opponent_axis = [i for i in range(0, num_rounds, freq)]
        ret_opponent_axis = []
        population_axis = [i for i in range(0, population_size)]
        # Enforce doing evaluation for the last generation
        if(agent_axis[-1] != num_rounds-1):
            agent_axis.append(num_rounds-1)
        if(opponent_axis[-1] != num_rounds-1):
            opponent_axis.append(num_rounds-1)
        self.evaluation_matrix = np.zeros((len(agent_axis), len(opponent_axis)))

        # Enumerate in order to correctly place in the matrix
        for ei, i in enumerate(agent_axis): #, freq):
            startswith_keyword = f"{prefix}{i}_"
            agent_model = None
            sampled_agent = None

            # Get 1st agent
            if(not self.OS):
                # Get the agents that has models' names starts with "history_<i>_"
                sampled_agent_startswith = utlst.get_startswith(models_names, startswith=startswith_keyword)
                # Get the latest agent in this round/generation/population
                # sampled_agent = utlst.get_latest(sampled_agent_startswith)[0]
                # Get the latest agent in this round/generation for this specific population
                sampled_agent = utlst.get_latest(sampled_agent_startswith, population_idx=self.population_idx)[0] 
                # Load the model
                agent_model = self.archive.load(name=sampled_agent, env=self.eval_env, algorithm_class=algorithm_class)
            # The same but for OS stored
            else:
                # sampled_agent_startswith = utos.get_startswith(self.save_path, startswith=startswith_keyword)
                # sampled_agent = utos.get_latest(self.save_path, startswith=startswith_keyword)[0]
                sampled_agent = utos.get_latest(sampled_agent_startswith, population_idx=self.population_idx)[0]
                sampled_agent_path = os.path.join(agents_path, sampled_agent)  # Join it with the agent path
                agent_model = algorithm_class.load(sampled_agent_path, env=self.eval_env)

            ret_agent_axis.append(get_model_label(sampled_agent))
            agent_names.append(sampled_agent)
            for ej, j in enumerate(opponent_axis):
                clilog.info("------------------------------")
                clilog.info(f"Round: {i} vs {j}")
                opponent_startswith_keyword = f"{prefix}{j}_"
                
                # win_rates = []
                scores = []
                # For each opponent from different population, we evaluate the agent of the specific population against all the populations
                for ep, population_idx in enumerate(population_axis):
                    # To ensure that each evalaution is determinsitic and not affected by the previous one somehow
                    # make_deterministic(seed_value=self.seed_value, cuda_check=False)
                    sampled_opponent = None
                    # Get 2nd agent
                    if(not self.OS):
                        # if(len(models_names) == 0): Not possible as we are evaluating after training
                        sampled_opponent_startswith = utlst.get_startswith(opponent_models_names, startswith=opponent_startswith_keyword)
                        # Get the latest agent in this round/generation/population
                        # sampled_opponent = utlst.get_latest(sampled_opponent_startswith)[0]
                        # Get the latest opponent in this round/generation for this specific population
                        sampled_opponent = utlst.get_latest(sampled_opponent_startswith, population_idx=population_idx)[0] 
                    else:
                        # sampled_opponent_startswith = utos.get_startswith(self.eval_sample_path, startswith=opponent_startswith_keyword)
                        sampled_opponent = utos.get_latest(self.eval_sample_path, startswith=opponent_startswith_keyword, population_idx=self.population_idx)[0]
                        sampled_opponent = os.path.join(opponents_path, sampled_opponent)
                    if(ei == 0):
                        ret_opponent_axis.append(get_model_label(sampled_opponent))
                    clilog.info("---------------")
                    clilog.info(f"Model {sampled_agent} vs {sampled_opponent}")

                    # Run evaluation n_eval_rep for each opponent
                    eval_model_list = [sampled_opponent]
                    # The current model vs the iterated model from the opponent (last opponent in each generation/round)
                    score = self._get_score(agent_model, n_eval_rep, deterministic, eval_model_list, self.eval_matrix_method, True)
                    scores.append(score)
                # Save the result to a matrix (nxm) -> n -agent, m -opponents -> Index by round number
                # Add this matrix to __init__
                # It will be redundent to have 2 matrices but it is fine
                # mean_win_rate = np.mean(win_rates)
                # self.evaluation_matrix[ei, ej] = mean_win_rate
                # print(f"win rate: {mean_win_rate}")
                mean_score = np.mean(scores)
                # This already done
                if(self.eval_matrix_method == "length" and negative_indicator):
                    mean_score = self.eval_env.max_num_steps - mean_score #.get_attr("max_num_steps",0)[0] - mean_score
                self.evaluation_matrix[ei, ej] = mean_score
                clilog.info(f"Mean score ({self.eval_matrix_method}): {mean_score}")
        return [ret_agent_axis, ret_opponent_axis], agent_names

    # This last agent
    # Post evaluate the model against all the opponents from opponents_path
    # TODO: enable retrieving the agents from the archive
    def post_eval(self, opponents_path, startswith_keyword="history", n_eval_rep=3, deterministic=None, population_size=None):
        # TODO: fix it in order to print the labels correctly for the x-axis -> as it is now, just enumerate them
        #           But what it should be <round_num>:<idx>:<population_number> -> such that the idx is the order of it in the checkpoint of the round
        deterministic = self.deterministic if deterministic is None else deterministic  # https://stackoverflow.com/questions/66455636/what-does-deterministic-true-in-stable-baselines3-library-means
        population_axis = [i for i in range(0, population_size)]
        population_eval_return_list = [] # Store for each population of opponents vs single population of agents
        for ep, population_idx in enumerate(population_axis):
            opponents_models_names = utos.get_sorted(opponents_path, startswith_keyword, utsrt.sort_steps, population_idx=population_idx)
            # sampled_agent = utos.get_latest(sampled_agent_startswith, population_idx=self.population_idx)[0]

            opponents_models_path = [os.path.join(opponents_path, f) for f in opponents_models_names]
            eval_return_list = []
            self.eval_env.OS = True
            # self.eval_env.set_attr("OS", True)
            self.OS = True
            for i, o in enumerate(opponents_models_path):   # For all the opponents saved for this population
                # make_deterministic(seed_value=self.seed_value, cuda_check=False)
                eval_model_list = [o for _ in range(n_eval_rep)]
                # Doing this as only logger.record doesn't work, I think I need to call something else for Wandb callback
                # TODO: Fix the easy method (the commented) without using evaluate() function to make the code better
                score = self._get_score(self.model, n_eval_rep, deterministic, eval_model_list, self.eval_matrix_method, True)
                evaluation_result = score
                # wandb.log({f"{self.agent_name}/post_eval/opponent_idx": i})
                # wandb.log({f"{self.agent_name}/post_eval/win_rate": evaluation_result})

                
                # self.logger.record("post_eval/opponent_idx", i)
                # evaluation_result = self._evaluate_policy_core(logger_prefix="post_eval", 
                #                                                 n_eval_episodes=n_eval_opponent,
                #                                                 deterministic=deterministic,
                #                                                 sampled_opponents=eval_model_list,
                #                                                 override=True)
                eval_return_list.append(evaluation_result)
            population_eval_return_list.append(eval_return_list)
        self.eval_env.OS = False
        # self.eval_env.set_attr("OS", False)
        self.OS = False
        return np.array(eval_return_list)

    # [Not used]
    def agentVopponentOS(self, agent_path, opponent_path, n_eval_rep=5, deterministic=None, algorithm_class=None):
        deterministic = self.deterministic if deterministic is None else deterministic  # https://stackoverflow.com/questions/66455636/what-does-deterministic-true-in-stable-baselines3-library-means

        self.eval_env.set_attr("OS", True)
        self.OS = True

        agent_model = algorithm_class.load(agent_path, env=self.eval_env)
        eval_model_list = [opponent_path]
        res =  self._evaluate(agent_model, n_eval_episodes=n_eval_rep,
                              deterministic=deterministic,
                              sampled_opponents=eval_model_list)


        self.eval_env.set_attr("OS", False)
        self.OS = False

        return res
        
    # [Not used]
    def agentVopponentArchive(self, agent_name, opponent_name, n_eval_rep=5, deterministic=None, algorithm_class=None):
        deterministic = self.deterministic if deterministic is None else deterministic  # https://stackoverflow.com/questions/66455636/what-does-deterministic-true-in-stable-baselines3-library-means

        self.eval_env.set_attr("OS", False)
        self.OS = False

        agent_model = self.archive.load(name=agent_name, env=self.eval_env, algorithm_class=algorithm_class)
        eval_model_list = [opponent_name]
        res =  self._evaluate(agent_model, n_eval_episodes=n_eval_rep,
                              deterministic=deterministic,
                              sampled_opponents=eval_model_list)

        self.eval_env.set_attr("OS", True)
        self.OS = True

        return res