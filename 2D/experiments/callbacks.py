# Generation and round are being used as the same meaning
from random import sample
from stable_baselines3.common.callbacks import EvalCallback, EventCallback
import os
import bach_utils.sorting as utsrt
import bach_utils.os as utos
import bach_utils.list as utlst
import bach_utils.sampling as utsmpl

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, sync_envs_normalization, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvIndices
from copy import deepcopy
import wandb


OS = False   # This flag just for testing now in order not to break the compatibility and the working of the code

# Based on: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/dummy_vec_env.py
# Only used for evaluation not training envs
class DummyVecEnvSelfPlay(DummyVecEnv):
    def __init__(self, *args, **kwargs):
        super(DummyVecEnvSelfPlay, self).__init__(*args, **kwargs)

    def set_sampled_opponents(self, sampled_opponents):
        self.sampled_opponents = sampled_opponents
    
    def set_opponents_indicies(self, opponents_indicies):
        self.opponents_indicies = opponents_indicies

    def change_opponent(self, env_idx):
        self.opponents_indicies[env_idx] += 1
        opponent_index_idx = min(self.opponents_indicies[env_idx], len(self.sampled_opponents)-1) # if it is reached the maximum, then this is the last reset for this environment without doing any other steps later. This min function is only used to protect against crashing
        opponent_policy_filename = self.sampled_opponents[opponent_index_idx]
        # print(f"Load evaluation's model: {opponent_policy_filename} with index {self.opponents_indicies[env_idx]}")
        self.envs[env_idx].set_target_opponent_policy_filename(opponent_policy_filename)

    # This is modified in order to change the opponent before the env's automatic reseting
    def step_wait(self) -> VecEnvStepReturn:
        # print("Modified")
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                self.change_opponent(env_idx)
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    # Modified to set different value (opponent) for different environment in the vectorized environment
    def set_attr(self, attr_name: str, values: Any, indices: VecEnvIndices = None, different_values=False, values_indices=None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for i, env_i in enumerate(target_envs):
            value = None
            if(different_values):
                value = values[values_indices[i]]
            else:
                value = values
            setattr(env_i, attr_name, value)


# Select opponent for training
class TrainingOpponentSelectionCallback(EventCallback):
    def __init__(self, *args, **kwargs):
        self.sample_path = kwargs["sample_path"]
        self.startswith_keyword = "history"
        self.env = kwargs["env"]
        self.opponent_selection = kwargs["opponent_selection"]
        self.sample_after_rollout = kwargs["sample_after_rollout"]
        self.num_sampled_per_round = kwargs["num_sampled_per_round"]
        self.archive = kwargs["archive"]

        del kwargs["sample_path"]
        del kwargs["env"]
        del kwargs["opponent_selection"]
        del kwargs["num_sampled_per_round"]
        del kwargs["sample_after_rollout"]
        # del kwargs["archive"]
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
        # if(not self.sampled_per_round):
        print("training started")
        opponent = None
        if(not OS):
            # print("Not OS")
            archive = self.archive.get_sorted(self.opponent_selection)
            models_names = archive[0]
            self.sampled_per_round = utsmpl.sample_opponents(models_names, self.num_sampled_per_round, selection=self.opponent_selection, sorted=True)
        if(OS):
            self.sampled_per_round = utsmpl.sample_opponents_os(self.sample_path, self.startswith_keyword, self.num_sampled_per_round, selection=self.opponent_selection)
        # If it is not updated with every rollout, only updated at the begining
        if(self.num_sampled_per_round == 1):
            self.env.set_target_opponent_policy_filename(self.sampled_per_round[0])
        super(TrainingOpponentSelectionCallback, self)._on_training_start()

    # def _on_training_end(self):
    #     self.sampled_per_round = []
    #     super(TrainingOpponentSelectionCallback, self)._on_training_start()

    # With every rollout
    def _on_rollout_start(self):
        print("Rollout")
        if(self.sample_after_rollout):
            opponent = None
            if(not OS):
                archive = self.archive.get_sorted(self.opponent_selection)
                models_names = archive[0]
                opponent = utsmpl.sample_opponents(models_names, self.num_sampled_per_round, selection=self.opponent_selection, sorted=True)[0]
            if(OS):
                opponent = utsmpl.sample_opponents_os(self.sample_path, self.startswith_keyword, self.num_sampled_per_round, selection=self.opponent_selection)[0]
            self.env.set_target_opponent_policy_filename(opponent)
        
        if(self.num_sampled_per_round > 1):
            print("Change sampled agent")
            self.env.set_target_opponent_policy_filename(self.sampled_per_round[self.sampled_idx % self.num_sampled_per_round]) # as a circular buffer
            self.sampled_idx += 1
        super(TrainingOpponentSelectionCallback, self)._on_rollout_start()


# This is the newer version based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/evaluation.py
# With support to vectorized environment
def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    sampled_opponents = None
    ) -> Union[Tuple[float, float, float], Tuple[List[float], List[int], List[float]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    # Not working, TODO: make it work instead the line in __init__ EvalSaveCallback
    # if not isinstance(env, DummyVecEnvSelfPlay):
    #     env = DummyVecEnvSelfPlay([lambda: env])

    # if not isinstance(env, VecEnv):
    #     env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    win_rate = np.zeros(n_envs, dtype="int")

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
    opponents_indicies = np.zeros(n_envs, dtype="int")
    for i in range(1, n_envs):
        opponents_indicies[i] = opponents_indicies[i-1] + episode_count_targets[i-1]

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    env.set_sampled_opponents(sampled_opponents)
    env.set_opponents_indicies(opponents_indicies) # To be used later inside reset()
    env.set_attr("target_opponent_policy_filename", sampled_opponents, different_values=True, values_indices=opponents_indicies)
    # print(f"Load evaluation models for {n_envs} vectorized env")
    observations = env.reset()
    states = None
    # print("Evaluation started --------------------")
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(observations, state=states, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]: 
                    if(info["win"] > 0):
                        win_rate[i] += 1
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    if states is not None:
                        states[i] *= 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    win_rate = np.sum(win_rate)/n_eval_episodes
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, win_rate
    return mean_reward, std_reward, win_rate

# Based on: https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_selfplay.py
# Based on: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py
class EvalSaveCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        self.save_path = kwargs["save_path"]
        self.eval_metric = kwargs["eval_metric"]
        self.eval_opponent_selection = kwargs["eval_opponent_selection"]
        self.eval_sample_path = kwargs["eval_sample_path"]
        self.save_freq = kwargs["save_freq"]
        self.archive = kwargs["archive"]["self"]  # pass it by reference
        self.opponent_archive = kwargs["archive"]["opponent"]  # pass it by reference
        self.agent_name = kwargs["agent_name"]
        self.num_rounds = kwargs["num_rounds"]
        self.name_prefix = None
        self.startswith_keyword = "history"
        self.OS = OS
        self.win_rate = None
        self.best_mean_reward = None
        self.last_mean_reward = None

        del kwargs["save_path"]
        del kwargs["eval_metric"]
        del kwargs["eval_opponent_selection"]
        del kwargs["eval_sample_path"]
        del kwargs["save_freq"]
        del kwargs["agent_name"]
        del kwargs["num_rounds"]
        # del kwargs["archive"]
        new_kwargs = {}
        for k in kwargs.keys():
            if(k == "archive"):
                continue
            new_kwargs[k] = kwargs[k]

        super(EvalSaveCallback, self).__init__(*args, **new_kwargs)
        if not isinstance(self.eval_env, DummyVecEnvSelfPlay):
            self.eval_env.__class__ = DummyVecEnvSelfPlay   # This works fine, the other solution is commented



        # self.opponent_archive = self.eval_env.archive

        self.evaluation_matrix = np.zeros((self.num_rounds, self.num_rounds))
        # Just for debugging, TODO: remove it
        # for i in range(self.num_rounds):
        #     for j in range(self.num_rounds):
        #         self.evaluation_matrix[i,j] = -1

        # if isinstance(self.eval_env, DummyVecEnv):
        #     eval_env = DummyVecEnvMod([lambda: eval_env])

    def set_name_prefix(self, name_prefix):
        self.name_prefix = name_prefix


    def _evaluate(self, model, n_eval_episodes, deterministic, sampled_opponents):
        # Sync training and eval env if there is VecNormalize
        sync_envs_normalization(self.training_env, self.eval_env)

        # Reset success rate buffer
        self._is_success_buffer = []
        return evaluate_policy(
                                model,
                                self.eval_env,
                                n_eval_episodes=n_eval_episodes,
                                render=self.render,
                                deterministic=deterministic,
                                return_episode_rewards=True,
                                warn=self.warn,
                                callback=self._log_success_callback,
                                sampled_opponents=sampled_opponents
                              )

    def _evaluate_policy_core(self, logger_prefix, n_eval_episodes, deterministic, sampled_opponents, override=False) -> bool:
        episode_rewards, episode_lengths, win_rate = self._evaluate(self.model, n_eval_episodes, deterministic, sampled_opponents)

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

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        self.last_mean_reward = mean_reward
        self.win_rate = win_rate # ratio [0,1]

        if self.verbose > 0:
            print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        # Add to current Logger
        self.logger.record(f"{logger_prefix}/mean_reward", float(mean_reward))
        self.logger.record(f"{logger_prefix}/mean_ep_length", mean_ep_length)
        self.logger.record(f"{logger_prefix}/record_timesteps", self.num_timesteps)
        if self.verbose > 0:
            print(f"Win rate: {100 * win_rate:.2f}%")
        self.logger.record(f"{logger_prefix}/win_rate", win_rate)

        if len(self._is_success_buffer) > 0:
            success_rate = np.mean(self._is_success_buffer)
            if self.verbose > 0:
                print(f"Success rate: {100 * success_rate:.2f}%")
            self.logger.record(f"{logger_prefix}/success_rate", success_rate)

        if mean_reward > self.best_mean_reward:
            if self.verbose > 0:
                print("New best mean reward!")
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
            if(not self.OS):
                # print("Sample models for evaluation")
                # print(self.opponent_archive.archive_dict.keys())
                archive = self.opponent_archive.get_sorted(self.eval_opponent_selection)
                models_names = archive[0]
                # print(len(models_names))
                sampled_opponents = utsmpl.sample_opponents(models_names, self.n_eval_episodes, selection=self.eval_opponent_selection, sorted=True)
            if(self.OS):
                sampled_opponents = utsmpl.sample_opponents_os(self.eval_sample_path, self.startswith_keyword, self.n_eval_episodes, selection=self.eval_opponent_selection)
            return self._evaluate_policy_core(logger_prefix="eval", 
                                            n_eval_episodes=self.n_eval_episodes,
                                            deterministic=self.deterministic,
                                            sampled_opponents=sampled_opponents)
        return True

    def _save_model_core(self):
        metric_value = None
        if(self.eval_metric == "steps"):
            metric_value = self.num_timesteps
        elif(self.eval_metric == "bestreward"):
            metric_value = self.best_mean_reward 
        elif(self.eval_metric == "lastreward"):
            metric_value = self.last_mean_reward
        elif(self.eval_metric == "winrate"):
            metric_value = self.win_rate
        # history_<num-round>_<reward/points/winrate>_m_<value>_s_<num-step>
        name = f"{self.name_prefix}_{self.eval_metric}_m_{metric_value}_s_{self.num_timesteps}"
        path = os.path.join(self.save_path, name)
        self.model.save(path)
        if(not self.OS):
            self.archive.add(name, self.model) # Add the model to the archive
        if self.verbose > 1:
            print(f"Saving model checkpoint to {path}")
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
    def _on_training_end(self) -> None:
        if(self.save_freq == 0 and self.eval_freq == 0):
            self.eval_freq = self.n_calls   # There is a problem when I do not set it, thus, I have made this setting (The plots are not being reported in wandb)
            print("Evaluating the model according to the metric and save it")
            result = self._evaluate_policy(force_evaluation=True)
            name = self._save_model(force_saving=True)
            self.eval_freq = 0   # There is a problem when this line is not

        super(EvalSaveCallback, self)._on_training_end()

    # TODO: Add a feature that it will use the correct sorted from the archive if the metric for the archive is steps!
    def compute_eval_matrix_aggregate(self, prefix, round_num, opponents_path=None, agents_path=None, n_eval_rep=5, deterministic=False, algorithm_class=None):
        models_names = None
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


        for j in range(round_num+1):
            print(f"Round: {i} vs {j}")
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

            # Run evaluation n_eval_rep for each opponent
            eval_model_list = [sampled_opponent]
            # The current model vs the iterated model from the opponent (last opponent in each generation/round)
            _, _, win_rate = self._evaluate(agent_model, n_eval_episodes=n_eval_rep,
                                            deterministic=deterministic,
                                            sampled_opponents=eval_model_list)
            # Save the result to a matrix (nxm) -> n -agent, m -opponents -> Index by round number
            # Add this matrix to __init__
            # It will be redundent to have 2 matrices but it is fine
            self.evaluation_matrix[i,j] = win_rate

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
            print(f"Round: {i} vs {j}")

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

            # Run evaluation n_eval_rep for each opponent
            # The current model vs the iterated model from the opponent (last opponent in each generation/round)
            _, _, win_rate = self._evaluate(agent_model, n_eval_episodes=n_eval_rep,
                                            deterministic=deterministic,
                                            sampled_opponents=eval_model_list)
            # Save the result to a matrix (nxm) -> n -agent, m -opponents -> Index by round number
            self.evaluation_matrix[i,j] = win_rate
            

    # Evaluate the whole matrix
    def compute_eval_matrix(self, prefix, num_rounds, opponents_path=None, agents_path=None, n_eval_rep=5, deterministic=False, algorithm_class=None):
        models_names = None
        # self.evaluation_matrix.append([])
        # Evaluate the model and save it (+ Regular evaluation)
        # self._save_model()
        # Evaluate the current model vs all the previous opponents
        # Get the list of all the previous opponents
        if(self.OS and (opponents_path is None or agents_path is None)):
            raise ValueError("Wrong value for opponent/agent path")

        if(not self.OS):
            opponent_archive = self.opponent_archive.get_sorted("random")
            opponent_models_names = opponent_archive[0] # 0 index to get just the names
            archive = self.archive.get_sorted("random")
            models_names = archive[0]

        else:
            self.eval_env.set_attr("OS", True)


        self.evaluation_matrix = np.zeros((num_rounds, num_rounds))
        for i in range(num_rounds):
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


            for j in range(num_rounds):
                print(f"Round: {i} vs {j}")
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

                # Run evaluation n_eval_rep for each opponent
                eval_model_list = [sampled_opponent]
                # The current model vs the iterated model from the opponent (last opponent in each generation/round)
                _, _, win_rate = self._evaluate(agent_model, n_eval_episodes=n_eval_rep,
                                                deterministic=deterministic,
                                                sampled_opponents=eval_model_list)
                # Save the result to a matrix (nxm) -> n -agent, m -opponents -> Index by round number
                # Add this matrix to __init__
                # It will be redundent to have 2 matrices but it is fine
                self.evaluation_matrix[i,j] = win_rate
        
        # For each generation/round previous to the current generation/round and including this round
        # for i in range(round_num+1):    # round_num+1 In order to include itself round
        #     startswith_keyword = f"{prefix}{round_num}"
        #     sampled_opponents = None            
        #     if(not self.OS):
        #         # if(len(models_names) == 0): Not possible as we are evaluating after training
        #         sampled_opponents_startswith = utlst.get_startswith(models_names, startswith=startswith_keyword)
        #         sampled_opponents = utlst.get_latest(sampled_opponents_startswith)[0]
        #     else:
        #         sampled_opponents_startswith = utos.get_startswith(self.eval_sample_path, startswith=startswith_keyword)
        #         sampled_opponents = utos.get_latest(sampled_opponents_startswith)[0]

        #     # Run evaluation n_eval_rep for each opponent
        #     eval_model_list = [sampled_opponents]
        #     # The current model vs the iterated model form the opponent (last opponent in each generation/round)
        #     _, _, win_rate = self._evaluate(self.model, n_eval_episodes=n_eval_rep,
        #                                     deterministic=deterministic,
        #                                     sampled_opponents=eval_model_list)
        #     # Save the result to a matrix (nxm) -> n -agent, m -opponents -> Index by round number
        #     # Add this matrix to __init__
        #     # It will be redundent to have 2 matrices but it is fine
        #     self.evaluation_matrix[-1].append(win_rate)

            # --------------------- if each version in the generation/round of 1st agent vs each version in the generation of the 2nd agent ---------------------
            # if(not self.OS):
            #     # if(len(models_names) == 0): Not possible as we are evaluating after training
            #     sampled_opponents = utlst.get_startswith(models_names, startswith=startswith_keyword)
            # if(self.OS):
            #     sampled_opponents = utos.get_startswith(self.eval_sample_path, startswith=startswith_keyword)
            # For each model in the generation/round for opponent
            # for opponent in sampled_opponents:
            #     # For each model in the generation/round for self (the agent that I am evaluating)
            #     for model in sampled_self:
            #         eval_model_list = [opponent for j in range]
            #         # To make it work, change this function to include model
            #         _, _, win_rate = self._evaluate(model, n_eval_episodes=n_eval_rep,
            #                                         deterministic=deterministic,
            #                                         sampled_opponents=eval_model_list)
            # ---------------------------------------------------------------------------------------------


        # return self.eval
        # Save result of this matrix to wandb as heatmap with the current sizes
        # TODO
        # Save the round number with the name of the agent in a list
        # Save this list to a txt file -> This shows the generation/round number and the agent name 

    # This last agent
    # Post evaluate the model against all the opponents from opponents_path
    # TODO: enable retrieving the agents from the archive
    def post_eval(self, opponents_path, startswith_keyword="history", n_eval_rep=3, deterministic=False):
        opponents_models_names = utos.get_sorted(opponents_path, startswith_keyword, utsrt.sort_steps)
        opponents_models_path = [os.path.join(opponents_path, f) for f in opponents_models_names]
        eval_return_list = []
        # self.eval_env.OS = True
        self.eval_env.set_attr("OS", True)
        self.OS = True
        for i, o in enumerate(opponents_models_path):
            eval_model_list = [o for _ in range(n_eval_rep)]
            # Doing this as only logger.record doesn't work, I think I need to call something else for Wandb callback
            # TODO: Fix the easy method (the commented) without using evaluate() function to make the code better
            _, _, win_rate = self._evaluate(self.model, n_eval_episodes=n_eval_rep,
                                            deterministic=deterministic,
                                            sampled_opponents=eval_model_list)
            evaluation_result = win_rate
            wandb.log({f"{self.agent_name}/post_eval/opponent_idx": i})
            wandb.log({f"{self.agent_name}/post_eval/win_rate": evaluation_result})

            
            # self.logger.record("post_eval/opponent_idx", i)
            # evaluation_result = self._evaluate_policy_core(logger_prefix="post_eval", 
            #                                                 n_eval_episodes=n_eval_opponent,
            #                                                 deterministic=deterministic,
            #                                                 sampled_opponents=eval_model_list,
            #                                                 override=True)
            
            eval_return_list.append(evaluation_result)
        data = [[x, y] for (x, y) in zip([i for i in range(len(opponents_models_path))], eval_return_list)]
        table = wandb.Table(data=data, columns = ["opponent idx", "win-rate"])
        wandb.log({f"{self.agent_name}/post_eval/table": wandb.plot.line(table, "opponent idx", "win-rate", title=f"Post evaluation {self.agent_name}")})
        # self.eval_env.OS = False
        self.eval_env.set_attr("OS", False)
        self.OS = False
        return eval_return_list

    def agentVopponentOS(self, agent_path, opponent_path, n_eval_rep=5, deterministic=False, algorithm_class=None):
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

    def agentVopponentArchive(self, agent_name, opponent_name, n_eval_rep=5, deterministic=False, algorithm_class=None):
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