from stable_baselines3.common.callbacks import EvalCallback
import os
from bach_utils.os import *

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, sync_envs_normalization, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvIndices
from copy import deepcopy
import wandb


# Based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/evaluation.py
# It is changed to load different opponents or the same opponents for the same agent
def evaluate_policy_old(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    sampled_opponents=None
    ) -> Union[Tuple[float, float, float], Tuple[List[float], List[int], float]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
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
    from stable_baselines3.common.env_util import is_wrapped
    from stable_baselines3.common.monitor import Monitor

    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"
        is_monitor_wrapped = env.env_is_wrapped(Monitor)[0]
    else:
        is_monitor_wrapped = is_wrapped(env, Monitor)

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    episode_rewards, episode_lengths = [], []
    not_reseted = True
    win_rate = 0
    while len(episode_rewards) < n_eval_episodes:
        # Number of loops here might differ from true episodes
        # played, if underlying wrappers modify episode lengths.
        # Avoid double reset, as VecEnv are reset automatically.
        if not isinstance(env, VecEnv) or not_reseted:
            opponent_policy_filename = sampled_opponents[len(episode_rewards)]
            if(opponent_policy_filename is not None):
                print(f"Load evaluation's model: {opponent_policy_filename}")
                env.set_attr("target_opponent_policy_filename", opponent_policy_filename)
            obs = env.reset()
            not_reseted = False
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)
            # unpack values so that the callback can access the local variables
            reward = rewards[0]
            done = dones[0]
            info = infos[0]
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        if(info["win"] > 0):
            win_rate += 1
        if is_monitor_wrapped:
            # Do not trust "done" with episode endings.
            # Remove vecenv stacking (if any)
            if isinstance(env, VecEnv):
                info = info[0]
            if "episode" in info.keys():
                # Monitor wrapper includes "episode" key in info if environment
                # has been wrapped with it. Use those rewards instead.
                episode_rewards.append(info["episode"]["r"])
                episode_lengths.append(info["episode"]["l"])
        else:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    # print("-----------------")
    # print(win_rate, n_eval_episodes)
    win_rate = win_rate/n_eval_episodes
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, win_rate
    return mean_reward, std_reward, win_rate


# Based on: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/dummy_vec_env.py
# This is modified in order to change the opponent before the env
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
        self.envs[env_idx].target_opponent_policy_filename = opponent_policy_filename


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

    # Set different value (opponent) for different environment in the vectorized environment
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
        self.name_prefix = None
        self.startswith_keyword = "history"
        # eval_env = deepcopy(kwargs["eval_env"])

        del kwargs["save_path"]
        del kwargs["eval_metric"]
        del kwargs["eval_opponent_selection"]
        del kwargs["eval_sample_path"]
        del kwargs["save_freq"]
        super(EvalSaveCallback, self).__init__(*args, **kwargs)
        
        if not isinstance(self.eval_env, DummyVecEnvSelfPlay):
            self.eval_env.__class__ = DummyVecEnvSelfPlay   # This works fine, the other solution is commented

        # if isinstance(self.eval_env, DummyVecEnv):
        #     eval_env = DummyVecEnvMod([lambda: eval_env])


    def _sample_opponents(self, num_sampled_opponents):
        sampled_opponents_filenames = []
        files_list = get_startswith(self.eval_sample_path, self.startswith_keyword)
        if(len(files_list) == 0):
            sampled_opponents_filenames = [None for _ in range(num_sampled_opponents)]
        else:
            if(self.eval_opponent_selection == "random"):
                sampled_opponents_filenames = [get_random_from(files_list)[0] for _ in range(num_sampled_opponents)]  # TODO: Take care about pigonhole principle -> if the set is too small, then the likelihood for selection each element in the list will be relatively large
            
            elif(self.eval_opponent_selection == "latest"):
                sort_steps(files_list) # TODO: Take care about this computation bottelneck here O(NlogN), it will be a headach for large archives
                latest = files_list[-1]
                sampled_opponents_filenames = [latest for _ in range(num_sampled_opponents)]
            
            elif(self.eval_opponent_selection == "highest"):
                sort_metric(files_list)
                target = files_list[-1]
                sampled_opponents_filenames = [target for _ in range(num_sampled_opponents)]

            elif(self.eval_opponent_selection == "Lowest"):
                sort_metric(files_list)
                target = files_list[0]
                sampled_opponents_filenames = [target for _ in range(num_sampled_opponents)]

            sampled_opponents_filenames = [os.path.join(self.eval_sample_path, f) for f in sampled_opponents_filenames]
        return sampled_opponents_filenames


    def set_name_prefix(self, name_prefix):
        self.name_prefix = name_prefix

    def _evaluate_policy_param(self, logger_prefix, n_eval_episodes, deterministic, sampled_opponents, override=False) -> bool:
        if override or (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0):
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []
            episode_rewards, episode_lengths, win_rate = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=n_eval_episodes,
                render=self.render,
                deterministic=deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
                sampled_opponents=sampled_opponents
            )

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

    # In order to keep the oringinal callback functions, this is just a wrapper for the parameterized _evaluate_policy()
    def _evaluate_policy(self) -> bool:
        return self._evaluate_policy_param(logger_prefix="eval", 
                                           n_eval_episodes=self.n_eval_episodes,
                                           deterministic=self.deterministic,
                                           sampled_opponents=self._sample_opponents(self.n_eval_episodes))

    def _save_model(self):
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
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
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.eval_metric}_m_{metric_value}_s_{self.num_timesteps}")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")

    def _on_step(self) -> bool:
        # 1. evaluate the policy
        result = self._evaluate_policy()#super(EvalSaveCallback, self)._on_step()
        # 2. Save the results
        self._save_model()
        return result


    # Post evaluate the model against all the opponents from opponents_path
    def post_eval(self, agent_name, opponents_path, startswith_keyword="history", n_eval_opponent=3, deterministic=False):
        def evaluate(n_eval_episodes, deterministic, sampled_opponents):
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []
            _, _, win_rate = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=n_eval_episodes,
                render=self.render,
                deterministic=deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
                sampled_opponents=sampled_opponents
            )

            win_rate = win_rate # ratio [0,1]
            return win_rate

        opponents_models_names = get_sorted(opponents_path, startswith_keyword, sort_steps)
        opponents_models_path = [os.path.join(opponents_path, f) for f in opponents_models_names]
        eval_return_list = []
        for i, o in enumerate(opponents_models_path):
            eval_model_list = [o for _ in range(n_eval_opponent)]
            # Doing this as only logger.record doesn't work, I think I need to call something else for Wandb callback
            # TODO: Fix the easy method (the commented) without using evaluate() function to make the code better
            evaluation_result = evaluate(n_eval_episodes=n_eval_opponent,
                                         deterministic=deterministic,
                                         sampled_opponents=eval_model_list)
            wandb.log({f"{agent_name}/post_eval/opponent_idx": i})
            wandb.log({f"{agent_name}/post_eval/win_rate": evaluation_result}, step=i)

            
            # self.logger.record("post_eval/opponent_idx", i)
            # evaluation_result = self._evaluate_policy_param(logger_prefix="post_eval", 
            #                                                 n_eval_episodes=n_eval_opponent,
            #                                                 deterministic=deterministic,
            #                                                 sampled_opponents=eval_model_list,
            #                                                 override=True)
            
            eval_return_list.append(evaluation_result)
        # data = [[x, y] for (x, y) in zip([i for i in range(len(opponents_models_path))], eval_return_list)]
        # table = wandb.Table(data=data, columns = ["x", "y"])
        # wandb.log({f"{agent_name}/post_eval/": wandb.plot.line(table, "win-rate", "opponent idx", title=f"Post evaluation {agent_name}")})

        return eval_return_list