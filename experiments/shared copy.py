from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvIndices
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gym
from stable_baselines3.common import base_class
from copy import deepcopy
import warnings
from time import sleep
from bach_utils.shared import make_deterministic
from datetime import datetime
from bach_utils.sorting import population_key, round_key, checkpoint_key


def get_model_label(s):
    # history_10_winrate_m_0.53_s_565_c_1_p_0
    checkpoint_key_val = checkpoint_key(s)
    round_key_val = round_key(s)
    return f"{round_key_val:02d}.{checkpoint_key_val:01d}"

def normalize_reward(reward, mn=-1010, mx=1010):
    return (reward - mn)/(mx-mn)
    # return reward
    
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
        opponent_policy_name = self.sampled_opponents[opponent_index_idx]
        # print(f"Load evaluation's model: {opponent_policy_name} with index {self.opponents_indicies[env_idx]}")
        self.envs[env_idx].set_target_opponent_policy_name(opponent_policy_name)

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
    sampled_opponents = None,
    render_extra_info = None,
    render_callback = None,
    seed_value=None,
    ):
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
    render_ret = None
    # win_rate = np.zeros(n_envs, dtype="int")
    win_rates = []

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
    env.set_attr("target_opponent_policy_name", sampled_opponents, different_values=True, values_indices=opponents_indicies)
    seed_value = datetime.now().microsecond//1000 if seed_value is None else seed_value
    old_seed = seed_value#env[0].seed_val
    env.set_attr("seed_val", [i+seed_value for i in range(n_envs)], different_values=True, values_indices=[i for i in range(n_envs)])

    # print(f"Load evaluation models for {n_envs} vectorized env")
    observations = env.reset()
    states = None
    # print("Evaluation started --------------------")
    while (episode_counts < episode_count_targets).any():
        # print(f"Evaluate: {deterministic}")
        actions, states = model.predict(observations, state=states, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            # print(f"Seed: {env.get_attr('seed_val', i)}")
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]: 
                    if(int(info["win"]) > 0):
                        # win_rate[i] += 1
                        win_rates.append(1)
                    else:
                        win_rates.append(0)
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
            render_ret = env.render(extra_info=render_extra_info)
            if(render_callback is not None):
                render_ret = render_callback(render_ret)
                if(render_ret == -1):
                    break 

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    win_rate = np.mean(win_rates)#np.sum(win_rate)/n_eval_episodes
    std_win_rate = np.std(win_rates)

    env.set_attr("seed_val", [old_seed for i in range(n_envs)], different_values=True, values_indices=[i for i in range(n_envs)])


    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, win_rates, std_win_rate, render_ret
    return mean_reward, std_reward, win_rate, std_win_rate, render_ret


def evaluate_policy_simple(
    model: "base_class.BaseAlgorithm",
    env,#: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 1,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    sampled_opponents = None,
    render_extra_info = None,
    render_callback = None,
    sleep_time=0.0001,
    seed_value=None
    ):
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
    episodes_reward = []
    episodes_length = []
    render_ret = None
    vis_speed_status = "\t(Normal visualization speed)"
    # win_rate = np.zeros(n_envs, dtype="int")
    win_rates = []

    env.set_target_opponent_policy_name(sampled_opponents[0])

    # print(f"Load evaluation models for {n_envs} vectorized env")
    seed_value = datetime.now().microsecond//1000 if seed_value is None else seed_value
    old_seed = env.seed_val
    env.set_seed(seed_value)
    env.seed(seed_value)
    for i in range(n_eval_episodes):
        # TODO: add functionality for the seed 
        # if(seed == "random"):
            
        # if(seed is not None):
        #     env.seed(seed)
        # env.seed(seed_value)
        env.set_seed(seed_value)
        env.seed(seed_value)
        seed_value += 1
        # make_deterministic(seed_value, cuda_check=False)

        observations = env.reset()
        state = None
        # print("Evaluation started --------------------")
        done = False
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(observations, state=state, deterministic=deterministic)
            # action = env.action_space.sample()
            # print(action)
            observations, reward, done, info = env.step(action)
            # print(observations)

            episode_reward += reward
            episode_length += 1

            if ((isinstance(done, dict) and done["__all__"]) or (isinstance(done, bool) and done)):
                if(int(info["win"]) > 0):
                    # win_rate[i] += 1
                    win_rates.append(1)
                else:
                    win_rates.append(0)
                break
            if render:
                render_ret = env.render(extra_info=render_extra_info+vis_speed_status)
                sleep(sleep_time)
                if(render_callback is not None):
                    render_ret = render_callback(render_ret)
                    if(render_ret == 2):
                        sleep_time /=10
                        vis_speed_status = "\t(Faster visualiaztion speed)"
                    elif(render_ret == 3):
                        sleep_time *= 10
                        vis_speed_status = "\t(Slower visualiaztion speed)"
                    elif(render_ret == 8):
                        status = "\t(Visaulization is stopped)"
                        while True:
                            render_ret = env.render(extra_info=render_extra_info+status)
                            if(render_ret == 8):
                                break
                    elif(render_ret == 1):
                        vis_speed_status = "\t(Skipping)"
                        env.render(extra_info=render_extra_info+vis_speed_status)
                        sleep(0.25)
                        done = True
                    elif(render_ret == -1):
                        win_rates.append(0)
                        done = True 
        episodes_reward.append(episode_reward)
        episodes_length.append(episode_length)

    mean_reward = np.mean(episodes_reward)
    std_reward = np.std(episodes_reward)
    win_rate = np.mean(win_rates)#np.sum(win_rate)/n_eval_episodes
    std_win_rate = np.std(win_rates)
    env.set_seed(old_seed)
    env.seed(old_seed)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episodes_reward, episodes_length, win_rates, std_win_rate, render_ret
    return mean_reward, std_reward, win_rate, std_win_rate, render_ret




def get_best_agent_from_eval_mat(evaluation_matrix, agent_names, axis, maximize=False):
    print(evaluation_matrix.shape)
    score_vector = np.mean(evaluation_matrix.T, axis=axis)
    return get_best_agent_from_vector(score_vector, agent_names, maximize)

def get_best_agent_from_vector(score_vector, agent_names, maximize=False):
    best_score_idx = None
    # print(dict(zip(score_vector, agent_names)))
    if(bool(maximize)):
        best_score_idx = np.argmax(score_vector)
    else:
        best_score_idx = np.argmin(score_vector)
    best_score_agent_name = agent_names[best_score_idx]
    best_score = score_vector[best_score_idx]

    return best_score_agent_name, best_score

