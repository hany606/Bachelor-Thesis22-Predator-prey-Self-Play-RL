# Note:
# Based on: https://github.com/utiasDSL/gym-pybullet-drones/blob/master/experiments/learning/multiagent.py
# Resources:
# - https://medium.com/@vermashresth/craft-and-solve-multi-agent-problems-using-rllib-and-tensorforce-a3bd1bb6f556

"""Learning script for 1v1 behavior problem.

Example
-------
To run the script, type in a terminal:

    $ python train_1v1.py

Notes
-----
Use:
    $ tensorboard --logdir ./results/save-<env>-<algo>-<obs>-<act>-<time-date>/tb/
to see the tensorboard results at:
    http://localhost:6006/
"""

import os
import time
from datetime import datetime
import subprocess
import numpy as np
import gym

from gym.spaces import Box, Dict
import torch
import torch.nn as nn
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.multi_agent_env import ENV_STATE

from gym_predprey.envs.PredPrey import PredPrey
# from PredPrey import PredPrey
import time

#### Useful links ##########################################
# Workflow: github.com/ray-project/ray/blob/master/doc/source/rllib-training.rst
# ENV_STATE example: github.com/ray-project/ray/blob/master/rllib/examples/env/two_step_game.py
# Competing policies example: github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py

OBS = "xy"
ACT = "vel"
ENV = "1v1"
WORKERS = 6
ALGO = "ppo"
OWN_OBS_VEC_SIZE = 2
ACTION_VEC_SIZE = 2

if __name__ == "__main__":
    #### Save directory ########################################
    filename = os.path.dirname(
        os.path.abspath(__file__)
    ) + '/results/save-' + ENV + '-' + ALGO + '-' + OBS + '-' + ACT + '-' + datetime.now(
    ).strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename + '/')

    #### Print out current git commit hash #####################
    try:
        git_commit = subprocess.check_output(["git", "describe",
                                              "--tags"]).strip()
        with open(filename + '/git_commit.txt', 'w+') as f:
            f.write(str(git_commit))
    except:
        with open(filename + '/git_commit.txt', 'w+') as f:
            f.write("NO TAG")
    ########################################################################################################

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    ########################################################################################################

    #### Register the custom centralized critic model ##########
    # ModelCatalog.register_custom_model("cc_model", CustomTorchCentralizedCriticModel)

    #### Register the environment #################################
    env_name = "predprey"
    register_env(env_name, lambda _: PredPrey())
    #### Unused env to extract the act and obs spaces ##########
    temp_env = PredPrey()
    # observer_space = Dict({
    #     "own_obs": temp_env.observation_space[0],
    #     "opponent_obs": temp_env.observation_space[1],
    #     "opponent_action": temp_env.action_space[1],
    # })
    # action_space = temp_env.action_space[0]


    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    num_agents = temp_env.nrobots
    def gen_policy():
        return (None, obs_space, act_space, {})
    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()
    def policy_mapping_fn(agent_id):
            return 'agent-' + str(agent_id)

    # print("[INFO] Action space:", temp_env.action_space)
    # print("[INFO] Observation space:", temp_env.observation_space)

    ########################################################################################################
    config={
        "log_level": "WARN",
        "num_workers": WORKERS,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        # "num_cpus_for_driver": 1,
        # "num_cpus_per_worker": 1,
        "lr": 5e-3,
        "model":{"fcnet_hiddens": [8, 8]},
        "multiagent": {
            "policies": policy_graphs,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "env": env_name,
        "framework": "torch",

    }


    #### Set up the trainer's config ###########################
    # config = ppo.DEFAULT_CONFIG.copy(
    # )  # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    # config = {
    #     "env": env_name,
    #     "num_workers": 0 + WORKERS,
    #     "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
    #     "batch_mode": "complete_episodes",
    #     # "callbacks": FillInActions,
    #     "framework": "torch",
    # }
    # #### Set up the model parameters of the trainer's config ###
    # # config["model"] = {
    # #     "custom_model": "cc_model",
    # # }
    # #### Set up the multiagent params of the trainer's config ##
    # config["multiagent"] = {
    #     "policies": {
    #         "pol0": (None, observer_space, action_space, {
    #             "agent_id": 0,
    #         }),
    #         "pol1": (None, observer_space, action_space, {
    #             "agent_id": 1,
    #         }),
    #     },
    #     "policies_to_train": ["pol0", "pol1"],
    #     "policy_mapping_fn": lambda x: "pol0" if x == 0 else "pol1",  # # Function mapping agent ids to policy ids
    #     # "observation_fn": central_critic_observer,  # See rllib/evaluation/observation_function.py for more info
    # }

    #### Ray Tune stopping conditions ##########################
    stop = {
        # "timesteps_total": 120000, # 100000 ~= 10'
        # "episode_reward_mean": -250,
        "training_iteration": 100,
    }
    ########################################################################################################

    #### Train #################################################
    results = tune.run(
        "PPO",
        stop=stop,
        config=config,
        verbose=True,
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir=filename,
    )
    ########################################################################################################

    #### Save agent ############################################
    checkpoints = results.get_trial_checkpoints_paths(
        trial=results.get_best_trial('episode_reward_mean', mode='max'),
        metric='episode_reward_mean')
    with open(filename + '/checkpoint.txt', 'w+') as f:
        f.write(checkpoints[0][0])

    #### Shut down Ray #########################################
    ray.shutdown()
    ########################################################################################################