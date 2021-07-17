# Based on: https://github.com/utiasDSL/gym-pybullet-drones/blob/master/experiments/learning/test_multiagent.py

"""Test script for multiagent problems.

This scripts runs the best model found by one of the executions of `multiagent.py`

Example
-------
To run the script, type in a terminal:

    $ python test_multiagent.py --exp ./results/save-<env>-<num_drones>-<algo>-<obs>-<act>-<time_date>

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import numpy as np
import pybullet as p
import pickle
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Dict
import torch
import torch.nn as nn
import ray
from ray import tune
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.logger import pretty_print


from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Pred, PredPrey1v1Super
from gym_predprey.envs.PredPrey1v1 import Behavior

def create_environment(_):
    import gym_predprey
    from gym_predprey.envs.PredPrey1v1 import Behavior
    env = gym.make('PredPrey-Pred-v0')
    behavior = Behavior()
    env.reinit(prey_behavior=behavior.fixed_prey)
    return env


OBS = "full"
ACT = "vel"
ENV = "PredPrey-Pred-v0"
WORKERS = 1
ALGO = "PPO"
############################################################
if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--exp',    type=str,       help='Help (default: ..)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    #### Register the environment ##############################
    register_env(ENV, create_environment)


    #### Set up the trainer's config ###########################
    config = ppo.DEFAULT_CONFIG.copy() # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    config = {
        "env": ENV,
        "num_workers":WORKERS,
        # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")), # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        "batch_mode": "complete_episodes",
        "framework": "torch",
    }

    #### Restore agent #########################################
    agent = ppo.PPOTrainer(config=config)
    # with open(ARGS.exp+'/checkpoint.txt', 'r+') as f:
    #     checkpoint = f.read()
    checkpoint = ARGS.exp
    agent.restore(checkpoint)

    #### Extract and print policies ############################
    # policies = [agent.get_policy(f"pol{i}") for i in range(NUM_DRONES)]
    # for i in range(NUM_DRONES):
    #     print(f"action model {i}", policies[i].model.action_model)
    #     print(f"value model {i}", policies[i].model.value_model)

    #### Create test environment ###############################
    test_env = create_environment(None)
    
    #### Show, record a video, and log the model's performance #
    obs = test_env.reset()
    action = test_env.action_space.sample()
    done = False
    start = time.time()
    # for i in range(test_env.max_num_steps): # Up to 6''
    while not done:
        time.sleep(0.01)
        action = agent.compute_action(obs)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        # if done["__all__"]: obs = test_env.reset() # OPTIONAL EPISODE HALT
    print(test_env.num_steps)
    test_env.close()

    #### Shut down Ray #########################################
    ray.shutdown()
