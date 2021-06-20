# Based on: https://github.com/utiasDSL/gym-pybullet-drones/blob/master/experiments/learning/test_multiagent.py

"""Test script for multiagent problems.

This scripts runs the best model found by one of the executions of `agent.py`

Example
-------
To run the script, type in a terminal:

    $ python test_agent.py --exp ./results/save-<env>-<num_drones>-<algo>-<obs>-<act>-<time_date>

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

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

from TrajAviary import TrajAviary

import shared_constants

# Source: https://github.com/caelan/pybullet-planning/blob/6af327ba03eb32c0c174656cca524599c076e754/pybullet_tools/utils.py#L4415
def add_line(start, end, color=[0,0,0], width=1, lifetime=None, parent=-1, parent_link=-1):
    assert (len(start) == 3) and (len(end) == 3)
    return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width,
                              parentObjectUniqueId=parent, parentLinkIndex=parent_link)
                              
def draw_point(point, size=0.01, **kwargs):
    lines = []
    for i in range(len(point)):
        axis = np.zeros(len(point))
        axis[i] = 1.0
        p1 = np.array(point) - size/2 * axis
        p2 = np.array(point) + size/2 * axis
        lines.append(add_line(p1, p2, **kwargs))
    return lines


############################################################
if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--exp',    type=str,       help='Help (default: ..)', metavar='')
    ARGS = parser.parse_args()

    #### Parameters to recreate the environment ################
    OBS = ObservationType.KIN if ARGS.exp.split("-")[3] == 'kin' else ObservationType.RGB
    if ARGS.exp.split("-")[4] == 'rpm':
        ACT = ActionType.RPM
    elif ARGS.exp.split("-")[4] == 'dyn':
        ACT = ActionType.DYN
    elif ARGS.exp.split("-")[4] == 'pid':
        ACT = ActionType.PID
    elif ARGS.exp.split("-")[4] == 'vel':
        ACT = ActionType.VEL
    elif ARGS.exp.split("-")[4] == 'one_d_rpm':
        ACT = ActionType.ONE_D_RPM
    elif ARGS.exp.split("-")[4] == 'one_d_dyn':
        ACT = ActionType.ONE_D_DYN
    elif ARGS.exp.split("-")[4] == 'one_d_pid':
        ACT = ActionType.ONE_D_PID

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)


    #### Register the environment ##############################
    temp_env_name = "this-aviary-v0"
    
    if ARGS.exp.split("-")[1] == 'trajtrack':
        register_env(temp_env_name, lambda _: TrajAviary(aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                                        obs=OBS,
                                                        act=ACT
                                                        )
                     )
    else:
        print("[ERROR] environment not yet implemented")
        exit()

    #### Unused env to extract the act and obs spaces ##########
    if ARGS.exp.split("-")[1] == 'trajtrack':
        temp_env = TrajAviary(aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                             obs=OBS,
                             act=ACT
                             )
    else:
        print("[ERROR] environment not yet implemented")
        exit()

    observer_space = temp_env.observation_space
    action_space = temp_env.action_space

    #### Set up the trainer's config ###########################
    config = ppo.DEFAULT_CONFIG.copy() # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    config = {
        "env": temp_env_name,
        "num_workers": 0, #0+ARGS.workers,
        "num_gpus": 0,# int(os.environ.get("RLLIB_NUM_GPUS", "0")), # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        "batch_mode": "complete_episodes",
        "framework": "torch",
    }

    
    #### Restore agent #########################################
    agent = ppo.PPOTrainer(config=config)
    with open(ARGS.exp+'/checkpoint.txt', 'r+') as f:
        checkpoint = f.read()
    agent.restore(checkpoint)

    #### Create test environment ###############################
    if ARGS.exp.split("-")[1] == 'trajtrack':
        test_env = TrajAviary(aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                             obs=OBS,
                             act=ACT,
                             gui=True,
                             record=True
                             )
    else:
        print("[ERROR] environment not yet implemented")
        exit()
    
    #### Show, record a video, and log the model's performance #
    obs = test_env.reset()
    logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                    num_drones=1
                    )

    PYB_CLIENT = test_env.getPyBulletClient()
    obs = test_env.reset()
    start = time.time()
    for i in range(int(3*test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS)): # Up to 6''
        draw_point(test_env.TARGET_POSITION[i])
        # print(test_env.TARGET_POSITION[i])
        action = agent.compute_action(obs)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        if OBS==ObservationType.KIN:
            logger.log(drone=0,
                       timestamp=i/test_env.SIM_FREQ,
                       state= np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
                       control=np.zeros(12)
                       )

        sync(np.floor(i*test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
        # if done["__all__"]: obs = test_env.reset() # OPTIONAL EPISODE HALT
    test_env.close()
    logger.save_as_csv("sa") # Optional CSV save
    logger.plot()

    #### Shut down Ray #########################################
    ray.shutdown()
