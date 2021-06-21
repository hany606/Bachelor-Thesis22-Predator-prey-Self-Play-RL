# Note:
# Passing argument is based on: https://github.com/utiasDSL/gym-pybullet-drones/blob/master/experiments/learning/multiagent.py
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

import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy

from gym_predprey.envs.PredPrey import PredPrey
import gym
import time

OBS = "xy"
ACT = "vel"
ENV = "1v1"
WORKERS = 5
ALGO = "ppo"

if __name__ == "__main__":
    #### Save directory ########################################
    filename = os.path.dirname(os.path.abspath(__file__))+'/results/save-'+ENV+'-'+ALGO+'-'+OBS+'-'+ACT+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Print out current git commit hash #####################
    try:
        git_commit = subprocess.check_output(["git", "describe", "--tags"]).strip()
        with open(filename+'/git_commit.txt', 'w+') as f:
            f.write(str(git_commit))
    except:
        with open(filename+'/git_commit.txt', 'w+') as f:
                    f.write("NO TAG")
    ########################################################################################################

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    ########################################################################################################


    #### Register the environment #################################
    env_name = "predprey"
    register_env(env_name, lambda _: PredPrey())
    #### Unused env to extract the act and obs spaces ##########
    # temp_env = PredPrey()

    # print("[INFO] Action space:", temp_env.action_space)
    # print("[INFO] Observation space:", temp_env.observation_space)
    
    ########################################################################################################


    #### Set up the trainer's config ###########################
    config = ppo.DEFAULT_CONFIG.copy() # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    config = {
        "env": env_name,
        "num_workers": 0 + WORKERS,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")), # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        "batch_mode": "complete_episodes",
        "framework": "torch",
    }
    #### Ray Tune stopping conditions ##########################
    stop = {
        # "timesteps_total": 120000, # 100000 ~= 10'
        "episode_reward_mean": -250,
        # "training_iteration": 1000,
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
    checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean',
                                                                                   mode='max'
                                                                                   ),
                                                      metric='episode_reward_mean'
                                                      )
    with open(filename+'/checkpoint.txt', 'w+') as f:
        f.write(checkpoints[0][0])

    #### Shut down Ray #########################################
    ray.shutdown()
    ########################################################################################################