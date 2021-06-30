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
from ray.tune.logger import pretty_print


import wandb
from gym_predprey.envs.PredPrey import PredPrey
# from PredPrey import PredPrey
import time

#### Useful links ##########################################
# Workflow: github.com/ray-project/ray/blob/master/doc/source/rllib-training.rst
# ENV_STATE example: github.com/ray-project/ray/blob/master/rllib/examples/env/two_step_game.py
# Competing policies example: github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py

############################################################
class Logging(DefaultCallbacks):
    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        """Called at the end of Trainable.train().

        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """

        if self.legacy_callbacks.get("on_train_result"):
            self.legacy_callbacks["on_train_result"]({
                "trainer": trainer,
                "result": result,
            })
        print("Mean:")
        print(pretty_print(result["policy_reward_mean"]))
        print("Max:")
        print(pretty_print(result["policy_reward_max"]))
        print("Min:")
        print(pretty_print(result["policy_reward_min"]))
        print(f"episode_len_mean: {result['episode_len_mean']}")
        print(f"iterations_since_restore:{result['iterations_since_restore']}")
        print("-------------------------------------------------")




OBS = "xy"
ACT = "vel"
ENV = "1vfixed"
WORKERS = 6
ALGO = "PPO"
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
    ModelCatalog.register_custom_model("cc_model", CustomTorchCentralizedCriticModel)

    #### Register the environment #################################
    env_name = "predprey"
    register_env(env_name, lambda _: PredPrey())
    #### Unused env to extract the act and obs spaces ##########
    temp_env = PredPrey()
    observer_space = Dict({
        "own_obs": temp_env.observation_space,
        "opponent_obs": temp_env.observation_space,
        "opponent_action": temp_env.action_space,
    })
    action_space = temp_env.action_space

    # print("[INFO] Action space:", temp_env.action_space)
    # print("[INFO] Observation space:", temp_env.observation_space)

    ########################################################################################################

    #### Set up the trainer's config ###########################
    config = ppo.DEFAULT_CONFIG.copy()  # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    config = {
        "env": env_name,
        "num_workers": 0 + WORKERS,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        # "batch_mode": "complete_episodes",
        "callbacks": Logging,
        "framework": "torch",
    }
    

    #### Ray Tune stopping conditions ##########################
    stop = {
        # "timesteps_total": 120000, # 100000 ~= 10'
        # "episode_reward_mean": -250,
        "training_iteration": 1000,
    }
    ########################################################################################################

    #### Train #################################################
    results = tune.run(
        ALGO,
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
    # checkpoints = results.get_trial_checkpoints_paths(
    #     trial=results.get_best_trial('episode_len_mean', mode='min'),
    #     metric='episode_len_mean')
    # with open(filename + '/checkpoint.txt', 'w+') as f:
    #     f.write(checkpoints[0][0])


    #### Shut down Ray #########################################
    ray.shutdown()
    ########################################################################################################