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
from gym.core import Env
import numpy as np
import gym
import wandb

from gym.spaces import Box, Dict
import torch
import torch.nn as nn
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
from ray.tune.integration.wandb import WandbLoggerCallback
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


# import gym_predprey
from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Super
from gym_predprey.envs.PredPrey1v1 import Behavior

#### Useful links ##########################################
# Workflow: github.com/ray-project/ray/blob/master/doc/source/rllib-training.rst
# ENV_STATE example: github.com/ray-project/ray/blob/master/rllib/examples/env/two_step_game.py
# Competing policies example: github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py

############################################################
class ResultsCallback(DefaultCallbacks):
    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        """Called at the end of Trainable.train().

        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        # print(f"reward: {result['reward_mean']}")
        print(f"episode_len_mean: {result['episode_len_mean']}")
        print(f"iterations_since_restore:{result['iterations_since_restore']}")
        print("-------------------------------------------------")


OBS = "full"
ACT = "vel"
ENV = 'PredPrey-Superior-1v1-v0'
WORKERS = 2
ALGO = "PPO"

def create_environment(_):
    import gym_predprey
    from gym_predprey.envs.PredPrey1v1 import Behavior
    env = gym.make(ENV)
    behavior = Behavior()
    env.reinit(prey_behavior=behavior.fixed_prey)
    return env





# wandb.init(project="Behavioral-Learning-Thesis",
#            group="Evorobotpy2-Predetor-Prey",
#            config={
#                 "info": "1Pred-1Fixed_prey",
#                 "algo": ALGO,
#                 "env": ENV,
#                 "obs": OBS
#                 }
#             )



if __name__ == "__main__":
    #### Save directory ########################################
    filename = os.path.dirname(os.path.abspath(__file__)) + '/results/save-' + ENV + '-' + ALGO + '-' + OBS + '-' + ACT + '-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename + '/')

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    ########################################################################################################

    #### Register the environment #################################
    register_env(ENV, create_environment)
    ########################################################################################################

    #### Set up the trainer's config ###########################
    config = ppo.DEFAULT_CONFIG.copy()  # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    # config = dict(config, **dict({
    #     "env": ENV,
    #     "num_workers": 0 + WORKERS,
    #     "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
    #     # "batch_mode": "complete_episodes",
    #     # "callbacks":
    #         # ResultsCallback,
    #         # WandbLoggerCallback(
    #         #     project="Behavioral-Learning-Thesis",
    #         #     group="Evorobotpy2-Predetor-Prey",
    #         #     # log_config=True,
    #         #     config={
    #         #         "info": "1Pred-1Fixed_prey",
    #         #         "algo": ALGO,
    #         #         "env": ENV,
    #         #         "obs": OBS
    #         #         }
    #         # )
    #     # ],
    #     "framework": "torch",
    # }))
    config = {
        "env": ENV,
        "num_workers": 0 + WORKERS,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        "batch_mode": "complete_episodes",
        # "callbacks":
        #     # ResultsCallback,
        #     WandbLoggerCallback(
        #         project="Behavioral-Learning-Thesis",
        #         group="Evorobotpy2-Predetor-Prey",
        #         # log_config=True,
        #         config={
        #             "info": "1Pred-1Fixed_prey",
        #             "algo": ALGO,
        #             "env": ENV,
        #             "obs": OBS
        #             }
        #     ),
        # # ],
        # "logger_config":{
        #     "wandb": {
        #         "project": "Behavioral-Learning-Thesis",
        #         "group": "Evorobotpy2-Predetor-Prey",
        #         # "config":{
        #         #     "info": "1Pred-1Fixed_prey",
        #         #     "algo": ALGO,
        #         #     "env": ENV,
        #         #     "obs": OBS
        #         # }
        #     }
            
        # },
        "framework": "torch",
    }
    print(pretty_print(config))
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
        # verbose=True,
        checkpoint_freq=5,
        checkpoint_at_end=True,
        local_dir=filename,
        loggers=DEFAULT_LOGGERS #+ (WandbLogger, )
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