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


from gym_predprey.envs.PredPrey import PredPrey
# from PredPrey import PredPrey
import time

#### Useful links ##########################################
# Workflow: github.com/ray-project/ray/blob/master/doc/source/rllib-training.rst
# ENV_STATE example: github.com/ray-project/ray/blob/master/rllib/examples/env/two_step_game.py
# Competing policies example: github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py

OWN_OBS_VEC_SIZE = None  # Modified at runtime
ACTION_VEC_SIZE = None  # Modified at runtime


############################################################
class CustomTorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized value function.

    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).

    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.action_model = FullyConnectedNetwork(
            Box(low=0, high=3000, shape=(OWN_OBS_VEC_SIZE, )), action_space,
            num_outputs, model_config, name + "_action")
        self.value_model = FullyConnectedNetwork(obs_space, action_space, 1,
                                                 model_config, name + "_vf")
        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        self._model_in = [input_dict["obs_flat"], state, seq_lens]
        return self.action_model({"obs": input_dict["obs"]["own_obs"]}, state,
                                 seq_lens)

    def value_function(self):
        value_out, _ = self.value_model({"obs": self._model_in[0]},
                                        self._model_in[1], self._model_in[2])
        return torch.reshape(value_out, [-1])


############################################################
class FillInActions(DefaultCallbacks):
    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id,
                                  policies, postprocessed_batch,
                                  original_batches, **kwargs):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 1 if agent_id == 0 else 0
        action_encoder = ModelCatalog.get_preprocessor_for_space(
            # Box(-np.inf, np.inf, (ACTION_VEC_SIZE,), np.float32) # Unbounded
            Box(-1, 1, (ACTION_VEC_SIZE, ), np.float32)  # Bounded
        )
        _, opponent_batch = original_batches[other_id]
        # opponent_actions = np.array([action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]]) # Unbounded
        opponent_actions = np.array([
            action_encoder.transform(np.clip(a, -1, 1))
            for a in opponent_batch[SampleBatch.ACTIONS]
        ])  # Bounded
        to_update[:, -ACTION_VEC_SIZE:] = opponent_actions
    
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
        print("episode_len_mean:")
        print(pretty_print(result["episode_len_mean"]))
        print("iterations_since_restore:")
        print(pretty_print(result["iterations_since_restore"]))
        print("-------------------------------------------------")


############################################################
def central_critic_observer(agent_obs, **kw):
    new_obs = {
        0: {
            "own_obs": agent_obs[0],
            "opponent_obs": agent_obs[1],
            "opponent_action":
            np.zeros(ACTION_VEC_SIZE),  # Filled in by FillInActions
        },
        1: {
            "own_obs": agent_obs[1],
            "opponent_obs": agent_obs[0],
            "opponent_action":
            np.zeros(ACTION_VEC_SIZE),  # Filled in by FillInActions
        },
    }
    return new_obs


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
    config = ppo.DEFAULT_CONFIG.copy(
    )  # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    config = {
        "env": env_name,
        "num_workers": 0 + WORKERS,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        "batch_mode": "complete_episodes",
        "callbacks": FillInActions,
        "framework": "torch",
    }
    #### Set up the model parameters of the trainer's config ###
    config["model"] = {
        "custom_model": "cc_model",
    }
    #### Set up the multiagent params of the trainer's config ##
    config["multiagent"] = {
        "policies": {
            "pol0": (None, observer_space, action_space, {"agent_id": 0,}),
            "pol1": (None, observer_space, action_space, {"agent_id": 1,}),
        },
        "policy_mapping_fn": lambda x: "pol0" if x == 0 else "pol1",  # # Function mapping agent ids to policy ids
        "observation_fn": central_critic_observer,  # See rllib/evaluation/observation_function.py for more info
    }

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