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


from gym_predprey.envs.PredPrey import PredPrey

OWN_OBS_VEC_SIZE = None # Modified at runtime
ACTION_VEC_SIZE = None # Modified at runtime

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
WORKERS = 1
ALGO = "ppo"
OWN_OBS_VEC_SIZE = 2
ACTION_VEC_SIZE = 2

############################################################
if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--exp',    type=str,       help='Help (default: ..)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    #### Register the custom centralized critic model ##########
    ModelCatalog.register_custom_model("cc_model", CustomTorchCentralizedCriticModel)

    #### Register the environment ##############################
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

    #### Set up the trainer's config ###########################
    config = ppo.DEFAULT_CONFIG.copy() # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    config = {
        "env": env_name,
        "num_workers":WORKERS,
        # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")), # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
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
        # "policies": {f"pol{i}": (None, observer_space, action_space, {"agent_id": i}) for i in range(NUM_DRONES)},
        "policies": {
            "pol0": (None, observer_space, action_space, {"agent_id": 0,}),
            "pol1": (None, observer_space, action_space, {"agent_id": 1,}),
        },
        "policy_mapping_fn": lambda x: "pol0" if x == 0 else "pol1", # # Function mapping agent ids to policy ids
        "observation_fn": central_critic_observer, # See rllib/evaluation/observation_function.py for more info
    }

    #### Restore agent #########################################
    agent = ppo.PPOTrainer(config=config)
    with open(ARGS.exp+'/checkpoint.txt', 'r+') as f:
        checkpoint = f.read()
    agent.restore(checkpoint)

    #### Extract and print policies ############################
    # policies = [agent.get_policy(f"pol{i}") for i in range(NUM_DRONES)]
    # for i in range(NUM_DRONES):
    #     print(f"action model {i}", policies[i].model.action_model)
    #     print(f"value model {i}", policies[i].model.value_model)
    policy0 = agent.get_policy("pol0")
    print("action model 0", policy0.model.action_model)
    print("value model 0", policy0.model.value_model)
    policy1 = agent.get_policy("pol1")
    print("action model 1", policy1.model.action_model)
    print("value model 1", policy1.model.value_model)

    #### Create test environment ###############################
    test_env = PredPrey()
    
    #### Show, record a video, and log the model's performance #
    obs = test_env.reset()
    action = {i: np.array([0, 0]) for i in range(2)}
    done = {"__all__": False}
    start = time.time()
    # for i in range(test_env.max_num_steps): # Up to 6''
    while not (True in done.values()):
        time.sleep(0.01)
        #### Deploy the policies ###################################
        temp = {}
        temp[0] = policy0.compute_single_action(np.hstack([action[1], obs[1], obs[0]])) # Counterintuitive order, check params.json
        temp[1] = policy1.compute_single_action(np.hstack([action[0], obs[0], obs[1]]))
        action = {0: temp[0][0], 1: temp[1][0]}
        # temp[0] = policies[0].compute_single_action(np.hstack([action[1], obs[1], obs[0]])) # Counterintuitive order, check params.json
        # for i in range(1, NUM_DRONES):
        #     temp[i] = policies[i].compute_single_action(np.hstack([action[0], obs[0], obs[1]]))
        # action = {i: temp[0][i] for i in range(NUM_DRONES)}
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        # if done["__all__"]: obs = test_env.reset() # OPTIONAL EPISODE HALT
    print(test_env.num_steps)
    test_env.close()

    #### Shut down Ray #########################################
    ray.shutdown()
