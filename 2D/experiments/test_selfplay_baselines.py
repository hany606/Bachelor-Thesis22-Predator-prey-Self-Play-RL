# Training script for self-play using Stable baselines3
# Based on: https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_selfplay.py
import os
import argparse
from datetime import datetime
import numpy as np

import torch

import gym

from stable_baselines3 import PPO
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import EvalCallback
from shutil import copyfile # keep track of generations


from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Pred
from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Prey
from gym_predprey.envs.PredPrey1v1 import Behavior

from time import sleep

OBS = "full"
ACT = "vel"
ENV = "PredPrey-Pred-v0"
WORKERS = 1#3
ALGO = "PPO"
SEED_VALUE = 3
EVAL_EPISODES = 5
# PRED_TRAINING_EPOCHS = 5
# PREY_TRAINING_EPOCHS = 5
LOG_DIR = None
# TRAINING_ITERATION = 1000
NUM_TIMESTEPS = int(25e3)#int(1e9)
EVAL_FREQ = int(1e3)
RENDER_MODE = True
BEST_THRESHOLD = 0.5 # must achieve a mean score above this to replace prev best self
NUM_ROUNDS = 50
# selfplay_policies = None

# TODO: Initialize Wandbai

# Source: https://github.com/rlturkiye/flying-cavalry/blob/main/rllib/main.py
def make_deterministic(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    # see https://github.com/pytorch/pytorch/issues/47672
    cuda_version = torch.version.cuda
    if cuda_version is not None and float(torch.version.cuda) >= 10.2:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'
    else:
        torch.set_deterministic(True)  # Not all Operations support this.
    # This is only for Convolution no problem
    torch.backends.cudnn.deterministic = True

class SelfPlayPredEnv(PredPrey1v1Pred):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self):
        super(SelfPlayPredEnv, self).__init__()
        self.prey_policy = self # It replaces the policy for the other agent with the best policy that found during reset (This class have it)
        self.best_model = None
        self.best_model_filename = None

    # TODO: This works fine for identical agents but different agents will not work as they won't have the same action spaec
    def compute_action(self, obs): # the policy
        if self.best_model is None:
            return self.action_space.sample() # return a random action
        else:
            action, _ = self.best_model.predict(obs) # it is predict because this is PPO from stable-baselines not rllib
            return action

    # Change to search only for the prey
    def reset(self):
        self.prey_policy = PPORLLIB.load(os.path.join(LOG_DIR, "prey", "final_model"), env=self)
        # self.prey_behavior = Behavior().cos_1D
        self.best_model = PPORLLIB.load(os.path.join(LOG_DIR, "pred", "final_model"), env=self)
        return super(SelfPlayPredEnv, self).reset()

class SelfPlayPreyEnv(PredPrey1v1Prey):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self):
        super(SelfPlayPreyEnv, self).__init__()
        self.pred_policy = self # It replaces the policy for the other agent with the best policy that found during reset (This class have it)
        self.best_model = None
        self.best_model_filename = None

    # TODO: This works fine for identical agents but different agents will not work as they won't have the same action spaec
    def compute_action(self, obs): # the policy
        action, _ = self.best_model.predict(obs) # it is predict because this is PPO from stable-baselines not rllib
        return action

    # Change to search only for the prey
    def reset(self):
        return super(SelfPlayPreyEnv, self).reset()

def rollout(env, policy):
    # obs = env.reset()
    # done = False
    # while not done:
    # # for _ in range (1000):
    #     action = env.action_space.sample()
    #     # action[0] = [0,0]
    #     # action[1] = 1
    #     # action[2] = 1
    #     # action[3] = 1
    #     observation, reward, done, info = env.step(action)
    #     print(observation)
    #     # print(reward)
    #     env.render()
    #     sleep(0.01)
    #     # print(done)
    # env.close()
    """ play one agent vs the other in modified gym-style loop. """
    obs = env.reset()

    done = False
    total_reward = 0

    while not done:

        action, _states = policy.predict(obs)
        # action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

        total_reward += reward

        if RENDER_MODE:
            env.render()
            sleep(0.01)


    return total_reward


class PPORLLIB(PPO):
    def __init__(self, *args, **kwargs):
        super(PPORLLIB, self).__init__(*args, **kwargs)

    def compute_action(self, obs):
        return super(PPORLLIB, self).predict(obs)[0]

def test(log_dir):
    # train selfplay agent
    logger.configure(folder=log_dir)
    pred_env = SelfPlayPredEnv()
    pred_env.seed(SEED_VALUE)
    # make_deterministic(SEED_VALUE)


    # prey_env = SelfPlayPreyEnv()
    # prey_env.seed(SEED_VALUE)
    # prey_model = PPORLLIB("MlpPolicy", prey_env)

    pred_model = PPORLLIB.load(os.path.join(log_dir, "pred", "final_model"), pred_env)
    # prey_model.load(os.path.join(log_dir, "prey", "final_model"))

    # pred_env.prey_policy = prey_model
    # pred_env.prey_behavior = Behavior().fixed_prey
    # print(pred_env.prey_policy)
    rollout(pred_env, pred_model)
    # pred_env.close()
    # prey_env.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--exp',    type=str,       help='Help (default: ..)', metavar='')
    ARGS = parser.parse_args()
    LOG_DIR = ARGS.exp
    test(ARGS.exp)