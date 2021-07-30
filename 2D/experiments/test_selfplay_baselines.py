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

from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPredEnv
from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPreyEnv
from time import sleep

OBS = "full"
ACT = "vel"
ENV = "PredPrey-Pred-v0"
WORKERS = 1#3
ALGO = "PPO"
SEED_VALUE = 3
EVAL_EPISODES = 5
LOG_DIR = None
NUM_TESTING_EPISODES = 5
RENDER_MODE = True

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

def rollout(env, policy):
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
            sleep(0.005)


    return total_reward


# Interface to match the same one as in RayRLlib
class PPORLlibInterface(PPO):
    def __init__(self, *args, **kwargs):
        super(PPORLlibInterface, self).__init__(*args, **kwargs)

    def compute_action(self, obs):
        return super(PPORLlibInterface, self).predict(obs)[0]

    # To fix issue while loading when loading from different versions of pickle and python from the server and the local machine
    def load(model_path, env):
        custom_objects = {
            "lr_schedule": lambda x: .003,
            "clip_range": lambda x: .02
        }
        return PPO.load(model_path, env, custom_objects=custom_objects)

def test(log_dir):
    # train selfplay agent
    logger.configure(folder=log_dir)
    # prey model is being loaded inside the environment while reset
    pred_env = SelfPlayPredEnv(log_dir=log_dir, algorithm_class=PPORLlibInterface)
    # pred_env.set_render_flag(True)
    pred_env.seed(SEED_VALUE)
    # make_deterministic(SEED_VALUE)


    # prey_env = SelfPlayPreyEnv()
    # prey_env.seed(SEED_VALUE)
    # prey_model = PPORLlibInterface("MlpPolicy", prey_env)

    pred_model = PPORLlibInterface.load(os.path.join(log_dir, "pred", "final_model"), pred_env)
    
    # prey_model.load(os.path.join(log_dir, "prey", "final_model"))

    # pred_env.prey_policy = prey_model
    # pred_env.prey_behavior = Behavior().fixed_prey
    # print(pred_env.prey_policy)
    rewards = []
    for i in range(NUM_TESTING_EPISODES):
        rewards.append(rollout(pred_env, pred_model))
    
    # pred_env.close()
    # prey_env.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--exp',    type=str,       help='Help (default: ..)', metavar='')
    ARGS = parser.parse_args()
    LOG_DIR = ARGS.exp
    test(LOG_DIR)