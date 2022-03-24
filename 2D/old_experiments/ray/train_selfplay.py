# Buggy
import os
from datetime import datetime
import numpy as np

import torch

import gym

import ray
from ray import tune
from ray.tune.logger import pretty_print
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import DEFAULT_LOGGERS


from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Pred
from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Prey
 

OBS = "full"
ACT = "vel"
ENV = "PredPrey-Pred-v0"
WORKERS = 1#3
ALGO = "PPO"
SEED_VALUE = 3
TRAINING_ITERATION = 1000
PRED_TRAINING_EPOCHS = 5
PREY_TRAINING_EPOCHS = 5
checkpoint_dir = None
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


# Just a wrapper for the gym environment to have the same interface of compute_action
class InitAgent:
    def __init__(self, env):
        self.env = env

    def compute_action(self, _):
        return self.env.action_space.sample()

# Class to store, initialize, sample policeis for SelfPlay algorithm

class SelfPlayPolicies:
    # initialize the policy
    # def __init__(self, num_policies=2, initialization_policy=RandomPolicy, keys=None):
    def __init__(self, initialization_policy, num_policies=2, keys=None):
        self.num_policies = num_policies
        if(keys is None):
            self.keys = [i for i in range(num_policies)]
        else:
            if(num_policies != len(keys)):
                print("Number of policies is not equal to number of keys provideed to map the policy to the dictionary")
                raise ValueError
            self.keys = keys

        self.policies = {self.keys[i]: [{"policy":initialization_policy[self.keys[i]], "path":None}] for i in range(num_policies)}

    # Random sampling for the agents
    def sample(self, num_sampled_policies=1):
        policies = {}
        for i in range(self.num_policies):
            key = self.keys[i]
            policies[key] = np.random.choice(self.policies[key], num_sampled_policies)
        return policies
    
    def store(self, policies, path):
        for i in range(self.num_policies):
            key = self.keys[i]
            self.policies[key].append({"policy": policies[key], "path": path[key]})

    def get_num_policies(self):
        return len(self.policies[self.keys[0]])

# Algorithm flow: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_train_fn.py

# Bug in passing the shared selfplay_policies TODO: Fix it
def selfplay_train_func(config, reporter):
    selfplay_policies = config["env_config"]["something"]
    sampled_policies = selfplay_policies.sample()
    sampled_pred = sampled_policies["pred"][0]["policy"]
    sampled_prey = sampled_policies["prey"][0]["policy"]
    # Train for n iterations with high LR
    agent1 = PPOTrainer(env="CartPole-v0", config=config)
    for _ in range(10):
        result = agent1.train()
        result["phase"] = 1
        reporter(**result)
        phase1_time = result["timesteps_total"]
    state = agent1.save()
    agent1.stop()

    # Train for n iterations with low LR
    config["lr"] = 0.0001
    agent2 = PPOTrainer(env="CartPole-v0", config=config)
    agent2.restore(state)
    for _ in range(10):
        result = agent2.train()
        result["phase"] = 2
        result["timesteps_total"] += phase1_time  # keep time moving forward
        reporter(**result)
    agent2.stop()



def selfplay_train_func_test(config, reporter):
    # Initialize policies [DONE]
    # Sample predator and prey
    sampled_policies = selfplay_policies.sample()
    sampled_pred = sampled_policies["pred"][0]["policy"]
    sampled_prey = sampled_policies["prey"][0]["policy"]
    print("-------------- Train Predator --------------------")
    # Train the predator against the sampled prey
    config["env_config"] = {"prey_policy": sampled_prey}
    print(pretty_print(config))
    register_env("Pred", lambda _: PredPrey1v1Pred(prey_policy=sampled_prey))
    pred_agent = PPOTrainer(env="Pred", config=config)   # https://github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    if(selfplay_policies.get_num_policies() > 1): # restore the policy
        pred_agent.restore(sampled_policies["pred"][0]["path"])
    
    # Train for multiple epochs
    for pred_epoch in range(PRED_TRAINING_EPOCHS):
        result = pred_agent.train()
        result["phase"] = "Predator"
        reporter(**result)
        pred_time = result["timesteps_total"]
    state = pred_agent.save()
    pred_save_checkpoint = pred_agent.save_checkpoint(checkpoint_dir+"-pred")
    pred_agent.stop()

    # Train the prey against the sampled predator (optimize)
    print("-------------- Train Prey --------------------")
    config["env_config"] = {"pred_policy": sampled_pred}
    register_env("Prey", lambda _: PredPrey1v1Prey(pred_policy=sampled_pred))
    prey_agent = PPOTrainer(PPOTrainer="Prey", config=config)
    # TODO: Remove the restore and just pick the last one
    if(selfplay_policies.get_num_policies() > 1): # restore the policy
        prey_agent.restore(sampled_policies["prey"][0]["path"])
    for prey_epoch in range(PREY_TRAINING_EPOCHS):
        result = prey_agent.train()
        result["phase"] = "Prey"
        result["timesteps_total"] += pred_time
        reporter(**result)
    state = prey_agent.save()
    prey_save_checkpoint = prey_agent.save_checkpoint(checkpoint_dir+"-prey")
    prey_agent.stop()
    # Save the predator and save the prey to the policies
    # selfplay_policies.store({"pred":pred_agent.get_policy(), "prey":prey_agent.get_policy}, path = {"pred": pred_save_checkpoint, "prey": prey_save_checkpoint})
    selfplay_policies.store({"pred":pred_agent, "prey":prey_agent}, path = {"pred": pred_save_checkpoint, "prey": prey_save_checkpoint})
    print("------------------------------------------------------")
    # Repeate

if __name__ == '__main__':
    # Source: https://github.com/rlturkiye/flying-cavalry/blob/main/rllib/main.py
    if torch.cuda.is_available():
        print("## CUDA available")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("## CUDA not available")
    checkpoint_dir = os.path.dirname(os.path.abspath(__file__)) + '/selfplay-results/save-' + ENV + '-' + ALGO + '-' + OBS + '-' + ACT + '-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir + '/')

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    #### Register the environment #################################
    pred_agent = InitAgent(env=PredPrey1v1Pred())
    prey_agent = InitAgent(env=PredPrey1v1Prey())
    selfplay_policies = SelfPlayPolicies(initialization_policy={"pred":pred_agent, "prey":prey_agent},
                                         num_policies=2, keys=["pred", "prey"])
    # register_env("Pred_orig", lambda _: PredPrey1v1Pred())
    # register_env("Prey_orig", lambda _: PredPrey1v1Prey())

    # Initialize policies
    # Each history has two versions: version for pred and for prey -> pred is a Trainer and inside it a trainer for prey but is only used to compute actions not to train and the same for the prey
    # pred_agent = PPOTrainer(env="Pred_orig", config=ppo.DEFAULT_CONFIG.copy()) # Trainer is not sampling from the environment, the trainer is only used to compute actions
    # prey_agent = PPOTrainer(env="Prey_orig", config=ppo.DEFAULT_CONFIG.copy())
    # pred_agent = InitAgent(env=PredPrey1v1Pred())
    # prey_agent = InitAgent(env=PredPrey1v1Prey())
    # selfplay_policies = SelfPlayPolicies(initialization_policy={"pred":pred_agent, "prey":prey_agent},
    #                                      num_policies=2, keys=["pred", "prey"])


    ppo_default_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config = ppo_default_config
    # ppo_config = {**ppo_default_config,
    #               **{

    #                 }
    #             }

    config = {
        # "env": ENV,
        "num_workers": 0 + WORKERS,
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        "batch_mode": "complete_episodes",
        "seed": SEED_VALUE,
        "framework": "torch",
        "env_config":{"something":selfplay_policies}
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
    }
    pretty_print(config)
    #### Ray Tune stopping conditions ##########################
    stop = {
        # "timesteps_total": 120000, # 100000 ~= 10'
        # "episode_reward_mean": -250,
        "training_iteration": TRAINING_ITERATION,
    }
    ########################################################################################################


    #### Train #################################################
    config={**ppo_config, **config}
    results = tune.run(
        # ALGO,
        selfplay_train_func,
        stop=stop,
        config=config,
        # verbose=True,
        log_to_file=["logs_out.txt", "logs_err.txt"],
        checkpoint_freq=5,
        checkpoint_at_end=True,
        local_dir=checkpoint_dir,
        resources_per_trial=PPOTrainer.default_resource_request(config),#.to_json(),
        loggers=DEFAULT_LOGGERS #+ (WandbLogger, )
    )
    ########################################################################################################

    #### Save agent ############################################
    checkpoints = results.get_trial_checkpoints_paths(
        trial=results.get_best_trial('episode_reward_mean', mode='max'),
        metric='episode_reward_mean')
    with open(checkpoint_dir + '/checkpoint.txt', 'w+') as f:
        f.write(checkpoints[0][0])

    #### Shut down Ray #########################################
    ray.shutdown()
    ########################################################################################################


