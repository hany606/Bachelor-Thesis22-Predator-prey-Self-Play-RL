# TODO
import os
import ray
from ray import tune
import gym
from ray.tune.logger import pretty_print
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.agents.callbacks import DefaultCallbacks
import torch
import numpy as np

OBS = "full"
ACT = "vel"
ENV = "PredPrey-Pred-v0"
WORKERS = 6
ALGO = "PPO"
SEED_VALUE = 3
TRAINING_ITERATION = 1000

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

def create_environment(config):
    import gym_predprey
    # TODO: Which environment?
    env = gym.make('PredPrey-Pred-v0')
    return env


# Class to store, initialize, sample policeis for SelfPlay algorithm
class SelfPlayPolicies:
    # initialize the policy
    def __init__(self, num_policies=2, initialization_policy=RandomPolicy, keys=None):
        self.num_policies = num_policies
        if(keys is None):
            self.keys = [i for i in range(num_policies)]
        else:
            if(num_policies != len(keys)):
                print("Number of policies is not equal to number of keys provideed to map the policy to the dictionary")
                raise ValueError
            self.keys = keys

        self.policies = {self.keys[i]: [initialization_policy] for i in range(num_policies)}

    # Random sampling for the agents
    def sample(self, num_sampled_policies=1):
        policies = {}
        for i in range(self.num_policies):
            key = self.keys[i]
            policies[key] = np.random.choice(self.policies[key], num_sampled_policies)
        return policies
    
    def store(self, policies):
        for i in range(self.num_policies):
            key = self.keys[i]
            self.policies[key].append(policies[key])


if __name__ == '__main__':
    # Source: https://github.com/rlturkiye/flying-cavalry/blob/main/rllib/main.py
    if torch.cuda.is_available():
        print("## CUDA available")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("## CUDA not available")
    checkpoint_dir = os.path.dirname(os.path.abspath(__file__)) + '/results/save-' + ENV + '-' + ALGO + '-' + OBS + '-' + ACT + '-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir + '/')

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    #### Register the environment #################################
    register_env(ENV, create_environment)
    ppo_default_config = ppo.DEFAULT_CONFIG.copy()

    ppo_config = {**ppo_default_config,
                  **{

                    }
                }

    config = {
        "env": ENV,
        "num_workers": 0 + WORKERS,
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        "batch_mode": "complete_episodes",
        "seed": SEED_VALUE,
        "framework": "torch",
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
    print(pretty_print(config))
    #### Ray Tune stopping conditions ##########################
    stop = {
        # "timesteps_total": 120000, # 100000 ~= 10'
        # "episode_reward_mean": -250,
        "training_iteration": TRAINING_ITERATION,
    }
    ########################################################################################################


    #### Train #################################################
    results = tune.run(
        ALGO,
        stop=stop,
        config={**config, **ppo_config},
        # verbose=True,
        log_to_file=["logs_out.txt", "logs_err.txt"],
        checkpoint_freq=5,
        checkpoint_at_end=True,
        local_dir=checkpoint_dir,
        # loggers=DEFAULT_LOGGERS #+ (WandbLogger, )
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

    # Algorithm flow
    # Initialize policies
    # Sample pred and prey
    # Train the predator against the sampled prey
    # Train the 

    # callbacks
    # At the start of the training -> create the selfpolicy object and pass it through the callbacks

