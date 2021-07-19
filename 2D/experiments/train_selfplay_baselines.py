# Training script for self-play using Stable baselines3
# Based on: https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_selfplay.py
import os
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
from stable_baselines3.common.callbacks import CheckpointCallback


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
RENDER_MODE = False
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
        # load model if it's there
        modellist = [f for f in os.listdir(os.path.join(LOG_DIR, "prey")) if f.startswith("history")]
        modellist.sort()
        if len(modellist) > 0:
            filename = os.path.join(LOG_DIR, "prey", modellist[-1]) # the latest best model
            if filename != self.best_model_filename:
                print("loading model: ", filename)
                self.best_model_filename = filename
                if self.best_model is not None:
                    del self.best_model
                self.best_model = PPO.load(filename, env=self)
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
        if self.best_model is None:
            return self.action_space.sample() # return a random action
        else:
            action, _ = self.best_model.predict(obs) # it is predict because this is PPO from stable-baselines not rllib
            return action

    # Change to search only for the prey
    def reset(self):
        # load model if it's there
        modellist = [f for f in os.listdir(os.path.join(LOG_DIR, "pred")) if f.startswith("history")]
        modellist.sort()
        if len(modellist) > 0:
            filename = os.path.join(LOG_DIR, "pred", modellist[-1]) # the latest best model
            if filename != self.best_model_filename:
                print("loading model: ", filename)
                self.best_model_filename = filename
                if self.best_model is not None:
                    del self.best_model
                self.best_model = PPO.load(filename, env=self)
        return super(SelfPlayPreyEnv, self).reset()

class SelfPlayCallback(EvalCallback):
  # hacked it to only save new version of best model if beats prev self by BEST_THRESHOLD score
  # after saving model, resets the best score to be BEST_THRESHOLD
    def __init__(self, *args, **kwargs):
        super(SelfPlayCallback, self).__init__(*args, **kwargs)
        self.best_mean_reward = BEST_THRESHOLD
        self.generation = 0
        self.agent_name = None


    # def _on_training_end(self) -> None:
    #     super(SelfPlayCallback, self)._on_training_end()

    # def _on_step(self) -> bool:
    #     result = super(SelfPlayCallback, self)._on_step()
    #     if result: #and self.best_mean_reward > BEST_THRESHOLD:
    #         self.generation += 1
    #         print("SELFPLAY: mean_reward achieved:", self.best_mean_reward)
    #         print("SELFPLAY: new best model, bumping up generation to", self.generation)
    #         source_file = os.path.join(LOG_DIR, self.agent_name, "best_model.zip")
    #         backup_file = os.path.join(LOG_DIR, self.agent_name, "history_"+str(self.generation).zfill(8)+".zip")
    #         copyfile(source_file, backup_file)
    #         self.best_mean_reward = BEST_THRESHOLD
    #     return result

    # def set_agent_save_name(self, name):
    #     self.agent_name = name

def rollout(env, policy):
    """ play one agent vs the other in modified gym-style loop. """
    obs = env.reset()

    done = False
    total_reward = 0

    while not done:

        action, _states = policy.predict(obs)
        obs, reward, done, _ = env.step(action)

        total_reward += reward

        if RENDER_MODE:
            env.render()

    return total_reward


# TODO: make it more easy to train and switch the agent that is trainin
def train(log_dir):
    # train selfplay agent
    logger.configure(folder=log_dir)
    pred_env = SelfPlayPredEnv()
    pred_env.seed(SEED_VALUE)
    make_deterministic(SEED_VALUE)
    pred_model = PPO("MlpPolicy", pred_env, 
                     clip_range=0.2, ent_coef=0.0,
                     learning_rate=3e-4, batch_size=64, gamma=0.99, verbose=2)

    pred_eval_callback = SelfPlayCallback(pred_env,
                                        best_model_save_path=os.path.join(LOG_DIR, "pred"),
                                        log_path=os.path.join(LOG_DIR, "pred"),
                                        eval_freq=EVAL_FREQ,
                                        n_eval_episodes=EVAL_EPISODES,
                                        deterministic=True)

    prey_env = SelfPlayPreyEnv()
    prey_env.seed(SEED_VALUE)
    prey_model = PPO("MlpPolicy", prey_env, 
                     clip_range=0.2, ent_coef=0.0,
                     learning_rate=3e-4, batch_size=64, gamma=0.99, verbose=2)

    prey_eval_callback = SelfPlayCallback(prey_env,
                                        best_model_save_path=os.path.join(LOG_DIR, "prey"),
                                        log_path=os.path.join(LOG_DIR, "prey"),
                                        eval_freq=EVAL_FREQ,
                                        n_eval_episodes=EVAL_EPISODES,
                                        deterministic=True)

    pred_checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=os.path.join(LOG_DIR, "pred"),
                                                    name_prefix='history')

    prey_checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=os.path.join(LOG_DIR, "prey"),
                                                    name_prefix='history')
    # Here alternate training
    for round_num in range(NUM_ROUNDS):
        print(f"------------------- Pred {round_num+1}--------------------")
        pred_model.learn(total_timesteps=NUM_TIMESTEPS, callback=[pred_checkpoint_callback,pred_eval_callback])
        print(f"------------------- Prey {round_num+1}--------------------")
        prey_model.learn(total_timesteps=NUM_TIMESTEPS, callback=[prey_checkpoint_callback, prey_eval_callback])
        pred_model.save(os.path.join(LOG_DIR, "pred", "final_model")) # probably never get to this point.
        prey_model.save(os.path.join(LOG_DIR, "prey", "final_model")) # probably never get to this point.
    pred_env.close()
    prey_env.close()

if __name__=="__main__":

    # Source: https://github.com/rlturkiye/flying-cavalry/blob/main/rllib/main.py
    if torch.cuda.is_available():
        print("## CUDA available")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("## CUDA not available")
    LOG_DIR = os.path.dirname(os.path.abspath(__file__)) + '/selfplay-results/save-' + ENV + '-' + ALGO + '-' + OBS + '-' + ACT + '-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR + '/')
    if not os.path.exists(os.path.join(LOG_DIR, "pred")):
        os.makedirs(os.path.join(LOG_DIR, "pred") + '/')
    if not os.path.exists(os.path.join(LOG_DIR, "prey")):
        os.makedirs(os.path.join(LOG_DIR, "prey") + '/')

    train(LOG_DIR)
