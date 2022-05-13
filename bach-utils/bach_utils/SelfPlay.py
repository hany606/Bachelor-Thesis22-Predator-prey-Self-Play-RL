from bach_utils.logger import get_logger
clilog = get_logger()

import os

from stable_baselines3.ppo.ppo import PPO as sb3PPO
from stable_baselines3.sac.sac import SAC as sb3SAC

import bach_utils.sampling as utsmpl

OS = False#True

# Parent class for all using SB3 functions to predict
class SelfPlayEnvSB3:
    def __init__(self, algorithm_class, archive, sample_after_reset, sampling_parameters):
        self.opponent_algorithm_class = algorithm_class  # algorithm class for the opponent
        self.opponent_policy = None             # The policy itself after it is loaded
        self.opponent_policy_name = None    # Current loaded policy name -> File -> as it was implement first to be stored on disk (But now in cache=archive) 
        self.target_opponent_policy_name = None
        self._name = None
        self.archive = archive  # opponent archive
        self.OS = OS
        self.sample_after_reset = sample_after_reset
        self.sampling_parameters = sampling_parameters
        self.reset_counter = 0

        if(archive is None):
            self.OS = True
        self.states = None
        

    def set_target_opponent_policy_name(self, policy_name):
        self.target_opponent_policy_name = policy_name

    # TODO: This works fine for identical agents but different agents will not work as they won't have the same action spaec
    # Solution: environment of Pred will have a modified step function that will call the parent environment step and then take what observation it is related to  
    # Compute actions for the opponent agent in the environment (Note: that the action for )
    # This is only be called for the opponent agent
    # TODO: This should be renamed -> compute_opponent_action or opponent_compute_policy-> Change them in PredPrey1v1.py
    def compute_action(self, obs): # the policy for the opponent
        if self.opponent_policy is None:
            return self.action_space.sample() # return a random action
        else:
            action = None
            deterministic = None # False through the training procedure, and True during the evaluation
            if(self._name in ["Training", "Evaluation"]):
                deterministic = False
            else:   # For testing
                deterministic = True
            # deterministic = True
            # if(isinstance(self.opponent_policy, sb3SAC)):
            #     deterministic = False
            if(isinstance(self.opponent_policy, sb3PPO) or isinstance(self.opponent_policy, sb3SAC)):
                # For determinisitic flag: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
                # print(f"Opponent (Training?{'Training' in self._name}): {deterministic}")
                action, self.states = self.opponent_policy.predict(obs, state=self.states, deterministic=deterministic) #it is predict because this is PPO from stable-baselines not rllib
            # if(isinstance(self.opponent_policy, rllibPPO)):
                # action, _ = self.opponent_policy.compute_action(obs) #it is predict because this is PPO from stable-baselines not rllib
            return action

    def _load_opponent(self, opponent_name):
        # print(f"Wants to load {opponent_name}")
        # Prevent reloading the same policy again or reload empty policy -> empty policy means random policy
        if opponent_name is not None:
            if("Training" in self._name):
                clilog.debug(f"Add frequency +1 for {opponent_name}")
                self.archive.add_freq(opponent_name, 1) 
        
            # To prevent the time for reloading it and it is already loaded
            if(opponent_name != self.opponent_policy_name):
                self.opponent_policy_name = opponent_name
                if self.opponent_policy is not None:
                    del self.opponent_policy
                # if(isinstance(self.opponent_algorithm_class, sb3PPO) or isinstance(super(self.opponent_algorithm_class), sb3PPO)):
                if(not self.OS):
                    self.opponent_policy = self.archive.load(name=opponent_name, env=self, algorithm_class=self.opponent_algorithm_class) # here we load the opponent policy
                if(self.OS):
                    self.opponent_policy = self.opponent_algorithm_class.load(opponent_name, env=self) # here we load the opponent policy
                clilog.debug(f"loading opponent model: {opponent_name}, {self.opponent_policy}, {self}")


    def reset(self):
        self.states = None
        # if sample_after_reset flag is set then we need to sample from the archive here
        if(self.sample_after_reset):
            clilog.debug("Sample after reset the environment")

            opponent_selection = self.sampling_parameters["opponent_selection"]
            sample_path = self.sampling_parameters["sample_path"]
            startswith_keyword = "history"
            randomly_reseed_sampling = self.sampling_parameters["randomly_reseed_sampling"]

            sampled_opponent = None
            if(not self.OS):
                # print("Not OS")
                archive = self.archive.get_sorted(opponent_selection) # this return [sorted_names, sorted_policies]
                models_names = archive[0]
                sampled_opponent = utsmpl.sample_opponents(models_names, 1, selection=opponent_selection, sorted=True, randomly_reseed=randomly_reseed_sampling, delta=self.archive.delta, idx=self.reset_counter)[0]
            if(self.OS):
                sampled_opponent = utsmpl.sample_opponents_os(sample_path, startswith_keyword, 1, selection=opponent_selection, randomly_reseed=randomly_reseed_sampling, delta=self.archive.delta, idx=self.reset_counter)[0]
            self.target_opponent_policy_name = sampled_opponent
        
        if(self.OS):
            clilog.debug(f"Reset, env name: {self._name}, OS, target_policy: {self.target_opponent_policy_name} ({str(self.opponent_algorithm_class)})")
        # if(not self.OS):
        else:
            clilog.debug(f"Reset, env name: {self._name}, archive_id: {self.archive.random_id}, target_policy: {self.target_opponent_policy_name} ({str(self.opponent_algorithm_class)})")
        
        self._load_opponent(self.target_opponent_policy_name)
        self.reset_counter += 1
