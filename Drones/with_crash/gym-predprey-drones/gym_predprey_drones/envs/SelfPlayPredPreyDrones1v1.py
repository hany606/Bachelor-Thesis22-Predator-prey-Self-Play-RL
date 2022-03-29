# Copy file from hany606/gym-predprey
import os
from gym_predprey_drones.envs.PredPreyDrones1v1 import PredPrey1v1PredDrone
from gym_predprey_drones.envs.PredPreyDrones1v1 import PredPrey1v1PreyDrone
from stable_baselines3.ppo.ppo import PPO as sb3PPO
from stable_baselines3.sac.sac import SAC as sb3SAC

import bach_utils.os as utos
import bach_utils.list as utlst
import bach_utils.sorting as utsrt
import bach_utils.sampling as utsmpl

OS = False#True

# Parent class for all using SB3 functions to predict
class SelfPlayEnvSB3:
    def __init__(self, algorithm_class, archive, sample_after_reset, sampling_parameters):
        self.algorithm_class = algorithm_class  # algorithm class for the opponent
        self.opponent_policy = None             # The policy itself after it is loaded
        self.opponent_policy_name = None    # Current loaded policy name -> File -> as it was implement first to be stored on disk (But now in cache=archive) 
        self.target_opponent_policy_name = None
        self._name = None
        self.archive = archive  # opponent archive
        self.OS = OS
        self.sample_after_reset = sample_after_reset
        self.sampling_parameters = sampling_parameters

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
    def compute_action(self, obs): # the policy
        if self.opponent_policy is None:
            return self.action_space.sample() # return a random action
        else:
            action = None
            if(isinstance(self.opponent_policy, sb3PPO) or isinstance(self.opponent_policy, sb3SAC)):
                action, self.states = self.opponent_policy.predict(obs, state=self.states) #it is predict because this is PPO from stable-baselines not rllib
            # if(isinstance(self.opponent_policy, rllibPPO)):
                # action, _ = self.opponent_policy.compute_action(obs) #it is predict because this is PPO from stable-baselines not rllib

            return action

    def _load_opponent(self, opponent_name):
        # print(f"Wants to load {opponent_name}")
        # Prevent reloading the same policy again or reload empty policy -> empty policy means random policy
        if opponent_name is not None and opponent_name != self.opponent_policy_name:
            # print("loading model: ", opponent_name)
            self.opponent_policy_name = opponent_name
            if self.opponent_policy is not None:
                del self.opponent_policy
            # if(isinstance(self.algorithm_class, sb3PPO) or isinstance(super(self.algorithm_class), sb3PPO)):
            if(not self.OS):
                print(self.algorithm_class)
                self.opponent_policy = self.archive.load(name=opponent_name, env=self, algorithm_class=self.algorithm_class) # here we load the opponent policy
            if(self.OS):
                self.opponent_policy = self.algorithm_class.load(opponent_name, env=self) # here we load the opponent policy


    def reset(self):
        self.states = None
        # if sample_after_reset flag is set then we need to sample from the archive here
        if(self.sample_after_reset):
            print("Sample after reset the environment")

            opponent_selection = self.sampling_parameters["opponent_selection"]
            sample_path = self.sampling_parameters["sample_path"]
            startswith_keyword = "history"
            randomly_reseed_sampling = self.sampling_parameters["randomly_reseed_sampling"]

            sampled_opponent = None
            if(not self.OS):
                # print("Not OS")
                archive = self.archive.get_sorted(opponent_selection) # this return [sorted_names, sorted_policies]
                models_names = archive[0]
                sampled_opponent = utsmpl.sample_opponents(models_names, 1, selection=opponent_selection, sorted=True, randomly_reseed=randomly_reseed_sampling)[0]
            if(self.OS):
                sampled_opponent = utsmpl.sample_opponents_os(sample_path, startswith_keyword, 1, selection=opponent_selection, randomly_reseed=randomly_reseed_sampling)[0]
            self.target_opponent_policy_name = sampled_opponent
        
        if(self.OS):
            print(f"Reset, env name: {self._name}, OS, target_policy: {self.target_opponent_policy_name}")
        # if(not self.OS):
        else:
            print(f"Reset, env name: {self._name}, archive_id: {self.archive.random_id}, target_policy: {self.target_opponent_policy_name}")
        self._load_opponent(self.target_opponent_policy_name)

class SelfPlayPredDroneEnv(SelfPlayEnvSB3, PredPrey1v1PredDrone):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, *args, **kwargs):
        seed_val = kwargs.pop('seed_val')
        gui = kwargs.pop('gui', False)
        reward_type = kwargs.pop('reward_type', 'normal')
        SelfPlayEnvSB3.__init__(self, *args, **kwargs)  # env_opponent_name="prey"
        PredPrey1v1PredDrone.__init__(self, seed_val=seed_val, gui=gui, reward_type=reward_type)
        self.prey_policy = self # It replaces the policy for the other agent with the best policy that found during reset (This class have it)

    # Change to search only for the prey
    def reset(self):
        SelfPlayEnvSB3.reset(self)
        return PredPrey1v1PredDrone.reset(self)


class SelfPlayPreyDroneEnv(SelfPlayEnvSB3, PredPrey1v1PreyDrone):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, *args, **kwargs):
        seed_val = kwargs.pop('seed_val')
        gui = kwargs.pop('gui', False)
        reward_type = kwargs.pop('reward_type', 'normal')
        SelfPlayEnvSB3.__init__(self, *args, **kwargs)  # env_opponent_name="pred"
        PredPrey1v1PreyDrone.__init__(self, seed_val=seed_val, gui=gui, reward_type=reward_type)
        self.pred_policy = self # It replaces the policy for the other agent with the best policy that found during reset (This class have it)

    # Change to search only for the prey
    def reset(self):
        SelfPlayEnvSB3.reset(self)
        return PredPrey1v1PreyDrone.reset(self)