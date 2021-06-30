# Note: this environment is based on https://github.com/snolfi/evorobotpy2
# Resources:
# https://medium.com/@vermashresth/craft-and-solve-multi-agent-problems-using-rllib-and-tensorforce-a3bd1bb6f556

import numpy as np
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from gym_predprey.envs import renderWorld
from copy import deepcopy
import os

# TODO:
# [X] Make action space
# [X] Make observation space
# [X] Modify the observations from cpp file
# [X] Modify the action space from cpp file
# [ ] Parameterize the number of robots
# [ ] Reward function for 1v1 behavior of predetor and prey robots (Make it similar to OpenAI)
# [X] Done criteria function for maximum number of steps
# [ ] (Make it work for scripts out of this directory) Make the directory as it is made in tensegrity gym to solve problems in importing the files
# [ ] Answer this question: what is dt for the system?
# [ ] Parameterize the initial positions for the robots
# [ ] Make dictionary (As in multi-agent env) for the actions: red is predetor and green is prey

class PredPrey(gym.Env, MultiAgentEnv):
    def __init__(self, max_num_steps=1000):
        # ErProblem = __import__("/home/hany606/repos/research/Drones-PEG-Bachelor-Thesis-2022/2D/evorobotpy2/gym-predprey/gym_predprey/envs/ErPredprey")
        # ErProblem = __import__(os.path.join(os.environ["EVOROBOTPY_PREDPREY_BI"], "ErPredprey"))
        ErProblem = __import__("ErPredprey")
        self.env = ErProblem.PyErProblem()
        self.ninputs = self.env.ninputs               # only works for problems with continuous observation space
        self.noutputs = self.env.noutputs             # only works for problems with continuous observation space
        self.nrobots = 2

        self.ob = np.arange(self.ninputs * self.nrobots, dtype=np.float32)  # allocate observation vector
        self.ac = np.arange(self.noutputs * self.nrobots, dtype=np.float32) # allocate action vector
        self.done = np.arange(1, dtype=np.int32) # allocate a done vector
        self.objs = np.arange(1000, dtype=np.float64) 
        self.objs[0] = -1

        self.env.copyObs(self.ob)                     # pass the pointer to the observation vector to the Er environment
        self.env.copyAct(self.ac)                     # pass the pointer to the action vector to the Er environment
        self.env.copyDone(self.done)                  # pass the pointer to the done vector to the Er environment    
        self.env.copyDobj(self.objs)

        self.num_steps = 0

        self.nhiddens = 25   # number of hiddens
        self.nlayers = 1     # number of hidden layers 
        self.nact = np.arange((self.ninputs + (self.nhiddens * self.nlayers) + self.noutputs) * self.nrobots, dtype=np.float64)


        # x,y for each robot
        low = np.array([0, 0])
        high = np.array([self.env.worldx, self.env.worldy])
        # self.observation_space = gym.spaces.Dict({i:gym.spaces.Box(low=low, high=high, dtype=np.float32) for i in range(self.nrobots)})
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # two motors for each robot
        low = np.array([self.env.low, self.env.low])
        high = np.array([self.env.high, self.env.high])
        # self.action_space = gym.spaces.Dict({i:gym.spaces.Box(low=low, high=high, dtype=np.float32) for i in range(self.nrobots)})
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.max_num_steps = max_num_steps
        self.caught = False

    def reset(self):
        self.env.reset()
        self.num_steps = 0
        self.caught = False
        return self._get_observation()

    def step(self, action):
        self.num_steps += 1
        self.ac = self._preprocess_action(action)
        self.env.step()
        obs = self._get_observation()
        reward = self._compute_reward(obs)
        done  = self._compute_done(obs)
        info = self._compute_info()
        return obs, reward, done, info
    
    def _preprocess_action(self, action):
        ac = deepcopy(action) # copy the actions
        ac[1] = np.array([0,0], dtype=np.float32) # fixed - no motion
        # ac[1] = np.array([np.cos(self.num_steps/20), np.cos(self.num_steps/20)], dtype=np.float32) # periodic motion
        return np.array([ac[0], ac[1]]).reshape((self.nrobots*2,))

    def _compute_info(self):
        return {i: {} for i in range(self.nrobots)}


    # Adapted from: https://github.com/openai/multiagent-competition/blob/master/gym-compete/gym_compete/new_envs/you_shall_not_pass.py
    # Two teams: predetor and prey
    # Prey needs to run away and the predetor needs to catch it
    # Rewards:
    #   If predetor caught the prey:
    #       It is finished and predetor gets +1000 and prey -1000
    #   If the predetor did not catch the prey:
    #       The prey gets +10 and the predetor -10
    #   if the episode finished the prey get +1000 and predetor -1000

    # OpenAI human blocker reward function
    #     Some Walker reaches end:
    #         walker which did touchdown: +1000
    #         all blockers: -1000
    #     No Walker reaches end:
    #         all walkers: -1000
    #         if blocker is standing:
    #             blocker gets +1000
    #         else:
    #             blocker gets 0
    def _compute_reward(self, obs):
        # if the predetor(pursuer) caught the prey(evader) then the predetor takes good reward and done
        # if the predetor couldn't catch the prey then it will take negative reward
        prey_reward = 10
        predetor_reward = -10
        dist = np.linalg.norm(obs[0] - obs[1])
        eps = 200
        # print(f"distance: {dist}")
        prey_reward = +dist
        predetor_reward = -dist
        if (dist < eps):
            self.caught = True
        #     prey_reward = -1000
        #     predetor_reward = 1000
        # if(self.num_steps > self.max_num_steps):
        #     prey_reward = 1000
        #     predetor_reward = -1000
        # "predetor": 0 -> red, "prey": 1 -> green
        return {0:predetor_reward, 1:prey_reward}

    def _compute_done(self, obs):
        bool_val = True if(self.caught or self.num_steps > self.max_num_steps) else False
        done = {i: bool_val for i in range(self.nrobots)}
        done["__all__"] = True if True in done.values() else False
        return done
    
    def _get_observation(self):
        # it should work but it make some weird behavior:
        # observation = np.array([[self.ob[i*3+1], self.ob[i*3+2]] for i in range(self.nrobots)]).reshape(self.nrobots*2)
        #ob = deepcopy(self.ob)[:4]
        # return deepcopy(self.ob)
        # Regrdless this weird way for putting the values in another variable to return it, but this is the one that worked somehow, others made some weird behavior for the system
        ob = deepcopy(self.ob)
        observation = np.empty((self.nrobots*2))
        for i in range(self.nrobots):
            observation[i*2] = ob[i*3+1]
            observation[i*2+1] = ob[i*3+2]
        obs = {i:observation[i*2:i*2+2] for i in range(self.nrobots)}
        # obs = {i:np.array([-1,-1]) for i in range(self.nrobots)} # for testing the assertion
        # print(obs)

        # if not self.observation_space.contains(obs):
        #     raise Exception("The provided action is out of allowed space.")
        return obs

    def render(self):
        self.env.render()
        info = f'Step: {self.num_steps}'
        renderWorld.update(self.objs, info, self.ob, self.ac, self.nact)

class PredPreySingle(gym.Env):
    def __init__(self, max_num_steps=1000):
        # ErProblem = __import__("/home/hany606/repos/research/Drones-PEG-Bachelor-Thesis-2022/2D/evorobotpy2/gym-predprey/gym_predprey/envs/ErPredprey")
        # ErProblem = __import__(os.path.join(os.environ["EVOROBOTPY_PREDPREY_BI"], "ErPredprey"))
        ErProblem = __import__("ErPredprey")
        self.env = ErProblem.PyErProblem()
        self.ninputs = self.env.ninputs               # only works for problems with continuous observation space
        self.noutputs = self.env.noutputs             # only works for problems with continuous observation space
        self.nrobots = 2

        self.ob = np.arange(self.ninputs * self.nrobots, dtype=np.float32)  # allocate observation vector
        self.ac = np.arange(self.noutputs * self.nrobots, dtype=np.float32) # allocate action vector
        self.done = np.arange(1, dtype=np.int32) # allocate a done vector
        self.objs = np.arange(1000, dtype=np.float64) 
        self.objs[0] = -1

        self.env.copyObs(self.ob)                     # pass the pointer to the observation vector to the Er environment
        self.env.copyAct(self.ac)                     # pass the pointer to the action vector to the Er environment
        self.env.copyDone(self.done)                  # pass the pointer to the done vector to the Er environment    
        self.env.copyDobj(self.objs)

        self.num_steps = 0

        self.nhiddens = 25   # number of hiddens
        self.nlayers = 1     # number of hidden layers 
        self.nact = np.arange((self.ninputs + (self.nhiddens * self.nlayers) + self.noutputs) * self.nrobots, dtype=np.float64)


        # x,y for each robot
        low = np.array([0, 0])
        high = np.array([self.env.worldx, self.env.worldy])
        # self.observation_space = gym.spaces.Dict({i:gym.spaces.Box(low=low, high=high, dtype=np.float32) for i in range(self.nrobots)})
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # two motors for each robot
        low = np.array([self.env.low, self.env.low])
        high = np.array([self.env.high, self.env.high])
        # self.action_space = gym.spaces.Dict({i:gym.spaces.Box(low=low, high=high, dtype=np.float32) for i in range(self.nrobots)})
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.max_num_steps = max_num_steps
        self.caught = False

    def reset(self):
        self.env.reset()
        self.num_steps = 0
        self.caught = False
        return self._get_observation()

    def step(self, action):
        self.num_steps += 1
        self.ac = self._preprocess_action(action)
        self.env.step()
        obs = self._get_observation()
        reward = self._compute_reward(obs)
        done  = self._compute_done(obs)
        info = self._compute_info()
        return obs, reward, done, info
    
    def _preprocess_action(self, action):
        ac = deepcopy(action) # copy the actions
        ac1 = np.array([0,0], dtype=np.float32) # fixed - no motion
        # ac[1] = np.array([np.cos(self.num_steps/20), np.cos(self.num_steps/20)], dtype=np.float32) # periodic motion
        return np.array([ac, ac1]).reshape((self.nrobots*2,))
        return ac
        # ac[1] = np.array([0,0], dtype=np.float32) # fixed - no motion
        # # ac[1] = np.array([np.cos(self.num_steps/20), np.cos(self.num_steps/20)], dtype=np.float32) # periodic motion
        # return np.array([ac[0], ac[1]]).reshape((self.nrobots*2,))

    def _compute_info(self):
        # return {i: {} for i in range(self.nrobots)}
        return {}


    # Adapted from: https://github.com/openai/multiagent-competition/blob/master/gym-compete/gym_compete/new_envs/you_shall_not_pass.py
    # Two teams: predetor and prey
    # Prey needs to run away and the predetor needs to catch it
    # Rewards:
    #   If predetor caught the prey:
    #       It is finished and predetor gets +1000 and prey -1000
    #   If the predetor did not catch the prey:
    #       The prey gets +10 and the predetor -10
    #   if the episode finished the prey get +1000 and predetor -1000

    # OpenAI human blocker reward function
    #     Some Walker reaches end:
    #         walker which did touchdown: +1000
    #         all blockers: -1000
    #     No Walker reaches end:
    #         all walkers: -1000
    #         if blocker is standing:
    #             blocker gets +1000
    #         else:
    #             blocker gets 0
    def _compute_reward(self, obs):
        # if the predetor(pursuer) caught the prey(evader) then the predetor takes good reward and done
        # if the predetor couldn't catch the prey then it will take negative reward
        prey_reward = 10
        predetor_reward = -10
        dist = np.linalg.norm(obs[:2] - obs[2:])
        eps = 200
        # print(f"distance: {dist}")
        prey_reward = +dist
        predetor_reward = -dist
        if (dist < eps):
            self.caught = True
        #     prey_reward = -1000
        #     predetor_reward = 1000
        # if(self.num_steps > self.max_num_steps):
        #     prey_reward = 1000
        #     predetor_reward = -1000
        # "predetor": 0 -> red, "prey": 1 -> green
        # return {0:predetor_reward, 1:prey_reward}
        return predetor_reward

    def _compute_done(self, obs):
        bool_val = True if(self.caught or self.num_steps > self.max_num_steps) else False
        # done = {i: bool_val for i in range(self.nrobots)}
        # done["__all__"] = True if True in done.values() else False
        # return done
        return bool_val
    
    def _get_observation(self):
        # it should work but it make some weird behavior:
        # observation = np.array([[self.ob[i*3+1], self.ob[i*3+2]] for i in range(self.nrobots)]).reshape(self.nrobots*2)
        #ob = deepcopy(self.ob)[:4]
        # return deepcopy(self.ob)
        # Regrdless this weird way for putting the values in another variable to return it, but this is the one that worked somehow, others made some weird behavior for the system
        ob = deepcopy(self.ob)
        observation = np.empty((self.nrobots*2))
        for i in range(self.nrobots):
            observation[i*2] = ob[i*3+1]
            observation[i*2+1] = ob[i*3+2]
        # obs = {i:observation[i*2:i*2+2] for i in range(self.nrobots)}
        # obs = {i:np.array([-1,-1]) for i in range(self.nrobots)} # for testing the assertion
        # print(obs)

        # if not self.observation_space.contains(obs):
        #     raise Exception("The provided action is out of allowed space.")

        # return obs
        return observation

    def render(self):
        self.env.render()
        info = f'Step: {self.num_steps}'
        renderWorld.update(self.objs, info, self.ob, self.ac, self.nact)



if __name__ == '__main__':
    import gym_predprey
    import gym
    import time
    import os
    # env = gym.make('gym_predprey:predprey-v0')
    # # print(f"Action space: {env.action_space.shape}\nObservation space: {env.observation_space.shape}")
    # obs = env.reset()
    # print(obs)
    # done = {"__all__": False}
    # reward = 0
    # # for i in range(100):
    # while not (True in done.values()):
    #     time.sleep(0.1)
    #     action_prey = np.zeros(env.noutputs , dtype=np.float32)#Policy(obs) #4 np.random.randn(4)#
    #     action_pred = env.action_space.sample()
    #     action = {0: action_pred, 1: action_prey}
    #     # print(action)
    #     # action[0] = np.zeros((2,),dtype=np.float32)
    #     # action[1] = np.zeros((2,),dtype=np.float32)

    #     # action[0] = 0.5#np.random.rand()*2 - 1
    #     # action[1] = np.random.rand()*2 - 1
    #     # action[2] = np.random.rand()*2 - 1
    #     # action[3] = 0.5#np.random.rand()*2 - 1
    #     # print(action)
    #     # print(action.shape, np.zeros(env.noutputs * env.nrobots, dtype=np.float32).shape)
    #     obs, r, done, _ = env.step(action)
    #     reward += r[0]
    #     # print(done)
    #     # print(f"Reward: {reward}")
    #     # print(obs[0]) # why just printing destroys everything
    #     # print(type(obs))
    #     env.render()
    # obs = env.reset()
    # for i in range(100):
    # # while not (True in done.values()):
    #     time.sleep(0.1)
    #     action_prey = env.action_space.sample()#np.zeros(env.noutputs * env.nrobots, dtype=np.float32)#Policy(obs) #4 np.random.randn(4)#
    #     action_pred = env.action_space.sample()
    #     # print(action_prey)
    #     action = {0: action_pred, 1: action_prey}
    #     # print(action)
    #     # action[0] = np.zeros((2,),dtype=np.float32)
    #     # action[1] = np.zeros((2,),dtype=np.float32)

    #     # action[0] = 0.5#np.random.rand()*2 - 1
    #     # action[1] = np.random.rand()*2 - 1
    #     # action[2] = np.random.rand()*2 - 1
    #     # action[3] = 0.5#np.random.rand()*2 - 1
    #     # print(action)
    #     # print(action.shape, np.zeros(env.noutputs * env.nrobots, dtype=np.float32).shape)
    #     obs, r, done, _ = env.step(action)
    #     reward += r[0]
    #     print(done)
    #     # print(f"Reward: {reward}")
    #     # print(obs)
    #     # print(type(obs))
    #     env.render()
    # ------------------------------------------------------
    env = gym.make('gym_predprey:predpreysingle-v0')
    obs = env.reset()
    # print(obs)
    # done = {"__all__": False}
    done = False
    reward = 0
    # for i in range(100):
    # while not (True in done.values()):
    while not done:
        time.sleep(0.1)
        action_prey = np.zeros(env.noutputs , dtype=np.float32)#Policy(obs) #4 np.random.randn(4)#
        action_pred = env.action_space.sample()
        # action = {0: action_pred, 1: action_prey}
        action = action_pred
        # print(action)
        # action[0] = np.zeros((2,),dtype=np.float32)
        # action[1] = np.zeros((2,),dtype=np.float32)

        # action[0] = 0.5#np.random.rand()*2 - 1
        # action[1] = np.random.rand()*2 - 1
        # action[2] = np.random.rand()*2 - 1
        # action[3] = 0.5#np.random.rand()*2 - 1
        # print(action)
        # print(action.shape, np.zeros(env.noutputs * env.nrobots, dtype=np.float32).shape)
        obs, r, done, _ = env.step(action)
        # reward += r[0]
        print(obs)
        # print(done)
        # print(f"Reward: {reward}")
        # print(obs[0]) # why just printing destroys everything
        # print(type(obs))
        env.render()
    