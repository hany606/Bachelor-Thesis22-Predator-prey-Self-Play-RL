# Note: this environment is based on https://github.com/snolfi/evorobotpy2
import numpy as np
import gym

import renderWorld
from copy import deepcopy

# TODO:
# [] Make action space
# [] Make observation space
# [] Modify the observations from cpp file
# [] Modify the action space from cpp file

class PredPrey(gym.Env):
    def __init__(self):
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
        low = np.array([[0, 0] for i in range(self.nrobots)]).reshape(self.nrobots*2)
        high = np.array([[self.env.worldx, self.env.worldy] for i in range(self.nrobots)]).reshape(self.nrobots*2)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # two motors for each robot
        low = np.array([self.env.low for _ in range(2*self.nrobots)])
        high = np.array([self.env.high for _ in range(2*self.nrobots)])
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        self.env.reset()

    def step(self, action):
        self.num_steps += 1
        self.ac = action # copy the actions
        self.env.step()
        obs = self._get_observation()
        reward = self._compute_reward(obs)
        done  = self._compute_done(obs)
        info = None
        return obs, reward, done, info

    def _compute_reward(self, obs):
        return -1

    def _compute_done(self, obs):
        return 0
    
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
        return observation
    def render(self):
        self.env.render()
        info = f'Step: {self.num_steps}'
        renderWorld.update(self.objs, info, self.ob, self.ac, self.nact)


if __name__ == '__main__':
    import gym_predprey
    import gym
    import time
    env = gym.make('gym_predprey:predprey-v0')
    # print(f"Action space: {env.action_space.shape}\nObservation space: {env.observation_space.shape}")
    obs = env.reset()
    for i in range(100):
        time.sleep(0.1)
        action = np.zeros(env.noutputs * env.nrobots, dtype=np.float32)#Policy(obs) #4
        action[0] = 0.5
        action[1] = -1
        obs, reward, done, _ = env.step(action)
        print(obs)
        print(type(obs))
        env.render()