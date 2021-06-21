# Note: this environment is based on https://github.com/snolfi/evorobotpy2
import numpy as np
import gym

from gym_predprey.envs import renderWorld
from copy import deepcopy

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
class PredPrey(gym.Env):
    def __init__(self, max_num_steps=1000):
        # ErProblem = __import__("/home/hany606/repos/research/Drones-PEG-Bachelor-Thesis-2022/2D/evorobotpy2/gym-predprey/gym_predprey/envs/ErPredprey")
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

        self.max_num_steps = max_num_steps
        self.caught = False

    def reset(self):
        self.env.reset()
        return self._get_observation()

    def step(self, action):
        self.num_steps += 1
        self.ac = deepcopy(action) # copy the actions
        self.env.step()
        obs = self._get_observation()
        reward = self._compute_reward(obs)
        done  = self._compute_done(obs)
        info = {}
        return obs, reward, done, info

    def _compute_reward(self, obs):
        # if the predetor(pursuer) caught the prey(evader) then the predetor takes good reward and done
        # if the predetor couldn't catch the prey then it will take negative reward
        dist = np.linalg.norm(obs[:2] - obs[2:])
        eps = 200
        print(f"distance: {dist}")
        if (dist < eps):
            self.caught = True
            return 100
        return -1

    def _compute_done(self, obs):
        if(self.caught or self.num_steps >= self.max_num_steps):
            return 1
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
    done = False
    # for i in range(100):
    while not done:
        time.sleep(0.1)
        action = np.zeros(env.noutputs * env.nrobots, dtype=np.float32)#Policy(obs) #4 np.random.randn(4)#
        action[0] = 0.5
        action[1] = -1
        action[2] = np.random.rand()
        action[3] = np.random.rand()
        print(action)
        # print(action.shape, np.zeros(env.noutputs * env.nrobots, dtype=np.float32).shape)
        obs, reward, done, _ = env.step(action)
        # print(obs)
        # print(type(obs))
        env.render()