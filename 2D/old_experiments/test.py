import gym
import gym_predprey
from time import sleep

from gym_predprey.envs.PredPrey1v1 import Behavior

env = gym.make('PredPrey-Superior-1v1-v0')

# behavior = Behavior(**{"amplitude":0.5, "freq":15})
# env.reinit(prey_behavior=behavior.cos_1D)

behavior = Behavior()
env.reinit(prey_behavior=behavior.fixed_prey)


env.reset()
for _ in range (1000):
    action = env.action_space.sample()
    # action[0] = [0,0]
    # action[1] = 1
    # action[2] = 1
    # action[3] = 1
    observation, reward, done, info = env.step(action)
    print(observation.shape)
    # print(reward)
    env.render()
    sleep(0.01)
    # print(done)
    if ((isinstance(done, dict) and done["__all__"]) or (isinstance(done, bool) and done)):
        break
env.close()