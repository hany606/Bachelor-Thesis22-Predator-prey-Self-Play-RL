import gym
import gym_predprey

env = gym.make('PredPrey-1v1-v0')
env.reset()
for _ in range (1000):
    # observation, reward, done, info = env.step(env.actions_space.sample())
    # print(observation)
    # print(reward)
    env.render()
    # if (done):
    #     break
env.close()