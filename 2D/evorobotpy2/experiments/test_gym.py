import gym_predprey
import gym
import time
import os
import sys
import importlib
filename = os.path.join(os.environ["EVOROBOTPY_PREDPREY_BIN"], "ErPredprey")
sys.path.append(os.path.dirname(filename))
mname = os.path.splitext(os.path.basename(filename))[0]
imported = importlib.import_module(mname)                       
sys.path.pop()

filename = os.path.join(os.environ["EVOROBOTPY_PREDPREY_BIN"], "ErPredprey.ini")
sys.path.append(os.path.dirname(filename))
mname = os.path.splitext(os.path.basename(filename))[0]
imported = importlib.import_module(mname)                       
sys.path.pop()

# filename = os.path.join(os.environ["EVOROBOTPY_PREDPREY_BIN"], "khepera-cylinder.sample")
# sys.path.append(os.path.dirname(filename))
# mname = os.path.splitext(os.path.basename(filename))[0]
# imported = importlib.import_module(mname)                       
# sys.path.pop()

# filename = os.path.join(os.environ["EVOROBOTPY_PREDPREY_BIN"], "khepera-light.sample")
# sys.path.append(os.path.dirname(filename))
# mname = os.path.splitext(os.path.basename(filename))[0]
# imported = importlib.import_module(mname)                       
# sys.path.pop()

# filename = os.path.join(os.environ["EVOROBOTPY_PREDPREY_BIN"], "khepera-wall.sample")
# sys.path.append(os.path.dirname(filename))
# mname = os.path.splitext(os.path.basename(filename))[0]
# imported = importlib.import_module(mname)                       
# sys.path.pop()


# filename = os.path.join(os.environ["EVOROBOTPY_PREDPREY_BIN"], "khepera-scylinder.sample")
# sys.path.append(os.path.dirname(filename))
# mname = os.path.splitext(os.path.basename(filename))[0]
# imported = importlib.import_module(mname)                       
# sys.path.pop()

# filename = os.path.join(os.environ["EVOROBOTPY_PREDPREY_BIN"], "marxbot-cylinder.sample")
# sys.path.append(os.path.dirname(filename))
# mname = os.path.splitext(os.path.basename(filename))[0]
# imported = importlib.import_module(mname)                       
# sys.path.pop()

# filename = os.path.join(os.environ["EVOROBOTPY_PREDPREY_BIN"], "marxbot-scylinder.sample")
# sys.path.append(os.path.dirname(filename))
# mname = os.path.splitext(os.path.basename(filename))[0]
# imported = importlib.import_module(mname)                       
# sys.path.pop()

# filename = os.path.join(os.environ["EVOROBOTPY_PREDPREY_BIN"], "marxbot-wall.sample")
# sys.path.append(os.path.dirname(filename))
# mname = os.path.splitext(os.path.basename(filename))[0]
# imported = importlib.import_module(mname)                       
# sys.path.pop()

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
