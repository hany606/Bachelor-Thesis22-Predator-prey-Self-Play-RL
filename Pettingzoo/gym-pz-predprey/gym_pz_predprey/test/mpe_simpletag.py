# from pettingzoo.mpe import simple_tag_v2

# # def sample_action(env):

# env = simple_tag_v2.env()
# env.reset()
# # print(env.action_space().sample_)
# print(env.action_space("adversary_0"))
# for agent in env.agent_iter():
#     print(agent)
#     observation, reward, done, info = env.last()
#     # action = {"adversary_0":0, "adversary_1":0, "adversary_2":0, "agent_0":0}
#     action = 0#[0,0,0,0,0]
#     if(done):
#         action = None
#     env.step(action)
#     env.render()
# #     print(env.action_space)
#     # action
#     # action = env.action_space.sample(observation)#policy(observation, agent)
#     # env.step(action)

from pettingzoo.mpe import simple_tag_v2
from time import sleep
import numpy as np

parallel_env = simple_tag_v2.parallel_env(num_good=1, num_adversaries=2, num_obstacles=0, max_cycles=1000, continuous_actions=True)

observations = parallel_env.reset()
max_cycles = 500
dones = {"a":False}
# for step in range(max_cycles):
while not all(dones.values()):
    # actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
    actions = { 'adversary_0': np.array([0, 1, 0, 0, 0 ], dtype=np.float32),
                'adversary_1': np.array([0, 0, 0, 0, 0 ], dtype=np.float32),
                'agent_0': np.array([0 , 0, 0, 0, 0 ], dtype=np.float32)}

      # [self_vel, self_pos, other_agent_rel_positions, other_agent_velocities]
      # Zero for all (noone is moving)
      # {'adversary_0': array([ 0.        ,  0.        , -0.03663022,  0.6247733 , -0.38995707,
      #  -0.09468317,  0.39243072, -0.609876  ,  0.        , -0.        ], dtype=float32), 
      # 'adversary_1': array([-0.        , -0.        , -0.42658728,  0.53009015,  0.38995707,
      #   0.09468317,  0.7823878 , -0.5151928 ,  0.        , -0.        ],
      # dtype=float32), 
      # 'agent_0': array([ 0.        , -0.        ,  0.35580048,  0.01489731, -0.39243072,
      #   0.609876  , -0.7823878 ,  0.5151928 ], dtype=float32)}

      # Only agent_0 is moving
      # {'adversary_0': array([-0.        ,  0.        ,  0.3240594 , -0.5181404 ,  0.12626518,
      #  -0.35424265,  0.42426148,  1.3634    ,  0.9192388 ,  0.9192388 ],
      # dtype=float32), 
      # 'adversary_1': array([ 0.        , -0.        ,  0.45032457, -0.872383  , -0.12626518,
      #   0.35424265,  0.29799628,  1.7176427 ,  0.9192388 ,  0.9192388 ],
      # dtype=float32), 
      # 'agent_0': array([ 0.9192388 ,  0.9192388 ,  0.7483209 ,  0.8452596 , -0.42426148,
      #  -1.3634    , -0.29799628, -1.7176427 ], dtype=float32)}

      # adversary_0 is moving
      # {'adversary_0': array([ 0.525     , -0.        ,  0.18808459, -0.8391163 ,  0.27561897,
      #   1.0883422 ,  0.57984304,  0.04107413,  0.        ,  0.        ],
      # dtype=float32), 
      # 'adversary_1': array([ 0.        ,  0.        ,  0.46370357,  0.24922593, -0.27561897,
      #  -1.0883422 ,  0.30422407, -1.0472682 ,  0.        ,  0.        ],
      # dtype=float32), 
      # 'agent_0': array([ 0.        ,  0.        ,  0.76792765, -0.7980422 , -0.57984304,
      #  -0.04107413, -0.30422407,  1.0472682 ], dtype=float32)}
    # print(actions)
    observations, rewards, dones, infos = parallel_env.step(actions)
    print(observations)
    # print(observations, rewards, dones, infos)
    parallel_env.render()
    sleep(0.1)