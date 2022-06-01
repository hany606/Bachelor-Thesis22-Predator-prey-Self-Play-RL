# These environment are  single agent envs while the others agents are preloaded polices
from bach_utils.logger import get_logger
clilog = get_logger()

from pettingzoo.mpe import simple_tag_v2

import numpy as np
from copy import deepcopy

import gym
from gym import spaces
from gym.utils import seeding
from torch import seed
import time
from math import copysign

class Behavior: # For only prey for now, we need to make it configured for the predator also :TODO:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fixed_prey(self, action, time, observation):
        if(isinstance(action, dict)):
            action["agent_0"] = [0, 0, 0, 0, 0]
        else:
            # action[5:] = [0, 0, 0, 0, 0]
            action[2:] = [0,0]#[0, 0, 0, 0, 0]

        return action

    def fixed_pred(self, action, time, observation):
        if(isinstance(action, dict)):
            action["adversary_0"] = [0, 0, 0, 0, 0]
        else:
            # action[:5] = [0, 0, 0, 0, 0]
            action[:2] = [0,0]#[0, 0, 0, 0, 0]

        return action

# TODO: custom class Scenario from scenarios/simple_tag.py for enabling the feature for fixed positions
# TODO: custom class raw_env from simple_tag_v2.py 
# TODO: make the adversary faster -> Sol. decrease the actions of the prey


# TODO: make the code works for more than adversary -> 
#       Either make all the adversary as one environment
#       Or make each adversary as a different environment 

# TODO: Refactor the code and make all things internally in dictionaries
# Here they have joined everything in terms of lists not dictionary as one single agent that controls all the agents
class PZPredPrey(gym.Env):
    def __init__(self,  max_num_steps=1000, 
                        pred_behavior=None, 
                        prey_behavior=None, 
                        pred_policy=None, 
                        prey_policy=None, 
                        seed_val=3, 
                        reward_type=None,
                        caught_distance=0.001,
                        gui=False,  # for compatibility with other envs
                        reseed=True,
                        specific_pos=
                        {
                                        "adversary_0": np.array([-0.82870167, -0.52637899]),
                                        "agent_0": np.array([0.60254893, 0]),
                                        "landmark":[
                                                    np.array([-0.73056844, -0.12037151]),
                                                    np.array([-0.03770766, -0.61246995]),
                                                    np.array([ 0.42223887, -0.69539036])
                                    ]
                        }
                ):
        # adversary_0 -> predator, "agent_0" -> prey
        self.agent_keys = ["adversary_0", "agent_0"]
        self.nrobots = len(self.agent_keys)
        self.num_obstacles = 3
        # Fixed pose from seed=3
        # adversary_0 [-0.82870167 -0.52637899]
        # agent_0 [0.60254893 0.16432407]
        # Landmark 0: [-0.73056844 -0.12037151]
        # Landmark 1: [-0.03770766 -0.61246995]
        # Landmark 2: [ 0.42223887 -0.69539036]
        self.specific_pos = specific_pos


        self.env = simple_tag_v2.parallel_env(num_good=1, num_adversaries=1, num_obstacles=self.num_obstacles, max_cycles=max_num_steps, continuous_actions=True, specific_pos=self.specific_pos)

        self.seed_val = seed_val
        self.reseed = reseed
        self.seed_val = self.seed(seed_val)[0]

        # [no_action, move_left, move_right, move_up, move_down]
        # (x,y)
        self.noutputs = 2#self.env.action_space("adversary_0").shape[0]   # for single agent
        # TODO: low, high can be written like: action = np.array([action, [0,0]]).flatten()
        low = []
        high = []
        for i in range(self.nrobots):
            # low.extend([0 for i in range(self.noutputs)])
            low.extend([-1 for i in range(self.noutputs)])

            high.extend([1 for i in range(self.noutputs)])
        self.action_space_      = spaces.Box(low=np.array(low),
                                            high=np.array(high),
                                            dtype=np.float32)
        self.action_space = deepcopy(self.action_space_)

        low = []
        high = []
        
        # positive, negative
        rel_pos_limits = 4#[4,4]
        rel_vel_limits = 1#[1,1]
        # [ (limit_x, limit_y) ]
        self.normalized_obs_limits = [
                                            0.5,0.5,  # self vel
                                            2,2,      # self pos
        ]
        self.normalized_obs_limits.extend([rel_pos_limits for _ in range(self.num_obstacles*2)]) # Landmarks rel pos
        self.normalized_obs_limits.extend([rel_pos_limits for _ in range(1*2)]) # Other agents rel pos
        self.normalized_obs_limits.extend([rel_vel_limits for _ in range(1*2)]) # Other agents rel vel
        self.normalized_obs_limits.append(1) # Time

        #   2           2           #*2                     #*2                     #*2
        # [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]
        self.ninputs = self.env.observation_space("adversary_0").shape[0]+1    # for single agent
        for _ in range(self.nrobots):
            # low.extend([-np.float32(np.inf) for i in range(self.ninputs)])
            # high.extend([np.float32(np.inf) for i in range(self.ninputs)])
            low.extend([-1 for i in range(self.ninputs)])
            high.extend([1 for i in range(self.ninputs)])

        self.observation_space_      = spaces.Box(low=np.array(low),
                                            high=np.array(high),
                                            dtype=np.float32)
        self.observation_space = deepcopy(self.observation_space_)

        self.caught_distance = caught_distance
        self.max_num_steps = max_num_steps
        self.pred_behavior = pred_behavior
        self.prey_behavior = prey_behavior
        self.pred_policy = pred_policy #-> it is actually "self"
        self.prey_policy = prey_policy
        self.reward_type = "normal" if reward_type is None else reward_type
        self._set_env_parameters()
        self.caught = False
        self.steps_done = False
        self.observation = None
        self._posx_lim = [-1.8,1.8]
        self._posy_lim = [-1.8,1.8]



    def _set_env_parameters(self):
        self.num_steps = 0
        self.caught = False
        self.steps_done = False
        self._pred_reward = None
        self._prey_reward = None


    def reinit(self, max_num_steps=1000, pred_behavior=None, prey_behavior=None):
        self.max_num_steps = max_num_steps
        self.prey_behavior = prey_behavior
        self.pred_behavior = pred_behavior

    def set_seed(self, seed_val):
        if(not self.reseed):
            self.seed_val = seed_val

    def seed(self, seed_val=None):
        self.np_random, seed_val = seeding.np_random(seed_val)
        clilog.debug(f"Seed (env): {self.seed_val}")
        # self.env.seed(seed)
        # This is due to some problems, I do not know the reason that it make seed when it is not called
        clilog.warn(f"Warn: if you want to seed with different value, change seed_value of env first")
        print("Reset the env with seed function")
        self.env.reset(seed=self.seed_val, specific_pos=self.specific_pos)
        # self.reset()
        return [self.seed_val]

    def reset(self):
        # self.env.seed(self.seed_val)
        obs = None
        if(self.reseed):
            clilog.debug(f"Reseed env with the initial seed: {self.seed_val}")
            obs = self.env.reset(seed=self.seed_val, specific_pos=self.specific_pos)
        else:
            obs = self.env.reset(specific_pos=self.specific_pos)
        if(self.specific_pos is not None):
            clilog.debug(f"Initialize the env with specific positions")
        self.num_steps = 0
        self.observation, self.whole_observation = self._process_observation(obs)
        return self.observation

    def _get_agent_observation(self, obs):
        return obs

    def _get_opponent_observation(self, obs):
        raise NotImplementedError("_get_opponent_observation() Not implemented")
    
    # Transform action to the original env actions
    # From (x,y) to (stop, right, left, up, down)
    def _transform_action(self, a):
        new_a = [0,0,0,0,0]
        idx_map = [(1,2), (3,4)]
        for i in range(2):
            idx = None
            if(int(copysign(1,a[i])) > 0):  # positive
                idx = 0
            else:
                idx = 1
            new_a[idx_map[i][idx]] = abs(a[i])
        return new_a


    def _process_action(self, action, observation):
        """
        Change the actions generated by the policy (List) to the base (PettingZoo) environment datatype (Dict)
        ----------
        Parameters
        ----------
        action : ndarray or list
            Action from the policy
        observation: list
            Observations to be used by other agents to infer their actions as this environment is a single agent env while the others agents are preloaded polices
        ----------
        Returns
        -------
        dict[string, ndarray]
        """
        # return action
        ac = deepcopy(action)
        # Fill the actions of the other agents
        # action already passed with the selection of the actions for the agent (main agent) of the environment
        #   For example, for pred_env -> the main is the predator and the opponenet is the prey and vice versa
        # Contruct the actions for the opponent
        if(self.prey_behavior is not None):
            ac = self.prey_behavior(ac, self.num_steps, observation)
            # print(ac)
        if(self.pred_behavior is not None):
            ac = self.pred_behavior(ac, self.num_steps, observation)
            # print(ac)

        if(self.pred_policy is not None):
            # print("policy pred")
            # Changed the observation input
            ac[:self.noutputs] = self.pred_policy.compute_action(self._get_opponent_observation(observation))
            # ac[:self.noutputs] = self.pred_policy.compute_action(self.ob)
            # ac[:self.noutputs] = self.pred_policy.compute_action(self.ob[:self.ninputs])
            # ac[:self.noutputs] = self.pred_policy.compute_action(self.ob[self.ninputs:])

        if(self.prey_policy is not None):
            # print("policy prey")
            # Changed the observation input
            ac[self.noutputs:] = self.prey_policy.compute_action(self._get_opponent_observation(observation))
            # ac[self.noutputs:] = self.prey_policy.compute_action(self.ob)  # The agent gets the full observations
            # ac[self.noutputs:] = self.prey_policy.compute_action(self.ob[self.ninputs:]) # The agent gets its own observations
            # ac[self.noutputs:] = self.prey_policy.compute_action(self.ob[:self.ninputs]) # The agent gets the opponent observations

        ac = [a for a in ac]
        action_dict = {self.agent_keys[i]:np.array(self._transform_action(ac[self.noutputs*i:self.noutputs*(i+1)]), dtype=np.float32) for i in range(self.nrobots)}
        # Actions are amplified
        # Divide the speed of the adversary (predator) by 2 -> to slow it down
        # for i in range(len(action_dict[self.agent_keys[0]])):
        #     action_dict[self.agent_keys[0]][i] *= 0.8
        return action_dict
        

    def _normalize_obs(self, obs):
        def normalize(o, mn, mx):
            # https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
            return 2*(o - mn)/(mx-mn) - 1
    
        normalized_obs = []
        for i,o in enumerate(obs):
            # print(self.normalized_obs_limits[i])
            mn = -self.normalized_obs_limits[i]
            mx = -mn
            normalized_obs.append(normalize(o, mn, mx))
        return np.array(normalized_obs)

            
    def _process_observation(self, obs):
        """
        Change from PZ environment's observations (dict) to list of observations
        ----------
        Parameters:
        ----------
            obs: dict[string, ndarray]
        ----------
        Returns:
        ----------
            obs_list: ndarray or list
        ----------
        """
        obs_list = []
        for i in range(self.nrobots):
            num_steps = self.num_steps/self.max_num_steps
            extended_obs = [num_steps]
            if(i == self.nrobots-1):
                extended_obs = [0 for _ in range(2)]
                extended_obs.append(num_steps)
            tmp_obs = np.append(obs[self.agent_keys[i]], extended_obs)
            normalized_obs = self._normalize_obs(tmp_obs)
            obs_list.extend(normalized_obs)
        # Originally:
        # adversary_0: (2)self_vel, (2)self_pos, (2*other agents)agent_i_rel_position, (2*other agents)agent_i_vel
        # agent_0: (2)self_vel, (2)self_pos, (2*other_adversaries)adversary_i_rel_position
        # However, we will add the adversary_rel_vel to the agent_0
        # TODO: refactor it and remove the hardcoded part
        # obs_list.extend(obs["adversary_0"][:2])
        # obs_list.insert(self.ninputs+1, self.num_steps)
        # obs_list.append(self.num_steps)
        # New:
        # adversary_0: (2)self_vel, (2)self_pos, (2*other agents)agent_i_rel_position, (2*other agents)agent_i_vel
        # agent_0: (2)self_vel, (2)self_pos, (2*other_adversaries)adversary_i_rel_position, (2*other other_adversaries)adversary_i_vel
        ret_obs = np.array(obs_list, dtype=np.float32).flatten()
        return (ret_obs, ret_obs)

    # def _compute_relative_distance(self, obs):
    #     pos0 = np.array(obs[2:4])
    #     pos1 = np.array(obs[2+self.ninputs:2+self.ninputs+2])
    #     dist = np.linalg.norm(pos0 - pos1)
    #     return dist

    # def _compute_caught(self, obs):
    #     """
    #     ----------
    #     Parameters:
    #         obs: list
    #     ----------
    #     ----------
    #     Returns:
    #     ----------
    #         caught: bool -> indicate if the predator caught the prey
    #     ----------
    #     """
    #     # obs is a list
    #     dist = self._compute_relative_distance(obs)
    #     print(dist)
    #     if(dist <= self.caught_distance):
    #         return True
    #     return False

    def _process_reward(self, obs, action, reward_dict):
        # # OLD when the boundaries where not made and the bounds where compensated in the reward function
        # #       This was not giving good trainings I do not know why
        # # To take the advantage of the bound reward that is already implemented
        # prey_reward, predator_reward = reward_dict["agent_0"], reward_dict["adversary_0"]
        # # Survival rewards
        # prey_reward += 1
        # predator_reward += -1
        # # The increment to take into account the boundaries violations
        # if(self.caught):   # if the predator caught the prey before finishing the time
        #     prey_reward += -10
        #     predator_reward = 10    # predator it does not matter if go out of the boundary or not in case of catching
        # if(self.steps_done):
        #     prey_reward += 10   # if the prey got out of circle he will get positive and will not caught in immediate step
        #     predator_reward += -10
        # --------------------------------------------------

        prey_reward, predator_reward = reward_dict["agent_0"], reward_dict["adversary_0"]
        # prey_reward, predator_reward = 0, 0
        # delta_pos = obs[self.num_obstacles*2+4:self.num_obstacles*2+6] #np.array(obs[2:4]) - np.array(obs[self.num_obstacles*2+4:self.num_obstacles*2+6])
        # dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist = 0
        timestep_reward = 3*self.num_steps/self.max_num_steps
        prey_reward += 1 + timestep_reward + dist
        predator_reward += -1 - timestep_reward - dist
        if(self.caught):   # if the predator caught the prey before finishing the time
            prey_reward = -1000
            predator_reward = 1000    # predator it does not matter if go out of the boundary or not in case of catching
        if(self.steps_done):
            prey_reward = 1000   # if the prey got out of circle he will get positive and will not caught in immediate step
            predator_reward = -1000

        self._pred_reward, self._prey_reward = predator_reward, prey_reward # to be used for the info
        return predator_reward, prey_reward
        # prey_reward, predator_reward = None, None
        # # Survival rewards
        # prey_reward = 1
        # predator_reward = -1

        # if(self.caught):   # if the predator caught the prey before finishing the time
        #     prey_reward = -10
        #     predator_reward = 10
        # if(self.steps_done):
        #     prey_reward = 10
        #     predator_reward = -10

        # self._pred_reward, self._prey_reward = predator_reward, prey_reward # to be used for the info
        # return predator_reward, prey_reward

    # Assuming there is only one opponent (1v1)
    def _compute_caught(self, obs):
        delta_pos = obs[self.num_obstacles*2+4:self.num_obstacles*2+6] #np.array(obs[2:4]) - np.array(obs[self.num_obstacles*2+4:self.num_obstacles*2+6])
        # print(obs[self.num_obstacles*2+4:self.num_obstacles*2+6])
        # print(self.env.state().shape)
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # dist_min = agent1.size + agent2.size
        dist_min = 0.04
        # print(f"Dist: {dist}")
        return True if dist < dist_min else False

    def _process_done(self, obs, done_dict, reward_dict):
        # As I am not sure about what is their termination criteria as it seems it only related with time
        # print(self._compute_caught(obs))
        
        self.caught = self._compute_caught(obs) #True if reward_dict["adversary_0"] > 1 else False
        self.steps_done = self.num_steps >= self.max_num_steps
        done = True if self.caught or self.steps_done else False
        return done

    def who_won(self):
        if(self.caught):
            return "pred"
        if(self.steps_done):
            return "prey"
        return ""
    
    def _process_info(self, obs_dict):
        # TODO: clean that part later
        self.pred_pos = obs_dict[self.agent_keys[0]][2:4]
        self.prey_pos = obs_dict[self.agent_keys[1]][2:4]

        return {"win":self.who_won(), "reward": (self._pred_reward, self._prey_reward), "num_steps": self.num_steps, "pred_pos":self.pred_pos, "prey_pos":self.prey_pos}

    def step(self, action):
        self.num_steps += 1
        action_dict = self._process_action(action, self.whole_observation)
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)
        obs, whole_obs = self._process_observation(obs_dict)
        self.obs = obs
        self.whole_observation = whole_obs

        done = self._process_done(whole_obs, done_dict, reward_dict)
        reward = self._process_reward(obs, action, reward_dict)
        info = self._process_info(obs_dict)
        if(done):
            # self.render()
            # input("Input!!!!")
            clilog.debug(info)
        # self.render()
        return obs, reward, done, info

    def render(self, mode='human', extra_info=None):
        extra_info = f"{self.num_steps}" if(extra_info is None) else f"{self.num_steps}, "+extra_info
        self.env.render(mode, extra_info)

    def close(self):
        time.sleep(0.3)
        self.env.close()

# Single-agent environment for the predator
class PZPredPreyPred(PZPredPrey):
    def __init__(self, **kwargs):
        PZPredPrey.__init__(self, **kwargs)
        self.action_space      = spaces.Box(low=self.action_space_.low[:self.noutputs],
                                            high=self.action_space_.high[:self.noutputs],
                                            dtype=np.float32)
        
        # Split the observation space
        # self.ninputs += 1
        self.observation_space = spaces.Box(low=self.observation_space_.low[:self.ninputs],
                                            high=self.observation_space_.high[:self.ninputs],
                                            dtype=np.float32)
        # print(self.action_space)
        # print(self.observation_space)

    def _process_action(self, action, observation):
        if(self.prey_behavior is None and self.prey_policy is None):
            raise ValueError("prey_behavior or prey_policy should be specified")

        action = np.array([action, [0 for _ in range(self.noutputs)]], dtype=np.float32).flatten()
        return PZPredPrey._process_action(self, action, observation)

    def _process_observation(self, observation):
        # Get a subset of the whole observations (Only one agent observations)
        # _process_observation changes the dictionary to list
        obs,_ = PZPredPrey._process_observation(self, observation)  # this function change from dictionary to list
        # Return agent observations and the whole observations
        return (self._get_agent_observation(obs), obs)

    def _get_agent_observation(self, observation):
        return observation[:self.ninputs]

    def _get_opponent_observation(self, observation):
        return observation[self.ninputs:]

    def who_won(self):
        if(self.caught):
            return 1
        if(self.steps_done):
            return -1
        return 0

    def _process_reward(self, obs, action, reward_dict):
        predator_reward, prey_reward = PZPredPrey._process_reward(self, obs, action, reward_dict)
        return predator_reward

# Single-agent environment for the prey
class PZPredPreyPrey(PZPredPrey):
    def __init__(self, **kwargs):
        PZPredPrey.__init__(self, **kwargs)
        self.action_space      = spaces.Box(low=self.action_space_.low[self.noutputs:],
                                            high=self.action_space_.high[self.noutputs:],
                                            dtype=np.float32)
        # Split the observation space
        # self.ninputs += 1
        self._ninputs = self.ninputs #12
        self.observation_space = spaces.Box(low=self.observation_space_.low[self.ninputs:self.ninputs+self._ninputs],
                                            high=self.observation_space_.high[self.ninputs:self.ninputs+self._ninputs],
                                            dtype=np.float32)
        # print(self.action_space)
        # print(self.observation_space)

    def _process_action(self, action, observation):
        if(self.pred_behavior is None and self.pred_policy is None):
            raise ValueError("pred_behavior or pred_policy should be specified")
        action = np.array([[0 for _ in range(self.noutputs)], action]).flatten()
        return PZPredPrey._process_action(self, action, observation)

    def _process_observation(self, observation):
        # Get a subset of the whole observations (Only one agent observations)
        # _process_observation changes the dictionary to list
        obs,_ = PZPredPrey._process_observation(self, observation) # this function change from dictionary to list
        return (self._get_agent_observation(obs), obs)

    def _get_agent_observation(self, observation):
        return observation[self.ninputs:self.ninputs+self._ninputs]

    def _get_opponent_observation(self, observation):
        return observation[:self.ninputs]

    def who_won(self):
        if(self.caught):
            return -1
        if(self.steps_done):
            return 1
        return 0

    def _process_reward(self, obs, action, reward_dict):
        predator_reward, prey_reward = PZPredPrey._process_reward(self, obs, action, reward_dict)
        return prey_reward

def print_obs(obs, n_landmarks):
    # print(obs)
    print(f"Self vel: {obs[0:2]}")
    print(f"Self pos: {obs[2:4]}")
    print(f"Landmark rel pos: {obs[4:4+n_landmarks*2]}")
    # Assuming there is only one agent more
    print(4+n_landmarks*2,4+n_landmarks*2+2)
    print(f"Other agents rel pos: {obs[4+n_landmarks*2:4+n_landmarks*2+2]}")
    print(f"Other agents rel vel: {obs[4+n_landmarks*2+2:4+n_landmarks*2+4]}")
    print(f"Time: {obs[4+n_landmarks*2+4:]}")

if __name__ == '__main__':
    import gym
    from time import sleep

    # env = PZPredPrey()
    # # # exit()
    # observation = env.reset()
    # done = False
    # total_reward = 0
    # # for i in range(1000):
    # while not done:
    #     # actions = {agent: env.action_space(agent).sample() for agent in env.env.agents}
    #     actions = env.action_space.sample()
    #     # actions = np.zeros(2*5)
    #     # actions = {'adversary_0': np.array([0, 1, 1, 0, 0 ],
    #     # dtype=np.float32), 'agent_0': np.array([1 , 0, 0, 0, 0 ],
    #     # dtype=np.float32)}

    #     # print(actions)
    #     observation, reward, done, info = env.step(actions)
    #     # print(env.num_steps)
    #     # print(observation)
    #     env.render()
    #     # sleep(0.001)
    
    # env = PZPredPreyPred(seed_val=3)
    # behavior = Behavior()
    # env.reinit(prey_behavior=behavior.fixed_prey)

    # observation = env.reset()
    # done = False

    # while not done:
    #     # action = {0: np.array([0.5, 0, 0.6]), 1: np.array([0, 0, 0])}#env.action_space.sample()
    #     # action = env.action_space.sample()
    #     action = [0,0,0,0,0]
    #     # action = [0,0,1]
    #     # action = [-observation[0]+observation[6],-observation[1]+observation[7],-observation[2]+observation[8]]
    #     # print(f"Actions: {action}")
    #     print_obs(observation, 3)

    #     # action[0] = [0,0]
    #     # action[1] = 1
    #     # action[2] = 1
    #     # action[3] = 1
    #     observation, reward, done, info = env.step(action)
    #     # print(observation)
    #     # print(reward, info, done)
    #     env.render()
    #     # sleep(0.01)
    #     # print(done)
    #     # if ((isinstance(done, dict) and done["__all__"]) or (isinstance(done, bool) and done)):
    #     #     break
    # env.close()
    from matplotlib import pyplot as plt
    env = PZPredPreyPrey(seed_val=3)
    behavior = Behavior()
    env.reinit(pred_behavior=behavior.fixed_pred)
    for i in range(1):
        observation = env.reset()
        # print(env.observation_space.shape)
        # print(observation.shape)
        # exit()
        done = False
        rewards = []
        while not done:
        # for i in range(500):
            # action = {0: np.array([0.5, 0, 0.6]), 1: np.array([0, 0, 0])}#env.action_space.sample()
            # action = env.action_space.sample()
            # print(action)
            # pos, rel, .., .., ..
            # action = observation[10:12]   # P controller
            action = [-0.5,0]
            # print_obs(observation)
            # action = [-0,+0.1]#[0,0,-0.3,0,0]
            # action = [0,0,1]
            # action = [-observation[0]+observation[6],-observation[1]+observation[7],-observation[2]+observation[8]]
            # print(f"Actions: {action}")
            # action[0] = [0,0]
            # action[1] = 1
            # action[2] = 1
            # action[3] = 1
            observation, reward, done, info = env.step(action)
            # print_obs(observation, 3)
            print(env.prey_pos)
            rewards.append(info["reward"][1])
            # print(observation.shape)
            # print(info)
            # print(reward, info, done)
            env.render(extra_info="test")
            sleep(0.01)
            # print(done)
            # if ((isinstance(done, dict) and done["__all__"]) or (isinstance(done, bool) and done)):
            #     break
        env.close()
    print(f"Sum: {sum(rewards)}")
    print(f"Max: {max(rewards)}")
    print(f"Min: {min(rewards)}")
    plt.plot(rewards)
    plt.show()
    


# self Vel 0.5,0.5
# self Pos 2,2
# rel pos 4
# rel 