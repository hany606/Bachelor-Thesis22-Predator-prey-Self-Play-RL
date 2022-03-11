# These environment are  single agent envs while the others agents are preloaded polices
from pettingzoo.mpe import simple_tag_v2

import numpy as np
from copy import deepcopy

import gym
from gym import spaces
from gym.utils import seeding
from torch import seed

class Behavior: # For only prey for now, we need to make it configured for the predator also :TODO:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fixed_prey(self, action, time, observation):
        if(isinstance(action, dict)):
            action["agent_0"] = [0, 0, 0, 0, 0]
        else:
            action[5:] = [0, 0, 0, 0, 0]
        return action

    def fixed_pred(self, action, time, observation):
        if(isinstance(action, dict)):
            action["adversary_0"] = [0, 0, 0, 0, 0]
        else:
            action[:5] = [0, 0, 0, 0, 0]
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
                        reward_type="normal",
                        caught_distance=0.001):
        # adversary_0 -> predator, "agent_0" -> prey
        self.agent_keys = ["adversary_0", "agent_0"]
        self.nrobots = len(self.agent_keys)
        self.num_obstacles = 3
        self.env = simple_tag_v2.parallel_env(num_good=1, num_adversaries=1, num_obstacles=self.num_obstacles, max_cycles=max_num_steps, continuous_actions=True)


        self.seed(seed_val)

        # [no_action, move_left, move_right, move_down, move_up]
        self.noutputs = self.env.action_space("adversary_0").shape[0]   # for single agent
        # TODO: low, high can be written like: action = np.array([action, [0,0]]).flatten()
        low = []
        high = []
        for i in range(self.nrobots):
            low.extend([0 for i in range(self.noutputs)])
            high.extend([1 for i in range(self.noutputs)])
        self.action_space_      = spaces.Box(low=np.array(low),
                                            high=np.array(high),
                                            dtype=np.float32)
        self.action_space = deepcopy(self.action_space_)

        low = []
        high = []
        # [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]
        self.ninputs = self.env.observation_space("adversary_0").shape[0]    # for single agent
        for i in range(self.nrobots):
            low.extend([-np.inf for i in range(self.ninputs)])
            high.extend([np.inf for i in range(self.ninputs)])
        self.observation_space_      = spaces.Box(low=np.array(low),
                                            high=np.array(high),
                                            dtype=np.float32)
        self.observation_space = deepcopy(self.observation_space_)

        self.caught_distance = caught_distance
        self.max_num_steps = max_num_steps
        self.pred_behavior = pred_behavior
        self.prey_behavior = prey_behavior
        self.pred_policy = pred_policy
        self.prey_policy = prey_policy
        self.reward_type = reward_type
        self._set_env_parameters()
        self.caught = False
        self.steps_done = False
        self.observation = None

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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        print(f"Seed: {seed}")
        self.env.seed(seed)
        return [seed]

    def reset(self):
        obs = self.env.reset()
        self.num_steps = 0
        self.observation, self.whole_observation = self._process_observation(obs)
        return self.observation

    def _get_agent_observation(self, obs):
        return obs

    def _get_opponent_observation(self, obs):
        raise NotImplementedError("_get_opponent_observation() Not implemented")
    
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

        # Actions are amplified
        ac = [a for a in ac]
        action_dict = {self.agent_keys[i]:np.array(ac[self.noutputs*i:self.noutputs*(i+1)]) for i in range(self.nrobots)}
        return action_dict
        
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
            obs_list.extend(obs[self.agent_keys[i]]) 
        # Originally:
        # adversary_0: (2)self_vel, (2)self_pos, (2*other agents)agent_i_rel_position, (2*other agents)agent_i_vel
        # agent_0: (2)self_vel, (2)self_pos, (2*other_adversaries)adversary_i_rel_position
        # However, we will add the adversary_rel_vel to the agent_0
        # TODO: refactor it and remove the hardcoded part
        obs_list.extend(obs["adversary_0"][:2])
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
        prey_reward, predator_reward = None, None
        # Survival rewards
        prey_reward = 1
        predator_reward = -1

        if(self.caught):   # if the predator caught the prey before finishing the time
            prey_reward = -10
            predator_reward = 10
        if(self.steps_done):
            prey_reward = 10
            predator_reward = -10

        self._pred_reward, self._prey_reward = predator_reward, prey_reward # to be used for the info
        return predator_reward, prey_reward

    def _process_done(self, obs, done_dict, reward_dict):
        # As I am not sure about what is their termination criteria as it seems it only related with time
        self.caught = True if reward_dict["adversary_0"] > 0 else False #self._compute_caught(obs)
        self.steps_done = self.num_steps >= self.max_num_steps
        done = True if self.caught or self.steps_done else False
        return done

    def who_won(self):
        if(self.caught):
            return "pred"
        if(self.steps_done):
            return "prey"
        return ""
    
    def _process_info(self):
        return {"win":self.who_won(), "reward": (self._pred_reward, self._prey_reward), "Num. steps": self.num_steps}


    def step(self, action):
        self.num_steps += 1
        action_dict = self._process_action(action, self.whole_observation)
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)
        obs, whole_obs = self._process_observation(obs_dict)
        self.obs = obs
        self.whole_observation = whole_obs

        done = self._process_done(obs, done_dict, reward_dict)
        reward = self._process_reward(obs, action, reward_dict)
        info = self._process_info()
        if(done):
            print(info)
        return obs, reward, done, info

    def render(self, mode='human'):
        self.env.render()

# Single-agent environment for the predator
class PZPredPreyPred(PZPredPrey):
    def __init__(self, **kwargs):
        PZPredPrey.__init__(self, **kwargs)
        self.action_space      = spaces.Box(low=self.action_space_.low[:self.noutputs],
                                            high=self.action_space_.high[:self.noutputs],
                                            dtype=np.float32)
        # Split the observation space
        self.observation_space = spaces.Box(low=self.observation_space_.low[:self.ninputs],
                                            high=self.observation_space_.high[:self.ninputs],
                                            dtype=np.float32)
        # print(self.action_space)
        # print(self.observation_space)

    def _process_action(self, action, observation):
        if(self.prey_behavior is None and self.prey_policy is None):
            raise ValueError("prey_behavior or prey_policy should be specified")

        action = np.array([action, [0 for _ in range(self.noutputs)]]).flatten()
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
        self.observation_space = spaces.Box(low=self.observation_space_.low[self.ninputs:],
                                            high=self.observation_space_.high[self.ninputs:],
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
        return observation[self.ninputs:]

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

if __name__ == '__main__':
    import gym
    from time import sleep

    # env = PZPredPrey()
    # # exit()
    # observation = env.reset()
    # done = False
    # total_reward = 0
    # # for i in range(1000):
    # while not done:
    #     # actions = {agent: env.action_space(agent).sample() for agent in env.env.agents}
    #     # actions = env.action_space.sample()
    #     actions = np.zeros(2*5)
    #     # actions = {'adversary_0': np.array([0, 1, 1, 0, 0 ],
    #     # dtype=np.float32), 'agent_0': np.array([1 , 0, 0, 0, 0 ],
    #     # dtype=np.float32)}

    #     # print(actions)
    #     observation, reward, done, info = env.step(actions)
    #     print(env.num_steps)
    #     env.render()
    #     # sleep(0.001)
    
    # env = PZPredPreyPred(seed_val=3)
    # behavior = Behavior()
    # env.reinit(prey_behavior=behavior.fixed_prey)

    # observation = env.reset()
    # done = False

    # while not done:
    #     # action = {0: np.array([0.5, 0, 0.6]), 1: np.array([0, 0, 0])}#env.action_space.sample()
    #     action = env.action_space.sample()
    #     # action = [0,0,1]
    #     # action = [-observation[0]+observation[6],-observation[1]+observation[7],-observation[2]+observation[8]]
    #     print(f"Actions: {action}")
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
    
    env = PZPredPreyPrey(seed_val=3)
    behavior = Behavior()
    env.reinit(pred_behavior=behavior.fixed_pred)

    observation = env.reset()
    print(observation.shape)
    exit()
    done = False

    while not done:
        # action = {0: np.array([0.5, 0, 0.6]), 1: np.array([0, 0, 0])}#env.action_space.sample()
        action = env.action_space.sample()
        # action = [0,0,1]
        # action = [-observation[0]+observation[6],-observation[1]+observation[7],-observation[2]+observation[8]]
        print(f"Actions: {action}")
        # action[0] = [0,0]
        # action[1] = 1
        # action[2] = 1
        # action[3] = 1
        observation, reward, done, info = env.step(action)
        print(observation.shape)
        # print(reward, info, done)
        env.render()
        # sleep(0.01)
        # print(done)
        # if ((isinstance(done, dict) and done["__all__"]) or (isinstance(done, bool) and done)):
        #     break
    env.close()
    
