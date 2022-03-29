# File to train a single agent of drone
from copy import deepcopy

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import BaseSingleAgentAviary, ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync
import pybullet as p
import os
from gym_pybullet_drones.utils.Logger import Logger
import time

# Note: 0:pred, 1:prey

ACTION2D = False#True

def draw_point(point, size=0.1, **kwargs):
    lines = []
    for i in range(len(point)):
        axis = np.zeros(len(point))
        axis[i] = 1.0
        p1 = np.array(point) - size/2 * axis
        p2 = np.array(point) + size/2 * axis
        lines.append(add_line(p1, p2, **kwargs))
    return lines

def add_line(start, end, color=[0,0,0], width=1, lifetime=None, parent=-1, parent_link=-1):
    assert (len(start) == 3) and (len(end) == 3)
    return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width,
                              parentObjectUniqueId=parent, parentLinkIndex=parent_link)


class Behavior: # For only prey for now, we need to make it configured for the predator also :TODO:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fixed_prey(self, action, time, observation):
        if(isinstance(action, dict)):
            action[1] = [0, 0, 0]
        else:
            action[3:] = [0, 0, 0]
        return action

    def fixed_pred(self, action, time, observation):
        if(isinstance(action, dict)):
            action[0] = [0, 0, 0]
        else:
            action[:3] = [0, 0, 0]
        return action

    def cos_1D(self, action, time, observation):
        freq = 5#self.kwargs["freq"]
        amplitude = 1#self.kwargs["amplitude"]
        if(isinstance(action, dict)):
            action[1] = [0, 0]
        else:
            sin_wave = amplitude*np.cos(time/freq)
            action[2:] = [sin_wave, sin_wave]
        return action

class MotionPrimitive:
    def __init__(self, n_steps=5, max_val=0.5):
        self.n_steps = n_steps
        self.max_val = max_val

        self.directions = {
                            0:[0,0,0],
                            1:[+1,0,0], # +x
                            2:[-1,0,0], # -x
                            3:[0,+1,0], # +y
                            4:[0,-1,0], # -y
                            5:[0,0,+1], # +z
                            6:[0,0,-1], # -z
                          }
        self.num_motion_primitives = len(self.directions.keys())

    
    def compute_motion(self, idx):
        return np.array(self.directions[idx],dtype=np.float32)*self.max_val

    def stop(self):
        return self.compute_motion(0)

    def pos_x(self):
        return self.compute_motion(1)
    def neg_x(self):
        return self.compute_motion(2)

    def pos_y(self):
        return self.compute_motion(3)
    def neg_y(self):
        return self.compute_motion(4)

    def pos_z(self):
        return self.compute_motion(5)
    def neg_z(self):
        return self.compute_motion(6)
#  0.0003473364494069103

# This is a wrapper above the mutli-agent drones environment 
class _DroneReach(BaseSingleAgentAviary):
    def __init__(self,
                 caught_distance=0.01, #0.015
                 max_num_steps=1000,
                 crashing_max_angle=np.pi/4,
                 seed_val=45, 
                 reward_type="reach",
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.PID,
                 logger=False,
                 ):
        if initial_xyzs is None:
            initial_xyzs = np.vstack((
                                    np.zeros((1,3)),
                                    ))

        initial_xyzs = np.array([[0,-0.5,0.2]])
        BaseSingleAgentAviary.__init__(  self, 
                                        drone_model=drone_model,
                                        initial_xyzs=initial_xyzs,
                                        initial_rpys=initial_rpys,
                                        physics=physics,
                                        freq=freq,
                                        aggregate_phy_steps=aggregate_phy_steps,
                                        gui=gui,
                                        record=record, 
                                        obs=obs,
                                        act=act
                                     )

        # exit()
        # p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+self.URDF,
        #                                       self.INIT_XYZS[i,:],
        #                                       p.getQuaternionFromEuler(self.INIT_RPYS[i,:]),
        #                                       flags = p.URDF_USE_INERTIA_FROM_FILE,
        #                                       physicsClientId=self.CLIENT
        #                                       ) for i in range(self.NUM_DRONES)]
        self.reward_type = reward_type if reward_type is not None else "normal" # treating None value as normal reward type
        self.seed(seed_val)

        # Convert from dictionary action/observation spaces made by the environment to the list -> to be compatible with SB3 work that we have used
        # print(self.observation_space)
        # print(self.action_space)
        # Note: action_space is amplified inside "process_action" to make the motion faster
        self.nrobots = 1
        if(ACTION2D):
            self.noutputs = 2#self.action_space[0].shape[0]   # for single drone
            low = []
            high = []
            a = self.action_space
            # X and Y only 2D action space
            low.extend([-1 for i in range(2)])
            high.extend([1 for i in range(2)])
            self.action_space      = spaces.Box(low=np.array(low),
                                                high=np.array(high),
                                                dtype=np.float32)
        else:
            self.noutputs = self.action_space.shape[0]   # for single drone
            low = []
            high = []
            a = self.action_space
            low.extend([l for l in a.low])
            high.extend([h for h in a.high])
            self.action_space      = spaces.Box(low=np.array(low),
                                                high=np.array(high),
                                                dtype=np.float32)

        #### KIN Observation vector ### X Y Z Q1 Q2 Q3 Q4 R P Y VX VY VZ WX WY WZ P0 P1 P2 P3 (Without Q1,Q2,Q3,Q4, P0,P1,P2,P3 -> quaternions and rpms)
        # The observations from gym_pybullet_drones is being masked to remove unnecessary 
        # In the mask 1 means -> remove
        # self.observation_mask = [0,0,0, 1,1,1, 0,0,0, 1,1,1] # Mask for the observation for a single drone in order to get rid of the unnecessary observations
        self.observation_mask = [0,0,0, 0,0,0, 0,0,0, 0,0,0] # Mask for the observation for a single drone in order to get rid of the unnecessary observations

        # TODO: Improve the following by making it more numpy way
        low = []
        high = []
        o = self.observation_space
        l = self.masking_observations([l for l in o.low])
        h = self.masking_observations([h for h in o.high])
        low.extend(list(l))
        high.extend(list(h))
        # For both drones together
        self.observation_space = spaces.Box(low=np.array(low),
                                            high=np.array(high),
                                            dtype=np.float32)
        
        self.ninputs = len(self.observation_mask)-sum(self.observation_mask) #self.observation_space[0].shape[0]    # for single drone

        # print(self.observation_space)
        # print(self.action_space)
        # exit()
        # self.motion = MotionPrimitive()
        # self.action_space = spaces.MultiDiscrete([self.motion.num_motion_primitives for _ in range(self.nrobots)])

        self.max_num_steps = max_num_steps
        self._set_env_parameters()
        self.caught_distance = caught_distance
        self.crashing_max_angle = crashing_max_angle
        self.observation = None # This is created to keep track the previous observations to feed for the opponent policy when selecting an action
        self.logger = Logger(logging_freq_hz=int(self.SIM_FREQ/self.AGGR_PHY_STEPS), num_drones=self.nrobots) if logger else None

        # [0,-0.5,0.2]
        self.reach_goal = np.array([0.5,0,0.5])
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        print(f"Seed: {seed}\treward_type:{self.reward_type}")
        # TODO Seed Environment
        return [seed]

    def _addObstacles(self):
        p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/box.urdf",
                                              [0,0,0],
                                              p.getQuaternionFromEuler([0,0,0]),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT
                                              )

        BaseSingleAgentAviary._addObstacles(self)

    def _set_env_parameters(self):
        self.num_steps = 0
        self.caught = False
        self.steps_done = False
        self.start_time = time.time()
        self.crashed = False
        self._reward = None

    def _add_whiskers(self):
        pass

    def reset(self):
        observation = BaseSingleAgentAviary.reset(self)
        self._set_env_parameters()
        # Get the observation
        # Process the observation to get it in np format
        self.observation = self._process_observation(observation)

        self.log()
        if(self.GUI):
            draw_point(self.reach_goal)

        return self.observation

    # get action from the network
    # process the action to be passed to the environment
    def _process_action(self, action):
        """
        Parameters
        ----------
        action : ndarray or list
            The input action for all drones with empty/dummy actions for non-trained drone
        Returns
        -------
        dict[int, ndarray]
            (NUM_DRONES, 3)-shaped array of ints containing to clipped delta target positions
        """
        ac = deepcopy(action)
        # Actions are amplified
        ac = [a*6 for a in ac]
        if(ACTION2D):
            # For making the action space only 2D
            z_pos = self.pos[0,2]
            # print(pred_z)
            ac.append(6*(-z_pos + 0.2))
        ac = np.array(ac)
        return ac

    def masking_observations(self, observations):
        masked_observations = np.ma.masked_array(observations, mask=self.observation_mask)
        result_observations = masked_observations.compressed()
        return result_observations

    def _process_observation(self, observation):
        # Change from dictionary to list
        ob = []
        ob.extend(self.masking_observations(observation))
        return np.array(ob)

    def _process_reward(self, obs, action):
        reward = None
        if(self.reward_type == "reach"):
            dist = self._compute_relative_distance(obs)
            # reward = 1/dist#-dist
            reward = -10 * dist**2
            if(self.caught):
                reward = 300#10 # more than the maximum possible reward
        # ----------------- Crashing rewards -----------------
        # [0] -> predator, [1] -> prey
        # Predator has no access to the reward of the prey neither the other
        # Return the rest of the steps as a value to the reward -> negative reward

        # steps_left = (self.max_num_steps - self.num_steps +1)
        # if(all(self.crashed)):  # -> both of them crashed
        #     prey_reward = -10-1*steps_left #-10000
        #     predator_reward = -10-1*steps_left #-10000
        # # TODO: try later to remove the positive rewards for the other agents
        # elif(self.crashed[0]): # -> Predator crashed
        #     prey_reward = 10+steps_left
        #     predator_reward = -10-1*steps_left
        # elif(self.crashed[1]): # -> Prey crashed
        #     prey_reward = -10-1*steps_left
        #     predator_reward = 10+steps_left

        self._reward = reward
        
        return reward
        
    def _compute_relative_distance(self, obs):
        pos0 = self.pos[0,0:3] # Not normalized
        dist = np.linalg.norm(pos0 - self.reach_goal)
        return dist

    def _compute_caught(self, obs):
        # obs is a list where the first inputs are the xyz and then other observations and then the second agent and so on
        # The following comment might does not make sense now, but maybe later will make sense or not :)
        # It would be better to pass the obs_dict instead but I would like to keep it this way for now -> as other environment does not return observations as dictionary and I want this here to be compatible with others and we just changed the obs that is getting outputed to the RL agent but everything should be the same with lists 
        if(self.reward_type == "reach"):
            dist = self._compute_relative_distance(obs)
            if(dist <= self.caught_distance):
                return True
            return False
    
    # Get extra data not from the RL agent observations but internal drone state
    def _compute_crash(self):
        crashed = False
        # return crashed
        roll, pitch, yaw = self.rpy[0, :]
        # Rotations angles crashing criteria
        if abs(roll) >= self.crashing_max_angle or abs(pitch) >= self.crashing_max_angle:
            crashed = True
        return crashed

    def _process_done(self, obs):
        # TODO: Compute caught flag
        self.caught = self._compute_caught(obs)
        self.crashed = self._compute_crash()
        self.steps_done = self.num_steps > self.max_num_steps
        done = True if self.caught or self.steps_done or (self.crashed) else False
        return done

    def who_won(self):
        if(self.caught):
            # return "winner"
            return 1
        if(self.steps_done):
            # return "loser"
            return -1
        elif(self.crashed): # -> Predator crashed
            # return "crashed"
            return 0
        return ""

    def _process_info(self):
        return {"win":self.who_won(), "crash": self.crashed, "reward": (self._reward), "caught": self.caught, "dist": self._compute_relative_distance(self.observation)}

    # Dummy functions for the sake of the implementation of the parent gym
    # We did not use it, as for a reason, the function to compute the reward, it requires to compute first the observation & action as parameters, then we compute the reward
    # TODO: think more -> we can make obs -> self.obs
    def _computeReward(self):
        return 0
    def _computeDone(self):
        return False
    def _computeInfo(self):
        return {}
    
    def step(self, action):
        self.num_steps += 1
        action = self._process_action(action)
        # Take a step in the environment
        obs, _, _, _ = BaseSingleAgentAviary.step(self, action)
        self.observation = obs
                
        done = self._process_done(obs)
        reward = self._process_reward(obs, action)
        info = self._process_info()
        if(done):
            print(f"Inside step function: {info}")

        self.log()

        return obs, reward, done, info

    def log(self):
        if(self.logger is not None):
            for j in range(self.nrobots):
                state = self._getDroneStateVector(j)
                self.logger.log(drone=j,
                                timestamp=self.num_steps/self.SIM_FREQ,
                                state=state)
    def save_log(self):
        self.logger.save()
        self.logger.save_as_csv("pid") # Optional CSV save
    def show_log(self):
        self.logger.plot()

    def render(self, mode='human', extra_info=None):
        # Render environment
        BaseSingleAgentAviary.render(self, mode)
        # Render is called after the step function
        sync(min(0,self.num_steps), self.start_time, self.TIMESTEP)
        # info = f'Step: {self.num_steps}'
        

    ################################################################################
    # Copied and paste from https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/multi_agent_rl/LeaderFollowerAviary.py
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 2

        MAX_XY = 1.8#MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = 1#MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY) #state[0:2]
        clipped_pos_z = np.clip(state[2], 0, MAX_Z) #state[2]
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_Z
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)
        # Don't get confused, this function is used in the parent class and then the extra observations are being cut

        return norm_and_clipped
        
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        # pass
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in _clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in _clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in _clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in _clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in _clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))

# just a wrapper to have the same api as in the self-play envs in order to work with the same training scripts with the minimal changes
class DroneReach(_DroneReach):
    def __init__(self, *args, **kwargs):
        super(DroneReach, self).__init__(*args, **kwargs)
        self.target_opponent_policy_name = None

    def set_target_opponent_policy_name(self, *args, **kwargs):
        pass

    def set_sampled_opponents(*args, **kwargs):
        pass
    def set_opponents_indicies(*args, **kwargs):
        pass

if __name__ == '__main__':
    import gym
    from time import sleep
    from math import cos, sin, tan
    env = DroneReach(seed_val=45, gui=True, logger=False)#, reward_type="relative_distance")

    observation = env.reset()
    done = False
    total_reward = 0
    # for s in range (5000):
    while not done:
        # action = {0: np.array([0.5, 0, 0.6]), 1: np.array([0, 0, 0])}#env.action_space.sample()
        action = env.action_space.sample()
        action = [0.5*(-env.pos[0][i]+env.reach_goal[i]) for i in range(3)]
        # action = [0,0,0]
        # action = [0,1,0,0,-1,0] # Crashing together -> predator is catching the prey
        # action = [0,-1,0,0,1,0] # Crashing to the walls (moving right)
        # action = [1,0,0,1,0,0]  # Crashing to the front walls (moving forward)
        # action = [0,0,1,0,0,0] # predator is crashing
        # action = [-observation[0]+observation[6],-observation[1]+observation[7],-observation[2]+observation[8], 1*cos(env.num_steps*0.001), 1*sin(env.num_steps*0.01), sin(env.num_steps*1)]#sin(3*2*env.num_steps*0.01)/2.0,cos(3*env.num_steps*0.01),0]
        # action = [0,-1,0,0,0,0]
        # action = [0.2,0,0.2,0]
        # action = [-observation[0]+observation[6],-observation[1]+observation[7],action[2], action[3]]#-observation[2]+observation[8], #1*cos(env.num_steps*0.001), 1*sin(env.num_steps*0.01), sin(env.num_steps*1)]#sin(3*2*env.num_steps*0.01)/2.0,cos(3*env.num_steps*0.01),0]
        # if(env.num_steps < 200):
        #     action = [0,1,0,0,-1,0]
        #     print("Up")

        # action[0][3] = 0.8
        # action[0][0] = 0
        # action[0][1] = 0
        # action[0][2] = 0
        # print(f"Actions: {action}")
        # action[0] = [0,0]
        # action[1] = 1
        # action[2] = 1
        # action[3] = 1
        observation, reward, done, info = env.step(action)
        total_reward += reward
        # print(observation)
        print(info)
        # print(reward, info, done)
        # print(done)
        # env.render()
        # sleep(0.01)
        # print(done)
        # if ((isinstance(done, dict) and done["__all__"]) or (isinstance(done, bool) and done)):
        #     break
    env.close()
    print(env.num_steps)
    print(total_reward)
    # env.show_log()