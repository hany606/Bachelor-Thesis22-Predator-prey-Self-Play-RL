from copy import deepcopy

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary
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
class PredPreyDrones(BaseMultiagentAviary):
    def __init__(self,
                 caught_distance=0.13, #0.015
                 max_num_steps=1000,
                 crashing_max_angle=np.pi/4,
                 pred_behavior=None, 
                 prey_behavior=None, 
                 pred_policy=None, 
                 prey_policy=None, 
                 seed_val=45, 
                 reward_type="normal",
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_pred_drones: int=1,
                 num_prey_drones: int=1,
                 neighbourhood_radius: float=np.inf,
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
        self.nrobots = num_pred_drones + num_prey_drones
        if initial_xyzs is None:
            initial_xyzs = np.vstack((
                                    np.zeros((num_pred_drones,3)),
                                    np.zeros((num_prey_drones,3))
                                    ))

        initial_xyzs = np.array([[0,-0.5,0.2], [0,0.5,0.2]])
        BaseMultiagentAviary.__init__(  self, 
                                        drone_model=drone_model,
                                        num_drones=self.nrobots,
                                        neighbourhood_radius=neighbourhood_radius,
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
        if(ACTION2D):
            self.noutputs = 2#self.action_space[0].shape[0]   # for single drone
            low = []
            high = []
            for i in range(self.nrobots):
                a = self.action_space[i]
                # X and Y only 2D action space
                low.extend([-1 for i in range(2)])
                high.extend([1 for i in range(2)])
            self.action_space      = spaces.Box(low=np.array(low),
                                                high=np.array(high),
                                                dtype=np.float32)
        else:
            self.noutputs = self.action_space[0].shape[0]   # for single drone
            low = []
            high = []
            for i in range(self.nrobots):
                a = self.action_space[i]
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
        for i in range(self.nrobots):
            o = self.observation_space[i]
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
        self.pred_behavior = pred_behavior
        self.prey_behavior = prey_behavior
        self.pred_policy = pred_policy
        self.prey_policy = prey_policy
        self._set_env_parameters()
        self.caught_distance = caught_distance
        self.crashing_max_angle = crashing_max_angle
        self.observation = None # This is created to keep track the previous observations to feed for the opponent policy when selecting an action
        self.logger = Logger(logging_freq_hz=int(self.SIM_FREQ/self.AGGR_PHY_STEPS), num_drones=self.nrobots) if logger else None

        # [0,-0.5,0.2]
        self.reach_goal = np.array([0.5,0,0.5])


    def reinit(self, max_num_steps=1000, prey_behavior=None, pred_behavior=None):
        self.max_num_steps = max_num_steps
        self.prey_behavior = prey_behavior
        self.pred_behavior = pred_behavior

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

        BaseMultiagentAviary._addObstacles(self)

    def _set_env_parameters(self):
        self.num_steps = 0
        self.caught = False
        self.steps_done = False
        self.start_time = time.time()
        self.crashed = [False for i in range(self.nrobots)]
        self._pred_reward = None
        self._prey_reward = None

    def _add_whiskers(self):
        pass

    def reset(self):
        observation = BaseMultiagentAviary.reset(self)
        self._set_env_parameters()
        # Get the observation
        # Process the observation to get it in np format
        self.observation = self._process_observation(observation)

        self.log()
        if(self.GUI):
            draw_point(self.reach_goal)

        return self.observation

    def _get_agent_observation(self, obs):
        return obs

    def _get_opponent_observation(self, obs):
        return obs

    # get action from the network
    # process the action to be passed to the environment
    def _process_action(self, action, observation):
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

        # # Something like virtual wall
        # pos0 = np.array(observation[0:3])
        # pos1 = np.array(observation[self.ninputs:self.ninputs+3])
        # # If the distance for the agent is more than threshold in a specific axis, then stop in that axis 
        # for i,p in enumerate(pos0):
        #     if(p >= self.max_distance):
        #         ac[i] = 0.0
        # for i,p in enumerate(pos1):
        #     if(p >= self.max_distance):
        #         ac[self.noutputs+i] = 0.0
        # print(ac)

        # Actions are amplified
        ac = [a*6 for a in ac]
        if(not ACTION2D):
            action_dict = {i:np.array(ac[self.noutputs*i:self.noutputs*(i+1)]) for i in range(self.nrobots)}
        else:
            # For making the action space only 2D
            pred_z = self.pos[0,2]
            prey_z = self.pos[1,2]
            # print(pred_z)
            ac.insert(2,6*(-pred_z + 0.2))
            ac.append(6*(-prey_z + 0.2))
            action_dict = {i:np.array(ac[(self.noutputs+1)*i:(self.noutputs+1)*(i+1)]) for i in range(self.nrobots)}

        # Change from list of actions to dictionary
        # for i in range(len(ac)):
        #     ac[i] *=6
        # action_dict = {}
        # for i in range(self.nrobots):
        #     action_dict[i] = np.array(ac[self.noutputs*i:self.noutputs*(i+1)])#self.motion.compute_motion(action[i])
        return action_dict

    def masking_observations(self, observations):
        masked_observations = np.ma.masked_array(observations, mask=self.observation_mask)
        result_observations = masked_observations.compressed()
        return result_observations

    def _process_observation(self, observation):
        # Change from dictionary to list
        ob = []
        for i in range(self.nrobots):
            ob.extend(self.masking_observations(observation[i]))
        return np.array(ob)

    def _process_reward(self, obs, action):
        norm_action_predator = np.tanh(np.linalg.norm(action[:self.noutputs]))/3
        norm_action_prey     = np.tanh(np.linalg.norm(action[self.noutputs:]))/3
        # Dense reward based on catching without taking into consideration the distance between them
        prey_reward, predator_reward = None, None
        if(self.reward_type == "normal"):
            prey_reward = 1
            predator_reward = -1
        elif(self.reward_type == "action_norm_pen"): # action_norm_penalization
            prey_reward = 1 - norm_action_prey
            predator_reward = -1 - norm_action_predator
        elif(self.reward_type == "relative_distance"):
            dist = self._compute_relative_distance(obs) - self.caught_distance*0.8 # to have that distance when the predator is nearly zero for the reward computation
            # Tanh is non-linear bounded function
            prey_reward = np.tanh(dist)
            predator_reward = -np.tanh(dist)
            # print(dist)
        
        if(self.caught):   # if the predator caught the prey before finishing the time
            # self.caught = True
            prey_reward = -10
            predator_reward = 10
        if(self.steps_done):
            prey_reward = 10
            predator_reward = -10

        if(self.reward_type == "reach"):
            pos0 = np.array(obs[0:3])
            pos1 = np.array(obs[self.ninputs:self.ninputs+3])
            dist_pred = np.linalg.norm(pos0 - self.reach_goal)
            dist_prey = np.linalg.norm(pos1 - self.reach_goal)
            prey_reward = -dist_prey
            predator_reward = -dist_pred
            # print(predator_reward, prey_reward)
            if(dist_pred < 0.2):
                predator_reward = 10
            if(dist_prey < 0.2):
                prey_reward = 10

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

        self._pred_reward, self._prey_reward = predator_reward, prey_reward # to be used for the info
        
        return predator_reward, prey_reward

    def _compute_relative_distance(self, obs):
        pos0 = np.array(obs[0:3])
        pos1 = np.array(obs[self.ninputs:self.ninputs+3])
        dist = np.linalg.norm(pos0 - pos1)
        return dist

    def _compute_caught(self, obs):
        # obs is a list where the first inputs are the xyz and then other observations and then the second agent and so on
        # The following comment might does not make sense now, but maybe later will make sense or not :)
        # It would be better to pass the obs_dict instead but I would like to keep it this way for now -> as other environment does not return observations as dictionary and I want this here to be compatible with others and we just changed the obs that is getting outputed to the RL agent but everything should be the same with lists 
        dist = self._compute_relative_distance(obs)
        if(dist <= self.caught_distance):
            return True

        if(self.reward_type == "reach"):
            pos0 = np.array(obs[0:3])
            pos1 = np.array(obs[self.ninputs:self.ninputs+3])
            dist_pred = np.linalg.norm(pos0 - self.reach_goal)
            dist_prey = np.linalg.norm(pos1 - self.reach_goal)
            if(dist_pred < 0.2 or dist_prey < 0.2):
                return True
        # # If the distance for the prey is more than 0.5 then it is caught -> damaged to the wall -> just a naiive approach for now for them
        # if(np.linalg.norm(pos1) >= self.max_distance):
        #     return True
        return False
    
    # Get extra data not from the RL agent observations but internal drone state
    def _compute_crash(self, obs):
        crashed = [False for i in range(self.nrobots)]
        # return crashed
        for i in range(self.nrobots):
            # state = self._getDroneStateVector(i)
            # quat = self.quat[i, :]
            roll, pitch, yaw = self.rpy[i, :]
            # vel = self.vel[i, :]
            # print(f"Quat: {quat}")
            # print(f"RPY: {roll, pitch, yaw}")
            # print(f"Vel: {vel}")
            # Rotations angles crashing criteria
            if abs(roll) >= self.crashing_max_angle or abs(pitch) >= self.crashing_max_angle:
                crashed[i] = True
            # TODO: Think more do we need "Ground-level crashing criteria"
        return crashed

    def _process_done(self, obs):
        # TODO: Compute caught flag
        self.caught = self._compute_caught(obs)
        self.crashed = self._compute_crash(obs)
        self.steps_done = self.num_steps > self.max_num_steps
        done = True if self.caught or self.steps_done else False #or any(self.crashed) else False
        return done

    def who_won(self):
        if(self.caught):
            return "pred"
        if(self.steps_done):
            return "prey"
        if(all(self.crashed)):  # -> both of them crashed
            return "none"
        elif(self.crashed[0]): # -> Predator crashed
            return "prey"
        elif(self.crashed[1]): # -> Prey crashed
            return "pred"
        return ""

    def _process_info(self):
        return {"win":self.who_won(), "crash": self.crashed, "reward": (self._pred_reward, self._prey_reward)}

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
        action_dict = self._process_action(action, self.observation)
        # Take a step in the environment
        obs_dict, _, _, _ = BaseMultiagentAviary.step(self, action_dict)
        obs = self._process_observation(obs_dict)    # self.ob has changed
        self.observation = obs
        
        # if(obs.shape != self.observation_space.shape):
        #     raise ValueError("Observation space is incorrect")
        
        done = self._process_done(obs)
        reward = self._process_reward(obs, action)
        info = self._process_info()
        if(done):
            print(info)

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
        BaseMultiagentAviary.render(self, mode)
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
            
class PredPrey1v1PredDrone(PredPreyDrones, gym.Env):
    def __init__(self, **kwargs):
        PredPreyDrones.__init__(self, **kwargs)
        self.action_space      = spaces.Box(low=self.action_space.low[:self.noutputs],
                                            high=self.action_space.high[:self.noutputs],
                                            dtype=np.float32)

        # This one is exactly the same as the parent
        # self.observation_space = spaces.Box(low=np.array([0     for _ in range(self.ninputs*self.nrobots)]),
        #                                     high=np.array([1000 for _ in range(self.ninputs*self.nrobots)]),
        #                                     dtype=np.float64)

    def _process_action(self, action, observation):
        if(self.prey_behavior is None and self.prey_policy is None):
            raise ValueError("prey_behavior or prey_policy should be specified")

        if(ACTION2D):
            action = np.array([action, [0,0]]).flatten()
        else:
            action = np.array([action, [0,0,0]]).flatten()
        return PredPreyDrones._process_action(self, action, observation)

    def _process_observation(self, observation):
        # Get the whole observations for all agents
        return PredPreyDrones._process_observation(self, observation)   # this function change from dictionary to list
        # TODO: later we can define different observations for each agent and then return it
    # TODO: later we can define different observations (split observations) for the oppponents
    def _get_opponent_observation(self, observation):
        return observation

    def who_won(self):
        if(self.caught):
            return 1
        if(self.steps_done):
            return -1
        if(all(self.crashed)):  # -> both of them crashed
            return 0
        elif(self.crashed[0]): # -> Predator crashed
            return -1
        elif(self.crashed[1]): # -> Prey crashed
            return 1

        return 0

    def _process_reward(self, ob, action):
        predator_reward, prey_reward = PredPreyDrones._process_reward(self, ob, action)
        return predator_reward

class PredPrey1v1PreyDrone(PredPreyDrones, gym.Env):
    def __init__(self, **kwargs):
        PredPreyDrones.__init__(self, **kwargs)
        self.action_space      = spaces.Box(low=self.action_space.low[self.noutputs:],
                                            high=self.action_space.high[self.noutputs:],
                                            dtype=np.float32)
        # This one is exactly the same as the parent class
        # self.observation_space = spaces.Box(low=np.array([0     for _ in range(self.ninputs*self.nrobots)]),
        #                                     high=np.array([1000 for _ in range(self.ninputs*self.nrobots)]),
        #                                     dtype=np.float64)

    def _process_action(self, action, observation):
        if(self.pred_behavior is None and self.pred_policy is None):
            raise ValueError("prey_behavior or prey_policy should be specified")

        if(ACTION2D):
            action = np.array([[0,0], action]).flatten()
        else:
            action = np.array([[0,0,0], action]).flatten()
        return PredPreyDrones._process_action(self, action, observation)

    def _process_observation(self, observation):
        # Get the whole observations for all agents
        return PredPreyDrones._process_observation(self, observation)
    def _get_opponent_observation(self, observation):
        return observation

    def who_won(self):
        if(self.caught):
            return -1
        if(self.steps_done):
            return 1
        if(all(self.crashed)):  # -> both of them crashed
            return 0
        elif(self.crashed[0]): # -> Predator crashed
            return 1
        elif(self.crashed[1]): # -> Prey crashed
            return -1
        return 0
    
    def _process_reward(self, ob, action):
        predator_reward, prey_reward = PredPreyDrones._process_reward(self, ob, action)
        return prey_reward

    

if __name__ == '__main__':
    import gym
    from time import sleep
    from math import cos, sin, tan
    env = PredPreyDrones(seed_val=45, gui=True, logger=False, reward_type="reach")#, reward_type="relative_distance")

    observation = env.reset()
    done = False
    total_reward = 0
    # for s in range (5000):
    while not done:
        # action = {0: np.array([0.5, 0, 0.6]), 1: np.array([0, 0, 0])}#env.action_space.sample()
        action = env.action_space.sample()
        action = [0.5,0,0, 0,0,0]
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
        print(f"Actions: {action}")
        # action[0] = [0,0]
        # action[1] = 1
        # action[2] = 1
        # action[3] = 1
        observation, reward, done, info = env.step(action)
        total_reward += reward[0]
        print(observation)
        print(reward, info, done)
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

    exit()

    env = PredPrey1v1PredDrone(seed_val=45, gui=True)
    behavior = Behavior()
    env.reinit(prey_behavior=behavior.fixed_prey)

    observation = env.reset()
    done = False

    while not done:
        # action = {0: np.array([0.5, 0, 0.6]), 1: np.array([0, 0, 0])}#env.action_space.sample()
        # action = env.action_space.sample()
        # action = [0,0,1]
        action = [-observation[0]+observation[6],-observation[1]+observation[7],-observation[2]+observation[8]]
        if(env.num_steps < 200):
            action = [0,1,0]
            print("Up")

        # action[0][3] = 0.8
        # action[0][0] = 0
        # action[0][1] = 0
        # action[0][2] = 0
        print(f"Actions: {action}")
        # action[0] = [0,0]
        # action[1] = 1
        # action[2] = 1
        # action[3] = 1
        observation, reward, done, info = env.step(action)
        print(observation)
        print(reward, info, done)
        env.render()
        # sleep(0.01)
        # print(done)
        # if ((isinstance(done, dict) and done["__all__"]) or (isinstance(done, bool) and done)):
        #     break
    env.close()
    

        
    env = PredPrey1v1PreyDrone(seed_val=45, gui=True)
    behavior = Behavior()
    env.reinit(pred_behavior=behavior.fixed_pred)

    env.reset()
    done = False

    while not done:
        # action = {0: np.array([0.5, 0, 0.6]), 1: np.array([0, 0, 0])}#env.action_space.sample()
        # action = env.action_space.sample()
        action = [0,0,1]
        if(env.num_steps < 200):
            action = [0,1,0]
            print("Up")

        # action[0][3] = 0.8
        # action[0][0] = 0
        # action[0][1] = 0
        # action[0][2] = 0
        print(f"Actions: {action}")
        # action[0] = [0,0]
        # action[1] = 1
        # action[2] = 1
        # action[3] = 1
        observation, reward, done, info = env.step(action)
        print(observation)
        print(reward, info, done)
        env.render()
        # sleep(0.01)
        # print(done)
        # if ((isinstance(done, dict) and done["__all__"]) or (isinstance(done, bool) and done)):
        #     break
    env.close()
    
