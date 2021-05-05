# Note: this environment is based on https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/multi_agent_rl/LeaderFollowerAviary.py
# This environment is made to simulate the environment for pursuit and evasio games using drones
import math
from math import sin, cos
from random import shuffle
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary

def mobius_strip(R:float=4, r:float=1, n:int=1):
    x_func = lambda t: R*cos(t) + r*sin(n*t/2)*cos(t)
    y_func = lambda t: R*sin(t) + r*sin(n*t/2)*sin(t)
    z_func = lambda t: R*cos(n*t/2)
    traj_func = lambda t: [x_func(t), y_func(t), z_func(t)]
    return traj_func

def mobius_strip_trajs(num:int=1, **args):
    return [mobius_strip(**args) for i in range(num)]

def plot_traj(self):
    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D
    from math import sin, cos

    fig = pyplot.figure()
    ax = Axes3D(fig)

    x, y, z = mobius_strip()

    time = np.linspace(0,4* np.pi,100)

    xs, ys, zs = [], [], []
    for t in time:
        xs.append(x(t))
        ys.append(y(t))
        zs.append(z(t))

    ax.plot(xs, ys, zs, label='parametric curve')
    ax.legend()
    pyplot.show()

# Note: here it doesn't have the same functions as openai gym as it already inhereted the interface from the base class BaseAviary
# [0,NUM_EVADER_DRONES) -> Evaders agents, [NUM_EVADER_DRONES, NUM_EVADER_DRONES+NUM_PURSUER_DRONES) -> Pursuers agent
class PEGDrones(BaseMultiagentAviary):
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_pursuer_drones: int=1,
                 num_evader_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 evaders_trajectory=None):

        super().__init__(drone_model=drone_model,
                         num_drones=num_pursuer_drones+num_evader_drones,
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
        self.NUM_EVADER_DRONES = num_evader_drones
        self.NUM_PURSUER_DRONES = num_pursuer_drones
        self.evaders_trajectory = mobius_strip_trajs(self.NUM_EVADER_DRONES) if evaders_trajectory is None else evaders_trajectory
        self.captured_flags = [False for i in range(self.NUM_EVADER_DRONES)]
        idx_evader_drones = [i for i in range(self.NUM_EVADER_DRONES)]
        shuffle([i for i in range(self.NUM_EVADER_DRONES)])
        self.chased_drone_idxs = dict(zip([i for i in range(self.NUM_EVADER_DRONES, self.NUM_DRONES)], idx_evader_drones))

    def _getChasedDroneIdx(self, idx):
        """Get the index of the closest available drone to that specific drnoe with index=idx
        
        Returns
        -------
        int
            index of the evader chased drone
        """
        self.adjacency_mat = self._getAdjacencyMatrix()
        # Currently, it is just randomly from the begining of the simulation
        # chased_drone_idx = self.chased_drone_idxs[idx]
        # TODO: Check if it works properly with indexing and sub matrix part
        chased_drone_idx = np.argmax(self.adjacency_mat[idx, self.NUM_EVADER_DRONES:self.NUM_DRONES])
        return chased_drone_idx

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone. For each drone it is different
            For evader:
                It is L2 norm for the trajectory (Trajectory Tracking problem)
            For pursuers:
                It is L2 norm with the closest available evader
        """
        rewards = {}
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(0, self.NUM_EVADER_DRONES):
            ref_traj = np.array(self.evaders_trajectory[i](self.step_counter/self.SIM_FREQ))
            rewards[i] = -(1/self.NUM_EVADER_DRONES) * np.linalg.norm(np.array([states[i, 0], states[i, 1], states[i, 2]]) - ref_traj)**2
        for i in range(self.NUM_EVADER_DRONES, self.NUM_DRONES):
            chased_drone_idx = self._getChasedDroneIdx(i)
            rewards[i] = -(1/self.NUM_PURSUER_DRONES) * np.linalg.norm(np.array([states[i, 0], states[i, 1], states[i, 2]]) - states[chased_drone_idx, 0:3])**2
        return rewards
    
    def _computeDone(self):
        """Computes the current done value(s).
        
        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and 
            one additional boolean value for key "__all__".
            For evaders: it is true if the evader has been capture -> entered the adjacency matrix with entry = 1
            For pursuers: it is true for all if they have captured all the evaders
        """
        # File "/home/hany606/.local/lib/python3.6/site-packages/ray/rllib/evaluation/collectors/simple_list_collector.py", line 661, in postprocess_episode
        # "from a single trajectory.", pre_batch)
        # ValueError: ('Batches sent to postprocessing must only contain steps from a single trajectory.', SampleBatch({'obs': array([[ 0.0000000e+00,  1.0586667e-02,  1.0586667e-02, ...,

        # time_bool_val = True if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC else False
        # self.captured_flags = [np.any(self.adjacency_mat[i, self.NUM_EVADER_DRONES:self.NUM_DRONES]) for i in range(self.NUM_EVADER_DRONES)]
        # done = {**{i: self.captured_flags[i] for i in range(self.NUM_EVADER_DRONES)},
        #         **{i: np.all(self.captured_flags) for i in range(self.NUM_EVADER_DRONES, self.NUM_DRONES)}}
        # done["__all__"] = time_bool_val  # True if True in done.values() else False

        bool_val = True if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC else False
        done = {i: bool_val for i in range(self.NUM_DRONES)}
        done["__all__"] = True if True in done.values() else False
        return done

    def _computeInfo(self):
        """
        Unused.
        """
        return {i: {} for i in range(self.NUM_DRONES)}


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
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
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
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
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
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
        
if __name__ == '__main__':
    pass