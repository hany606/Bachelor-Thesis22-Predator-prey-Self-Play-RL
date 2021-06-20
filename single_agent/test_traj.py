# Based on: https://github.com/utiasDSL/gym-pybullet-drones/blob/master/examples/fly.py
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType


from TrajAviary import TrajAviary
import shared_constants


def plot_traj(traj_func):
    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D
    from math import sin, cos

    fig = pyplot.figure()
    ax = Axes3D(fig)

    traj = traj_func()

    # time = np.linspace(0,4* np.pi,100)
    time = [i for i in range(len(traj))]

    xs, ys, zs = [], [], []
    for t in time:
        x, y, z = traj[t]
        xs.append(x)
        ys.append(y)
        zs.append(z)

    ax.plot(xs, ys, zs, label='parametric curve')
    ax.legend()
    pyplot.show()


# Source: https://github.com/caelan/pybullet-planning/blob/6af327ba03eb32c0c174656cca524599c076e754/pybullet_tools/utils.py#L4415
def draw_point(point, size=0.01, **kwargs):
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

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##

    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--exp',    type=str,       help='Help (default: ..)', metavar='')
    ARGS = parser.parse_args()


    #### Parameters to recreate the environment ################
    OBS = ObservationType.KIN if ARGS.exp.split("-")[3] == 'kin' else ObservationType.RGB
    if ARGS.exp.split("-")[4] == 'rpm':
        ACT = ActionType.RPM
    elif ARGS.exp.split("-")[4] == 'dyn':
        ACT = ActionType.DYN
    elif ARGS.exp.split("-")[4] == 'pid':
        ACT = ActionType.PID
    elif ARGS.exp.split("-")[4] == 'vel':
        ACT = ActionType.VEL
    elif ARGS.exp.split("-")[4] == 'one_d_rpm':
        ACT = ActionType.ONE_D_RPM
    elif ARGS.exp.split("-")[4] == 'one_d_dyn':
        ACT = ActionType.ONE_D_DYN
    elif ARGS.exp.split("-")[4] == 'one_d_pid':
        ACT = ActionType.ONE_D_PID

    env = TrajAviary(aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                        obs=OBS,
                        act=ACT,
                        gui=True,
                        record=True
                        )
    # plot_traj(lambda : env.TARGET_POSITION)
    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Run the simulation ####################################
    start = time.time()
    for i in range(int(5*env.SIM_FREQ/env.AGGR_PHY_STEPS)): # Up to 6''
        draw_point(env.TARGET_POSITION[i])
        env.render()

        sync(np.floor(i*env.AGGR_PHY_STEPS), start, env.TIMESTEP)

    #### Close the environment #################################
    env.close()