# Source: https://github.com/caelan/pybullet-planning/blob/6af327ba03eb32c0c174656cca524599c076e754/pybullet_tools/utils.py#L4415
import numpy as np
import pybullet as p
def add_line(start, end, color=[0,0,0], width=1, lifetime=None, parent=-1, parent_link=-1):
    assert (len(start) == 3) and (len(end) == 3)
    return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width,
                              parentObjectUniqueId=parent, parentLinkIndex=parent_link)

def draw_point(point, size=0.01, **kwargs):
    lines = []
    for i in range(len(point)):
        axis = np.zeros(len(point))
        axis[i] = 1.0
        p1 = np.array(point) - size/2 * axis
        p2 = np.array(point) + size/2 * axis
        lines.append(add_line(p1, p2, **kwargs))
    return lines