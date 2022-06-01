# import bach_utils.list as utlst
# import bach_utils.sorting as utsrt

# l = ["3", "2", "1"]

# print(utlst.get_latest(l))
# print(l)
# print(utlst.get_first(l))
# print(l)
# print(utlst.get_sorted(l, utsrt.sort_nicely))
# print(l)
import logging
logger = logging.getLogger("self-play-rl")

def testing():
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

if __name__ == "__main__":
    import numpy as np
    from bach_utils.heatmapvis import traj_vis

    x = {"pred":list(np.load("pred_x.npy")), "prey":    list(np.load("prey_x.npy"))}
    y = {"pred":list(np.load("pred_y.npy")), "prey":    list(np.load("prey_y.npy"))}
    
    traj_vis(x, y)