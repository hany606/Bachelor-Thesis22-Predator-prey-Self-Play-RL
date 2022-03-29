import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='PredPrey-Drones-v0',
    entry_point='gym_predprey_drones.envs:PredPreyDrones',
)

register(
    id='PredPrey-Pred-Drone-v0',
    entry_point='gym_predprey_drones.envs:PredPrey1v1PredDrone',
)

register(
    id='PredPrey-Prey-Drone-v0',
    entry_point='gym_predprey_drones.envs:PredPrey1v1PreyDrone',
)

register(
    id='SelfPlay1v1-Pred-Drone-v0',
    entry_point='gym_predprey_drones.envs:SelfPlayPredDroneEnv',
    kwargs={"log_dir":None, "algorithm_class":None}
)

register(
    id='SelfPlay1v1-Prey-Drone-v0',
    entry_point='gym_predprey_drones.envs:SelfPlayPreyDroneEnv',
    kwargs={"log_dir":None, "algorithm_class":None}

)

register(
    id='Drone-Reach-v0',
    entry_point='gym_predprey_drones.envs:DroneReach',
)
