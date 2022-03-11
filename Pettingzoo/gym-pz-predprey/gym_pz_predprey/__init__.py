import logging
from gym.envs.registration import register

# PZ: PettingZoo

logger = logging.getLogger(__name__)

register(
    id='PZ-PredPrey-v0',
    entry_point='gym_pz_predprey.envs:PZPredPrey',
)

register(
    id='PZ-PredPrey-Pred-v0',
    entry_point='gym_pz_predprey.envs:PZPredPreyPred',
)

register(
    id='PZ-PredPrey-Prey-v0',
    entry_point='gym_pz_predprey.envs:PZPredPreyPrey',
)


register(
    id='SelfPlayPZ-Pred-v0',
    entry_point='gym_pz_predprey.envs:SelfPlayPZPredEnv',
    kwargs={"log_dir":None, "algorithm_class":None}
)

register(
    id='SelfPlayPZ-Prey-v0',
    entry_point='gym_pz_predprey.envs:SelfPlayPZPreyEnv',
    kwargs={"log_dir":None, "algorithm_class":None}

)