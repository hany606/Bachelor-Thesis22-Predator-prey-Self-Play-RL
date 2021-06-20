from gym.envs.registration import register

register(
    id='predprey-v0',
    entry_point='gym_predprey.envs:PredPrey',
)