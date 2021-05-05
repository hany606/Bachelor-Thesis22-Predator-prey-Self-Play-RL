from gym.envs.registration import register

register(
    id='peg-drones-v0',
    entry_point='gym_peg_drones.envs:PEGDrones',
)