from setuptools import setup

setup(name='gym_peg_drones',
    version='0.0.1',
    install_requires=['numpy', 'Pillow', 'matplotlib', 'cycler', 'gym', 'pybullet', 'stable_baselines3', 'ray[rllib]', 'gym_pybullet_drones']
)
