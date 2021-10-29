# Drones-PEG-Bachelor-Thesis-2022
This repository is created for my thesis during the bachelor degree at Innopolis University. The topic for research is the realization of Pursuit and Evasion Games (PEG) using drones and learning the collective/cooperative behavior using RL and EA algorithms and trying to transfer from simulation to reality (Sim2Real)


## TODO:

- [ ] Create setup script to setup everything

## Challenge:
- [ ] At least 2 useful commits per day (Not update README.md neither update submodules)
- [ ] Submit +1 paper(s) in top-tier conferences by the end of the thesis


## Installation

### Clone the repository

```bash
git clone --recursive https://github.com/hany606/Drones-PEG-Bachelor-Thesis-2022.git

# Or through ssh
# First, need to register ssh key to github: https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
git clone --recursive git@github.com:hany606/Drones-PEG-Bachelor-Thesis-2022.git
```

Export the following variable that points to the directory of predpreylib that inside 2D/gym-predprey
```bash
# /home/hany606/repos/research/Drones-PEG-Bachelor-Thesis-2022/2D/gym-predprey/predpreylib
# /home/hany606/Drones-PEG-Bachelor-Thesis-2022/2D/gym-predprey/predpreylib
export PREDPREYLIB="absolute/path/to/the/root/folder/predpreylib"
```

or even better, put it inside ~/.bashrc or ~/.zshrc

### Install gym-predprey library

```bash
cd Drones-PEG-Bachelor-Thesis-2022/2D/gym-predprey
cd predpreylib
# Edit setupErPredprey.py
# include_gsl_dir, lib_gsl_dir
python3 setupErPredPrey.py build_ext --inplace  
cd ..
cd gym-predprey/gym_predprey/envs/
# Edit 6th line in PredPrey1v1.py
# sys.path.append('/home/hany606/repos/research/Drones-PEG-Bachelor-Thesis-2022/2D/gym-predprey/predpreylib')
pip3 install -e .
```

### Install other libraries that are necessary

```bash
# Stable-baselines 3: https://stable-baselines3.readthedocs.io/en/master/guide/install.html
pip3 install stable-baselines3[extra]

# Wandbai: https://docs.wandb.ai/quickstart
pip3 install wandb
wandb login
# & Setup API key

# Ray: https://docs.ray.io/en/stable/installation.html1
pip3 install ray
pip3 install ray[debug]
pip3 install ray[rllib]
```

### Install bach-utils module

```bash
# It is a local module to include multiple utilities

cd Drones-PEG-Bachelor-Thesis-2022/bach-utils/
pip3 install -e .
```

### Running training

Note: this will require [wandb.ai](http://wandb.ai) installation and authentication for visualization

```bash
cd Drones-PEG-Bachelor-Thesis-2022/2D/experiments/
python3 train_selfplay_baselines.py
```

### Running an experiment

```bash
cd Drones-PEG-Bachelor-Thesis-2022/2D/experiments/

python3 test_selfplay_baselines.py --exp <experiment-path>

An example:
python3 test_selfplay_baselines.py --exp selfplay-results/save-SelfPlay1v1-Pred_Prey-v0-PPO-full-vel-08.18.2021_01.08.37
```

## Notes:

1. Submodules
To update the submodules to be the same as the remote (master/main)

First, go to the submodules, **Check that you pushed everything to the main/master**

```git push origin HEAD:main```

Then, updates the submodules:

```git submodule update --remote --recursive```

Then, add, commit, and push

```bash 
  git add submodule
  git commit -m "Update submodules"
  git push
```

2. Remove changes in a file

For example:

```git stash -- PredPrey1v1.py```


3. Pull with the newest changes in submodules

```git pull --recurse-submodules```
