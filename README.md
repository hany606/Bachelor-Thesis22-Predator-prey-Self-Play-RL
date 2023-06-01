# Bachelor-Thesis22-Predator-prey-Self-Play-RL
This repository is created for my thesis during the bachelor degree at Innopolis University. The topic for research is Learning 
behavioral strategies for a multi-robot system in a predator-prey 
environment using Reinforcement Learning.

If you like this work and intend to use this software or the information inside has helped you in your work, I would be happy to cite it as below. (Hopefully a publication will be released later)

• Links: [Preprint in English](https://drive.google.com/file/d/1J1bmWlP1J9skfXmwQXqdyMFVNwNfsgto/view) & [presentation](https://drive.google.com/file/d/1mgLtZNa14XSOrtyRj58-buWwY9n79iME/view)

```
@software{Hamed_Learning_behavioral_strategies_2022,
  author = {Hamed, Hany and Klimchik, Alexandr and Nolfi, Stefano},
  doi = {10.5281/zenodo.1234},
  month = {6},
  title = {{Learning behavioral strategies for a predator and prey using Self-play Reinforcement Learning}},
  url = {https://github.com/hany606/Bachelor-Thesis22-Predator-prey-Self-Play-RL},
  version = {1.0.0},
  year = {2022}
}
```

## Packages to install:

- [ ] gym-predprey
- [ ] bach-utils
- [ ] submodule PettingZoo
- [ ] submodule gym-pybullet-drones
- [ ] gym-predprey-drones
- [ ] gym-pz-predprey

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
python3 setupErPredprey.py build_ext --inplace  
cd ..
#cd gym_predprey/envs/
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

4. To remove submodules: Check [https://stackoverflow.com/questions/1260748/how-do-i-remove-a-submodule](https://stackoverflow.com/questions/1260748/how-do-i-remove-a-submodule)


5. Fetch and merge with upstream from a fork: check [https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/merging-an-upstream-repository-into-your-fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/merging-an-upstream-repository-into-your-fork)

6. Merge with some conflicts: check [https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-on-github](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-on-github), it is better to use gitkraken

7. In order to run the experiment on the server that you connected to it through ssh, and then close the server. You can use screen or nohup

To use screen:

```bash

screen # creates new screen
screen -S <name-screen>
screen -ls # shows the ids and names of the opened screens
screen -r <screen-id>

ctrl+a+d to go out of the screen
ctrl+a+ESC to go up and down using scrolling and then press ESC to go out of the scrolling mode
```

To use nohup
```bash
nohup <cmd> &
```


8. To copy output of command directly to the clipboard:
```bash
<command> | xclip -selection clipboard 
```
