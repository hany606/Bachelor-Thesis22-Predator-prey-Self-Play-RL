FROM nvidia/cudagl:10.2-devel-ubuntu18.04

############################################
# Basic dependencies
############################################
RUN apt-get update --fix-missing && apt-get install -y \
      python3.7 \ 
      python3-numpy  python3.7-dev \
      python3-opengl python3-pip \
      cmake zlib1g-dev libjpeg-dev xvfb  \
      xorg-dev libboost-all-dev libsdl2-dev swig \
      git wget openjdk-8-jdk ffmpeg unzip\
    && apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

RUN update-alternatives --install /usr/local/bin/python3 python3 /usr/bin/python3.6 1

RUN update-alternatives --install /usr/local/bin/python3 python3 /usr/bin/python3.7 2

############################################
# OpenAI Gym
############################################
RUN python3.7 -m pip install pip
RUN pip3.7 install --upgrade pip
RUN pip3 install setuptools
RUN pip3.7 install 'gym==0.21.0' 'stable-baselines3[extra]' 'pybullet==3.2.1' 'ray[rllib]' 'wandb' seaborn plotly


############################################
# PyTorch and Tensorflow
############################################
RUN pip3.7 install torch torchvision pandas filelock
# tensorflow-gpu==1.14
############################################
# EvoRobotPy
############################################
RUN apt-get update && apt-get install libgsl0-dev -y
RUN pip3.7 install pyglet Cython tensorboard scikit-learn
# https://stackoverflow.com/questions/68433967/valueerror-not-a-tbloader-or-tbplugin-subclass-class-tensorboard-plugin-wit
RUN pip3.7 uninstall tensorboard-plugin-wit --yes

# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
# 
RUN apt-get update --fix-missing && apt-get install -y screen

