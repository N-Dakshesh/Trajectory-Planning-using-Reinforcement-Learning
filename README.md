# Trajectory Optimization using Reinforcement Learning
Note: USE LAPTOP BRANCH ONLY. LOOK INSIDE FINAL FOLDER. 
The 'FINAL' folder should be your directory when running code.
Read presentation inside 'FINAL', called trajectory planning dak.

With help from https://medium.com/deeplearningmadeeasy/advantage-actor-critic-continuous-case-implementation-f55ce5da6b4c for A2C implementation. Recommend learn about reinforcement learning, A2C, laplace transforms, convolution, pytorch, neural networks, RNN's, entropy in neural networks.


## Required Modules
numpy
scipy
matplotlib
gymnasium
tqdm
sympy
optuna
pytorch - cpu or cuda, look up instructions on pytorch website on how to install.

If using server, make sure packages are up to date. If unable to install packages due to lack of admin permissions on server, use pip --user install 'package_name'.

## How to use
Look at presentation. 

1) Choose path. Only form y = f(x) allowed. Only constant velocity paths (with discontinuties) or always accelerating paths supported. Note, must travel from left to right.
2) Create reward func using path gen. 
3) Run loop in testing_interface, after choosing parameters. Note: choose mode (constant, accelerate, or constant_experimental).
Note: 82% result was achieved when reward_func = path_func (in testing_interface). 

Testing interface is example code.


Will output multiple graphs: path/reward function with red line indiciating where you want it to go. 
History of total rewards per episode - hopefully increases. 
Graph of x and time.
Graph of y and time.
Contour plot of xy history and reward function, to see if agent followed path.



## File list

data_gen_toolsv3 holds tools linked to implementing transfer function model. See supervisor for more info. Model is first converted into time domain, and convolution used to convert inputs to outputs.
EnvA2C_v4 holds environment. If using a new environment, input new file name into Testing_Interface. Environment is where the agent gets rewards for new inputs, and calculates outputs. Very flexible.
A2C_main holds A2C reinforcement learning algorithm, implemented with RNN's (memory needed to learn integrals).
Training_loop trains agent - runs multiple 'episodes', and over time agent improves
Path_gen creates reward function, based which will follow any one to one function y = f(x) between an initial and final x point. Tilted towards higher values of x - make sure goal is at higher value of x than target. Make sure y = f(x) is inside bounds. +/- 1
Testing interface is file used most - example interface to run tests. Creates reward func (any shape y = f(x)), lets you set parameters, and runs. In a loop - runs as many times as you want, so you can see number of times it succeeds.
Tuner runs optuna on testing interface, to try automatically find ideal hyperparameters. Out of date, and slow. Use optuna distributed to take advantage of multiple cores, and update for most recent version of training_loop. Note: graphs must be disabled for optuna to work, use disable graphs features.


## Useful info
    mode : string, constant, accelerate or constant_experimental
        Chooses which transfer function to use. The default is 'constant'. constant_experimental combines both - but only valid for constant velocity with discontinuties.



## Hard coded Parameters:
Tracking error (T_val) and theta1 (theta) inside data_gen_toolsv3. These define parameters for the transfer function, and are set to 1. Change them manually if you want to use them. Note: tracking error must be a positive integer - rescale time if this is too restrictive.

State dimensions and size inside EnvA2C_v4 (the environment). Hard coded to be a 2D plane between -1 and 1 - recommend rescale instead of increasing size. Code will not support 3D without significant modification.

tolerance inside environment - set to complete if 0.1 away from goal.
Timestep is set to 0.1 (inside training_loop).
