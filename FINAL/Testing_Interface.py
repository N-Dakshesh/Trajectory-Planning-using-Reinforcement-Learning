'''
This is the main testing file you should use.
First section makes the path function (can choose between straight line and sine wave).
Second section trains RL agent and outputs best result, and repeats.
'''


import Training_loop as net
import numpy as np
import matplotlib.pyplot as plt
import Path_gen as pg
from EnvA2C_v4 import path_func



#%%

#Defines xy plane - use -1 to 1
x_range = np.linspace(-1, 1, 200)
y_range = np.linspace(-1, 1, 200)

#Straight line path
def path(x):
    if type(x) == np.ndarray:
        return np.zeros(len(x))
    else:
        return 0

#sin path
#def path(x):
#    return np.sin(np.pi*x)


#Start and end points
xstart = 0
xend = 0.5

#Creates the reward function + prints graph.
reward_func = pg.reward_func_maker(x_range, y_range, func = path, xstart = xstart, xend = xend,
                                   tilt_factor = 0.7, prob_factor = 2.8, disable_graphs = False)
plt.show()

#%%
#Modes are constant, accelerate, or auto
i = 0
j = 0

all_final = []

#Choose number of iterations here.
#Note, if you want presentation results, set reward_func = path_func
#Note optimised parameters for presentation are inside presentation.
while i < 2:

    results = net.training(lr = 0.000383, entropy_coeff = 0.4, hidden_dim = 128, num_layers = 5, gamma = 0.9,
                 batch_size = 50, episode_no = 1000, end_reward = 1.5, cuttoff_time = 15,
                 entry_point = "EnvA2C_v4:CustomEnv", reward_func = reward_func, mode = 'constant',
                           start = np.array([xstart, path(xstart)]), target = np.array([xend, path(xend)]),
                           disable_graphs = False)
    
    final_time, all_inputs, all_outputs, every_loss = results
    print(final_time)

    #Plots path the output followed on top of the reward function.
    plt.figure(figsize = [8, 6])
    plt.contourf(x_range, y_range, reward_func(x_range, y_range))
    plt.plot(all_outputs[:, 0], all_outputs[:, 1], label = 'Outputs')
    plt.legend()
    plt.colorbar()
    plt.title("Reward Function Plot and Output Path")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


    #If you want to see losses, plot this. Not very useful.
    # plt.figure(figsize = [8, 6])
    # plt.plot(every_loss)
    # plt.grid()
    # plt.xlabel('Episodes')
    # plt.ylabel('Losses')
    # plt.show()

    all_final.append(final_time)
    i = i + 1
    if final_time < 6.0:
        j = j + 1

print(all_final)
print(np.mean(np.array(all_final)))


