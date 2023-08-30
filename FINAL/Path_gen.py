'''
Input any function the the form y = f(x). Path must go in direction of increasing x. 
'''


import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter

# Define a function to calculate probabilities based on user-defined function and points
def calculate_probabilities(x_grid, y_grid, func, xstart, xend, tilt_factor = 0.1, prob_factor = 1):
    #xgrid, ygrid are from meshgrid

    ystart = func(xstart)
    yend = func(xend)

    if ystart == np.nan:
        raise Exception("Starting x value does not give a real y value. Use a different function or number.")
    if yend == np.nan:
        raise Exception("Ending x value does not give a real y value. Use a different function or number.")

    distances = np.abs(y_grid - func(x_grid))
    probabilities = 1 / (1 + distances)
    probabilities = np.nan_to_num(probabilities)
    probabilities /= np.sum(probabilities)

    probabilities = probabilities*np.heaviside(x_grid - xstart, 0)*(1 - np.heaviside(x_grid - xend, 0))

    probabilities = probabilities/np.max(probabilities)

    distances_start = np.sqrt((x_grid[x_grid < xstart] - xstart)**2 + (y_grid[x_grid < xstart] - ystart)**2)

    probabilities[x_grid < xstart] = 1/(1 + distances_start)

    distances_end = np.sqrt((x_grid[x_grid > xend] - xend)**2 + (y_grid[x_grid > xend] - yend)**2)

    probabilities[x_grid > xend] = 1/(1 + distances_end)


    return prob_factor*probabilities/np.max(probabilities) + tilt_factor*x_grid



def reward_func_maker(x_range, y_range, func, xstart, xend, tilt_factor = 0.2, prob_factor = 1, disable_graphs = True):
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    probabilities = calculate_probabilities(x_grid, y_grid, func, xstart, xend, tilt_factor, prob_factor)

    inter_prob = inter.RectBivariateSpline(y_range, x_range, probabilities)
    x = np.linspace(xstart, xend, 1000)

    if disable_graphs == False:
    # Plot the probability distribution
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)

        plt.figure(figsize = [12, 9])
        plt.contourf(x_grid, y_grid, inter_prob(y_grid, x_grid, grid = False), levels = 20)
        plt.colorbar()
        plt.plot(x, func(x), color='red')  # Overlay the user-defined function
        plt.xlabel('x', fontsize = 22)
        plt.ylabel('y', fontsize = 22)
        #plt.title('Reward Function')
        plt.show()


    def flipped_func(x, y):
        return inter_prob(y, x)

    return flipped_func
#%%
# x_range = np.linspace(-1, 1, 200)
# y_range = np.linspace(-1, 1, 200)

# def func(x):
#     return np.cos(np.pi*x)

# reward_func_maker(x_range, y_range, func, xstart = -0.5, xend = 0.5, tilt_factor = 0.2)

