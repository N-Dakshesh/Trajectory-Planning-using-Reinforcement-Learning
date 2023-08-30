'''
Input any function the the form y = f(x). Path must go in direction of increasing x. 
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter

# Define a function to calculate probabilities based on user-defined function and points
def calculate_probabilities(x_range, y_range, func, xstart, xend, tilt_factor = 0.1):

    ystart = func(xstart)
    yend = func(xend)

    x_grid, y_grid = np.meshgrid(x_range, y_range)
    distances = np.abs(y_grid - func(x_grid))
    probabilities = 1 / (1 + distances)
    probabilities /= np.sum(probabilities)

    probabilities = probabilities*np.heaviside(x_range - xstart, 0)*(1 - np.heaviside(x_range - xend, 0))

    probabilities = probabilities/np.max(probabilities)

    distances_start = np.sqrt((x_grid[x_grid < xstart] - xstart)**2 + (y_grid[x_grid < xstart] - ystart)**2)

    probabilities[x_grid < xstart] = 1/(1 + distances_start)

    distances_end = np.sqrt((x_grid[x_grid > xend] - xend)**2 + (y_grid[x_grid > xend] - yend)**2)

    probabilities[x_grid > xend] = 1/(1 + distances_end)


    return probabilities/np.max(probabilities) + 0.2*x_grid

# Define the range of x values
x_range = np.linspace(-1, 1, 200)
y_range = np.linspace(-1, 1, 200)

# Define the function you want to visualize (e.g., sine function)
def user_defined_function(x):
    return np.sin(np.pi*x)

# Calculate probabilities based on the user-defined function
probabilities = calculate_probabilities(x_range, y_range, user_defined_function, -0.5, 0.5)

# Plot the probability distribution
plt.figure(figsize = [8, 6])
plt.contourf(x_range, y_range, probabilities, levels=20)
plt.colorbar()
plt.plot(x_range, user_defined_function(x_range), color='red')  # Overlay the user-defined function
plt.xlabel('x')
plt.ylabel('y')
plt.title('Probability Distribution for User-Defined Function')
plt.show()

#%%
inter_prob = inter.RectBivariateSpline(y_range, x_range, probabilities)
x_grid, y_grid = np.meshgrid(x_range, y_range)
plt.figure(figsize = [8, 6])
plt.contourf(x_grid, y_grid, inter_prob(y_grid, x_grid, grid = False), levels = 20)
plt.colorbar()

