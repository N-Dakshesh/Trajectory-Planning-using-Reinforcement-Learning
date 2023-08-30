import numpy as np
import sympy as sp
import scipy.signal as sg

#Create gaussian distributed path function for a straight line, but works in 2D space
def pathfunc(x, y, sigma = 0.05, y0 = 0, xi = 0, xf = 5):
    #x, y = xy
    #Define to return - 1 at peak.

    output = np.array([])

    for j in range(0, len(x)):
        if abs(x[j] - xf) < 1 == True:
            out = -1000
            output = np.append(output, -10)
            #print('goal')
        elif x[j] <= xi:
            pwr = ((x[j] - xi)*(x[j]-xi) + (y[j] - y0)*(y[j]-y0))/(2*sigma*sigma)
            out = -np.exp(-pwr)
            output = np.append(output, out)
        elif x[j] > xf:
            pwr = ((x[j] - xf)*(x[j]-xf) + (y[j] - y0)*(y[j]-y0))/(2*sigma*sigma)
            out = -np.exp(-pwr)
            output = np.append(output, out)
        else:
            pwr = (y[j] - y0)*(y[j]-y0)/(2*sigma*sigma)
            out = -np.exp(-pwr)
            output = np.append(output, out)

    return output
#Test trajectory? Set u to high speed, and try.

def u_x(t):
    # Control function - instructions of x
    #t = np.asarray(t)  # Convert t to a NumPy array
    u_values = np.zeros_like(t)  # Initialize the output array

    # Apply the conditions to calculate control values for each element in t
    u_values[t >= 0] = 5 * t[t >= 0]
    u_values[t > 1] = 5
    return u_values

def u_y(t):
    return np.zeros(len(t))


import sympy as sp


s, t = sp.symbols('s t')

# n=6
# Tracking = 300e-6
# theta1 = 18468

# Gs_const =(Tracking*s/n + 1)**(-n)
# Gs_acc = s*s*(theta1*(Tracking*s/n + 1))**(-n)

# fx1 = sp.inverse_laplace_transform(Gs_const, s, t)
# fx2 = sp.inverse_laplace_transform(Gs_acc, s, t)

# trans_const = sp.lambdify(t, fx1, 'numpy')
# trans_acc = sp.lambdify(t, fx2, 'numpy')

# def transconst(inputs):
#     return abs(trans_const(inputs))

# def transacc(inputs):
#     return abs(trans_acc(inputs))

def transconst(t, T = 300e-6):
    numerator = 1944 * np.exp(-(6 * t) / T) * t**5
    #print(numerator)
    denominator = 5 * T**6
    #print(denominator)
    #print('const')
    result = numerator / denominator
    return result

def transconstdiff(t, T = 1):
    numerator = 1944 * np.exp(-(6 * t) / T) * t**4 * (-6 * t + 5 * T)
    denominator = 5 * T**7
    result = numerator / denominator
    return result

def transacc(t, T = 300e-6, theta = 18468):
    numerator = 7776 * np.exp(-(6 * t) / T) * t**3 * (9 * t**2 - 15 * t * T + 5 * T**2)
    denominator = 5 * T**8 * theta
    #print('acc')
    result = numerator / denominator
    return result

def transaccdiff(t, T = 1, theta = 1):
    numerator = 23328 * np.exp(-(6 * t) / T) * t**2 * (-18 * t**3 + 45 * t**2 * T - 30 * t * T**2 + 5 * T**3)
    denominator = 5 * T**9 * theta
    result = numerator / denominator
    return result

def acc(inputs):
    #Calculates acceleration, and adds 0 as the initial and final accelerations!
    tdiff = 10e-6
    # acc = (inputs[2:] - 2*inputs[1:-1] + inputs[:-2])/(tdiff*tdiff)
    # acc = np.append(acc, 0)
    # acc = np.insert(acc, 0, 0)

    v = np.gradient(inputs, tdiff)
    acc = np.gradient(v, tdiff)
    acc[abs(acc > 1)] = 0

    return acc


def trans_maker(times, inputs, tolerance = 1e-5):
    #Creates unique transfer function, depending on acceleration
    acceleration = acc(inputs)

    trans = np.array([])
    for i in range(0, len(inputs)):
        if abs(acceleration[i]) < tolerance: #i.e accerlation is almost 0
            transpoint = transconst(times[i])
            trans = np.append(trans, transpoint)
        else:
            transpoint = transacc(times[i])
            trans = np.append(trans, transpoint)

    return trans

def convolve(array1, array2):
    """
    Convolve two arrays.

    Parameters:
    array1 (list or tuple): The first input array.
    array2 (list or tuple): The second input array.

    Returns:
    list: The convolved output array.
    """
    m, n = len(array1), len(array2)
    result_length = m + n - 1
    result = [0] * result_length

    for i in range(result_length):
        for j in range(max(0, i - n + 1), min(m, i + 1)):
            result[i] += array1[j] * array2[i - j]

    return result
import matplotlib.pyplot as plt
#Final time given by timesteps
#inputs are u(t)
def outputs(times, inputs):
    #times = np.linspace(0, len(inputs)*10e-6, len(inputs)) #To make sure the transfer function is sampled as much as the input.

    #Construct discrete transfer function - if input has 0 accleration vs non 0
    #Calculate acceleration
    trans = trans_maker(times, inputs)
    output = np.convolve(trans, inputs, mode= 'same')

    #or try other way.
    #plt.plot(times, trans)
    #plt.xlabel('Time')
    #plt.ylabel('Transfer Function, Time domain')
    #plt.show()


    length = len(inputs)
    #Convolution doubles length, so restrict length to inputs (and therefore times to inputs!)
    #output = output[0:length]

    return output


def partial_derivatives(f, x, y, h=1e-5):
    """
    Calculate the partial derivatives of the function f with respect to x and y at the point (x, y).

    Parameters:
        f (function): The function to calculate the partial derivatives for.
        x (float): The x-coordinate at which to calculate the partial derivatives.
        y (float): The y-coordinate at which to calculate the partial derivatives.
        h (float, optional): The step size for numerical differentiation. Default is 1e-5.

    Returns:
        (float, float): The partial derivatives of f with respect to x and y, respectively.
    """
    # Calculate the partial derivative with respect to x (dx)
    dx = (f(x + h, y) - f(x - h, y)) / (2 * h)

    # Calculate the partial derivative with respect to y (dy)
    dy = (f(x, y + h) - f(x, y - h)) / (2 * h)

    return dx, dy


def Hamiltonian_diff(times, inputs_x, inputs_y, outputs_x, outputs_y, path_func = pathfunc):
    """
    Hamiltonian is based on lagrangian, which depends on the path func, and needs the real outputs to calculate.
    Whilst inputs should be different in x and y directions, H is identical

    Parameters
    ----------
    inputs_x
    inputs_y
    outputs_x : Array of x outputs
    outputs_y : Array of y outputs
    path_func : Gaussian Spread Path function

    Returns
    -------
    output : Derivative of Hamiltonian. 

    """
    #Inputs actually disappear from this! Only output needed

    trans_x = trans_maker(times, inputs_x)
    trans_y = trans_maker(times, inputs_y)

    def path_func_in(inputs_x, inputs_y):
        outputs_x = outputs(times, inputs_x)
        outputs_y = outputs(times, inputs_y)

        return path_func(outputs_x, outputs_y)

    partial_x, partial_y = partial_derivatives(path_func_in, inputs_x, inputs_y)

    #Need derivative of lagrangian - ignore 1, just f(x)
    #Partial, one in x and one in y

    H_x = sg.fftconvolve(trans_x, partial_x)
    H_y = sg.fftconvolve(trans_y, partial_y)

    length = len(times)

    return H_x[:length], H_y[:length]

# def Hamiltonian(times, inputs_x, inputs_y, path_func):
#     #Inputs actually disappear from this! Only output needed

#     trans_x = trans_maker(inputs_x)
#     trans_y = trans_maker(inputs_y)


#     #Path_func does not take these inputs of x, y!



#     partial_x = ndt.directionaldiff(path_func, np.array([inputs_x, inputs_y]), [1.0, 0.0]) #In x direction
#     partial_y = ndt.directionaldiff(path_func, np.array([inputs_x, inputs_y]), [0.0, 1.0]) #In y direction


#     #Need derivative of lagrangian - ignore 1, just f(x)
#     #Partial, one in x and one in y


#     output = sg.fftconvolve(trans_x, partial_x)[0:len(inputs_x)] * inputs_x + sg.fft.convolve(trans_y, partial_y)[0:len(inputs_x)] * inputs_y
#     return output

import scipy.integrate as it

def Lagrangian(x, y):
    return 1 + pathfunc(x, y)

def loss(times, x, y):
    #Now integrate
    time_diff = np.diff(times)
    L = Lagrangian(x, y)


    integral = 0.5 * (L[1:] + L[:-1]) *time_diff
    integral = np.sum(integral)
    return integral
    #Use Reimann sum (for now) to integrate across loss function.

#Constrains positions depending on max velocity
def constrainer(times, positions, max_v = 10):
    """
    Implements velocity constraints on list of positions. Returns updated list of positions. Use on inputs

    Parameters
    ----------
    times : Array of times
    positions : Arrays of positions (input)
    max_v : Maximum velocity

    Returns
    -------
    constrained_ux : Array of recalculated positions.

    """
    time_diff = np.diff(times)
    new_velocities = np.diff(positions)/time_diff #Differentiate, here we want average velocity for each one
    for i in range(0, len(new_velocities)):
        if new_velocities[i] > max_v:
            new_velocities[i] = max_v
        elif new_velocities[i] < -max_v:
            new_velocities[i] = -max_v


    constrained_ux = np.zeros_like(positions)
    #Now, use new_velocities to recalculate new u.
    constrained_ux[0] = positions[0]
    for i in range(1, len(positions)): #Initial u is the same
        constrained_ux[i] = constrained_ux[i-1] + time_diff[i-1] * new_velocities[i-1] #Distance travelled between.

    return constrained_ux




# #%%
# #Write algorithm?
# #Discretize into 10ms timesteps. Do while loop, which only ends when condition is met.


# close = np.array([False])
# end = 0.05
# i = 0
# times = np.array([0])

# while close.any() != True: #i.e. if one point is close to the end point, terminate!
#     times = np.append(times, 0.01 + times[-1])
#     ux = u_x(times)
#     x = x_out(ux)
#     close = np.isclose(x[-1], 5.0, atol = 0.2)
#     i = i + 1
#     print(x)

# print('done')
# #%%
# ux = u_x(times)
# x = x_out(ux)

# #For now, focus on x
# h_diff = Hamiltonian_diff(ux, 0, pathfunc)






# #%%
# i = 0
# times = np.array([0]) #First point is set by initial conditions.

# ux = np.array([0])
# uy = np.array([0])

# x = np.array([0])
# y = np.array([0])

# while close == False:
#     times = np.append(times, 10e-6 + times[-1])
#     x =
# #%%
# import matplotlib.pyplot as plt

# times = np.arange(0, 5, 0.01)
# inputs = u_x(times)
# outputs = x_out(inputs)
# plt.plot(times, outputs)


#%%






