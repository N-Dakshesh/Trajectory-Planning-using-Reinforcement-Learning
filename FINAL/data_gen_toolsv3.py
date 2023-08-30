"""
Implements Transfer function - converts inputs to outputs.
Also controls Tracking error (T_val) and theta - currently both set to 1.
Note tracking must be integer. 
"""


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Define the symbols
s, T, t, tau = sp.symbols('s T t tau', real=True)

# Define the value of T
T_val = 1  # You can change this to any desired value
theta = 1



# Define the Laplace domain expression
laplace_const = 1 / ((s * T_val / 6) + 1)**6
laplace_acc = s**2 / (theta * (s * T_val / 6 + 1)**6)

# Take the inverse Laplace transform
inverse_laplace_const = sp.inverse_laplace_transform(laplace_const, s, t)
inverse_laplace_acc = sp.inverse_laplace_transform(laplace_acc, s, t)

print(inverse_laplace_const)
print(inverse_laplace_acc)

# Convert the inverse Laplace transform expression to a Python function
inverse_laplace_constfunc = sp.lambdify(t, inverse_laplace_const, 'numpy')
inverse_laplace_accfunc = sp.lambdify(t, inverse_laplace_acc, 'numpy')


def acc(inputs, tdiff = 0.1):
    #Calculates acceleration, and adds 0 as the initial and final accelerations!
    tdiff = 0.1
    v = np.gradient(inputs, tdiff)
    acc = np.gradient(v, tdiff)
    #acc[abs(acc > 1)] = 0

    return acc



def trans_maker_const(times, inputs, tolerance = 1e-2):
    #Creates unique transfer function, depending on acceleration
    trans = np.array([])
    for i in range(0, len(inputs)):
        transpoint = inverse_laplace_constfunc(times[i])
        trans = np.append(trans, transpoint)


    return trans

def trans_maker_acc(times, inputs, tolerance = 1):
    #Creates unique transfer function, depending on acceleration
    trans = np.array([])
    for i in range(0, len(inputs)):
        transpoint = inverse_laplace_accfunc(times[i])
        trans = np.append(trans, transpoint)

    return trans

def outputs(times, inputs, mode = 'constant', tolerance = 1e-2):
    #Construct discrete transfer function - if input has 0 accleration vs non 0
    #Calculate acceleration

    if mode == 'constant_experimental':
        trans = trans_maker_const(times, inputs, tolerance)*(times[1] - times[0]) + trans_maker_acc(times, inputs, tolerance)*(times[1] - times[0])
        output = np.convolve(trans, inputs)
        length = len(inputs)
        #Convolution doubles length, so restrict length to inputs (and therefore times to inputs!)
        output = output[0:length]
    
        return output

    elif mode == 'constant':
        trans_maker = trans_maker_const
    elif mode == 'accelerate':
        trans_maker = trans_maker_acc
    else:
        raise Exception("Need to input 'constant_experimental', 'constant' or 'accelerate'.")


    trans = trans_maker(times, inputs, tolerance)*(times[1] - times[0])
    output = np.convolve(trans, inputs)
    length = len(inputs)
    #Convolution doubles length, so restrict length to inputs (and therefore times to inputs!)
    output = output[0:length]

    return output



