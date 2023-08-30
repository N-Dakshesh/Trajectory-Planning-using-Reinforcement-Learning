import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import Data_gen_tools as dgt
#%%
# Define the symbols
s, T, t, tau = sp.symbols('s T t tau', real=True)

# Define the value of T
T_val = 2  # You can change this to any desired value
theta = 1

# Define the Laplace domain expression
laplace_const = 1 / ((s * T_val / 6) + 1)**6
laplace_acc = s**2 / (theta * (s * T_val / 6 + 1)**6)

# Take the inverse Laplace transform
inverse_laplace_const = sp.inverse_laplace_transform(laplace_const, s, t)
inverse_laplace_acc = sp.inverse_laplace_transform(laplace_acc, s, t)

print(inverse_laplace_const)
print(inverse_laplace_acc)

sp.plot(inverse_laplace_acc, (t, 0, 5), title='Plot of Acceleration Transfer Function', xlabel='Time (s)', ylabel='Amplitude', legend=True)
sp.plot(inverse_laplace_const, (t, 0, 5), title='Plot of Constant Transfer Function', xlabel='Time (s)', ylabel='Amplitude', legend=True)


# Convert the inverse Laplace transform expression to a Python function
inverse_laplace_constfunc = sp.lambdify(t, inverse_laplace_const, 'numpy')
inverse_laplace_accfunc = sp.lambdify(t, inverse_laplace_acc, 'numpy')
#%%

def acc(inputs):
    #Calculates acceleration, and adds 0 as the initial and final accelerations!
    tdiff = 10e-6
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
            transpoint = inverse_laplace_constfunc(times[i])
            trans = np.append(trans, transpoint)
        else:
            transpoint = inverse_laplace_accfunc(times[i])
            trans = np.append(trans, transpoint)

    return trans


def outputs(times, inputs):
    #Construct discrete transfer function - if input has 0 accleration vs non 0
    #Calculate acceleration
    trans = trans_maker(times, inputs)*(times[1] - times[0])
    output = np.convolve(trans, inputs)
    length = len(inputs)
    #Convolution doubles length, so restrict length to inputs (and therefore times to inputs!)
    output = output[0:length]

    return output


#%%
def ramp_function(t):
    return np.maximum(0, t)

time_values = np.linspace(0, 5, 500)
inputs = ramp_function(time_values)
out = outputs(time_values, inputs)

plt.figure(figsize=(8, 6))
plt.plot(time_values, inputs, label = 'Input')
plt.plot(time_values, out, label = 'Output')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.title('Convolution of Inverse Laplace Transform with Ramp Function')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(time_values, trans_maker(time_values, inputs))


#%%
# Create an array of time values for plotting
time_values = np.linspace(0, 10, 500)

# Evaluate the inverse Laplace transform function at the time values
inverse_laplace_result = inverse_laplace_func(time_values)

# Define the ramp function as a Python function
def ramp_func(t):
    return np.where(t >= 0, t, 0)

# Perform the convolution using np.convolve
convolved_result = np.convolve(inverse_laplace_result, ramp_func(time_values), mode='full')[:len(time_values)] * (time_values[1] - time_values[0])

# Plot the result
plt.figure(figsize=(8, 6))
plt.plot(time_values, convolved_result, label='Convolved Result')
plt.plot(time_values, ramp_func(time_values), label = 'Input')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.title('Convolution of Inverse Laplace Transform with Ramp Function')
plt.legend()
plt.grid(True)
plt.show()