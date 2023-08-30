import numpy as np
import matplotlib.pyplot as plt
import data_gen_toolsv3 as dgt


#%%
#sin wave tests
times = np.arange(0, 10 + 1e-5, 0.1)
outputsx = dgt.outputs(times, np.sin(times), mode = 'constant')
plt.plot(times, np.sin(times), label = 'inputs')
plt.plot(times, outputsx, label = 'outputs')
plt.legend()

plt.show()

#%%
#sin wave tests
times = np.arange(0, 10 + 1e-5, 0.1)
outputsx = dgt.outputs(times, np.cos(times), mode = 'accelerate')
plt.plot(times, np.cos(times), label = 'inputs')
plt.plot(times, outputsx, label = 'outputs')
plt.legend()

plt.show()
#%%
#sin wave tests

times = np.arange(0, 10 + 1e-5, 0.1)
#tolerance is min acceleration to choose acc or const function. Play around with it.
outputsx = dgt.outputs(times, np.sin(times), mode = 'auto', tolerance = 0.1)
plt.plot(times, np.sin(times), label = 'inputs')
plt.plot(times, outputsx, label = 'outputs')
plt.legend()

plt.show()

#%%

#straight line tests
outputsx = dgt.outputs(times, times, mode = 'accelerate')
plt.plot(times, times, label = 'inputs')
plt.plot(times, outputsx, label = 'outputs')
plt.legend()

plt.show()
#%%
#straight line tests
outputsx = dgt.outputs(times, times, mode = 'constant')
plt.plot(times, times, label = 'inputs')
plt.plot(times, outputsx, label = 'outputs')
plt.legend()

plt.show()
#%%

#What about combining both?

#straight line tests
outputsx = dgt.outputs(times, times, mode = 'accelerate') + dgt.outputs(times, times, mode = 'constant')
plt.plot(times, times, label = 'inputs')
plt.plot(times, outputsx, label = 'outputs')
plt.legend()

plt.show()
#%%
#Flat line test
outputsx = dgt.outputs(times, np.ones(len(times)), mode = 'accelerate', tolerance = 1)
plt.plot(times, np.ones(len(times)), label = 'inputs')
plt.plot(times, outputsx, label = 'outputs')
plt.legend()

plt.show()
#%%
#What about stiched together straight line segments?

def inputs(input_array):
    output_array = np.zeros_like(input_array)
    
    for i, x in enumerate(input_array):
        if x < 0:
            output_array[i] = 0
        elif x < 5:
            output_array[i] = x
        elif x <= 10:
            output_array[i] = 5
        else:
            output_array[i] = 0
    
    return output_array

#Complex line
outputsx = dgt.outputs(times, inputs(times), mode = 'constant') + dgt.outputs(times, inputs(times), mode = 'accelerate')
#outputsx = dgt.outputs(times, inputs(times), mode = 'auto')
plt.plot(times, inputs(times), label = 'inputs')
plt.plot(times, outputsx, label = 'outputs')
plt.xlabel('Time')
plt.legend()

plt.show()
#%%
#What about combining for an accelerating signal.

times = np.arange(0, 10 + 1e-5, 0.1)
outputsx = dgt.outputs(times, np.sin(2*times), mode = 'constant') + dgt.outputs(times, np.sin(2*times), mode = 'accelerate')
plt.plot(times, np.sin(2*times), label = 'inputs')
plt.plot(times, outputsx, label = 'outputs')
plt.legend()

plt.show()
#%%
#Noisy data
times = np.arange(0, 10 + 1e-5, 0.1)
noise = np.random.normal(0, 0.2, times.shape)
inputs = times + noise
outputsx = dgt.outputs(times, inputs, mode = 'accelerate')
plt.plot(times, inputs, label = 'inputs')
plt.plot(times, outputsx, label = 'outputs')

plt.legend()

plt.show()
#%%
import data_gen_toolsv4 as dgt
#What about combining for an accelerating signal.
#%%

times = np.arange(0, 10 + 1e-5, 0.1)
outputsx = dgt.outputs(times, np.sin(times))
plt.plot(times, np.sin(times), label = 'inputs')
plt.plot(times, outputsx, label = 'outputs')
plt.legend()

plt.show()
#%%
#straight line tests
outputsx = dgt.outputs(times, times, tolerance = 1e-2)
plt.plot(times, times, label = 'inputs')
plt.plot(times, outputsx, label = 'outputs')
plt.legend()

plt.show()
#%%
#Noisy data
times = np.arange(0, 10 + 1e-5, 0.1)
noise = np.random.normal(0, 0.2, times.shape)
inputs = times + noise
outputsx = dgt.outputs(times, inputs, mode = 'constant') + dgt.outputs(times, inputs, mode = 'accelerate')
plt.plot(times, inputs, label = 'inputs')
plt.plot(times, outputsx, label = 'outputs')

plt.legend()

plt.show()
#%%
#What about stiched together straight line segments?

def inputs(input_array):
    output_array = np.zeros_like(input_array)
    
    for i, x in enumerate(input_array):
        if x < 0:
            output_array[i] = 0
        elif x < 5:
            output_array[i] = x
        elif x <= 10:
            output_array[i] = 5
        else:
            output_array[i] = 0
    
    return output_array

#Complex line
outputsx = dgt.outputs(times, inputs(times))
#outputsx = dgt.outputs(times, inputs(times), mode = 'auto')
plt.plot(times, inputs(times), label = 'inputs')
plt.plot(times, outputsx, label = 'outputs')
plt.xlabel('Time')
plt.legend()

plt.show()