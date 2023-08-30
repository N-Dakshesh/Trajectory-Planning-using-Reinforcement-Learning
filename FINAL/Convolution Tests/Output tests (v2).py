import numpy as np
import Data_gen_tools as dgt
import matplotlib.pyplot as plt
#%%
t = np.linspace(0, 1, 100)
u = dgt.u_x(t)

plt.plot(t, u)
plt.xlabel('Time')
plt.ylabel('Input')
plt.show()

#%%
output = dgt.outputs(t, u)


plt.plot(t, output)
plt.xlabel('Time')
plt.ylabel('Output')
plt.show()
#%%
def f(t):
    return np.heaviside(t - 500e-6, 0.5)

u = f(t)
plt.plot(t, u)
plt.xlabel('Time')
plt.ylabel('Input')
plt.show()


#%%
output = dgt.outputs(t, u)


plt.plot(t, output)
plt.xlabel('Time')
plt.ylabel('Output')
plt.show()
