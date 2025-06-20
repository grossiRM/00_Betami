#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# In this activity we will use python and some of the built in functions to model a solute diffusion process. A complex function that arrises in many analytical solutions is termed the [Error function](https://en.wikipedia.org/wiki/Error_function).
# 
# This is the first notebook where we will use the SciPy package. This should have been installed with the rest of the required packages during the [MODFLOW, Python, and FloPy Setup notebook](https://github.com/zahasky/Contaminant-Hydrogeology-Activities/blob/master/MODFLOW%2C%20Python%2C%20and%20FloPy%20Setup.ipynb). To test that you have SciPy properly install try running the following cell.

# In[ ]:


# Import only the math.erfc (complementary error function) and math.erf (error function) from the math Library
from scipy.special import erfc as erfc
from scipy.special import erf as erf

# Print the error function of a few different numbers
print (erfc(1))
print(erfc(0))


# We also need to import a few useful packages for working with vectors (numpy) and for plotting (matplotlib.pyplot)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'widget')
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt


# Now we can plot the error function and complementary error function (this as a function equal to 1-erf(x)) in this example to better understand their shape:

# In[ ]:


# define variable 'xe' with 100 values from -3 to 3
xe = np.linspace(-3, 3, num=100)
plt.figure(dpi=100)
# plot error function
plt.plot(xe, erf(xe), linewidth=3)
# plot complementary error function
plt.plot(xe, erfc(xe), '--')
plt.show()


# In[ ]:


plt.figure(dpi=100)
# plot equivalent to complementary error function
plt.plot(xe, 1-erf(xe), linewidth=3)
# plot complementary error function
plt.plot(xe, erfc(xe), '--')
plt.show()


# ## Diffusion function definition

# Now let's define a function that calculates the diffusion between a region with solute initially present with a concentration equal to one. This region is at x<0

# In[ ]:


def diffusion_fun(x, t_month, Df, C0):
    # convert time from months to seconds (same units as D)
    t = 60*60*24*t_month
    # Equation for concentration profile as a function of space (x) and time (t)
    C = C0*(erfc((x)/(2*np.sqrt(Df*t))))
    # Return the concentration (C) from this function
    return C


# Aside: The square root function also exists in the 'math' library. It is important to use the 'numpy' library if we want to take the square root of an array of numbers. For example:

# In[ ]:


np.sqrt(xe[-5:])


# Attempting the same operation with the math library will result in an error stating that 'only size-1 arrays can be converted to Python scalars'. This means you can only perform the operation on scalars.

# In[ ]:


import math
math.sqrt(xe[-5:])


# In[ ]:


# Define diffusion coefficient
Df = 5E-9
# Define spatial coordinates
x = np.linspace(0, 1, num=200)
# Define initial concentration
C0 = 1

# Profile after one tenth of a month
t = 1/10
C = diffusion_fun(x, t, Df, C0)

plt.figure(dpi=100)
plt.plot(x, C)
plt.show()


# What are the units of space, time, and the diffusion coefficient?

# In[ ]:


# Profile after one year
t = 12
C = diffusion_fun(x, t, Df, C0)

plt.figure(dpi=100)
plt.plot(x, C)
plt.show()


# Note that you can use the error function or the complementary error function to define your diffusion solution. This is demonstrated in the example below.

# In[ ]:


def diffusion_fun_erf(x, t_month, Df, C0):
    # Equation for concentration profile as a function of space (x) and time (t)
    t = 60*60*24*t_month
    C = C0*(1-erf((x)/(2*np.sqrt(Df*t))))
    # Return the concentration (C) from this function
    return C


# In[ ]:


# set up plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel('Distance from source')
ax.set_ylabel('Concentration')
ax.set_title('Diffusive transport as a function of time')
 
# Make interactive plot
@widgets.interact(t_month=(1e-4, 120, 1))
def update(t_month = 12):
    """Remove old lines from plot and plot new one"""
    [l.remove() for l in ax.lines]
    ax.plot(C, diffusion_fun_erf(C, t_month, Df, C0=1), color='k')


# ## Activity:
# 
# #### Using this code, test the impact of different diffusion coefficients. 
# 
# How do you expect this to change the shape of this curve? 
# 
# Plot the case of Df = 1E-9 after 1 day, 1 month, and 6 months. In a second cell plot the case of Df = 1E-10 after 1 day, 1 month, and 6 months. Note that the time in the function we defined above is in units of seconds.
# 
# In each plot that you generate make sure to add axis labels and a legend.

# In[ ]:


# cell 1 for calculating and plot concentration profiles for Df = 1E-9


# In[ ]:


# cell 2 for calculating and plot concentration profiles for Df = 1E-10


# In[ ]:




