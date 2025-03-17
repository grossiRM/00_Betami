#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This is a notebook designed to begin applying some of the concepts we discussed in the introductory notebook and to illustrate statistical moment analysis of discrete data.
# 
# First we import the libraries that we need

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# We will use numpy to build vector and matrix arrays and perform the summation calculations necessary for moment analysis.

# In[ ]:


# first define spatial coordinates
x = np.linspace(1, 20, num=8)
print(x)


# In[ ]:


# next lets define some concentration measurements (for example measured in wells)
C = np.array([0, 1, 4, 7, 8, 10, 2, 0])
print(C)


# In[ ]:


plt.plot(x,C)
plt.show() # Optional but prevents text from being printed with plot. This is also necessary if you have several plots in the same cell and what them to print separately.


# From this plot, where do you expect the center for mass to be? Remember the the center of mass is the location with half of the area under the curve on one side, and the half the area under the curve on the other side.

# The equation for the zero moment is: $m_{x,0} = \int C(x)  dx$
# 
# To perform the integration with discrete data we need to numerically integrate the concentration profile as a function of $x$. There are several ways to do this in Python. One of the easiest ways is with the `np.trapz` function. This function performs numerical integration using the composite trapezoidal rule. More details can be found in the [numpy documentation](https://numpy.org/doc/stable/reference/generated/numpy.trapz.html). Since the concentration measurements have equal spacing in $x$, we can either call the `np.trapz` function by defining that spacing or by just providing the entire $x$ array.

# In[ ]:


# Perform numerical integration by defining dx spacing
M0_dx = np.trapz(C, dx=x[1]-x[0])
# print result
print(M0_dx)

# Perform numerical integration by defining x array
M0_x = np.trapz(C, x)
print(M0_x)


# Do these results agree with what you would expect? Note that if $\delta x$ is not a constant (e.g. you have well observations that are not evenly spaced) then we need to rely on numerical integration using the $x$ array.
# 
# The equation for the first moment is: $m_{x,1} = \int x C(x) d x$
# 
# Like the zero moment, we can using numerical integration in Python to calculate the first moment. Run the cell below to calculate the first moment.

# In[ ]:


np.trapz(x*C, x)


# ## Activity:
# 
# The final step for calculating the normalized first spatial moment is to divide the first moment by the zero moment. Enter the correct equation in the box below.

# In[ ]:


# center_of_mass = ...
# print(center_of_mass)


# Now plot your result as a vertical dashed red line on the same plot as we saw earlier.

# In[ ]:




