#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# In this activity we will use Python to plot a common type of function used to describe capillary pressure. The curve was first described in [1980 by Martinus Th. van Genuchten](https://acsess.onlinelibrary.wiley.com/doi/abs/10.2136/sssaj1980.03615995004400050002x) (see Canvas for a copy of the PDF).
# 
# Let's begin by importing the packages we need.

# In[ ]:


import numpy as np
import os
import matplotlib.pyplot as plt


# ## van Genuchten function

# Depending on the field (i.e. soil science, hydrogeology, petroleum engineering) you will see the van Genuchten function rearranged a few different ways.

# In[ ]:


# Define residual water saturation (the residual water after drainage)
Swr = 0.1
# Define the residual gas saturation (defined as the water saturation after imbibition)
Snwr = 0
# define variable 'Sw' to describe water saturation
Sw = np.linspace((Swr + 0.001), (1 - Snwr - 0.001), num=100)
# capillary entry pressure in selected units
Pc_entry = 2 # kPa
m = 2
n = 3


# Here I've programmed this into a convient function with the different equation parameters required as function input.

# In[ ]:


# Van Genuchten function
def van_g_pc(Sw, Swr, Snwr, Pc_entry, m, n):
    # Now calculate the effective saturation (think of this as normalized saturation (ranges from 0-1))
    Se = (Sw - Swr)/((1 - Snwr) - Swr)
    Pc = Pc_entry*(Se**(-1/m)-1)**(1/n)
    return Pc


# In[ ]:


Pc_vg = van_g_pc(Sw, Swr, Snwr, Pc_entry, m, n)
Pc_vg2 = van_g_pc(Sw, Swr, Snwr, Pc_entry*3, m, n)


plt.plot(Sw, Pc_vg)
plt.plot(Sw, Pc_vg2)
plt.xlabel('Water Saturation', fontsize=18)
plt.ylabel('Capillary Pressure (kPa)', fontsize=18)
plt.show()


# ## Brooks-Corey function
# 
# Another common function is known as the Brooks-Corey function/model. It has been programmed in the following function:

# In[ ]:


# Brooks-Corey function
def brooks_corey_pc(Sw, Swr, Snwr, Pc_entry, m):
    # Now calculate the effective saturation (think of this as normalized saturation (ranges from 0-1))
    Se = (Sw - Swr)/((1 - Snwr) - Swr)
    Pc = Pc_entry*(Se**(-1/m))
    return Pc


# In[ ]:


m_bc = 3
Pc_entry_bc = 0.9
Pc_bc = brooks_corey_pc(Sw, Swr, Snwr, Pc_entry_bc, m_bc)

plt.plot(Sw, Pc_vg)
plt.plot(Sw, Pc_bc)
plt.xlabel('Water Saturation')
plt.ylabel('Capillary Pressure (kPa)')


# Add a legend to this plot. Adjust ```m_bc``` and ```Pc_entry``` so that the curves are as similiar as possible. What is the biggest difference between these two different functional forms?

# ## Activity: Plot another capillary pressure function
# Using Appendix B (page 147-158) of the [TOUGH3 user manual](https://tough.lbl.gov/assets//files/Tough3/TOUGH3_Users_Guide_v2.pdf) or section 2.3.1 (page 22-26) of the [Hydrus user manual](https://www.pc-progress.com/Downloads/Pgm_hydrus1D/HYDRUS1D-4.17.pdf), select another capillary pressure model and program it into a function. Once you have it programmed, find the parameters that give you the closest fit to the original van Genuchten model.

# In[ ]:


# New capillary pressure function
def new_pc(Sw, Swr, Snwr, Pc_entry):
    # define variable 'Sw' to describe water saturation
    Pc = 1
    return Pc


# In[ ]:




