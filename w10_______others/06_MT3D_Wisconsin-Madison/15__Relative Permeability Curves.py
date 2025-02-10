#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# In this activity we will plot a few common relative permeability functional forms and explore a common approach for calculating relative permeability from capillary pressure curves.
# 
# Start by importing our necessary packages:

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# ## Brooks-Corey relative permeability function
# 
# In the [Capillary Pressure Functions notebook](https://github.com/zahasky/Contaminant-Hydrogeology-Activities/blob/master/Capillary%20Pressure%20Functions.ipynb) we explored a few common capillay pressure functional forms. Both the van Genuchten and Brooks-Corey capillary pressure models have corresponding relative permeability functional forms. The Brooks-Corey empirical equations are programmed in the following function:

# In[ ]:


# Brooks-Corey function
def brooks_corey_rel_perm_fun(Sw, Swr, Snwr, krw0, krnw0, m):
    # Now calculate the effective saturation (think of this as normalized saturation (ranges from 0-1))
    Se = (Sw - Swr)/((1 - Snwr) - Swr)
    # Water relative permeability
    krw = krw0*Se**(2/m + 3)
    # Nonwetting phase relative permeability
    krnw = krnw0*(1-Se)**2*(1-Se**(2/m + 1))
    return krw, krnw


# When deciding which model to use it is best to start with the most simple model and move to more complex functions as neccessary to fit your measurement data. Often a modified (simplified) Brooks-Corey functional form is sufficient.

# In[ ]:


# Modified Brooks-Corey function
def mod_brooks_corey_rel_perm_fun(Sw, Swr, Snwr, krw0, krnw0, nw, nnw):
    # Now calculate the effective saturation (think of this as normalized saturation (ranges from 0-1))
    Se = (Sw - Swr)/((1 - Snwr) - Swr)
    # Water relative permeability
    krw = krw0*Se**nw
    # Nonwetting phase relative permeability
    krnw = krnw0*(1-Se)**nnw
    return krw, krnw


# As you can see there are strong similiarities with the classic Brooks-Corey model but because of the simplicity of the exponential form, this model is also more intuative. 

# In[ ]:


# Define residual water saturation (the residual water after drainage)
Swr = 0.1
# Define the residual gas saturation (defined as the water saturation after imbibition)
Snwr = 0
# define variable 'Sw' to describe water saturation
Sw = np.linspace((Swr + 0.001), (1 - Snwr - 0.001), num=100)

# End point relative permeability. 
# Water end point relative permeability, this is equal to the relative permeabiltiy at the residual gas saturation
krw0 = 1
# Nonwetting phase end point relative permeability
krnw0 = 1

# Call Brooks-Corey function
krw_bc, krnw_bc = brooks_corey_rel_perm_fun(Sw, Swr, Snwr, krw0, krnw0, 1)
# Call modified Brooks-Corey function
krw_mbc, krnw_mbc = mod_brooks_corey_rel_perm_fun(Sw, Swr, Snwr, krw0, krnw0, 3, 3)

# Plot the results
plt.plot(Sw, krw_bc, label='Brooks-Corey Water Rel Perm')
plt.plot(Sw, krnw_bc, label='Brooks-Corey Nonwetting Rel Perm')

plt.plot(Sw, krw_mbc, label='Mod BC Water Rel Perm')
plt.plot(Sw, krnw_mbc, label='Mod BC Nonwetting Rel Perm')

plt.xlabel('Water Saturation')
plt.ylabel('Relative Permeability')
plt.legend()
plt.xlim(0, 1)


# ##### Increase the exponent in the modified Brooks-Corey curves. How do the relative permeability curves change? How does this change in the relative permeability translate to the physical system? 

# Note that sometimes when the exponents are very high and the relative permeability values are very small, you may see relative permability plotted on log scale on the y-axis. You could explore the differences by adding the following code to your plot script:
# 
#     plt.yscale('log')
#     plt.ylim(0.001, 1) # adjust for data range

# ## Burdine's Method to calculate relative permeability from a capillary pressure curve
# It is usually much easier to measure capillary pressure than relative permeability so it is convenient to have a method to calculate relative permeability from a capillary pressure curve. Burdine's method is most commonly used to accomplish this. In this activity you will see how Burdine's method enables us to use a capillary pressure curve to calculate the corresponding relative permeability curves. You can see the theoretical derivation of this Brooks-Corey model in the coures notes.
# 
# Let's start with the Brooks-Corey capillary pressure function that we saw in the [Capillary Pressure Functions notebook](https://github.com/zahasky/Contaminant-Hydrogeology-Activities/blob/master/Capillary%20Pressure%20Functions.ipynb).

# In[ ]:


# Brooks-Corey capillary pressure function 
def brooks_corey_pc(Sw, Swr, Snwr, Pc_entry, m):
    # Now calculate the effective saturation (think of this as normalized saturation (ranges from 0-1))
    Se = (Sw - Swr)/((1 - Snwr) - Swr)
    Pc = Pc_entry*(Se**(-1/m))
    return Pc


# Now let's plot an example capillary pressure curve.

# In[ ]:


# Define residual water saturation (the residual water after drainage)
Swr = 0.1
# Define the residual gas saturation (defined as the water saturation after imbibition)
Snwr = 0
# define variable 'Sw' to describe water saturation
Sw = np.linspace((Swr + 0.001), (1 - Snwr - 0.001), num=100)

m_bc = 3
Pc_entry_bc = 0.9
Pc_bc = brooks_corey_pc(Sw, Swr, Snwr, Pc_entry_bc, m_bc)

plt.plot(Sw, Pc_bc)
plt.xlabel('Water Saturation')
plt.ylabel('Capillary Pressure (kPa)')


# Now define a function that takes in water saturation and capillary pressure and calculates wetting and nonwetting relative permeability using Burdine's method. The full equations for this are given in the relative permeability section of chapter 3 of the course notes.

# In[ ]:


# Burdine function
def burdine_fun(Sw, Swr, Snwr, Pc):
    # Normalized saturation
    Se = (Sw - Swr)/((1 - Snwr) - Swr)
    # both of the relative permeability integrals have the same fixed denominator
    denom = np.trapz(1/Pc**2, Sw)
    # preallocate the array for saving the values
    krw_burdine = np.zeros(np.shape(Sw))
    krnw_burdine = np.zeros(np.shape(Sw))
    
    # integrate from Swr to Sw
    for i in range(len(Sw)-1,0,-1):
        kw_numer = 1/Pc[:i]**2
        krw_burdine[i] = Se[i]**2*np.trapz(kw_numer, Sw[:i])/denom
    
    # integrate from Sw to 1    
    for i in range(len(Sw)):
        knw_numer = 1/Pc[i:]**2
        krnw_burdine[i] = (1-Se[i])**2*np.trapz(knw_numer, Sw[i:])/denom
    
        ## Add plot showing areas for visualization
    return krw_burdine, krnw_burdine


# Now let's see if this works by calculating the Brooks-Corey relative permeability using the same ```m``` as we used for defining capillary pressure (```Pc_bc```).

# In[ ]:


# Calculate brooks-corey rel perm
krw_bc, krnw_bc = brooks_corey_rel_perm_fun(Sw, Swr, Snwr, krw0, krnw0, m_bc)

# Calculate rel perm from brooks-corey capillary pressure curve
krw_burdine, krnw_burdine = burdine_fun(Sw, Swr, Snwr, Pc_bc)

plt.plot(Sw, krw_bc, label='Brooks-Corey Water Rel Perm')
plt.plot(Sw, krnw_bc, label='Brooks-Corey Nonwetting Rel Perm')

plt.plot(Sw, krw_burdine, '--', label='Burdine Water Rel Perm')
plt.plot(Sw, krnw_burdine, '--', label='Burdine Nonwetting Rel Perm')

plt.xlabel('Water Saturation')
plt.ylabel('Relative Permeability')
plt.legend()
plt.xlim(0, 1)

print(Sw)


# Pretty cool, right?!

# ## Activity: Use Burdine's method to calculate relative permeability from capillary pressure data
# 
# ##### Now, using the capillary pressure function that you fit to the capillary pressure data in the [Capillary Pressure Curves notebook](https://github.com/zahasky/Contaminant-Hydrogeology-Activities/blob/master/Capillary%20Pressure%20Curves.ipynb), calculate the corresponding relative permeability curves using Burdine's Method.

# In[ ]:




