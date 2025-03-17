#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# In this activity we will expand the use of capillary pressure functions that we covered in the [Capillary Pressure Functions notebook](https://github.com/zahasky/Contaminant-Hydrogeology-Activities/blob/master/Capillary%20Pressure%20Functions.ipynb) to illustrate how these functions can be scaled for different fluid pairs and how these can be fit to laboratory measurements of capillary pressure. 
# 
# Let's begin by importing the packages we need.

# In[ ]:


import numpy as np
import os
import matplotlib.pyplot as plt


# ## Brooks-Corey function

# Here is the same Brooks-Corey function as last time. This will be useful today because it has less variables in the equation.

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

# Brooks-Corey function
def brooks_corey_pc(Sw, Swr, Snwr, Pc_entry, m):
    # Now calculate the effective saturation (think of this as normalized saturation (ranges from 0-1))
    Se = (Sw - Swr)/((1 - Snwr) - Swr)
    Pc = Pc_entry*(Se**(-1/m))
    return Pc


# And a quick plot to make sure everything is working.

# In[ ]:


# shape fitting parameter
m_bc = 2
Pc_entry_bc = 0.9
Pc_bc = brooks_corey_pc(Sw, Swr, Snwr, Pc_entry_bc, m_bc)

plt.plot(Sw, Pc_bc)
plt.xlabel('Water Saturation')
plt.ylabel('Capillary Pressure (kPa)')


# ## Fitting the equations to real data
# 
# The data in the file 'berea_mercury_capillary_pressure.txt' are capillary pressure measurements made with a [Micromeritics Mercury Intrusion Porosimeter AutoPore IV](https://www.micromeritics.com/Repository/Files/AUTOPORE_BROCHURE.pdf). This method is often also refered to as mercury injection capillary pressure measurements (MICP or MIP). This aparatus uses air and mercury as the working fluids (more information on the theory is available [here](https://www.micromeritics.com/Repository/Files/Mercury_Porosemitry_Theory_poster_.pdf)). In this unique case gas is the wetting phase and mercury is the nonwetting phase. The data were collected on a Berea sandstone. The first column in the text file is the gas saturation, the second column is the capillary pressure in PSI.

# In[ ]:


# Import capillary pressure data
datafile_name = 'berea_mercury_capillary_pressure.txt'
# if the data is not in your current directly then add the path information
path_to_datafolder = 'data_for_models' 
# This should return a path to your current working directory
current_directory = os.getcwd()
# IN A MAC delete the path_to_datafolder variable and uncomment this:
# data_file_with_path = os.path.join(current_directory, datafile_name)
data_file_with_path = os.path.join(current_directory, path_to_datafolder, datafile_name)
print(data_file_with_path)
pc_data = np.loadtxt(data_file_with_path, delimiter='\t')

Sw_micp = pc_data[:,0]
Pc_micp = pc_data[:,1]

# Plot the data
plt.plot(Sw_micp, Pc_micp, 'o')
plt.xlabel('Gas Saturation', fontsize=18)
plt.ylabel('Capillary Pressure (PSI)', fontsize=18)


# Using the Brooks-Corey function above, let's fit the equation to this measured data. To do this robustly let's implement a [least squares minimization](https://en.wikipedia.org/wiki/Least_squares#:~:text=The%20method%20of%20least%20squares,results%20of%20every%20single%20equation), although there are many other minimization options.
# 
# To begin, we can see from the data that the maximum gas saturation is very near 1 for one of the data points. We therefore can assume that the residual nonwetting saturation is equal to zero. This measurement at a gas saturation near 1 also gives us an approximate capillary entry pressure.

# In[ ]:


# Since Sw = 1 at some point we know that the residual nonwetting saturation is equal to zero
Snwr = 0
# capillary entry pressure in selected units. A good starting point is the mimimum capillary pressure measurement
Pc_entry = np.min(Pc_micp) # PSI
print('approximation of capillary entry pressure = ' + str(Pc_entry) + ' PSI')


# The simplest way to perform the minimization is to define some parameter space for the Brooks-Corey exponent (```m```) and the minimum wetting phase saturation. This miminum wetting phase saturation should be very near the minimum value measured with the MICP as plotted above. Note that this is not always the case and depends on the laboratory method used to measure capillary pressure, the rock/soil type, and potential data processing done of the raw measurement data.

# In[ ]:


# Define a range of possible m values in the Brooks-Corey function to search
M = np.linspace(0.1, 2, num=100)
# Use the minimum value of the wetting saturation to calculate the residual wetting saturation
SWR = np.min(Sw_micp) + np.linspace(-0.1, 0.1, num=20)
# In the for loop below we will iteratively calculate the mismatch of the equation and when we find a closer fit 
# will update the least squares minimum and the corresponding m and Swr values
least_square_min = 10000

# Start the for loop
for m in M:
    for Swr in SWR:
        # call our Brooks-Corey function for the current value of m and Swr
        Pc_bc = brooks_corey_pc(Sw_micp, Swr, Snwr, Pc_entry, m)
        # least squares calculation
        least_square_m = np.sum((Pc_bc- Pc_micp)**2)
        # If the mismatch is lower the the previous least squares then update the m and Swr values
        if least_square_m < least_square_min:
            # update the minimum least squares value
            least_square_min = least_square_m
            min_m = m
            min_Swr = Swr
             
# Now calculate the capillary pressure based on our least squares result
Pc_fit = brooks_corey_pc(Sw_micp, min_Swr, Snwr, Pc_entry, min_m)
# Plot the data
plt.plot(Sw_micp, Pc_fit)
plt.plot(Sw_micp, Pc_micp, 'o')
plt.xlabel('Gas Saturation', fontsize=18)
plt.ylabel('Capillary Pressure (PSI)', fontsize=18)
    
print('The best fit Swr is equal to ' + str(min_Swr))
print('The best fit m is equal to ' + str(min_m))


# ## Activity: Scaling capillary pressure measurements
# 
# As we see with the data above, capillary pressure is usually measured in the laboratory with air-water or air-mercury fluid pairs. To apply these measurements to different fluid pairs, for example a NAPL and water, it is necessary to scale the capillary pressure data. As described in the course notes, the equation to scale capillary pressure is $Pc_{scaled} = Pc_{measured} \frac{\sigma_{scaled} \text{cos}\theta_{scaled}}{\sigma_{measured} \text{cos}\theta_{measured}}$. Here the subscript 'scaled' refers to the fluid pairs we want to calculate for and the subscript 'measured' applies to the fluid pairs used for the capillary pressure measurement.
# 
# Using the capillary function that you fit above, scale the data from air-mercury to air-water. Assume the contact angle of air-water is 10 degrees ($\theta_{scaled}$), the contact angle of air-mercury is 140 degrees ($\theta_{measured}$). The interfacial tension of air-water is 72 mN/m ($\sigma_{scaled}$), the interfacial tension of air-mercury is 485 mN/m ($\sigma_{measured}$).

# In[ ]:


# scale data
# capillary pressure scaling
def scale_pc_fun(Pc, sig_scaled, sig_measured, theta_scaled, theta_measured):
    # Pc_scaled =  ## Input scaling equation
    return Pc_scaled

# hint
print((np.cos(140*2*3.1415/360)))
print((np.cos((180-140)*2*3.1415/360)))

# Now call function
# Pc_scaled = scale_pc_fun(Pc_fit, other input...)
# Pc_micp_scaled = scale_pc_fun(Pc_micp, other input...)


# convert from psi to kPa
# Pc_scaled_kpa = Pc_scaled*6.89
# Pc_micp_scaled_kpa = Pc_micp_scaled*6.89


# Now plot the results!

# In[ ]:


# plt.plot(Sw_micp, Pc_scaled_kpa, label='raw data')
# plt.plot(Sw_micp, Pc_micp_scaled_kpa, 'o', label='scaled fit')
plt.plot(Sw_micp, Pc_fit*6.89, label='Unscaled Pc Fit')
plt.plot(Sw_micp, Pc_micp*6.89, 'o', label='Unscaled data')
plt.xlabel('Water Saturation')
plt.ylabel('Capillary Pressure (kPa)')
plt.legend()


# Plot the results for another fluid pair based on the NAPL contact angle and interfacial tension (IFT). How does decreasing the contact angle (changing wettability) change the capillary pressure curves? 

# In[ ]:




