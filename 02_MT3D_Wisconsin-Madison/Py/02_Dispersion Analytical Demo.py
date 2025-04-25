#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# In this activity we will use python to better understand the advection-dispersion equation and plot some of the solutions.

# In[1]:


# Import the neccesary libraries
# Import only the scipy.special.erfc (complementary error function) and 
#     scipy.special.erf (error function) from the scipy Library
from scipy.special import erfc as erfc
from scipy.special import erf as erf
import numpy as np
import math
import matplotlib.pyplot as plt


# ## Analytical model 
# ### Setup

# Now let's define some of the characteristics of the system that we want to model. For conceptual simplicity lets imagine a homogenous sand column, similiar to the ones you worked with in lab in GEOSCI/GEOENG 627. First define the initial (t=0) conditions and boundary conditions (inlet and outlet).

# In[2]:


# Observation time(time in seconds)
t = 60*3
# Injection rate (mL/min)
q = 5
# Injected concentration
C0 = 1
# Initial concentration
Ci = 0


# Now lets define some (arbitrary) column properties

# In[3]:


# X-coordinates from column inlet to outlet (at 50 cm)
x = np.linspace(0.1, 10, num=1000)
# column radius
col_radius = 2.5
# porosity
phi = 0.3;
# dispersivity [cm] 
dispersivity = 0.5


# Note that the 'dispersivity' is the alpha in the equations from the course notes. This must be multiplied by the advection velocity to get dispersion. Lets calculate the dispersion.

# In[4]:


# calculate advection velocity
area = math.pi*(col_radius)**2
v = q/60/area/phi # divide by 60 to get flow rate from cm/min to cm/sec
print("advection velocity: " + str(v))

# Dispersion
D = v*dispersivity 
print(D)


# #### What are the units of dispersion again?

# ### Analytical solution: Type 1 boundary conditions and continous solute injection

# In[5]:


# Analytical solution: See lecture slides or (Parker and van Genuchten, 1984) for details
def ADE_type1_fun(x, t, v, D, C0):
    # Note that the '\' means continued on the next line
    C = C0*((1/2)*erfc((x - v*t)/(2*np.sqrt(D*t))) + \
        (1/2)*np.exp(v*x/D)*erfc((x + v*t)/(2*np.sqrt(D*t))))
    # Return the concentration (C) from this function
    return C

# Now call our function
Conc_time_type1 = ADE_type1_fun(x, t, v, D, C0)
# Plot the results as a function of x
plt.plot(x, Conc_time_type1)
plt.show()


# Note that the complementary error function (erfc) is back. Do you remember the shape of this curve from the diffusion demo?

# #### What do we call this plot again (concentration profile or breakthrough curve)? Label the axis...
# 
# Now let's look at how this curve changes with different values of hydrodynamic dispersion

# In[6]:


# Now call our function with 4x dispersivity
Conc_D3x = ADE_type1_fun(x, t, v, D*3, C0)
# Now instead we explicity include diffusion (Dstar) D = v*dispersivity + Dstar
Dstar = 5e-8*(100**2) #[cm^2/sec]
Conc_D3x_w_diff = ADE_type1_fun(x, t, v, D*3 + Dstar, C0)
# Now instead dispersivity is equal to zero (not phyiscial) or D = v*dispersivity + Dstar, which simplifies to D = Dstar
Conc_diff = ADE_type1_fun(x, t, v, Dstar, C0)

# Solution to advection equation (no diffusion or dispersion)
C_advect = C0*np.ones(len(x))
ind = x>v*t
C_advect[ind] = 0

# Plot the results as a function of x
plt.figure(figsize=(6, 3), dpi=150)
plt.plot(x, Conc_time_type1, label =r'$\alpha$ = 0.5 cm')
plt.plot(x, Conc_D3x, label=r'$\alpha$ = 1.5 cm')
plt.plot(x, Conc_D3x_w_diff, '--', label = r'$\alpha$ = 1.5 cm, $D^*$= 5e-6 cm$^2$/s')
plt.plot(x, Conc_diff, label=r'$\alpha$ = 0 cm, $D^*$= 5e-6 cm$^2$/s')
plt.plot(x, C_advect, 'k', label=r'$\alpha$ = 0 cm, $D^*$= 0 cm$^2$/s')
plt.xlabel('Distance from constant concentration boundary [cm]')
plt.ylabel(r'C/C$_0$')
plt.legend()
plt.show()


# Is it clear why sometimes diffusion is simply neglected from the hydrodynamic dispersion term?

# ### Analytical solution: Type 3 boundary conditions and continous solute injection

# In[7]:


# Analytical solution: See lecture slides or (Parker and van Genuchten, 1984 eq 9b) for details
def ADE_type3_fun(x, t, v, D, C0):
    C = C0*((1/2)* erfc((x - v*t)/(2* np.sqrt(D*t))) + \
        np.sqrt((v**2*t)/(math.pi*D))* np.exp(-(x - v*t)**2/(4*D*t)) - \
        (1/2)*(1 + (v*x/D) + (v**2*t/D))* np.exp(v*x/D)* erfc((x + v*t)/(2* np.sqrt(D*t))))
    return C
    
# Now call our function
Conc_time_type3 = ADE_type3_fun(x, t, v, D, C0)

# Plot the results as a function of x
plt.plot(x, Conc_time_type1, label='Type 1 Inlet BC')
plt.plot(x, Conc_time_type3, label='Type 3 Inlet BC')

# Format the plots
plt.legend()
plt.show()


# Note that the input and system geometry is identical except for the boundary conditions!

# ## Activity:
# Using these functions, **evaluate how the shape of the curves depends on the dispersion**. Describe and discuss what you see. If two cores have identical geometry and flow rate conditions but different dispersion behavior, what does this mean? 
# 
# **How does the solute concentration profile near the inlet change as a function of dispersion?** Are there conditions under which Type 1 and Type 3 boundary conditions produce similiar results? 
# 
# In the cell(s) below generate a concentration profile with both boundary conditions at later time

# In[ ]:





# ### Principle of superposition 

# As we saw in GEOSCI 627, the superposition of analytical solutions is a powerful tool for describing systems with more complex initial conditions. The principle of superposition
# 
# To apply superposition we need a few conditions to be true. First, we need a “linear” differential equation. That is, one for which: a) the dependent variable and all its derivatives are of the first degree (i.e. the power on any dependent variable terms is one) b) the coefficients are constants or functions of only the independent variable(s) The second condition is that we the differential equation needs to be homogeneous. A linear differential equation is mathematically homogeneous if the term(s) without the dependent variable equal zero.
# 
# When the advection-diffusion equation (ADE) is linear and homogeneous, sums of the solutions are also linear. This important condition allows us to use the principle of superposition to analytically solve the ADE and the diffusion equation for relative complex (i.e. mixed or inhomogeneous) boundary and initial conditions. As described by Bear (1972) "The principle of superposition means that the presence of one boundary condition does not affect the response produced by the presence of other boundary or initial conditions. Therefore, to analyze the combined effect of a number of boundary conditions (excitations) we may start by solving for the effect of each individual excitation and then combine the results." For example, the principle of superposition is often used to analytically describe the transport behavior of finite pulse of solute.
# 
# Work through the example below to see an illustration of this principle.

# ### Analytical solution: Type 1 boundary conditions and pulse solute injection

# In[8]:


# define observation location (outlet)
xout = 10
# Define the time array
ta = np.linspace(0.1, 60*30, num=100)
# Define the length of time of the pulse injection
t_pulse = 60*3
# Define the time array shifted by the length of pulse injection
t_t0 = ta-t_pulse

# Now call our function
Conc_time_type1_t = ADE_type1_fun(xout, ta, v, D, C0)
Conc_time_type1_t_t0 = ADE_type1_fun(xout, t_t0, v, D, C0)

Conc_pulse_solution = Conc_time_type1_t - Conc_time_type1_t_t0

# Plot the results as a function of time
plt.plot(ta, Conc_pulse_solution, label='Type 1 pulse')
plt.plot(ta, Conc_time_type1_t, label='Type 1 continuous')
plt.plot(ta, Conc_time_type1_t_t0, label='Type 1 continuous + t$_0$')

# Format the plots
plt.legend()
plt.show()


# #### Side note
# If you started the `ta` array from zero and then subtracted some time of injection then `t_t0 = ta-t_pulse` will have negative numbers. Negative numbers in the `erfc` and `sqrt` function are what lead to the warning above. To make our function more robust we can use the ```np.real``` numpy function to our time variable inside the 'erfc' functions, or we can record the location of negative values of ```t``` (see 'indices_below_zero'). Set ```t``` at these locations equal to some arbitrary positive number, and then make sure to set the corresponding concentrations equal to zero.

# In[9]:


# Analytical solution without errors when negative times are input
def ADE_type1_real_fun(x, t, v, D, C0):
    # Identify location of negative values in time array
    indices_below_zero = t <= 0
    if indices_below_zero.any() == True:
        # set values equal to 1 (but this could be anything)
        t[indices_below_zero] = 1
    
    # Note that the '\' means continued on the next line
    C = C0*((1/2)*erfc((x - v*t)/(2*np.sqrt(D*t))) + \
         (1/2)*np.exp(v*x/D)*erfc((x + v*t)/(2*np.sqrt(D*t))))
    
    if indices_below_zero.any() == True:
        # Now set concentration at those negative times equal to 0
        C[indices_below_zero] = 0
        
    # Return the concentration (C) from this function
    return C


# In[10]:


# Now call our function
Conc_time_type1_t = ADE_type1_real_fun(xout, ta, v, D, C0)
Conc_time_type1_t_t0 = ADE_type1_real_fun(xout, t_t0, v, D, C0)

Conc_pulse_solution = Conc_time_type1_t - Conc_time_type1_t_t0

# Plot the results as a function of time
plt.plot(ta, Conc_pulse_solution, label='Type 1 pulse')
plt.plot(ta, Conc_time_type1_t, label='Type 1 continuous')
plt.plot(ta, Conc_time_type1_t_t0, label='Type 1 offset by pulse time')

# Format the plots
plt.legend()
plt.show()


# In the plot above the 'continous line' minus the 'offset by pulse time' equal the 'pulse' solution. This is the principle of superposition. Pretty cool huh!?

# #### How does changing ```t_pulse``` (in the first code cell of this activity) change the height and shape of the curve 'Type 1 pulse' curve? How would you determine the value of ```t_pulse``` if I asked you to model the case of 10 mL of solute injected?

# The answer is (double click this cell to edit) ...

# In[ ]:




