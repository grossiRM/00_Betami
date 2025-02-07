#!/usr/bin/env python
# coding: utf-8

# In[24]:


from scipy.special import erfc as erfc
from scipy.special import erf as erf
# Type 1 inlet conditions, infinite solution
def analytical_model_1D_t1(x, t, v, al):
    # Dispersion
    D = v*al
    # Analytical solution: See lecture slides or (Parker and van Genuchten, 1984) for details
    # Note that the '\' means continued on the next line
    Conc_time_type1 = (1/2)*erfc((x - v*t)/(2*np.sqrt(D*t))) + \
        (1/2)*np.exp(v*x/D)*erfc((x + v*t)/(2*np.sqrt(D*t)))
    
    return Conc_time_type1

# Type 1 inlet conditions, finite length solution
def analytical_model_1D_finite_t1(x, t, v, al, L):
    # Dispersion
    D = v*al
    # Analytical solution: Analytical solution based on Equation A3 in van Genuchtena and Alves, 1982.
    # Note that the '\' means continued on the next line
    Conc_time_type1_finite = (1/2)*erfc((x - v*t)/(2*np.sqrt(D*t))) + \
        (1/2)*np.exp(v*x/D)*erfc((x + v*t)/(2*np.sqrt(D*t))) + \
        (1/2)*(2 + (v*(2*L - x)/D) + v**2*t/D)* \
        np.exp(v*L/D)*erfc(((2*L - x)+ v*t)/(2*np.sqrt(D*t))) - \
        (v**2 *t/(3.1415*D))**(1/2) * np.exp(v*L/D - ((2*L - x + v*t)**2)/(4*D*t))
            
    return Conc_time_type1_finite

# Type 3 inlet conditions, infinite solution
def analytical_model_1D_t3(x, t, v, al):
    # Dispersion
    D = v*al
    # Analytical solution: See lecture slides or (Parker and van Genuchten, 1984 eq 9b) for details
    Conc_time_type3 = (1/2)* erfc((x - v*t)/(2* np.sqrt(D*t))) + \
    np.sqrt((v**2*t)/(3.1415*D))* np.exp(-(x - v*t)**2/(4*D*t)) - \
    (1/2)*(1 + (v*x/D) + (v**2*t/D))* np.exp(v*x/D)* erfc((x + v*t)/(2* np.sqrt(D*t)))
    
    return Conc_time_type3

