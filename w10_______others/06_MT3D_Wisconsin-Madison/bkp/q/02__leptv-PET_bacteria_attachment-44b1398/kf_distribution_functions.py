# -*- coding: utf-8 -*-
"""
kf_distribution_functions.py
Created on:

@author:Vy Le and Christopher Zahasky

This script is used to ... (add description)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Basic colloid filtration theory function
def eta_calc(dp, d, v, rho_p):
    # Constants
    # gravity
    g = 9.81 # m*s^-2 
    # boltzman constant
    k = 1.38064852E-23 # units: m^2*kg*s^-2*K^-1
    
    # Water properties
    # absolute temperature
    T = 298.15 # Kelvin
    # water viscosity
    mu = 8.90E-4 # Pa·s
    # water density
    rho_w = 997 # kg/m^3
    
    # collector efficiency 
    # diffusion single collector efficiency (Equation 6 from Yao 1971)
    eta_d = 0.9*(k*T/(mu*dp*d*v))**(2/3) # a good check of this equation is to calculate the units
    # interception single collector efficiency (Equation 7 from Yao 1971)
    eta_i = (3/2)*(dp/d)**2
    # sedimentation single collector efficiency (Equation 8 from Yao 1971)
    eta_g = (rho_p - rho_w)*g*dp**2/ (18*mu*v)
    # combined effect
    eta = eta_d + eta_i + eta_g
    
    return eta

#! TO DO: Put in flexibility/checks so that prsity can be a single value or 3D map
def depositional_map(Kf, prsity, conc, timearray):
    # rhob = 1.8e3 # this is the default used for flopyflopy.mt3d.Mt3dRct
    rhob = 1 # this is the default used for flopyflopy.mt3d.Mt3dRct
    conc_size = conc.shape
    
    S = np.zeros([conc_size[1], conc_size[2], conc_size[3]])
    
    for layer in range(0, conc_size[1]):
        for row in range(0, conc_size[2]):
            for col in range(0, conc_size[3]):
                cell_btc = conc[:, layer, row, col]
                M0i = integrate.trapz(cell_btc, timearray)
                if type(prsity)==np.ndarray: # if prsity is an array
                    if type(Kf)==np.ndarray: # if Kf is an array
                        # Equation 14 from Interpreting Deposition Patterns of Microbial Particles in Laboratory-Scale Column Experiments
                        S[layer, row, col] = prsity[layer, row, col]/rhob*Kf[layer, row, col]*M0i
                    else:
                        # Equation 14 from Interpreting Deposition Patterns of Microbial Particles in Laboratory-Scale Column Experiments
                        S[layer, row, col] = prsity[layer, row, col]/rhob*Kf*M0i
                else:
                    if type(Kf)==np.ndarray: # if Kf is an array
                        # Equation 14 from Interpreting Deposition Patterns of Microbial Particles in Laboratory-Scale Column Experiments
                        S[layer, row, col] = prsity/rhob*Kf[layer, row, col]*M0i
                    else:
                        S[layer, row, col] = prsity/rhob*Kf*M0i
    
    return S


# Lognormal distribution 
def lognormal_kf(mu, sigma, nlay, nrow, ncol,*args):
    KF = np.random.lognormal(mean = mu, sigma = sigma, size = (nlay,nrow,ncol))
    if 'plot' in args:
        histn, bin_edgesn = np.histogram(KF, 50)
        # normalize to convert from histogram to PDF
        pdf = histn/np.sum(histn)
        
        fig = plt.figure(dpi=300)
        plt.plot(bin_edgesn[2:]*1e1, pdf[1:], label='$\sigma_1$ ='+str(sigma)) 
        
        #! TO DO: correctly format the number in the legend
        plt.axvline(x=np.exp(mu)*1e1,linestyle ="--", label = "single $k_f$ ="+str(np.exp(mu)), color ='k' )
        plt.xlabel('$k_f$ [$10^{-1}$ $min^{-1}$]')
        plt.ylabel('Probability density ')
        plt.legend(loc ='best', prop={'size': 12})
        #plt.xlim([0,3])
        plt.title('Random log-normal $k_f$ distribution')
        
    return KF

# Gaussian distribution (NOT FINISHED)
def normal_kf(mu, sigma, nlay, nrow, ncol,*args):
    KF = np.random.normal(mean = mu, sigma = sigma, size = (nlay,nrow,ncol))
    if 'plot' in args:
        histn, bin_edgesn = np.histogram(KF, 50)
        # normalize to convert from histogram to PDF
        pdf = histn/np.sum(histn)
        
        fig = plt.figure(dpi=300)
        plt.plot(bin_edgesn[2:]*1e3, pdf[1:], label='$\sigma_1$ ='+str(sigma)) 
        
        #! TO DO: correctly format the number in the legend
        plt.axvline(x=np.exp(mu)*1e3,linestyle ="--", label = "single $k_f$ ="+str(np.exp(mu)), color ='k' )
        plt.xlabel('$k_f$ [$10^{-3}$ $min^{-1}$]')
        plt.ylabel('Probability density ')
        plt.legend(loc ='best', prop={'size': 12})
        #plt.xlim([0,3])
        plt.title('Random normal $k_f$ distribution')
        
    return KF

# Depth-dependent distribution 
# This function is based on equations 2-6 in Bradford et al 2003 (10.1021/es025899u).
# Essentially they add a depth dependent straining term to the standard
# attachment coefficient. The detachment term is neglected
def depth_dependent_kf(k_att, k_str, alpha, beta, dc, grid_size, nlay, nrow, ncol, *args): 
    # depth dependent attachement (in meters to match dc units)
    z = np.arange(grid_size[2]/2, grid_size[2]*ncol, grid_size[2])/100
    Psi_str = ((dc+z)/dc)**-beta
    # k_att is the attachement coefficient
    kf_1d = (Psi_str*k_str) + (alpha*k_att)
    # Create 3D array the varies along the axis of the core
    KF = np.tile(kf_1d, (nlay, nrow, 1))
    
    if 'plot' in args:
        histn, bin_edgesn = np.histogram(KF, 50)
        # normalize to convert from histogram to PDF
        pdf = histn/np.sum(histn)
        
        fig = plt.figure(dpi=300)
        plt.plot(bin_edgesn[2:]*1e3, pdf[1:], label='beta ='+str(beta)) 
        
        #! TO DO: correctly format the number in the legend
        plt.axvline(x=k_att*1e3,linestyle ="--", label = "single $k_f$ ="+str(k_att), color ='k' )
        plt.xlabel('$k_f$ [$10^{-3}$ $min^{-1}$]')
        plt.ylabel('Probability density ')
        plt.legend(loc ='best', prop={'size': 12})
        #plt.xlim([0,3])
        plt.title('Depth-dependent $k_f$ distribution')
        
    return KF

# test calls of these functions
# Kf = lognormal_kf(np.log(0.001), 0.25, 20, 20, 40, 'plot')