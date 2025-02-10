# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:51:04 2023

@author: vylep

For pulication purpose:
    - load pre-processed PET data (after we corrected)
    - Analyze for attachment S*/C0 and attachment coefficients kf
"""



## Import packages needed in this script
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import math
from scipy.stats import gamma
# from matplotlib.ticker import StrMethodFormatter
from scipy import integrate

# Set working directory to folder where all python and data files are stored
os.chdir("//fons.geology.wisc.edu/Active Data/Le/publication paper/code")

## Import functions file
from pet_analysis_functions import *

## Font setting
# package for making font bigger
rc('font',**{'family':'serif','serif':['Arial']}) ### ES&T requires Arial or Helevica
fs = 20
plt.rcParams['font.size'] = fs

## Load PET data (pre-processed)
tracer_file = 'tracer_PET_data_processed.npy' # tracer data 
bact_file = 'bacteria_PET_data_processed.npy'  # bacteria data
raw_data_tracer = np.load(tracer_file)
raw_data_bact = np.load(bact_file) # import csv version - Vy way
# raw_data= np.fromfile(filename, dtype=np.float32)

# # manually set the dimensions of the raw file
# img_dim = [34, 34, 159, 60]
# raw_data = np.reshape(raw_data, (34, 34, 159, 60)) #reshape array from 1D to 4D

# extract dimension 
r, c, z, t = np.shape(raw_data_tracer) #r = rows (y axis)' c= columsn( x axis), z = z-azis/slices, t= time ()
vox_size = np.array([0.15526, 0.15526, 0.1592])  # voxel dimension (cm)
# timestep size depending on reconstruction. This information can also be found in the header files
timestep_size = 60 # scan duration (minutes)

"""
Figure 2 plotting 
The 2D concentration maps of the tracer and bacteria pulse can be plotted using the code below. Time stamp t1 is selected at 2, 5, and 8 min to plot at different times.
Note that t1 is the time since pulse injection, which was 1 minute after when the PET scanning started. 
"""
## Plot 2D Center-slice concentration avg of tracer data(Figure 2, left)
t1 =2 # Time since pulse injection (min) --> time to plot map = t1+1
plot_2d(np.nanmean(raw_data_tracer[:,:,:,t1+1], axis=1), vox_size[0], vox_size[2], 'C/C$_0$', cmap='Reds')
plt.clim(0.0, 0.3)
plt.title('Center slice average '+ str(t1)+ ' min')
plt.xlabel('Distance from column inlet (cm)')
t1 =5
plot_2d(np.nanmean(raw_data_tracer[:,:,:,t1+1], axis=1), vox_size[0], vox_size[2], 'C/C$_0$', cmap='Reds')
plt.clim(0.0, 0.3)
plt.title('Center slice average '+ str(t1)+ ' min')
plt.xlabel('Distance from column inlet (cm)')
t1 =8
plot_2d(np.nanmean(raw_data_tracer[:,:,:,t1+1], axis=1), vox_size[0], vox_size[2], 'C/C$_0$', cmap='Reds')
plt.clim(0.0, 0.3)
plt.title('Center slice average '+ str(t1)+ ' min')
plt.xlabel('Distance from column inlet (cm)')
plt.show(), plt.close()

# Plot 2D center-slice avg of E.coli bacteria (Figure 2, right)
t1 =2 # Time since pulse injection (min) --> time to plot map = t1+1
plot_2d(np.nanmean(raw_data_bact[:,:,:,t1+1], axis=1), vox_size[0], vox_size[2], 'C/C$_0$', cmap='Reds')
plt.clim(0.0, 0.3)
plt.title('Center slice average '+ str(t1)+ ' min')
plt.xlabel('Distance from column inlet (cm)')
t1 =5
plot_2d(np.nanmean(raw_data_bact[:,:,:,t1+1], axis=1), vox_size[0], vox_size[2], 'C/C$_0$', cmap='Reds')
plt.clim(0.0, 0.3)
plt.title('Center slice average '+ str(t1)+ ' min')
plt.xlabel('Distance from column inlet (cm)')
t1 =8
plot_2d(np.nanmean(raw_data_bact[:,:,:,t1+1], axis=1), vox_size[0], vox_size[2], 'C/C$_0$', cmap='Reds')
plt.clim(0.0, 0.3)
plt.title('Center slice average '+ str(t1)+ ' min')
plt.xlabel('Distance from column inlet (cm)')
plt.show(), plt.close()
   

"""
FIND BACTERIA ATTACHMENT (S*/C0):
Bacteria attachment was analyzed at 14 minute since pulse injection (t1 = 14 min), when the mobile bacteria pulse exited the column.
The plotting code generates 2D experimental bacteria attachment map shown in Figure 5 (bottom).
"""
# Plume out of column time index 
t1 =14 # time since pulse injection (min)
S_index = t1+1 # the time of analysis is 1 minute extra because counts from the beginning of PET scan

# Find bacteria attachment
S_star = raw_data_bact[:,:,:, S_index]
data_size = S_star.shape

# Plot S*/C0 attached concentration 
plot_2d(np.nanmean(S_star, axis=1), vox_size[0], vox_size[2], 'S*/C$_0$', cmap='Greens')
plt.clim([0, 0.030]) # color scale bar
plt.xlabel('Distance from column inlet (cm)')
plt.title('2D center slice average attached concentration')

plot_2d_sub_profile(S_star, vox_size[2], vox_size[0], 0.03, 'Attached concentration: S*/C$_0$ (-)', cmap='Greens')
plt.plot([17*vox_size[2], 17*vox_size[2]], [0, 1], '--', color = 'Grey') # plot interface boundary lines btw inlet - mid layer
plt.plot([41*vox_size[2], 41*vox_size[2]], [0, 1], '--', color = 'Grey')

# Save 3D bacteria attachment profile (S*/C0) - for 3d model comparison 
save_filename = 'bacteria_attachment_map_3d.csv'
save_data = np.append(S_star.flatten('C'), [data_size, vox_size])
np.savetxt(save_filename, save_data, delimiter=',') # file saved in the chosen os.chdir() at the beginning, not where this script is saved 

"""
CALCULATE ATTACHMENT RATE COEFFICIENTS (kf)
Attachment coefficients are calculated based on Eq (4) in our paper, which requires the above bacteria attachment S*/C0 
and voxel-scale integration of bacteria breakthrough curves.
This plot Figure 3.
"""
# Integrate bacteria BTC over time in each voxel
times =  np.linspace(0, timestep_size*t, t)+(timestep_size/2) 
C_int_3d, C_int_fit_3d = btc_integrate_func(raw_data_bact, times, S_index)

# center_slice_cint = C_int_3d[:,8,:]
# plot_2d(center_slice_cint, vox_size[0], vox_size[2], 'Voxel C integration', cmap='viridis')
# plt.xlabel('Distance from inlet [cm]')

# # center_slice_cint = np.nanmean(C_int_fit_3d[:,:,:], axis=1)
# center_slice_cint = C_int_fit_3d[:,8,:]
# plot_2d(center_slice_cint, vox_size[0], vox_size[2], 'Voxel C fit integration', cmap='viridis')
# plt.xlabel('Distance from column inlet [cm]')

# Calculate attachment coefficeint (kf)
kf_3d, c_int3d = kc_calc_insitu(raw_data_bact, times, C_int_fit_3d, S_index, 3) # unit: 1/sec
data_size = kf_3d.shape
kf_3d = kf_3d.astype('float')
kf_3d[kf_3d == 0] = np.nan # Replace zeros with "nan" values

# Plot slice-average kf 2D map only
cmax = 0.050 # Max value of color scale bar for kf
plot_2d(np.nanmean(kf_3d, axis=1)*60, vox_size[0], vox_size[2], 'k$_f$ (1/min)', cmap='magma')
plt.clim([0, cmax]) # 0.045 (050922), 0.015 (Nov 2022)
plt.xlabel('Distance from column inlet (cm)')
plt.title('2D center slice average attachment coefficients')

# Plot center slice average kf map 2D and 1D 
plot_2d_sub_profile(kf_3d*60, vox_size[2], vox_size[0], cmax, 'Attachment coefficients k$_f$ (1/min)', cmap='magma')
plt.plot([17*vox_size[2], 17*vox_size[2]], [0, 1], '--', color = 'Grey') # slice 17: inlet-mid layer interface
plt.plot([41*vox_size[2], 41*vox_size[2]], [0, 1], '--', color = 'Grey') # slice 41: mid-outlet interface

# Save kf data to parameterize a 3d model in separate file
save_filename = 'kf_distribution_3d.csv'
save_data = np.append(kf_3d.flatten('C'), [data_size, vox_size])
np.savetxt(save_filename, save_data, delimiter=',')

"""
PLOT kf DISTRIBUTION (Figure 4)
Note: Division of 3 column slices corresponding to each sand layer was based on the sand mass added in each layer, which was 25g, 35g, and 21.1g from inlet to outlet.
"""
nhistbins = 25
# bin_centers is kf values, uden is probability density
udenf, bin_centersf = pdf_plot(kf_3d[:,:,1:14], nhistbins, 0)   # inlet M356 layer (excluding first slice) 0:17 to include interface zone
udenc, bin_centersc = pdf_plot(kf_3d[:,:,18:36], nhistbins, 0)  # middle M324 layer low_repeat: 14:41 (Chris chose 18:18+24)
udenb, bin_centersb = pdf_plot(kf_3d[:,:,42:-1], nhistbins, 0)  # outlet M356 layer (exlcuding last slice) low_repeat: 42:55 (end except last voxel) (Chris chose 43:)
uden_intf1,bin_centers_intf1 = pdf_plot(kf_3d[:,:,15:17], nhistbins, 0) # interface 1 (near inlet)
uden_intf2,bin_centers_intf2 = pdf_plot(kf_3d[:,:,37:41], nhistbins, 0) # interface 2 (near outlet)
# udenf, bin_centersf = pdf_plot(kc_3d[:,:,1:22], nhistbins, 0)
# udenc, bin_centersc = pdf_plot(kc_3d[:,:,23:-12], nhistbins, 0)
# udenb, bin_centersb = pdf_plot(kc_3d[:,:,-11:-1], nhistbins, 0)
# udenf, bin_centersf = pdf_plot(kc_3d[:,:,1:9], nhistbins, 0)   #low_repeat col: 1:9, hightolow: 1:13
# udenc, bin_centersc = pdf_plot(kc_3d[:,:,14:40], nhistbins, 0)  #low_repeat: 14:40, hightolow: 14:30
# udenb, bin_centersb = pdf_plot(kc_3d[:,:,46:-1], nhistbins, 0)  #low_repeat: 46:end, hightolow: 42:57


# Fit kf with log-normal distribution
def lognorm(x, mu, sigma):
   return 1/(np.sqrt(2*np.pi)*sigma*x)*np.exp(-((np.log(x)- mu)**2)/(2*sigma**2))

  ## Fit distribution of inlet layer
kf_interpf = np.linspace(0, bin_centersf.max(), 100)
p0=[-7, 0.4] # lognorm initial guess for the parameter(kf)
popt, pcov = curve_fit(lognorm, bin_centersf, udenf, p0=p0)     # Log-normal fit
k_fit_f = lognorm(kf_interpf, popt[0], popt[1])

    ## Fit distribution of middle layer
kf_interpc = np.linspace(0, bin_centersc.max(), 100)
popt, pcov = curve_fit(lognorm, bin_centersc, udenc, p0=p0) # lognormal fit
k_fit_c = lognorm(kf_interpc, popt[0], popt[1])

    ## Fit distribution of outlet layer
kf_interpb = np.linspace(0, bin_centersb.max(), 100)
udenb[udenb==0]=1e-2
popt, pcov = curve_fit(lognorm, bin_centersb, udenb, p0=p0) # log normal
k_fit_b = lognorm(kf_interpb, popt[0], popt[1])

## Fit distribution of interface layer 1
kf_interp_intf1 = np.linspace(0, bin_centers_intf1.max(), 100)
popt, pcov = curve_fit(lognorm, bin_centers_intf1, uden_intf1, p0=p0) # lognormal fit
k_fit_intf1 = lognorm(kf_interp_intf1, popt[0], popt[1])

## Fit distribution of interace layer 2
kf_interp_intf2 = np.linspace(0, bin_centers_intf2.max(), 100)
popt, pcov = curve_fit(lognorm, bin_centers_intf2, uden_intf2, p0=p0) # lognormal fit
k_fit_intf2 = lognorm(kf_interp_intf2, popt[0], popt[1])


# Plot kf distribution and log-normal fitting lines 
fig, axs = plt.subplots(1, 1, figsize=(13, 6.5), dpi=250) # define figure size
plt.plot(bin_centersf*60, udenf, 'o', color='grey', markersize = 5, label = 'Inlet layer (M356 sand)')    # multiply by 60 converts unit from 1/sec to 1/min
plt.plot(bin_centersc*60, udenc, '^', color='purple', markersize = 5, label = 'Middle layer (M324 sand)') # Vy modified
plt.plot(bin_centersb*60, udenb, 's', color='k', markersize = 5, label = 'Outlet layer (M356 sand)') # Vy modified
plt.plot(bin_centers_intf1*60, uden_intf1, "*", color = "green", markersize = 5, label = 'Interface 1 (inlet)',) # Vy modified
plt.plot(bin_centers_intf2*60, uden_intf2, "P", color = "orange", markersize = 5, label = 'Interface 2 (outlet)') # Vy modified
plt.plot(kf_interpf*60, k_fit_f, color='grey')   # inlet kf lognormal fitting line
plt.plot(kf_interpc*60, k_fit_c, color='purple', linestyle =':') # mid layer kf fitting line
plt.plot(kf_interpb*60, k_fit_b, color='k', linestyle ='--') # outlet layer kf fitting line 
plt.plot(kf_interp_intf1*60, k_fit_intf1, color='green', linestyle ='--') # interface 1 fiutting line
plt.plot(kf_interp_intf2*60, k_fit_intf2, color='orange', linestyle ='--') # interface 2 fitting line
plt.xticks(ticks =np.arange(0,0.12,step =0.015))
plt.ylim([0.5e1, 1e4])
plt.xlabel('k$_f$ (1/min)')
plt.ylabel('Probability density')
plt.yscale('log')
plt.title('Probability density of k$_f$ fitted with log-normal distribution', pad= 20)
plt.legend(loc='best', prop={'size': fs-3}), plt.show(), plt.close()

