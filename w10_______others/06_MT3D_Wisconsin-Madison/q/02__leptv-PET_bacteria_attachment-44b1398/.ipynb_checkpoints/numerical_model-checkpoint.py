# -*- coding: utf-8 -*-
"""
@author: Vy Le (vple@wisc.edu) and Chris Zahasky

The below script creates a 3D numerical model for bacteria attachment to validate the experimental result from PET column.

"""

# Import some python packages
import os
# import shutil
import numpy as np
# from scipy import integrate
import matplotlib.pyplot as plt
# import time
import copy
# import flopy 
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Arial']})
plt.rcParams['font.size'] = 20

# Import custom functions to run flopy stuff and analytical solutions
from first_order_attachment_3D_model_functions import *
# Import custom kf distribution functions 
from kf_distribution_functions import *
# Import custom plotting function
from plotting_functions import *

# Set main directory to save data
main_dir = "//fons.geology.wisc.edu/Active Data/Le/publication paper/code" # for Chris generated ready-to-use kc file cmc_kc_low.txt
os.chdir(main_dir)

#  Path to mf2005 and mt3dms 
exe_name_mf = "//fons.geology.wisc.edu/Active Data/Le/publication paper/code/MF2005.1_12/MF2005.1_12/bin/mf2005"
exe_name_mt = "//fons.geology.wisc.edu/Active Data/Le/publication paper/code/mt3dusgs1.1.0/mt3dusgs1.1.0/bin/mt3d-usgs_1.1.0_32"

# Load kf distribution data from PET
filename ='kf_distribution_3d'
import_data = np.loadtxt(filename + '.csv', delimiter=',')

# Extract metadata
dx = import_data[-3] # grid cell length in x-direction
dy = import_data[-2] 
dz = import_data[-1] 

nx = int(import_data[-6]) # number of cells in x-direction
ny = int(import_data[-5])
nz = int(import_data[-4])

import_data_kf = import_data[0:-6] 
kf = import_data_kf.reshape(nx, ny, nz)

kf = kf*60 # convert from 1/sec to 1/min

vox_size = np.array([0.15526, 0.15526, 0.1592]) # Voxel dimension (cm)
# =============================================================================
# Model geometry
# =============================================================================
# grid_size = [grid size in direction of Lx (layer thickness), 
    # Ly (left to right axis when looking down the core), Lz (long axis of core)]
grid_size = np.array([dx, dy, dz]) # selected units [cm]
# grid dimensions
nlay = nx # number of layers / grid cells
nrow = ny # number of columns 
ncol = nz # number of slices (parallel to axis of core)

# =============================================================================
# Core shape and input/output control
# =============================================================================
# Creates a cylindrical shape core. Ones are voxels with core and zeros are the area outside the core
# based on kf metadata structure
core_mask = np.squeeze(copy.deepcopy(kf[:,:,0]))
core_mask[core_mask>0] =1 
core_mask[np.isnan(core_mask)] = 0 # convert nan values to 0 (bc kc_3d imported has nan and 
                                    # core_mask should contains only 0 (no PET data) & 1 (column matrix)

# Output control for MT3dms
# nprs (int):  the frequency of the output. If nprs > 0 results will be saved at 
# the times as specified in timprs (evenly allocated between 0 and sim run length); 
# if nprs = 0, results will not be saved except at the end of simulation; if NPRS < 0, simulation results will be 
# saved whenever the number of transport steps is an even multiple of nprs. (default is 0).
nprs = 500 
# period length in selected units (min)
    # the first period is fluid pressure equilibration,
    # the second period is tracer/bacteria injection duration = 1ml/(1.826 ml/min)
    # the third period is tracer displacement
perlen_mt = [0.0001, 1./1.8262 , 30] 

# Numerical method flag
mixelm = -1


# =============================================================================
# General model parameters
# =============================================================================
# porosity
prsity = 0.39 
# dispersivity (cm)
al_t = 0.06 # tracer (from UV Vis exp)
al_b = 0.13 # bacteria (from UV vis exp)
# injection rate (ml/min)
injection_rate = 1.826 
# advection velocity (cm/min)
v = injection_rate/(3.1415*1.27**2*prsity)
# first order coefficient (Note units of minutes in flopy model)
kfmean = kf.astype('float')
kfbulk = np.nanmean(kfmean)


# =============================================================================
# Run numerical modelsfor 3 scenarios: tracer (kf =0), homogenous kf, heterogeneous kf
# =============================================================================
# Define homogeneous conductivity field equal to the geometric mean of the weakly anisotropic field
hom_hk = 10*np.ones([nlay, nrow, ncol]) #create 2D matrix with  #ballpark 10 darcy

# Crop outside of core
hom_hk_core = apply_core_mask(hom_hk, core_mask)

# Run numerical models and generate attachment maps
    # kf = 0 (for tracer)
mf, mt, conc, btc_solute_homogeneous, times = first_order_pulse_injection_sim(
                'tracer', hom_hk_core, prsity, al_t, 0, grid_size, v,
                perlen_mt, nprs, mixelm, exe_name_mf, exe_name_mt, main_dir)
    
    # single kf
mf, mt, conc_kf, btc_rc1_homog, times = first_order_pulse_injection_sim(
                    'tracer', hom_hk_core, prsity, al_b, kfbulk, grid_size, v,
                             perlen_mt, nprs, mixelm, exe_name_mf, exe_name_mt, main_dir)
S_homo = depositional_map(kfbulk, prsity, conc_kf, times)   
S_profile_kf_homo = np.sum(np.sum(S_homo[:, :, :], axis=0), axis=0)/core_mask.sum()

    # Heterogeneous kf
mf, mt, conc_kf_hetero, btc_rc1_cmc, times = first_order_pulse_injection_sim(
                    'tracer', hom_hk_core, prsity, al_b, kf, grid_size, v,
                             perlen_mt, nprs, mixelm, exe_name_mf, exe_name_mt, main_dir)

S_hetero = depositional_map(kf, prsity, conc_kf_hetero, times) 
S_profile_kf_hetero = np.nansum(np.nansum(S_hetero[:, :, :], axis=0), axis=0)/core_mask.sum() # 1D slice-avg attached bacteria
S_kf_hetero_2d = np.nanmean(S_hetero[:,:,:], axis = 1) # generate 2D maps of S*/C0 from numerical model


"""
COMPARE BACTERIA ATTACHMENT: NUMERICAL MODEL vs EXPERIMENTAL
The below codes generate Figure 5.
"""

# Load experimental attachment data (S*/C0) from PET
filename2 = "bacteria_attachment_map_3d.csv"  # WITHOUT slice avg
path2file = "//fons.geology.wisc.edu/Active Data/Le/publication paper/code"
path2data_Sstar = os.path.join(path2file, filename2)
import_data_Sstar = np.loadtxt(path2data_Sstar, delimiter=',')

# Extract metadata -- NO NEED TO DO if use Vy generated Sstar file that already slice-averaged
dx = import_data_Sstar[-3]
dy = import_data_Sstar[-2]
dz = import_data_Sstar[-1]
nx = int(import_data_Sstar[-6])
ny = int(import_data_Sstar[-5])
nz = int(import_data_Sstar[-4])
import_data_Sstar= import_data_Sstar[0:-6] # Remove metadata 
Sstar = import_data_Sstar.reshape(nx, ny, nz)
slice_avg_Sstar = np.nanmean(Sstar, axis = (0,1)) # calculate S*/C0 average in 1D
slice_avg_Sstar_2d = np.nanmean(Sstar, axis = 1)

# Extract grid cell centers
x = np.linspace(dz/2, dz*nz - dz/2,num=nz)

# Figure 5, top: Plot 1D slice-average S*/C0 comparison 
fig, axis = plt.subplots(1,1, figsize=(10, 4.2), dpi=300)
plt.plot(x, S_profile_kf_homo/prsity, color= 'grey', label='Homogeneous model, $k_f$=' +'{:.3f}'.format(kfbulk)+" min$^{-1}$", linewidth=2)
plt.plot(x, S_profile_kf_hetero/prsity, '--', color= 'k', label='Heterogeneous model', linewidth=2)
# plt.plot(x, import_data_Sstar, color ='red', label ='Measured', linewidth =2)
plt.plot(x, slice_avg_Sstar, color = 'r', label='Measured', linewidth=2)
plt.plot([17*vox_size[2], 17*vox_size[2]], [0, 1], '--', color = 'Grey') # slice 17: inlet-mid layer interface
plt.plot([41*vox_size[2], 41*vox_size[2]], [0, 1], '--', color = 'Grey') # slice 41: mid-outlet interface
plt.ylabel('Slice average $S*/C_0$ (-)')
plt.xlabel('Distance from column inlet (cm)')
plt.legend(loc ='best', prop ={'size':16})
plt.ylim([0, 0.03])
plt.xlim(left= 0, right = np.max(x))
plt.xticks(np.arange(0, 10, step=2))
plt.show(), plt.close()


# Figure 5, middle: Plot 2D S*/C0 map predicted by model based on heterogeneous kf distribution
vox_size = np.array([0.15526, 0.15526, 0.1592])# (These are the default values) # voxel size. This information can be found in the .hdr img file. (Fons drive > Permanent data)
plot_2d(S_kf_hetero_2d/prsity, vox_size[0], vox_size[2], 'S*/C$_0$', cmap='Greens')
plt.clim([0, 0.03])
plt.xlabel('Distance from column inlet (cm)')
plt.title('Numerical model prediction')
plt.show(),plt.close()

# Figure 5, bottom: Plot 2D S*/C0 map measured from PET
plot_2d(slice_avg_Sstar_2d, vox_size[0], vox_size[2], 'S*/C$_0$', cmap='Greens')
plt.clim([0, 0.030]) # color scale bar
plt.xlabel('Distance from column inlet (cm)')
plt.title('PET measured attachment')
plt.show(),plt.close()