# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:25:15 2021

@author: Czahasky
"""


# Only packages called in this script need to be imported
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import StrMethodFormatter

from scipy import integrate
from scipy.optimize import curve_fit


    
def btc_integrate_func(pet_data, times, S_index):
    times_interp = np.linspace(0, times[S_index], 300)
    # determine input data size
    petdim = pet_data.shape
    # preallocate arrays
    C_int = np.zeros((petdim[0:3]), dtype=float)
    C_int_fit = np.zeros((petdim[0:3]), dtype=float)
    
    n=0 
    # Loop through each voxel
    for cslice in range(0, petdim[2]):
        for row in range(0, petdim[0]):
            for col in range(0, petdim[1]):
                # check that voxel is inside of the column
                if np.isfinite(pet_data[row,col,cslice,0]):
                    # define breakthrough curve for voxel
                    vox_c = np.squeeze(pet_data[row, col, cslice,:])
                    # print(row, col, cslice)
                    ## fit Gaussian to this curve
                    max_ind = vox_c.argmax()
                    if n ==0:
                        # initial guess for parameters
                        p0 = [vox_c[max_ind], times[max_ind], times[max_ind]/10]
                    else:
                        p0 = popt
                    
                    popt, pcov = curve_fit(gauss, times[:max_ind+4], vox_c[:max_ind+4], p0=p0)
                    
                    
                    c_fit = gauss(times_interp, popt[0], popt[1], popt[2])
                    # integrate fit btc
                    c_vox_int_fit = np.trapz(c_fit, times_interp)
                    # save integration of concentration curve
                    C_int_fit[row, col, cslice] = c_vox_int_fit
                    
                    # integrate voxel btc
                    c_vox_int = np.trapz(vox_c[:S_index], times[:S_index])
                    # save integration of concentration curve
                    C_int[row, col, cslice] = c_vox_int
                    n +=1
     
    return C_int, C_int_fit

def coarsen_slices(array3d, coarseness):
    array_size = array3d.shape
    if len(array_size) ==3:
        coarse_array3d = np.zeros((int(array_size[0]/coarseness), int(array_size[1]/coarseness), int(array_size[2]/coarseness)))
        # for z in range(0, array_size[2]):
        for z in range(0, int(array_size[2]/coarseness)):
            sum_slices = np.zeros((int(array_size[0]/coarseness), int(array_size[1]/coarseness)))
            for zf in range(0, coarseness-1):
                array_slice = array3d[:,:, z*coarseness + zf]
                
                # array_slice = array3d[:,:, z]
                # coarsen in x-y plan
                temp = array_slice.reshape((array_size[0] // coarseness, coarseness,
                                        array_size[1] // coarseness, coarseness))
                smaller_slice = np.mean(temp, axis=(1,3))
                # record values for each slice to be averaged
                sum_slices = sum_slices + smaller_slice
            # after looping through slices to be averaged, calculate average and record values
            coarse_array3d[:,:, z] = sum_slices/coarseness
            
    elif len(array_size) == 4:
        coarse_array3d = np.zeros((int(array_size[0]/coarseness), 
                    int(array_size[1]/coarseness), int(array_size[2]/coarseness), int(array_size[3])))
        print('coarsening data assuming time is 4th dimension, with no time averaging')
        # loop through time
        for t in range(0, int(array_size[3])):
            for z in range(0, int(array_size[2]/coarseness)):
                sum_slices = np.zeros((int(array_size[0]/coarseness), int(array_size[1]/coarseness)))
                for zf in range(0, coarseness-1):
                    array_slice = array3d[:,:, z*coarseness + zf, t]
                    
                    # array_slice = array3d[:,:, z]
                    # coarsen in x-y plan
                    temp = array_slice.reshape((array_size[0] // coarseness, coarseness,
                                            array_size[1] // coarseness, coarseness))
                    smaller_slice = np.mean(temp, axis=(1,3))
                    # record values for each slice to be averaged
                    sum_slices = sum_slices + smaller_slice
                # after looping through slices to be averaged, calculate average and record values
                coarse_array3d[:,:, z, t] = sum_slices/coarseness
            
    return coarse_array3d

def crop_core(array4d):
    # crop outside of core and replace values with nans
    crop_dim = array4d.shape
    # crop_dim = raw_data.shape
    rad = (crop_dim[1]-1)/2
    cy = crop_dim[1]/2 
    dia = crop_dim[1]
    for ii in range(0, dia):
        yp = np.round(cy + math.sqrt(rad**2 - (ii-rad)**2) - 1E-8)
        ym = np.round(cy - math.sqrt(rad**2 - (ii-rad)**2) + 1E-8)
        
        if yp <= dia:
            array4d[ii, int(yp):, :, :] = np.nan
        
        if ym >= 0:
            array4d[ii, 0:int(ym), :, :] = np.nan

    return array4d

# A guassian function
def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def kc_calc_insitu(pet_data, times, C_int_matrix, S_timeframe, dim):
    # Use equation 14 from https://pubs.acs.org/doi/10.1021/es025871i to interpret despositional patterns
    # note that porosity and bulk density are neglected to make units consistent between S and C
    # Check if C_int_matrix is an array
    if type(C_int_matrix).__module__ == np.__name__:
        c_mat_dim = C_int_matrix.shape
    else:
        c_mat_dim = 0.0
            
    # determine input data size
    petdim = pet_data.shape
    # calculate kc in 1, 2, or 3 dimensions
    if int(dim) == 1 :
        kc = np.zeros((petdim[2]), dtype=float)
        C_int = np.zeros((petdim[2]), dtype=float)
        slice_sum_c = np.nansum(pet_data[:,:,:,:], axis=(0,1))

        for cslice in range(0, petdim[2]):
            # numerically integrate concentration in slice with respect to time
            if c_mat_dim[2] == petdim[2]:
                C_int[cslice] = np.nanmean(C_int_matrix[:, :, cslice])
            else:
                c_slice_int = np.trapz(slice_sum_c[cslice,:S_timeframe], times[:S_timeframe])
                C_int[cslice] = c_slice_int
            
            ### NOTE ####
            # There are two ways this could be done. Either use a single time frame to calculate S
            s_slice = slice_sum_c[cslice, S_timeframe]
            # or use the average of several late time frames
            s_slice = np.nanmean(slice_sum_c[cslice, S_timeframe:S_timeframe+5])
            kc[cslice] = s_slice/c_slice_int
            
    elif int(dim) == 2: 
        raise ValueError("kc map calculation has not yet been implemented in 2D")
        # Implement 2D kc calc...
        
    elif int(dim) == 3:
        # preallocate kc
        kc = np.zeros((petdim[0:3]), dtype=float)
        C_int = np.zeros((petdim[0:3]), dtype=float)
        # slice_sum_c = np.nansum(np.nansum(pet_data[:,:,:,:], axis=0), axis=0)

        for cslice in range(0, petdim[2]):
            # numerically integrate concentration in slice with respect to time
            # c_slice_int = np.trapz(slice_average_c[cslice,:], times)
            # c_slice_check = 0
            for row in range(0, petdim[0]):
                for col in range(0, petdim[1]):
                    # check that voxel is inside of the column
                    if np.isfinite(pet_data[row,col,cslice,0]):
                        # numerically integrate concentration in slice with respect to time
                        if c_mat_dim[2] == petdim[2]:
                            c_vox_int = C_int_matrix[row, col, cslice]
                        else:
                            # define breakthrough curve for voxel
                            vox_c = np.squeeze(pet_data[row, col, cslice,:])
                            # integrate voxel btc
                            c_vox_int = np.trapz(vox_c[:S_timeframe], times[:S_timeframe])
   
                        
                        # save integration of concentration curve
                        C_int[row, col, cslice] = c_vox_int
                        ### NOTE ####
                        # There are two ways this could be done. Either use a single time frame to calculate S
                        # s_vox = pet_data[row, col, cslice, S_timeframe]
                        # or use the average of several late time frames
                        s_vox = np.nanmean(pet_data[row, col, cslice, S_timeframe:S_timeframe+2])

                        kc[row, col, cslice] = s_vox/c_vox_int
                    
    else:
        raise ValueError("dim variable describes dimensionality of kc map, it must be 1, 2, or 3")
    return kc, C_int

# Calculate probability density functions of input 'data'
def pdf_plot(data, nbins, thresh):
    # remove all nan values
    data = data[~np.isnan(data)]
    # remove all zeros and calculate histogram
    den, b = np.histogram(data[data>thresh], bins=nbins, density=True)
    # calculate probility density when x-axis doesn't go from 0-1
    # uden = den / den.sum()
    bin_centers = b[1:]- ((b[2]- b[1])/2)
    
    return den, bin_centers


def plot_2d(map_data, dx, dy, colorbar_label, cmap, *args):

    r, c = np.shape(map_data)
    # define grid
    x_coord = np.linspace(0, dx*c, c+1)
    y_coord = np.linspace(0, dy*r, r+1)
    X, Y = np.meshgrid(x_coord, y_coord)
    
    # define figure and with high res
    # plt.figure(figsize=(10, 3), dpi=200)
    plt.figure(figsize=(10, 3), dpi=200)
    plt.pcolormesh(X, Y, map_data, cmap=cmap, shading = 'auto', edgecolor ='k', linewidth = 0.01)
    plt.gca().set_aspect('equal')  
    # add a colorbar
    cbar = plt.colorbar() 
    # label the colorbar
    cbar.set_label(colorbar_label)
    plt.tick_params(axis='both', which='major')
    plt.xlim((0, dx*c)) 
    plt.ylim((0, dy*r)) 
    
def plot_2d_sub_profile(map_data, dx, dy, cmax, colorbar_label, cmap):

    map_dim = np.shape(map_data)
    # define grid
    x_coord = np.linspace(0, dx*map_dim[2], map_dim[2])
    y_coord = np.linspace(0, dy*map_dim[0], map_dim[0])
    X, Y = np.meshgrid(x_coord, y_coord)
    
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 9), dpi=300, 
                            gridspec_kw={'height_ratios': [1.6, 1.0]})  #figsize =(8,7)
    # define overall title    
    fig.suptitle(colorbar_label)
    # extract center slice
    center_slice_data = np.nanmean(map_data, axis=1)  # axis =1: avergage across column (vertical) 
    # center_slice_data = np.nanmean(map_data, axis=0) # axis = 0: average across rows (horizontal, perpendicular to the vertical cross section)
    
    # define consistent max scale if not predefined in function
    if cmax == 0:
        cmax = np.nanmax(center_slice_data)
    
    # plot slice
    im1 = axs[0].pcolormesh(X, Y, center_slice_data, cmap=cmap, shading = 'auto', edgecolor ='k', linewidth = 0.01, vmin=0, vmax=cmax)
    # box = axs[0].get_position()
    # axColor = plt.axes([box.x0, box.y0 +box.height* 1.05, box.width, 0.01])
    fig.colorbar(im1, ax=axs[0], orientation = 'horizontal', aspect = 50, pad = 0.2)
    axs[0].set(xlabel='Distance from column inlet (cm)', xlim= (0, dx*map_dim[2]), ylim=(0, dy*map_dim[0]), aspect='equal') # , title='2D center slice average'
    
    # plot profile
    slice_average = np.nanmean(map_data, axis=(0,1))
    axs[1].plot(x_coord, slice_average, color='black', label='')
    axs[1].set(xlabel='Distance from column inlet (cm)', xlim= (0, dx*map_dim[2]), ylim=(0, cmax)) #, title='1D slice average'
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0e}'))
    plt.tight_layout()
