#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This is a notebook designed to illustrate the impact of model discretization in heterogeneous permeability fields. This notebook also provides some introductory resources for field generation.
# 
# First we import the libraries that we need

# In[155]:


# Import the flopy library
import flopy
# Import a few additional libraries
import sys
import pathlib
import os
import time
# In addition to our typical libraries
import numpy as np
import matplotlib.pyplot as plt


# First find where you have your MODFLOW and MT3D executables located on your system.

# In[156]:


# Path to MODFLOW executable, probably called 'mf2005'
exe_name_mf = 'C:\\Hydro\\MF2005.1_12\\bin\\mf2005'
# Print to make sure it is formated correctly
print(exe_name_mf)
# Path to MT3D-USGS executable, probably called 'mt3dms'
exe_name_mt = 'C:\\Hydro\\mt3dusgs1.1.0\\bin\\mt3d-usgs_1.1.0_64'
# Print to make sure it is formated correctly
print(exe_name_mt)

# This should return a path to your current working directory
current_directory = os.getcwd()
print(current_directory)

# if this is not where you want to save stuff then change your directory using 'os.chdir()'
# if this is not where you want to save stuff then change your directory using 'os.chdir()'
# define path
path = pathlib.Path('C:\\Users\\zahas\\Dropbox\\Teaching\\Contaminant hydro 629\\Notebooks_unpublished')
# if folder doesn't exist then make it 
path.mkdir(parents=True, exist_ok=True)
# set working directory to this new folder
os.chdir(path)
current_directory = os.getcwd()
print(current_directory)


# In[157]:


# now lets give a name to the directory to save data, this directory should be present in your 
# current working directory (but if it's not don't worry!)
directory_name = 'multiscale_heterogeneity_illustration'
# Let's add that to the path of the current directory
workdir = os.path.join('.', directory_name)

# if the path exists then we will move on, if not then create a folder with the 'directory_name'
if os.path.isdir(workdir) is False:
    os.mkdir(workdir) 
print("Directory '% s' created" % workdir) 
# directory to save data
datadir = os.path.join('..', directory_name, 'mt3d_test', 'mt3dms')


# Define model function nearly identical to the [FloPy 2D Macrodispersion Illustration](https://github.com/zahasky/Contaminant-Hydrogeology-Activities/blob/master/FloPy%202D%20Macrodispersion%20Illustration.ipynb) except now we keep the boundary condition of constant concentration fixed as well as the specific discharge.

# In[185]:


def model_2D(dirname, perlen_mt, hk, al, coarsen_factor):
    # Model workspace and new sub-directory
    model_ws = os.path.join(workdir, dirname)
    print(model_ws)
    
    # time units (itmuni in discretization package), unfortunately this entered differently into MODFLOW and MT3D
    # 1 = seconds, 2 = minutes, 3 = hours, 4 = days, 5 = years
    itmuni = 4 # MODFLOW
    mt_tunit = 'D' # MT3D units
    # length units (lenuniint in discretization package)
    # 0 = undefined, 1 = feet, 2 = meters, 3 = centimeters
    lenuni = 2 # MODFLOW units
    mt_lunit = 'M' # MT3D units
    
    # Modflow stress periods
    perlen_mf = [np.sum(perlen_mt)]
    # number of stress periods (MF input), calculated from period length input
    nper_mf = len(perlen_mf)
    
    # number of stress periods (MT input), calculated from period length input
    nper = len(perlen_mt)
    
    # Frequency of output, If nprs > 0 results will be saved at 
    #     the times as specified in timprs; 
    nprs = 100
    # timprs (list of float): The total elapsed time at which the simulation 
    #     results are saved. The number of entries in timprs must equal nprs. (default is None).
    timprs = np.linspace(0, np.sum(perlen_mf), nprs, endpoint=False)
    
    # Model information 
    hk_size = hk.shape
    nlay = hk_size[0] # number of layers
    nrow = hk_size[1] # number of rows
    ncol = hk_size[2] # number of columns
    delr = 0.25*coarsen_factor # grid size in direction of Lx
    delc = 1.0*coarsen_factor # grid size in direction of Ly, 
    delv = 0.25*coarsen_factor # grid size in direction of Lz
    laytyp = 0
    
    # length of model in selected units 
    Lx = (ncol - 1) * delr
    
    # porosity
    prsity = 0.3
    
    # Flow field boundary conditions
    # boundary conditions, <0 = specified head, 0 = no flow, >0 variable head
    ibound = np.ones((nlay, nrow, ncol), dtype=int)
    # index the inlet cell
    ibound[0, :, 0] = -1
    # index the outlet cell
    ibound[0, :, -1] = -1
    # constant head conditions
    strt = np.zeros((nlay, nrow, ncol), dtype=float)
    h1 = 1.5
    # index the inlet cell
    strt[0, :, 0] = h1
    print("Head difference across model: " + str(h1) + " meters")
    
    # Solute transport boundary conditions
    # Concentration at inlet boundary
    C_in = [1.0, 0.0]

    # Boundary conditions: if icbund = 0, the cell is an inactive concentration cell; 
    # If icbund < 0, the cell is a constant-concentration cell; 
    # If icbund > 0, the cell is an active concentration cell where the concentration value will be calculated.
    icbund = np.ones((nlay, nrow, ncol), dtype=int)

    # Initial conditions: concentration zero everywhere
    sconc = np.zeros((nlay, nrow, ncol), dtype=float)
    
    # MT3D stress period data, note that the indices between 'spd_mt' must exist in 'spd_mf' 
    # This is used as input for the source and sink mixing package
    # Itype is an integer indicating the type of point source, 2=well, 3=drain, -1=constant concentration
    itype = -1
    cwell_info = np.zeros((nrow, 5), dtype=float)
    # Nested loop to define every inlet face grid cell as a well
    for row in range(0, nrow):
        cwell_info[row] = [0, row, 0, C_in[0], itype] 
            
    # Second stress period        
    cwell_info2 = cwell_info.copy()   
    cwell_info2[:,3] = C_in[1] 
    # Now apply stress period info    
    spd_mt = {0:cwell_info, 1:cwell_info2}
    
    
    # Setup models
    # MODFLOW model name
    modelname_mf = dirname + '_mf'
    # MODFLOW package class
    mf = flopy.modflow.Modflow(modelname=modelname_mf, model_ws=model_ws, exe_name=exe_name_mf)
    # MODFLOW model discretization package class
    dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper_mf,
                                   delr=delr, delc=delc, top=0., botm=[0 - delv],
                                   perlen=perlen_mf, itmuni=itmuni, lenuni=lenuni)
    # MODFLOW basic package class
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    # MODFLOW layer properties flow package class
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, laytyp=laytyp)
    # MODFLOW preconditioned conjugate-gradient package class
    pcg = flopy.modflow.ModflowPcg(mf)
    # MODFLOW Link-MT3DMS Package Class (this is the package for solute transport)
    lmt = flopy.modflow.ModflowLmt(mf)
    # MODFLOW output control package
    oc = flopy.modflow.ModflowOc(mf)
    
    mf.write_input()
    mf.run_model(silent=True) # Set this to false to produce output in command window
    
    # RUN MT3dms solute tranport 
    modelname_mt = dirname + '_mt'
    # MT3DMS Model Class
    # Input: modelname = 'string', namefile_ext = 'string' (Extension for the namefile (the default is 'nam'))
    # modflowmodelflopy.modflow.mf.Modflow = This is a flopy Modflow model object upon which this Mt3dms model is based. (the default is None)
    mt = flopy.mt3d.Mt3dms(modelname=modelname_mt, model_ws=model_ws, 
                           exe_name=exe_name_mt, modflowmodel=mf)  
    
    
    # Basic transport package class
    btn = flopy.mt3d.Mt3dBtn(mt, icbund=icbund, prsity=prsity, sconc=sconc, 
                             tunit=mt_tunit, lunit=mt_lunit, nper=nper, 
                             perlen=perlen_mt, nprs=nprs, timprs=timprs)
    
    # mixelm is an integer flag for the advection solution option, 
    # mixelm = 0 is the standard finite difference method with upstream or central in space weighting.
    # mixelm = 1 is the forward tracking method of characteristics
    # mixelm = 2 is the backward tracking
    # mixelm = 3 is the hybrid method
    # mixelm = -1 is the third-ord TVD scheme (ULTIMATE)
    mixelm = -1
    
    adv = flopy.mt3d.Mt3dAdv(mt, mixelm=mixelm)
    
    dsp = flopy.mt3d.Mt3dDsp(mt, al=al)
    
    ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=spd_mt)
    
    gcg = flopy.mt3d.Mt3dGcg(mt)
    mt.write_input()
    fname = os.path.join(model_ws, 'MT3D001.UCN')
    if os.path.isfile(fname):
        os.remove(fname)
    mt.run_model(silent=True)
    
    # Extract head information
    fname = os.path.join(model_ws, modelname_mf+'.hds')
    hdobj = flopy.utils.HeadFile(fname)
    heads = hdobj.get_data()
    
    # Extract the 4D concentration values (t, x, y, z)
    fname = os.path.join(model_ws, 'MT3D001.UCN')
    ucnobj = flopy.utils.UcnFile(fname)
    # Extract the output time information, convert from list to np array
    times = np.array(ucnobj.get_times())
    # Extract the 4D concentration values (t, x, y, z)
    conc = ucnobj.get_alldata()
    
    return mf, mt, times, conc, heads


# Now let's import some permeability maps. These maps were generated with using the [gstools python toolbox](https://github.com/GeoStat-Framework/GSTools). See bottom of this notebook for example function for 3D field generation.

# In[181]:


# Import permeability map example
datafile_name = 'perm_2d_correlat_10_md_500_var_n0101.csv'
# if the data is not in your current directly then add the path information
path_to_datafile = 'C:\\Users\\zahas\\Dropbox\\Teaching\\Contaminant hydro 629\\Contaminant-Hydrogeology-Activities\\data_for_models'
data_file_with_path = os.path.join(current_directory, path_to_datafile, datafile_name)
print(data_file_with_path)
kdata_m2 = np.loadtxt(data_file_with_path, delimiter=',')

# If data is in the same folder as this notebook simply load the data (uncomment line below)
# kdata_m2 = np.loadtxt(datafile_name, delimiter=',')

# The last two values in this text file give the field dimensions
nrow = int(kdata_m2[-2]) # number of rows / grid cells
ncol = int(kdata_m2[-1]) # number of columns (parallel to axis of core)
# Print these row and column values
print('Number of rows in permeability map = ' + str(nrow))
print('Number of columns in permeability map = ' + str(ncol))

# Crop off these values and reshape column vector to matrix
kdata_m2 = kdata_m2[0:-2]
rawk_m2 = kdata_m2.reshape(1, nrow, ncol)

# Convert permeabiltiy (in m^2) to hydraulic conductivity in m/day
real1_cmsec = rawk_m2*(1000*9.81*3600*24/8.9E-4)


# Define a function for efficiently making the 2D plots.

# In[160]:


def plot_2d(map_data, dx, dy, colorbar_label, title, cmap):

    r, c = np.shape(map_data)
    # define grid
    x_coord = np.linspace(0, dx*c, c+1)
    y_coord = np.linspace(0, dy*r, r+1)
    X, Y = np.meshgrid(x_coord, y_coord)
    
    # define figure and with a littler higher res
    plt.figure(figsize=(10, 3), dpi=150)
    plt.pcolormesh(X, Y, map_data, cmap=cmap, shading = 'auto')
    plt.gca().set_aspect('equal')  
    # add a colorbar
    cbar = plt.colorbar() 
    # label the colorbar
    cbar.set_label(colorbar_label)
    plt.tick_params(axis='both', which='major')
    plt.xlim((0, dx*c)) 
    plt.ylim((0, dy*r))
    plt.title(title)


# In[182]:


# plot conductivity map
plot_2d(real1_cmsec[0,:,:], 1, 1, '[m/day]', 'Realization 1 hydraulic conductivity', 'viridis')


# Define a function that will coarsen our map by take a geometric average of some number of grid cells along two dimensions as defined by the ```coarseness``` variable.

# In[162]:


def coarsen_geomean(array2d, coarseness):
    array_size = array2d.shape
    # calculate if array is evenly divisible by level of coarsening
    rem0 = array_size[0] % coarseness
    rem1 = array_size[1] % coarseness
    if rem0 + rem1 > 0:
        raise NameError('array is not divisible by coarseness factor')
    
    # preallocate new array of values
    coarse_array = np.zeros([int(array_size[0]/coarseness), int(array_size[1]/coarseness)])
    n = coarseness**2
    # set row index
    rind = 0
    for i in range(0, array_size[0], coarseness):
        # reset column index
        cind = 0
        for j in range(0, array_size[1], coarseness):
            # calculation geometric mean of some group of grid cells
            geo_mean_cell = np.exp(np.sum(np.log(array2d[i:i+coarseness, j:j+coarseness]))/n)
            coarse_array[rind, cind] = geo_mean_cell
            # update column index
            cind += 1
        # update row index
        rind +=1
    
    # return the coarsened data
    return coarse_array
        


# Calculate the field geometric mean and generate homogenous field, field coarsened by a factor of 4, and the same field coarsened by a factor of 12.

# In[183]:


# geometric mean
geo_mean = np.exp(np.sum(np.log(real1_cmsec))/real1_cmsec.size)*np.ones([1, nrow, ncol])
# print("Geometric mean: " + str(geo_mean) + " m/day")

real1_4x = coarsen_geomean(real1_cmsec[0,:,:], 4)
# determine shape of coarsed array
cnrow, cncol = real1_4x.shape
# convert to 3D array (even though only 2D- this is necessary for MODFLOW model input)
real1_4x_3d = real1_4x.reshape(1, cnrow, cncol)
plot_2d(real1_4x_3d[0,:,:], 4, 4, '[m/day]', 'Realization 1 coarsened hydraulic conductivity', 'viridis')

real1_12x = coarsen_geomean(real1_cmsec[0,:,:], 12)
# determine shape of coarsed array
cnrow, cncol = real1_12x.shape
# convert to 3D array (even though only 2D- this is necessary for MODFLOW model input)
real1_12x_3d = real1_12x.reshape(1, cnrow, cncol)
plot_2d(real1_12x_3d[0,:,:], 12, 12, '[m/day]', 'Realization 1 coarsened hydraulic conductivity', 'viridis')


# ## Finally, we are ready to run some models!!

# In[186]:


# Directory name
dirname = 'multiscale1'
# Length of model run. Note that the model time was switched to days. So let's say we have a source that lasted for 30 days
perlen_mt = [7, 3*365]
# dispersivity is 1 meter
al = 1

mf, mt, times, conc_hom, heads = model_2D(dirname, perlen_mt, geo_mean, al, 1)
mf, mt, times1, conc1, heads1 = model_2D(dirname, perlen_mt, real1_cmsec, al, 1)
mf, mt, times1c4, conc_1c4, heads_1c4 = model_2D(dirname, perlen_mt, real1_4x_3d, al, 4)
mf, mt, times1c12, conc_1c12, heads_1c12 = model_2D(dirname, perlen_mt, real1_12x_3d, al, 12)


# In[187]:


# Plot the hydraulic heads
plot_2d(heads1[0,:,:], 1, 1, '[m]', 'Realization 1 steady state heads', 'Blues')


# In[188]:


# Plot the concentration field
plot_2d(conc1[16,0,:,:], 1, 1, '[C/C0]', 'Realization 1 concentration after 6 months', 'Reds')


# In[189]:


# Plot the coarsened concentration field
plot_2d(conc_1c4[16,0,:,:], 4, 4, '[C/C0]', 'Realization 1 concentration after 6 months', 'Reds')


# Now let's compare the breakthrough curves.

# In[190]:


# homogeneous model breakthrough curve
C_btc_r1_geomean = np.mean([conc_hom[:, 0, :, -1]], axis=2)
# uncoarsened
C_btc_r1 = np.mean([conc1[:, 0, :, -1]], axis=2)
# model coarsened by a factor of 4
C_btc_r1c4 = np.mean([conc_1c4[:, 0, :, -1]], axis=2)
# model coarsened by a factor of 4
C_btc_r1c12 = np.mean([conc_1c12[:, 0, :, -1]], axis=2)

plt.figure(figsize=(10, 3), dpi=150)
plt.plot(times, np.transpose(C_btc_r1_geomean), label='realization 1 geometric average')
plt.plot(times, np.transpose(C_btc_r1), label='realization 1')
plt.plot(times, np.transpose(C_btc_r1c4), label='realization 1, coarsened 4x')
plt.plot(times, np.transpose(C_btc_r1c12), label='realization 1, coarsened 12x')
plt.xlabel('Time [days]');
plt.legend()


# ### Comprison of different realizations (all generated with same statistical correlation statistics)
# Now compare the breakthrough curve results from the conductivity fields with the same statistical proerties but different realizations.

# In[191]:


# Import permeability second map example
datafile_name = 'perm_2d_correlat_10_md_500_var_n0102.csv'
# if the data is not in your current directly then add the path information
data_file_with_path = os.path.join(current_directory, path_to_datafile, datafile_name)
kdata_m2_2 = np.loadtxt(data_file_with_path, delimiter=',')

# Import permeability third map example
datafile_name = 'perm_2d_correlat_10_md_500_var_n0103.csv'
data_file_with_path = os.path.join(current_directory, path_to_datafile, datafile_name)
kdata_m2_3 = np.loadtxt(data_file_with_path, delimiter=',')

# Crop off these values and reshape column vector to matrix
kdata_m2_2 = kdata_m2_2[0:-2]
kdata_m2_2 = kdata_m2_2.reshape(1, nrow, ncol)
# Convert permeabiltiy (in m^2) to hydraulic conductivity in m/day
real2_cmsec = kdata_m2_2*(1000*9.81*3600*24/8.9E-4)

# Crop off these values and reshape column vector to matrix
kdata_m2_3 = kdata_m2_3[0:-2]
kdata_m2_3 = kdata_m2_3.reshape(1, nrow, ncol)
# Convert permeabiltiy (in m^2) to hydraulic conductivity in m/day
real3_cmsec = kdata_m2_3*(1000*9.81*3600*24/8.9E-4)

# plot conductivity map
plot_2d(real2_cmsec[0,:,:], 1, 1, '[m/day]', 'Realization 2 hydraulic conductivity', 'viridis')

# plot conductivity map
plot_2d(real3_cmsec[0,:,:], 1, 1, '[m/day]', 'Realization 3 hydraulic conductivity', 'viridis')


# In[192]:


mf, mt, times2, conc2, heads2 = model_2D(dirname, perlen_mt, real2_cmsec, al, 1)
mf, mt, times3, conc3, heads3 = model_2D(dirname, perlen_mt, real3_cmsec, al, 1)


# In[193]:


# uncoarsened
C_btc_r2 = np.mean([conc2[:, 0, :, -1]], axis=2)
C_btc_r3 = np.mean([conc3[:, 0, :, -1]], axis=2)

plt.figure(figsize=(10, 3), dpi=150)
# plt.plot(times, np.transpose(C_btc_r1_geomean), label='realization 1 geometric average')
plt.plot(times, np.transpose(C_btc_r1), label='realization 1')
plt.plot(times, np.transpose(C_btc_r2), label='realization 2')
plt.plot(times, np.transpose(C_btc_r3), label='realization 3')
plt.xlabel('Time [days]');
plt.legend()


# Bonus function to generate random fields using the [gstools python toolbox](https://github.com/GeoStat-Framework/GSTools).

# In[ ]:


# function to generate 3D random permeability fields 
def perm_field_generation(log_mD, log_var, correlat_len, ycorrelat_len, nlay, nrow, ncol, angle):
    x = np.arange(nlay)
    y = np.arange(nrow)
    z = np.arange(ncol)

    model = gs.Exponential(dim=3, var=10**log_var, len_scale=[1.0, ycorrelat_len, correlat_len], angles=[0.0, 0.0, angle])
    
    # If you specify the same seed then the generator will produce the same realization over and over
    # srf = gs.SRF(model, seed=25300)
    srf = gs.SRF(model)
    
    field = 10**(srf.structured([x, y, z]) + log_mD)
    
    print('Geometric mean: ' + str(np.log10(np.max(field)/np.min(field))) + ' mD')
    
    # convert from mD to km^2
    field_km2 = field*(9.869233E-13/1000)
    return field_km2

