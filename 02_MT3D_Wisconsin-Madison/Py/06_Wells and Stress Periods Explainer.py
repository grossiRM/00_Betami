#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Often times we will want to create more complex boundary conditions, such as those with a changing source concentration with time, or a source that is not at the edge of a model. This can get a little complicated so this notebook demonstrates how to set up these sources as 'wells' and how to edit the Modflow-MT3D stress periods accordingly. 
# 
# If you haven't installed Flopy, go back to the [MODFLOW and FloPy setup notebook](https://github.com/zahasky/Contaminant-Hydrogeology-Activities/blob/master/MODFLOW%2C%20Python%2C%20and%20FloPy%20Setup.ipynb) and the [FloPy Introduction notebook](https://github.com/zahasky/Contaminant-Hydrogeology-Activities/blob/master/FloPy%20Introduction.ipynb).
# 
# Import the standard libraries, plus a one we haven't seen ([deepcopy](https://docs.python.org/3/library/copy.html#copy.deepcopy)).

# In[ ]:


# Import the flopy library
import flopy
# Import a few additional libraries
import sys
import os
import pathlib
# In addition to our typical libraries
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as deepcopy


# First find where you have your MODFLOW and MT3D executables located on your system.

# In[ ]:


# Path to MODFLOW executable, probably called 'mf2005'
exe_name_mf = 'C:\\Hydro\\MF2005.1_12\\bin\\mf2005.exe'
# Print to make sure it is formated correctly
print(exe_name_mf)
# Path to MT3D-USGS executable, probably called 'mt3dms'
exe_name_mt = 'C:\\Hydro\\mt3dusgs1.1.0\\bin\\mt3d-usgs_1.1.0_64.exe'
# Print to make sure it is formated correctly
print(exe_name_mt)


# Let's use the same directory to save the data as the FloPy introduction and then create a path to this workspace. It may be useful to understand your current working directory, this should be whereever you have this notebook saved. You can double check this with the command 'os.getcwd()'.

# In[ ]:


# This should return a path to your current working directory
current_directory = os.getcwd()
print(current_directory)


# In[ ]:


# if this is not where you want to save stuff then change your directory using 'os.chdir()'
# define path
path = pathlib.Path('C:\\Users\\zahas\\Dropbox\\Teaching\\Contaminant hydro 629\\Notebooks_unpublished')
# if folder doesn't exist then make it 
path.mkdir(parents=True, exist_ok=True)
# set working directory to this new folder
os.chdir(path)
current_directory = os.getcwd()
print(current_directory)


# In[ ]:


# directory to save data
directory_name = 'well_demo'
# directory to save data
datadir = os.path.join('..', directory_name, 'mt3d_test', 'mt3dms')
workdir = os.path.join('.', directory_name)


# As usual, 'dirname' will be an input to our model function.

# ### 2D plot function
# Before jumping into the model, lets define a function for easily generating 2D plots. This will help us visualize what is happening as we build our model.

# In[ ]:


def plot_2d(map_data, dx, dy, colorbar_label, cmap):
    # shape information
    r, c = np.shape(map_data)
    # define plot coordinates
    x_coord = np.linspace(0, dx*c, c+1)
    y_coord = np.linspace(0, dy*r, r+1)
    
    X, Y = np.meshgrid(x_coord, y_coord)

    plt.figure(figsize=(8, 4), dpi=120)
    plt.pcolormesh(X, Y, map_data, cmap=cmap, shading = 'auto', edgecolor ='k', linewidth = 0.05)
    plt.gca().set_aspect('equal')  
    # add a colorbar
    cbar = plt.colorbar() 
    # plt.clim(cmin, cmax) 
    # label the colorbar
    cbar.set_label(colorbar_label)
    plt.tick_params(axis='both', which='major')
    plt.xlim((0, dx*c)) 
    plt.ylim((0, dy*r)) 
    plt.show()


# ## Well Demo Function
# So far when we have built models we have not messed around to much with things like the source concentration, timing, and location but this is often the first step in building more complex and site-specific models. Here we will first set up a function that pulls out these model inputs so that we can more carefully explore what is happening.
# 
# The first thing we do is setup the function. We will use nearly identical settings as we used in the [FloPy Introduction notebook](https://github.com/zahasky/Contaminant-Hydrogeology-Activities/blob/master/FloPy%20Introduction.ipynb) example, but now we are providing a few input variables that can be changed everytime we call the model. The input variables are:
# 
# ### Function Input:
# #### directory name
#     direname = 
# 
# #### period length 
# Time is in selected units, the model time length is the sum of this (for steady state flow it can be set to anything). The format for multi-period input: ```[60., 15*60]```
#  
#     perlen_mf = 
#     
# #### advection velocity
# Note that this is only an approximate advection flow rate in due to the way that the inlet boundary conditions are being assigned in the MODFLOW BAS6 - Basic Package. More rigorous constraint of constant flux boundaries require the Flow and Head Boundary Package, the Well Package, or the Recharge Package.
# 
#     v = 
#     
# #### dispersivity
# Set the longitudinal dispersivity in selected units. What are the units again?
# 
#     al = 
#     
# #### itype
# An integer indicating the type of solute source condition. ```itype = -1``` corresponds to a constant concentration boundary (first-type boundary conditions in our analytical solutions) and ```itype = 1``` is equivalent to third type boundary conditions in our analytical solution.  
# 
#     itype = 
#     

# In[ ]:


def model_with_wells(dirname, perlen_mf, al, ibound, strt, icbund, sconc, spd_mf, spd_mt, nlay, nrow, ncol):
#                      dirname, perlen_mf, al, ibound, strt, icbund, sconc, spd_mf, spd_mt, nlay, nrow, ncol
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
    
    # number of stress periods (MF input), calculated from period length input
    nper = len(perlen_mf)
    
    # Frequency of output, If nprs > 0 results will be saved at 
    #     the times as specified in timprs; 
    nprs = 100
    # timprs (list of float): The total elapsed time at which the simulation 
    #     results are saved. The number of entries in timprs must equal nprs. (default is None).
    timprs = np.linspace(0, np.sum(perlen_mf), nprs, endpoint=False)
    
    # hydraulic conductivity
    hk = 1.
    # porosity
    prsity = 0.3
    
    # Grid cell size in selected units
    delv = 1 # grid size for nlay
    delc = 1 # grid size for nrow
    delr = 1 # grid size for ncol

    # Setup models
    # MODFLOW model name
    modelname_mf = dirname + '_mf'
    # Assign name and create modflow model object
    mf = flopy.modflow.Modflow(modelname=modelname_mf, model_ws=model_ws, exe_name=exe_name_mf)
    # MODFLOW model discretization package class
    dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper,
                                   delr=delr, delc=delc, top=0., botm=[0 - delv],
                                   perlen=perlen_mf, itmuni=itmuni, lenuni=lenuni)
    # MODFLOW basic package class
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    # MODFLOW layer properties flow package class
    laytyp = 0
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, laytyp=laytyp)
    # MODFLOW well package class
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=spd_mf)
    # MODFLOW preconditioned conjugate-gradient package class
    pcg = flopy.modflow.ModflowPcg(mf)
    # MODFLOW Link-MT3DMS Package Class (this is the package for solute transport)
    lmt = flopy.modflow.ModflowLmt(mf)
    
    mf.write_input()
    mf.run_model(silent=True) # Set this to false to produce output in command window
    
    # RUN MT3dms solute tranport 
    modelname_mt = dirname + '_mt'
    # MT3DMS model object
    # Input: modelname = 'string', namefile_ext = 'string' (Extension for the namefile (the default is 'nam'))
    # modflowmodelflopy.modflow.mf.Modflow = This is a flopy Modflow model object upon which this Mt3dms model is based. (the default is None)
    mt = flopy.mt3d.Mt3dms(modelname=modelname_mt, model_ws=model_ws, 
                           exe_name=exe_name_mt, modflowmodel=mf)  
    
    
    # Basic transport package class
    btn = flopy.mt3d.Mt3dBtn(mt, icbund=icbund, prsity=prsity, sconc=sconc, 
                             tunit=mt_tunit, lunit=mt_lunit, nprs=nprs, timprs=timprs)
    
    # Advection package class
    # mixelm is an integer flag for the advection solution option, 
    # mixelm = 0 is the standard finite difference method with upstream or central in space weighting.
    # mixelm = 1 is the forward tracking method of characteristics, this produces minimal numerical dispersion.
    # mixelm = 2 is the backward tracking
    # mixelm = 3 is the hybrid method (HMOC)
    # mixelm = -1 is the third-ord TVD scheme (ULTIMATE)
    mixelm = -1
    # percel is the Courant number for numerical stability (≤ 1)
    adv = flopy.mt3d.Mt3dAdv(mt, mixelm=mixelm, percel=0.5)
    
    # Dispersion package class
    dsp = flopy.mt3d.Mt3dDsp(mt, al=al)
    # source/sink package
    ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=spd_mt)
    # matrix solver package, may want to add cclose=1e-6 to define the convergence criterion in terms of relative concentration
    gcg = flopy.mt3d.Mt3dGcg(mt, cclose=1e-6)
    # write mt3dms input
    mt.write_input()
    
    # run mt3dms
    mt.run_model(silent=True)

    # Extract output
    fname = os.path.join(model_ws, 'MT3D001.UCN')
    ucnobj = flopy.utils.UcnFile(fname)
    # Extract the output time information, convert from list to np array
    times = np.array(ucnobj.get_times())
    # Extract the 4D concentration values (t, z, y, x)
    conc = ucnobj.get_alldata()
    
    return mf, mt, times, conc


# ### Define stress periods
# While it is possible to define different stress period arrays for Modflow and MT3D, it is easier and likely less buggy to define them together.

# In[ ]:


# perlen (float or array of floats): An array of the stress period lengths.
perlen_mf = [1, 18, 10]
nper = len(perlen_mf)


# What are the stress period times here?

# ### Define model geometry
# Double check the function units. What are they?
# 
# Now define number of grid cells and the corresponding size of the grid cells.

# In[ ]:


# Number of grid cells
nlay = 1
nrow = 20
ncol = 40


# ### Define flow boundary conditions and intitial conditions

# In[ ]:


# Flow field boundary conditions (variables for the BAS package)
# boundary conditions, <0 = specified head, 0 = no flow, >0 variable head
ibound = np.ones((nlay, nrow, ncol), dtype=int)
# # index the cell all the way to the left
# ibound[0, 0, 0] = -1 # set to specified head
# index the cell all the way to the right
ibound[0, :, -1] = -1


# Note how we index the second index of ibound with the colon (:) to indicate 'all cells'. Lets see what this looks like in the cell below and demonstrate our handy 2d plot function.

# In[ ]:


print(ibound.shape)
print(ibound[0,:,:].shape)

plot_2d(ibound[0,:,:], 1, 1, 'ibound values', 'magma')


# In[ ]:


# Now flow initial conditions. All cells where ibound=1 will be solved in the flow model.
# constant head conditions
strt = np.zeros((nlay, nrow, ncol), dtype=float)
# All cells where ibound=-1 should be assigned a value


# As currently defined, the cell all the way to the right has a fixed head of zero and all other heads are solved

# ### Define flow wells

# In[ ]:


# total flow 
q = [0.5, 0.5, -0.5] # 0.5 meter per day

# Stress period well data for MODFLOW. Each well is defined through defintition
# of layer (int), row (int), column (int), flux (float). The first number corresponds to the stress period
# Example for 1 stress period: spd_mf = {0:[[0, 0, 1, q],[0, 5, 1, q]]}
# define well info structure
well_info = np.zeros((int(nrow), 4), dtype=float)
# set indices of left face of model
well_info[:,1] = range(0, nrow)
# set volumetric flow rate
well_info[:,3] = q[0]
# use copy.deepcopy (imported as 'deepcopy') to copy well_info array into dictonary
# note that if this deepcopy isn't made then when the flow rate it updated
# in well_info it will update all values copied to dictionary!
w = deepcopy(well_info)
# Now insert well information into the MODFLOW stress period data dictionary
spd_mf={0:w}

well_info[:,3] = q[1]
# use copy.deepcopy (imported as 'deepcopy') to copy well_info array into dictonary
# note that if this deepcopy isn't made then when the flow rate it updated
# in well_info it will update all values copied to dictionary!
w = deepcopy(well_info)
# Now insert well information into the MODFLOW stress period data dictionary
spd_mf.update({1:w})

well_info[:,3] = q[2]
# use copy.deepcopy (imported as 'deepcopy') to copy well_info array into dictonary
# note that if this deepcopy isn't made then when the flow rate it updated
# in well_info it will update all values copied to dictionary!
w = deepcopy(well_info)
# Now insert well information into the MODFLOW stress period data dictionary
spd_mf.update({2:w})

# Here is how you might set up a for loop to progressively update q from discrete time series information
# iterate through the stress periods to updated the flow rate
# for i in range(1,nper):
#     # print(q[i])
#     if isinstance(q, (list, tuple, np.ndarray)):
#         well_info[:,3] = q[i]/(nrow*nlay)
#     else:
#         well_info[:,3] = q/(nrow*nlay)

#     w = deepcopy(well_info)
#     # spd_mf = dict(spd_mf, {i: well_info)})
#     spd_mf.update({i:w})  


# In[ ]:


print(spd_mf)


# ### Define transport boundary conditions and intitial conditions

# In[ ]:


# Boundary conditions: if icbund = 0, the cell is an inactive concentration cell; 
# If icbund < 0, the cell is a constant-concentration cell; 
# If icbund > 0, the cell is an active concentration cell where the concentration value will be calculated.
icbund = np.ones((nlay, nrow, ncol), dtype=int)

# Initial conditions: initial concentration zero everywhere
sconc = np.zeros((nlay, nrow, ncol), dtype=float)


# ### Define solute conditions at wells for different stress periods

# In[ ]:


# Solute transport boundary conditions
# Concentration at well during first stress period
c = [1, 0, 0]

# MT3D stress period data, note that the indices between 'spd_mt' must exist in 'spd_mf' 
# This is used as input for the source and sink mixing package
# Itype is an integer indicating the type of point source, 2=well, 3=drain, -1=constant concentration
itype = -1
cwell_info = np.zeros((int(nrow), 5), dtype=float)
cwell_info[:,1] = range(0, nrow)
cwell_info[:,3] = c[0]
# assign itype
cwell_info[:,4] = itype
spd_mt = {0:cwell_info}

# Second stress period        
cwell_info2 = deepcopy(cwell_info)
cwell_info2[:,3] = c[1] 
# Now apply stress period info    
spd_mt.update({1:cwell_info2})

# Third stress period        
cwell_info2 = deepcopy(cwell_info)
cwell_info2[:,3] = c[2] 
# Now apply stress period info 
spd_mt.update({2:cwell_info2})


# Note that if you have more model stress periods than you have defined in the wells then it carries those conditions forward in remaining stress periods

# In[ ]:


print(spd_mt)


# What we have defined is a model that has three stress periods different conditions. This is plotted in the next cell.

# In[ ]:


print(perlen_mf)
model_time = np.cumsum(perlen_mf)
print(model_time)

plt.plot([0, model_time[0]], [q[0], q[0]], color='r', label= 'Flow rate [m/day]')
plt.plot([model_time[0], model_time[1]], [q[1], q[1]], color='r')
plt.plot([model_time[1], model_time[2]], [q[2], q[2]], color='r')

plt.plot([0, model_time[0]], [c[0], c[0]], color='b', label= 'Concentration')
plt.plot([model_time[0], model_time[1]], [c[1], c[1]], color='b')
plt.plot([model_time[1], model_time[2]], [c[2], c[2]], color='b')

plt.plot([model_time[0], model_time[0]], [-1, 1], '--k', label= 'Stress period boundary')
plt.plot([model_time[1], model_time[1]], [-1, 1], '--k', )
plt.plot([model_time[2], model_time[2]], [-1, 1], '--k', )

plt.xlabel('Time [days]')
plt.ylabel('Concentration / Flow rate')
plt.legend()
plt.show()


# #### How does the concentration change in each stress period? How is the flow magnitude and direction going to change?

# In[ ]:


dirname = 'run1'
al = 0.1 # m

# Call the FloPy model function with this well information
mf, mt, times, conc = model_with_wells(dirname, perlen_mf, al, ibound, strt, icbund, sconc, spd_mf, spd_mt, nlay, nrow, ncol)


# Now let's plot the 2D model output as a function of time

# In[ ]:


# To understand output size it may be useful to print the shape
print(conc.shape)


# In[ ]:


# early time
plot_2d(conc[1,0,:,:], 1, 1, 'C', 'Reds')


# In[ ]:


# around the switch in flow (stress period 1 - 2 boundary)
plot_2d(conc[58,0,:,:], 1, 1, 'C', 'Reds')


# In[ ]:


# last time step
plot_2d(conc[-1,0,:,:], 1, 1, 'C', 'Reds')
print(conc.shape)


# Alternatively we can plot the mean profiles parallel to the direction of flow

# In[ ]:


# Extract the model grid cell center location (in selected units, cm in this case)
ym, xm, zm = mf.dis.get_node_coordinates()
plt.plot(xm, np.mean(conc[1, 0, :, :], axis=0), label='ts = 1')
plt.plot(xm, np.mean(conc[58, 0, :, :], axis=0), label='ts = 58')
plt.plot(xm, np.mean(conc[-1, 0, :, :], axis=0), label='end of sim')
plt.xlabel('X [m]');
plt.legend()
plt.show()


# Explain why the end of sim solute profile is in between timestep 1 and timestep 58. Hint: look at the flow rate on the stress period plot.

# In[ ]:




