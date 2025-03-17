____________________________________________________________________________________________________________________________________________________01
#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The purpose of this exercise is to introduce the basics of using Flopy to construct, run and visualize a MODFLOW 6 model and its outputs. It is assumed you are familiar with MODFLOW 6.
# 
# We will cover the following:
#  - creating a Simulation Object
#  - creating a Model Object
#  - defining time and spatial discretisation
#  - adding Packages
#  - writting the MODFLOW files and running the model
#  - post-processing some results
# 
# This exercise is based on a simple groundwater system composed of two aquifer layers, separated by a thin low-permeability layer. A river flows across the center of the system in a straight line (not very natural, I know, but it keeps things simple for the tutorial), from West to East (left to right). The river only intersects the upper aquifer layer. The upper layer also receives recharge from rainfall.
# 
# We will represent the system using a classical structured grid. 

# In[1]:


# Import necessary libraries
# for the purposes of this course we are using frozen versions of flopy to avoid depedency failures.  
import os 
import sys
sys.path.append('../dependencies/')
import flopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## The Simulation
# FloPy requires that we first create a "simulation" object. This simulation can have multiple models. There are a couple of things that you will generaly have to assign:
# - a Simulation package
# - a TDIS package
# - one or more MF6 Models, which will generaly require:
#     - an IMS (i.e. the solver settings) package
#     - a spatial discretisation (DIS, DISV or DISU) package
#     - initial condition package
#     - hydraulic property package(s)
#     - boundary condition pacakge(s)
# 
# 
# A Simulation Object is instantiated with the command: *flopy.mf6.MFSimulation()*
# 
# Three inputs are usually necessary (they all have default values):
#  - the simulation name
#  - the path to the executable (in our case mf6.exe)
#  - the path to the simulation folder

# In[2]:


# you can customize the simulation name
sim_name = 'symple_ex01'

# if the MF6 executable is in your PATH, you can simply assign the string "mf6". 
# If not, you need to specify the path to the executable. 
# In the course repository folder, there is a folder named "bin" which contians all the necessary executables. 
# The MODFLOW 6 executable is named "mf6.exe":
exe_name = os.path.join('..','bin', 'mf6.exe')

# define the location of the model working directory; this is where all the MF6 files will be written
# the folder path is relative to the location in which python is running. In our case, relative to the location of the jupyter notebok file.
workspace = os.path.join('..','models','symple_ex01')

# Usually you will want to assing the Simulation Object to a variable (common practice is to name it "sim") so that it can be accessed later
sim = flopy.mf6.MFSimulation(sim_name=sim_name,
                            exe_name=exe_name,
                            version="mf6", 
                            sim_ws=workspace)


# ### The TDIS (time discretisation) object
# Time discretisation (i.e. the TDIS package) is defined at the simulation level. Let's instantiate a Tdis object. To do so, we need to define the stress period data.
# 
# Stress period data needs to be passed to the Tdis object as a list of tuples. The list needs a tuple for each stress period. Each tuple contains the period length, the number of time steps and the time-step multiplier:
#  \[(perlen, nstp, tsmult)]
# 
# We will have a single steady-state stress-period, so period length does not matter. In this case the number of time steps should be 1, and time-step multiplier does not matter.

# In[3]:


# define the time units for the simulation. All model inputs must have time-units of "day"
time_units = 'days'

#perioddata[perlen, nstp, tsmult]
perioddata = [(1.0, 1, 1.0)]

# the number of periods is 1
nper = 1 

# Create the TDIS Object
tdis = flopy.mf6.ModflowTdis(sim, pname="tdis",
                                  nper=nper, 
                                  perioddata=perioddata, 
                                  time_units=time_units)


# ## The Flow Model
#  Now we can create the FloPy MF6 Model Object and add the corresponding IMS package to the simulation.

# In[4]:


# Instantiate the Flopy groundwater flow (gwf) model object, refercing the "sim" Simulation Object
model_name = 'symp01'
gwf = flopy.mf6.ModflowGwf(sim,
                            modelname=model_name,
                            save_flows=True, print_flows=True)

# Instantiate a Flopy `IMS` Package object
# Here you can set all the solver settings.
ims = flopy.mf6.ModflowIms(sim,
                            pname="ims",
                            complexity="SIMPLE",
                            linear_acceleration="BICGSTAB",)

# lastly we need to register the MF6 model to an IMS package in the Simulation
sim.register_ims_package(ims, [gwf.name])


# ### The Model Grid
# So far we have created the Simulation, defined the Simulations time-discretisation and created a Model as part of that Simulation. 
# 
# Now we will start constructing the model itself. The Model will be built by adding packages to it that describe the features of the system. The first step is to define the spatial discretisation, as this is required before trying to assign any of the hydraulic property or stress packages. 
# 
# Recall that we will be using a classical structured grid. A Flopy DIS Object is created with *flopy.mf6.ModflowGwfdis()*. Note that DISV or DISU grids are created with their respective functions, which will be covered in other exercises.
# 
# To define sptial discretisation we require:
#  - number of layers (3)
#  - number of rows and columns
#  - row and column lenght
#  - elevation of the top of the model
#  - elevation of the bottom of each layer

# In[5]:


# set the length units. All model input values must have untis of length in "meters"
length_units = "METERS"
# number of layers
nlay = 3

# define the number of rows/columns
# our system covers a square area of 1x1.5 km. The extent in the x and y directions are 1000m and 1500m, respectively.
Lx = 1000
Ly = 1500
# we want our model grid to have cell sizes of 100 x 100 m
delr = 100 #row length
delc = 100 #column length
print(f'Cell size:{delr} x {delc}')
# we can compute the number of rows/columns:
ncol = int(Lx/delc)
nrow = int(Ly/delr)
print(f'Number of rows:{nrow}')
print(f'Number of columns:{ncol}')

# surface elevation of the systme is flat and equal to 50 m above some reference (e.g. sea-level)
top = 50
# the bottom of the upper-aquifer is at 40m; the aquitard at 35m and the lower aquifer at 0m
botm = [40, 35, 0]

# create the DIS object
dis = flopy.mf6.ModflowGwfdis(
                            gwf,
                            nlay=nlay,
                            nrow=nrow,
                            ncol=ncol,
                            delr=delr,
                            delc=delc,
                            top=top,
                            botm=botm)


# In[6]:


# Lets check out the layer bottom elevations
dis.botm


# ### Packages
# Now that we have the "skeleton" of the model (i.e. the grid) we can assign pacakges to define properties and stresses.
# 
# For this exercise, we will assign:
#  - initial condiction (IC) package to set initial conditions
#  - node property flow (NPF) package to set hydraulic properties
#  - recharge (RCH) pacakge to assign recharge to the upper layer
#  - river (RIV) package to define the river boundary condition
#  - the output control (OC) package to determine how model outputs are recorded
# 
# 
# 

# #### Array data

# In[7]:


# Create the initial conditions package
# you can set a single value for the entire model
strt = 50 
# or assign discrete values per layer. For example:
strt = [50, 50, 50]
# or even set the same value for each cell, by passing an array of shape (nlay, nrow, ncol). side note: we will go into dfferent ways to handle array data in a later exercise
# Lets do that, and set initial heads equal to the model top:
strt = top * np.ones((nlay, nrow, ncol))


# Now we can create the IC package
ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=strt)


# In[8]:


# Next, let's create the NPF pacakge to assign values of hydraulic conductivity (K)
# Each layer has different K, so we wish to assign diferent values to each layer. 
# set the value of k for each layer
k = [5, 0.1, 10]

# here we can also set the icelltype to determine whether a layer is trated as variably saturated or not.
# let us set the top layer as variably saturated (i.e. unconfiend) and the others as saturated
icelltype = [1, 0, 0]

npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=icelltype, k=k,
                              save_flows=True, 
                              save_specific_discharge=True) # so that we can plot specific discharge later


# In[9]:


# Create the recharge package.
# For this simple exercise we will simply assign a single uniform value for the entire model run. 
# To do so we can use the rechage array package.
# Other exercises will demonstrate other ways to assign recharge using list-type data
recharge = 50/1000/365 # 50mm/yr in m/d

# Note that this is flopy.mf6.ModflowGwfrcha(). Different from flopy.mf6.ModflowGwfrch()
rch = flopy.mf6.ModflowGwfrcha(gwf, pname='rch', recharge=recharge)


# #### List data

# In[10]:


# Lastly, we need to assign the river boundary condition in the upper layer using the RIV package.
# The river will be assigned to cells in the upper layer, in row 7 (the middle of the model domain in this case)
# Here we will use list data. We will go into greater detail on how to handle list data in a later exercise.

riv_row = 7
stage = top - 5
rbot = botm[0]
cond = 0.1 * delr*delc/1

riv_spd = []
for col in range(ncol):
    riv_spd.append(((0, riv_row, col), stage, cond, rbot))

riv = flopy.mf6.ModflowGwfriv(gwf, stress_period_data=riv_spd, boundnames=True)


# In[11]:


# create the output control (OC) package.
# Here we define how model outputs are recorded. 

# Save heads and budget output to binary files and 
# print heads to the model list file at the end of the stress period.

# the name of the binary head file
headfile = f"{gwf.name}.hds"
head_filerecord = [headfile]
# the name of the binary budget file
budgetfile = f"{gwf.name}.cbb"
budget_filerecord = [budgetfile]

# which outputs are recored to the binary files
saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
# which outputs are printed in the list file
printrecord = [("HEAD", "LAST")]
oc = flopy.mf6.ModflowGwfoc(gwf,
                            saverecord=saverecord,
                            head_filerecord=head_filerecord,
                            budget_filerecord=budget_filerecord,
                            printrecord=printrecord)


# ## Write the Model files

# In[12]:


# Write the model files by calling .write_simulation(). You can then inspect the workspace folder to see the MF6 input files written by Flopy. 
sim.write_simulation()


# ## Run the Model

# In[13]:


success, buff = sim.run_simulation()
if not success:
    raise Exception("MODFLOW 6 did not terminate normally.")


# # Post-process model outputs
# 
# For MODFLOW6, Flopy has built-in methods to get model outputs for some packages using the *.output* attribute. There are other ways in which you can access model ouputs, however for the purposes of this course we will try and keep it as simple as possible. 
# 
# Common ouputs which you will likely wish to access may include:
#  - heads at various times
#  - budget components
#  - specific discharge vectors
#  - and mass, density and so on in transport models 

# In[14]:


# You can check which output functions are available for any given package by using the output.methods() function.
# output functions available at the model level:
gwf.output.methods()


# In[15]:


# for example for the RIV package (although it will be empty because we havent recorded any observations yet):
gwf.riv.output.methods()


# ## Heads
# Depending on the settings in the OC package, simulated heads are recorded in the binary file (in our case named "symp01.hds"). 
# We configured outputs to be recorded at all timesteps in the first stress period. 
# As we only have a single steadystate stress period with 1 time step, heads are recorded only once. 

# In[16]:


# the head file output can be loaded from the model object:
hds = gwf.output.head()

# head data can then be accessed using the get_data() or get_alldata() functions
# get_alldata() returns an array with all recorded times
heads = hds.get_alldata()

# get_alldata() returns an array of shape (number of records, nlay, nrow, ncol)
heads.shape


# In[17]:


# get_data() returns an array with a single recorded time. Which time to read is passed by the user as either:
# an index value, 
heads = hds.get_data(idx=0)
# a tuple of stressperiod and timestep, 
heads = hds.get_data(kstpkper=(0,0))
# or a value of time
heads = hds.get_data(totim=1)

# get_data() returns an array of shape (nlay, nrow, ncol)
heads.shape


# ### Plot heads
# Flopy has built-in utilities to facilitate plotting. Use the PlotMapView() to plot model outputs.
# 
# Let us first create a plot of head in the upper layer:

# In[18]:


fig = plt.figure(figsize=(5, 5), constrained_layout=True)

# first instantiate a PlotMapView
mm = flopy.plot.PlotMapView(model=gwf)

# Plot heads
# plot the array of heads 
head_array = mm.plot_array(heads)
# add contour lines with contour_array()
contours = mm.contour_array(heads, colors='black')
# add labels to contour lines
plt.clabel(contours, fmt="%2.1f")
# add a color bar
cb = plt.colorbar(head_array, shrink=0.5, )
cb.ax.set_title('Heads')


# Plot grid 
# you can plot BC cells using the plot_bc() 
mm.plot_bc('RIV', color='blue')
# and plot the model grid if desired
mm.plot_grid(lw=0.5)


# ### Plot a cross-section
# Plot a cross section of heads along column 5.

# In[19]:


column = 5

# create the figure and subplots
fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)

# instantiate the a PlotCrosSection object, assign it to the axis and use the "line" attribute to define the crossection. ALternatively you could use {"row":rownumber} or {"line":array of (x, y) tuples with vertices of cross-section}
mm = flopy.plot.PlotCrossSection(model=gwf, ax=ax, line={"column": column})

# plot head array
head_array = mm.plot_array(heads)
# add color bar
cb = plt.colorbar(head_array, shrink=0.5, ax=ax)
# add cotour lines and labels
contours = mm.contour_array(heads, colors="black")
ax.clabel(contours, fmt="%2.1f")

# plot grid and BCs
quadmesh = mm.plot_bc("RIV")
linecollection = mm.plot_grid(lw=0.5, color="0.5")

# set the title
ax.set_title(f"Column {column}")


# ### Plot Specifc discharge
# 

# In[20]:


# get the specific discharge from the cell budget file
# first access the binary budget file
cbb = gwf.output.budget()

# read the specific discahrge data
spdis = cbb.get_data(text="SPDIS")[0]

# use Flopy's postprocessing functions to get specfici discharge x, y and z vectors
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)


# In[21]:


# Repeat the same steps as above, adding in the specific discharge quiver plot:

fig = plt.figure(figsize=(5, 5), constrained_layout=True)
# first instantiate a PlotMapView
mm = flopy.plot.PlotMapView(model=gwf)

# Plot heads
# plot the array of heads 
head_array = mm.plot_array(heads)
# add contour lines with contour_array()
contours = mm.contour_array(heads, colors='black')
# add labels to contour lines
plt.clabel(contours, fmt="%2.1f")
# add a color bar
cb = plt.colorbar(head_array, shrink=0.5, )
cb.ax.set_title('Heads')


# Plot grid 
# you can plot BC cells using the plot_bc() 
mm.plot_bc('RIV', color='blue')
# and plot the model grid if desired
mm.plot_grid(lw=0.5)

################# New Step ##########################
# add sepcific discharge vectors using plot_vector()
quiver = mm.plot_vector(qx, qy, normalize=False, color='blue')


# ## Read the List Budget file
# If you need to check the model buget

# In[22]:


mf_list = flopy.utils.Mf6ListBudget(os.path.join(workspace, f"{gwf.name}.lst"), timeunit='days')
# read as a list
incremental, cumulative = mf_list.get_budget()
# read as a Pandas DataFrame (much nicer)
incrementaldf, cumulativedf = mf_list.get_dataframes()

# inspect the incremental budget
incrementaldf.head()


# In[23]:


# inspect the cumulative budget. In this case they are the same as the model has a single stress period
cumulativedf.head()


____________________________________________________________________________________________________________________________________________________02
#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The purpose of this exercise is demosntrate how to load an existing model, inspect and alter it. 
# 
# We will:
#  - load the model constructed in exercise 01, 
#  - transfer the workspace to a new directory, 
#  - inspect the model packages, 
#  - alter the hydraulic conductivity in the upper layer, 
#  - re-write the model files and run the model, and
#  - plot the outputs

# In[1]:


# Import necessary libraries
# for the purposes of this course we are using frozen versions of flopy to avoid dependency failures.  
import os 
import sys
sys.path.append('../dependencies/')
import flopy
import matplotlib.pyplot as plt


# In[2]:


# As in the previous exercise, if you do not have MODFLOW 6 in yout system path, you must provide the location of the executable file
# The MODFLOW 6 executable is named "mf6.exe":
exe_name = os.path.join('..','bin', 'mf6.exe')

# define the location of the model working directory;  this is where the existing model is currently stored. We are going to load the model constructed during exercise 01.
org_workspace = os.path.join('..','models','symple_ex01')

# define a new model working directory
workspace = os.path.join('..','models','symple_ex02')


# In[3]:


# load the Simulation
sim = flopy.mf6.MFSimulation.load(sim_name='symple_ex02', exe_name=exe_name, sim_ws=org_workspace)


# In[4]:


# change the model workspace to a new folder
sim.set_sim_path(workspace)


# In[5]:


# get a list of model name sin the simulation. Names are used to load the model
model_names = list(sim.model_names)
for mname in model_names:
    print(mname)


# In[6]:


# access the Model Object using the model name
gwf = sim.get_model("symp01")


# In[7]:


# check what packages are in the model
pkg_list = gwf.get_package_list()
print(pkg_list)


# In[13]:


# access a pacakge 
npf = gwf.get_package('npf')

# inspect the values of k
npf.k


# In[9]:


# change k in the upper layer
npf.k.set_data(2.5, layer=0)

# inspect it again
npf.k


# In[10]:


# write the files and run the simulation. Inspect the .npf file in the new workspace folder. You should see K in layer 1 has changed.
sim.write_simulation()

sim.run_simulation()


# In[12]:


# plot outputs from the upper layer. The code below is the same as used in Exercise 01.

# load outputs
# the head file output can be loaded from the model object:
hds = gwf.output.head()
heads = hds.get_data(idx=0)

# get the specific discharge from the cell budget file
cbb = gwf.output.budget()
spdis = cbb.get_data(text="SPDIS")[0]
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)


# plot
fig = plt.figure(figsize=(5, 5), constrained_layout=True)
# first instantiate a PlotMapView
mm = flopy.plot.PlotMapView(model=gwf)

# Plot heads
# plot the array of heads 
head_array = mm.plot_array(heads)
# add contour lines with contour_array()
contours = mm.contour_array(heads, colors='black')
# add labels to contour lines
plt.clabel(contours, fmt="%2.1f")
# add a color bar
cb = plt.colorbar(head_array, shrink=0.5, )
cb.ax.set_title('Heads')

# Plot grid 
# you can plot BC cells using the plot_bc() 
mm.plot_bc('RIV', color='blue')
# and plot the model grid if desired
mm.plot_grid(lw=0.5)

# add specific discharge vectors using plot_vector()
quiver = mm.plot_vector(qx, qy, normalize=False, color='blue')


# In[ ]:




____________________________________________________________________________________________________________________________________________________03
#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This exercise addresses how to deal with data variables for MODFLOW 6 objects in FloPy. 
# FloPy handles MODFLOW 6 model data in a diffferent manner from other MODFLOW model variants. 
# 
# FloPy stores MODFLOW 6 model data in data objects. These data objects are accesible via simulation or model packages. 
# Data can be added to a package during construction or at a later stage through package attributes.
# 
# There are three (at the time of writting) types of model data objects:
#  - MFDataScalar
#  - MFDataArray
#  - MFDataList
# 
# The current exercise will focus on Scalar Data (MFDataScalar objects).
# 
# ## Scalar Data
# Scalar data are data that consist of a single integer or string, or a boolean flag (True/False). 
# Most model settings or package options are assigned with scalar data. For example, in exercise 01 scalar data were assigned to:
#  - nper, ncol, nrow, nlay (single integer)
#  - time and length units, complexity level in the IMS package (single string)
#  - in the NPF package save_flows and save_specific_discharge were assigned a boolean flag (True) to activate recording of flows and specific discharge
# 
# We will go through a few examples of how to set, view and change scalar data.

# In[1]:


# Import necessary libraries
# for the purposes of this course we are using frozen versions of flopy to avoid depenecy failures.  
import os 
import sys
sys.path.append('../dependencies/')
import flopy
import matplotlib.pyplot as plt


# # Build a Model
# The following cell constructs the same model developed in exercise 1. See if you can identify examples of each of the scalar data types.
# 
# We could also have simply loaded the existing model, as demonstrated in the previous exercise. However, we chose to include the entire code here to make it easier to follow.

# In[2]:


# simulation
sim_name = 'symple_ex03'
exe_name = os.path.join('..','bin', 'mf6.exe')
workspace = os.path.join('..','models','symple_ex03')

sim = flopy.mf6.MFSimulation(sim_name=sim_name,
                            exe_name=exe_name,
                            version="mf6", 
                            sim_ws=workspace)
# tdis
time_units = 'days'
perioddata = [(1.0, 1, 1.0)]
nper = len(perioddata)
tdis = flopy.mf6.ModflowTdis(sim, pname="tdis",
                                  nper=nper, 
                                  perioddata=perioddata, 
                                  time_units=time_units)
# model
model_name = 'symp03'
gwf = flopy.mf6.ModflowGwf(sim,
                            modelname=model_name,
                            save_flows=True, print_flows=True)
# ims pacakge
ims = flopy.mf6.ModflowIms(sim,
                            pname="ims",
                            complexity="SIMPLE",
                            linear_acceleration="BICGSTAB",)
sim.register_ims_package(ims, [gwf.name])

# dis package
length_units = "METERS"
nlay = 3
Lx = 1000
Ly = 1500
delr = 100 #row length
delc = 100 #column length
ncol = int(Lx/delc)
nrow = int(Ly/delr)
top = 50
botm = [40, 35, 0]

dis = flopy.mf6.ModflowGwfdis(
                            gwf,
                            nlay=nlay,
                            nrow=nrow,
                            ncol=ncol,
                            delr=delr,
                            delc=delc,
                            top=top,
                            botm=botm)

# IC package
ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=top)

# NPF package
k = [5, 0.1, 10]
icelltype = [1, 0, 0]

npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=icelltype, k=k,
                              save_flows=True, 
                              save_specific_discharge=True)

# RCH package
recharge = 50/1000/365
rch = flopy.mf6.ModflowGwfrcha(gwf, pname='rch', recharge=recharge)

# RIV package
riv_row = 7
stage = top - 5
rbot = botm[0]
cond = 0.1 * delr*delc/1

riv_spd = []
for col in range(ncol):
    riv_spd.append(((0, riv_row, col), stage, cond, rbot))

riv = flopy.mf6.ModflowGwfriv(gwf, stress_period_data=riv_spd, boundnames=True)

# OC package
# the name of the binary head file
headfile = f"{gwf.name}.hds"
head_filerecord = [headfile]
# the name of the binary budget file
budgetfile = f"{gwf.name}.cbb"
budget_filerecord = [budgetfile]

# which outputs are crecored to the binary files
saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
# which outputs are printed in the list file
printrecord = [("HEAD", "LAST")]
oc = flopy.mf6.ModflowGwfoc(gwf,
                            saverecord=saverecord,
                            head_filerecord=head_filerecord,
                            budget_filerecord=budget_filerecord,
                            printrecord=printrecord)


# ## Accessing Scalar Data
# 
# When we constructed the NPF pacakge, we set the option to "save_specific_discharge" by assigning a True value. In the next steps we will view the option and then change it to False.

# In[3]:


# to view the option in the package simply acces it using the npf package's attribute with the same name.
npf.save_specific_discharge.get_data()


# ## Editting Scalar Data

# In[4]:


# To change the scalar data value simply
npf.save_specific_discharge = False

# and then check it again
npf.save_specific_discharge.get_data()


# In[5]:


# the same applies for single string or integer scalar data
ims.complexity.get_data()


# In[6]:


# alter the IMS solver settings
ims.complexity = 'moderate'

ims.complexity.get_data()


# ## Write the model files 
# Write the model files. You can compare them to those in the exercise 01 folder to see how they have changed.

# In[7]:


sim.write_simulation()

____________________________________________________________________________________________________________________________________________________04
#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This exercise addresses how to deal with data variables for MODFLOW 6 objects in FloPy. 
# FloPy handles MODFLOW 6 model data in a diffferent manner from other MODFLOW model variants. 
# 
# FloPy stores MODFLOW 6 model data in data objects. These data objects are accesible via simulation or model packages. 
# Data can be added to a package during construction or at a later stage through package attributes.
# 
# There are three (at the time of writting) types of model data objects:
#  - MFDataScalar
#  - MFDataArray
#  - MFDataList
# 
# The current exercise will focus on Array Data (MFDataArray objects).
# 
# ## Array Data
# 
# Array data contains data in arrays with a dimension of 1 or larger. In FloPy these data are stored in "MFArray" or "MFTransientArray" objects. 
#  - MFArray objects house time-invariant arrays with one, two or three dimensions. 
#     - One and two dimensional arrays do not include a layer dimension. These are used for data which applies to a single alyer or is the same for all layers. Examples include an array of values for the "top" of the model (only applies to layer 1) or for column/row dimensions in a DIS grid which are the same for all layers.
#     - Three dimensional arrays additionaly contian a layer dimension. These usualy pertain to arrays of data applied to the entire model domain. 
#  - MFTransientArrays, as the name implies, house arrays of data which can change over time. These usualy pertain to data applied to the entire model domain, such as time-varying rechage arrays in the RCHA package. 
# 
# We will go through a few examples of how to set, view and change array data.

# In[1]:


# Import necessary libraries
# for the purposes of this course we are using frozen versions of flopy to avoid depenecy failures.  
import os 
import sys
sys.path.append('../dependencies/')
import flopy
import numpy as np
import matplotlib.pyplot as plt


# # Build a Model
# The following cell constructs the same model developed in exercise 1 with some modification. An additional stress period is added to the TDIS package so that MFTransientArrays can be demonstrated.

# In[2]:


# simulation
sim_name = 'symple_ex04'
exe_name = os.path.join('..','bin', 'mf6.exe')
workspace = os.path.join('..','models','symple_ex04')

sim = flopy.mf6.MFSimulation(sim_name=sim_name,
                            exe_name=exe_name,
                            version="mf6", 
                            sim_ws=workspace)
# tdis
time_units = 'days'
perioddata = [(1.0, 1, 1.0), (1.0, 1, 1.0)] # an additional stress period has been added
nper = len(perioddata)
tdis = flopy.mf6.ModflowTdis(sim, pname="tdis",
                                  nper=nper, 
                                  perioddata=perioddata, 
                                  time_units=time_units)
# model
model_name = 'symp03'
gwf = flopy.mf6.ModflowGwf(sim,
                            modelname=model_name,
                            save_flows=True, print_flows=True)
# ims pacakge
ims = flopy.mf6.ModflowIms(sim,
                            pname="ims",
                            complexity="SIMPLE",
                            linear_acceleration="BICGSTAB",)
sim.register_ims_package(ims, [gwf.name])

# dis package
length_units = "METERS"
nlay = 3
Lx = 1000
Ly = 1500
delr = 100 #row length
delc = 100 #column length
ncol = int(Lx/delc)
nrow = int(Ly/delr)
top = 50
botm = [40, 35, 0]

dis = flopy.mf6.ModflowGwfdis(
                            gwf,
                            nlay=nlay,
                            nrow=nrow,
                            ncol=ncol,
                            delr=delr,
                            delc=delc,
                            top=top,
                            botm=botm)

# IC package
strt = np.full((nlay, nrow, ncol), top)
ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=strt)

# NPF package
k = [5, 0.1, 10]
icelltype = [1, 0, 0]

npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=icelltype, k=k,
                              save_flows=True, 
                              save_specific_discharge=True)

# RCH package
recharge = 50/1000/365
rcha = flopy.mf6.ModflowGwfrcha(gwf, pname='rch', recharge=recharge)

# RIV package
riv_row = 7
stage = top - 5
rbot = botm[0]
cond = 0.1 * delr*delc/1

riv_spd = []
for col in range(ncol):
    riv_spd.append(((0, riv_row, col), stage, cond, rbot))

riv = flopy.mf6.ModflowGwfriv(gwf, stress_period_data=riv_spd, boundnames=True)

# OC package
# the name of the binary head file
headfile = f"{gwf.name}.hds"
head_filerecord = [headfile]
# the name of the binary budget file
budgetfile = f"{gwf.name}.cbb"
budget_filerecord = [budgetfile]

# which outputs are crecored to the binary files
saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
# which outputs are printed in the list file
printrecord = [("HEAD", "LAST")]
oc = flopy.mf6.ModflowGwfoc(gwf,
                            saverecord=saverecord,
                            head_filerecord=head_filerecord,
                            budget_filerecord=budget_filerecord,
                            printrecord=printrecord)


# ## **Specifying Array Data**
# 
# Array data were used during the construction of the model in the first exercise (and in the cell above). 
# 
# Grid array data can be specified in several ways:
#  - as a constant value (see for example the assignment of "top" in the DIS package)
#  - as an n-dimensional list (see for example the assignment of "k" for each layer in the NPF package)
#  - as a numpy array (see for example the assignment of "strt" for the IC package)
# 
#  The manner in which an array is assigned affects how it is written to the MODFLOW6 files and how it is stored by FloPy. The former has implications down-the-line if it relates to a parameter being adjusted during parameter estimation. The latter has implications on how the data can be accesed during model construction, should it be required.
# 
#  As an example we will assign array data to the k33 (vertical hydraulic conductivity) parameter in the NPF package. First, let's set the k33overk option to True, so that k33 represents the ratio of k33/k (i.e. the ratio of vertical to horizontal k). To do so, recall the lessons of the previous exercise on scaler data.

# In[3]:


# set k33overk to True in the NPF package
npf.k33overk = True


# ## **Array data as a constant value**
# Constant value for entire domain.

# In[4]:


# let us start by assigning a single cnstant value of k33 to the entire model. 
npf.k33 = 1


# In[5]:


# we can inspect how FloPy is storing the data 
npf.k33


# In[6]:


# although a single value is stored, we can still access values as an array if required
k33_arr = npf.k33.get_data()

# this array has dimensions (nlay, nrow, ncol).
# However, keep in mind the the model input files which FloPy will write (and MODFLOW will use) will have a single value. We will see this in a abit.
k33_arr.shape


# ## **Array data as a list**
# Unique constant values per layer.

# In[7]:


# set k22overk to True in the NPF package
npf.k22overk = True


# In[8]:


# for example, if we wish to assign a diferent constant value of k22 (horizontal anisotropy) ratio to each layer
npf.k22 = [1, 0.1, 2]

# now we can see FloPy stores different values for each layer. 
# Seperate values for each layer will also be printed to the NPF input file.
npf.k22


# In[9]:


# to update values for a specific layers 
npf.k22.set_data(3, layer=0)

npf.k22


# ## **Array data as an ndarray**
# Values on a cell by cell basis.

# In[10]:


# Array data can also be specifed on a cell-by-cell basis using an adeqautely shaped array
# For example, for K for all model layers an array of shape (nlay, nrow, ncol). This is simple to generate using numpy
# create an array of ones with the desired shape
k = np.ones((nlay, nrow, ncol))

# now you can specify values at the layer, row or column level using slicing:
# layer 1
k[0] = 5
#layer 2
k[1] = 0.1
# layer 3
k[2] = 10
#layer 3, top half
k[2, :7] = 3
#layer 3, bottom right
k[2, 7:, 5:] = 7

# update the NPF pacakge
npf.k = k

# plot the package 
npf.k.plot(colorbar=True)


# ## **Mixed array types**
# Working with layered arrays provides some flexibility. Consnta values can be specified for layers where hetergoeneity is not necesary, with arrays used fro layers in which they are. This reduces file sizes as memory requirements.

# In[11]:


# assign a constatn value for k in layer 1 and 3, but an array in layer 2
k1 = 5
k3 = 10
k2 = np.full((nrow, ncol), 0.1)

# pass the values and array to the package in a list
npf.k = [k1, k2, k3]

npf.k


# The examples above have demonstrate how to update an exising NPF package. However, the same principlies apply when constructing a package. Below we reconstruct the npf package to illustrate.

# In[12]:


# rebuild the npf pacakge
# k
k1 = 5
k2 = np.full((nrow, ncol), 0.1)
k3 = 10

k = [k1, k2, k3]

#k22
k22 = [1, 0.1, 2]
#k33
k33 = 0.1

npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=icelltype, 
                                    k=k,
                                    k22=k22, k33=k33,
                                    k22overk=True, k33overk=True,
                                    save_flows=True, 
                                    save_specific_discharge=True)


# ## **Write the model files**
# Write the model files. You can compare them to those in the exercise 01 folder to see how they have changed.

# In[13]:


sim.write_simulation()
sim.run_simulation()


# # Transient Array Data
# Transient data arrays for several stress periods are specified as a dictionary of arrays, in which the dictionary key is an integer matching the (zero-based) the stress period. The dictionary value is the array data.
# 
# As for other array types, single values, lists or ndarrays can be passed. 
# 
# The following example illustrates transient array data for the RCHA package. Recall that the model has two stress periods.

# In[19]:


# define recharge in stress period 1 wihth a single value 
rch_sp1 = 0.000137

# define recharge in stress period 2 with an array with recharge on only half the model domain
rch_sp2 = np.zeros((nrow, ncol)) 
rch_sp2[:7,:] = 0.000274

# construct the dictionary of stress periods and recharge arrays
rch_spd = {0: rch_sp1, 1: rch_sp2}


# In[20]:


# RCHA package
rcha = flopy.mf6.ModflowGwfrcha(gwf, pname='rch', recharge=rch_spd)


# In[21]:


sim.write_simulation()
sim.run_simulation()


# In[23]:


# plot outputs from the upper layer. The code below is the same as used in Exercise 01.

# load outputs
# the head file output can be loaded from the model object:
hds = gwf.output.head()
heads = hds.get_data(idx=-1) ### changed to read the last output

# get the specific discharge from the cell budget file
cbb = gwf.output.budget()
spdis = cbb.get_data(text="SPDIS")[-1]
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)


# plot
fig = plt.figure(figsize=(5, 5), constrained_layout=True)
# first instantiate a PlotMapView
mm = flopy.plot.PlotMapView(model=gwf)

# Plot heads
# plot the array of heads 
head_array = mm.plot_array(heads)
# add contour lines with contour_array()
contours = mm.contour_array(heads, colors='black')
# add labels to contour lines
plt.clabel(contours, fmt="%2.1f")
# add a color bar
cb = plt.colorbar(head_array, shrink=0.5, )
cb.ax.set_title('Heads')

# Plot grid 
# you can plot BC cells using the plot_bc() 
mm.plot_bc('RIV', color='blue')
# and plot the model grid if desired
mm.plot_grid(lw=0.5)

# add specific discharge vectors using plot_vector()
quiver = mm.plot_vector(qx, qy, normalize=False, color='blue')


# In[ ]:




____________________________________________________________________________________________________________________________________________________05
#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This exercise addresses how to deal with data variables for MODFLOW 6 objects in FloPy. 
# FloPy handles MODFLOW 6 model data in a diffferent manner from other MODFLOW model variants. 
# 
# FloPy stores MODFLOW 6 model data in data objects. These data objects are accesible via simulation or model packages. 
# Data can be added to a package during construction or at a later stage through package attributes.
# 
# There are three (at the time of writting) types of model data objects:
#  - MFDataScalar
#  - MFDataArray
#  - MFDataList
# 
# The current exercise will focus on List Data (MFDataList objects).
# 
# ## List Data
# 
# Some MODFLOW 6 data can conveniently be stored in tabular format, such as in a numpy recarray or pandas dataframe. These data are stored by FloPy in MFList or MFTransientList objects. 
# 
# MFList data can contain a single or multiple rows. Each column of the list contains the same data type. Single row MFLists usualy pertain to pacakge options (such as the BUY pacakge's pacakgedata). Multiple row MFLists are used for things like the 'conectiondata' for the MAW or Lake packages.
# 
# Perhaps most used are MFTransientList data, in which FloPy stores lists of stress period data. These are often used to assign stress package values at specifc cells and stress periods (i.e. pumping rates in the WEL package). 
# 
# In the following exercise we will adapt the model from excerise 01 by adding two new stress periods. We will then place a well in the lower aquifer using the WEL package. Diferent pumping rates will be assigned to each stress period to demosntrate the use of MFTranslientList data.

# In[1]:


# Import necessary libraries
# for the purposes of this course we are using frozen versions of flopy to avoid depenecy failures.  
import os 
import sys
sys.path.append('../dependencies/')
import flopy
import numpy as np
import matplotlib.pyplot as plt


# # Build a Model
# The following cell constructs the same model developed in exercise 1 with some modification. Two additional stress period are added to the TDIS package so that MFTransientList's can be demonstrated.

# In[2]:


# simulation
sim_name = 'symple_ex05'
exe_name = os.path.join('..','bin', 'mf6.exe')
workspace = os.path.join('..','models','symple_ex05')

sim = flopy.mf6.MFSimulation(sim_name=sim_name,
                            exe_name=exe_name,
                            version="mf6", 
                            sim_ws=workspace)
# tdis
time_units = 'days'
perioddata = [(1.0, 1, 1.0), 
                (1.0, 1, 1.0), # an additional stress period has been added
                (1.0, 1, 1.0)] # an additional stress period has been added
nper = len(perioddata)
tdis = flopy.mf6.ModflowTdis(sim, pname="tdis",
                                  nper=nper, 
                                  perioddata=perioddata, 
                                  time_units=time_units)
# model
model_name = 'symp03'
gwf = flopy.mf6.ModflowGwf(sim,
                            modelname=model_name,
                            save_flows=True, print_flows=True)
# ims pacakge
ims = flopy.mf6.ModflowIms(sim,
                            pname="ims",
                            complexity="SIMPLE",
                            linear_acceleration="BICGSTAB",)
sim.register_ims_package(ims, [gwf.name])

# dis package
length_units = "METERS"
nlay = 3
Lx = 1000
Ly = 1500
delr = 100 #row length
delc = 100 #column length
ncol = int(Lx/delc)
nrow = int(Ly/delr)
top = 50
botm = [40, 35, 0]

dis = flopy.mf6.ModflowGwfdis(
                            gwf,
                            nlay=nlay,
                            nrow=nrow,
                            ncol=ncol,
                            delr=delr,
                            delc=delc,
                            top=top,
                            botm=botm)

# IC package
strt = np.full((nlay, nrow, ncol), top)
ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=strt)

# NPF package
k = [5, 0.1, 10]
icelltype = [1, 0, 0]

npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=icelltype, k=k,
                              save_flows=True, 
                              save_specific_discharge=True)

# RCH package
recharge = 50/1000/365
rcha = flopy.mf6.ModflowGwfrcha(gwf, pname='rch', recharge=recharge)

# RIV package
riv_row = 7
stage = top - 5
rbot = botm[0]
cond = 0.1 * delr*delc/1

riv_spd = []
for col in range(ncol):
    riv_spd.append(((0, riv_row, col), stage, cond, rbot))

riv = flopy.mf6.ModflowGwfriv(gwf, stress_period_data=riv_spd, boundnames=True)

# OC package
# the name of the binary head file
headfile = f"{gwf.name}.hds"
head_filerecord = [headfile]
# the name of the binary budget file
budgetfile = f"{gwf.name}.cbb"
budget_filerecord = [budgetfile]

# which outputs are crecored to the binary files
saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
# which outputs are printed in the list file
printrecord = [("HEAD", "LAST")]
oc = flopy.mf6.ModflowGwfoc(gwf,
                            saverecord=saverecord,
                            head_filerecord=head_filerecord,
                            budget_filerecord=budget_filerecord,
                            printrecord=printrecord)


# ## **Specifying List Data**
# 
# For the purposes of this exercise we will only demonstrate the use of multirow list data to assign stress perdod data. Ths is the most commmon use case for list data. Single row cases tend to be package specific. These will be highlighted throught the course when they come up.
# 
# ### **Time-varying boundary conditions**
# When building stress packages (i.e. WEL, CHD, GHB, etc.), FloPy accepts stress period data as a dictionary of numpy recarrays, or a dictionary of lists of tuples, in which the ditionary key is the zero-based stress period. A dictionary of lists of tuples tends to be easier to generate. Internaly, FloPy stores the stress period data as a dictionary of numpy recarrays.
# 
# A recarray (or list of tuples) will have a specific structure depending on the package it is assigned to. The structure usualy relates to the structure of stress period data in the respective MODFLOW 6 package input file (see the MODFLOW6 manual for details). For example, for the WEL package default structure for each row (or tuple in the list) pertaining to a single stress period would be:
# 
#     (cellid, pumping_rate)
# 
# In which 'cellid' is a tuple contaning the cell identifier. In the case of a structured DIS grid:
#     
#     ((layer, row, col), pumping_rate)
# 
# Fortunately, FloPy has a built-in method to generate an empty recarray with the necessary structure for any pacakge, as is shown below:
# 

# In[3]:


# generate empty dictionary of recarays for the WEL pacakge
wel_spd = flopy.mf6.ModflowGwfwel.stress_period_data.empty(gwf)

# you can inspect the recarray dtypes. This is the format the recarray or list of tuples must have.
wel_spd[0].dtype


# In[4]:


# construct a dctionary of stress period data
# we will place a well in the lower layer (layer 2), at row 5 and column 3. 
# It will have pumping rate of -100 m3/d (negative is out).
# The well will pump during the first and last stress period, and be inactie during the second.
# In the last stress period, we will place a second well in the lower layer (layer 2), at row 10 and column 6 with -50 m3/d. 

# create an empty dictionary.
wel_spd = {}

# create the list of tuples for the first stress period
spd1 = [((2, 5, 3), -100)]

# adding an empty list deactivates pumping in the second stress period.
# If nothing were assinged, pumping would continue as per the previous stress period.
spd2= []

# add a second entry (tuple) to the list to add an additional well
spd3 = [((2, 5, 3), -100), # the first well
        ((2, 10, 6), -50)]  # the second well

# Assign easch list to the relevant key in the dictionary. 
# This could have been done directly, but in this way eah step is shown explicitly.
wel_spd[0] = spd1
wel_spd[1] = spd2
wel_spd[2] = spd3

# inspect the stress period data dictionary
wel_spd


# In[5]:


# construct the WEL package
wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_spd,
                                    print_input=True, 
                                    print_flows=True,
                                    save_flows=True)


# ## **Write the model files**
# Write the model files. You can compare them to those in the exercise 01 folder to see how they have changed.

# In[6]:


sim.write_simulation()
sim.run_simulation()


# # Plot the results at the end of each stress period

# In[8]:


# plot outputs from the upper layer. The code below is the same as used in Exercise 01.
# load outputs
# the head file output can be loaded from the model object:
hds = gwf.output.head()
# get the specific discharge from the cell budget file
cbb = gwf.output.budget()


# plot
fig = plt.figure(figsize=(13, 5))
x = 1
for totim in hds.get_times():
    ax = fig.add_subplot(1, 3, x, aspect='equal')
    ax.set_title(f'Layer 1; at time={totim}', fontsize=10)
    x += 1

    heads = hds.get_data(totim=totim)
    spdis = cbb.get_data(text="SPDIS",totim=totim)[-1]
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

    # first instantiate a PlotMapView
    mm = flopy.plot.PlotMapView(model=gwf, ax=ax)

    # Plot heads
    # plot the array of heads 
    head_array = mm.plot_array(heads)
    # add contour lines with contour_array()
    contours = mm.contour_array(heads, colors='black')
    # add labels to contour lines
    plt.clabel(contours, fmt="%2.1f")

    # Plot grid 
    # you can plot BC cells using the plot_bc() 
    mm.plot_bc('RIV', color='blue')
    # and plot the model grid if desired
    mm.plot_grid(lw=0.5)

    # add specific discharge vectors using plot_vector()
    quiver = mm.plot_vector(qx, qy, normalize=False, color='blue')


# In[ ]:




____________________________________________________________________________________________________________________________________________________06
#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This exercise addresses how to deal with the MODFLOW 6 Observation (OBS) Utility. The OBS utility provides options for extracting numeric values of interest generated during a model run (i.e. "observations").
# 
# Observations are output at the end of each time-step and represent the value used by MODFLOW 6 during the time-step. Types of available observations are listed in the MODFLOW6 manual. Commonly used observations are heads, concentrations (for mass transport models) and flows through boundary conditions.
# 
# The OBS utility can record outputs to either text or binary files. Text files are written in CSV format, making them easy to access using common spredsheet software or libraries (i.e. Pandas or Numpy).
# 
# In this exercise we will:
#  - configure observations of flow through the RIV boundary condition
#  - define observations of heads at specified locations in the model
#  - run the model and access simulated observation data using the .output method

# In[18]:


# Import necessary libraries
# for the purposes of this course we are using frozen versions of flopy to avoid depenecy failures.  
import os 
import sys
sys.path.append('../dependencies/')
import flopy
import numpy as np
import matplotlib.pyplot as plt


# # Build a Model
# The following cell constructs the same model developed in exercise 1 with some modification. A few changes have been introduced:
# 
#  - One additional stress period is added to the TDIS package;
#  - The new stress period is simulated under transient conditions for 365 days with 12 time-steps.

# In[19]:


# simulation
sim_name = 'symple_ex06'
exe_name = os.path.join('..','bin', 'mf6.exe')
workspace = os.path.join('..','models','symple_ex06')

sim = flopy.mf6.MFSimulation(sim_name=sim_name,
                            exe_name=exe_name,
                            version="mf6", 
                            sim_ws=workspace)


# Change the TDIS perioddata.

# In[20]:


# THis time we will add an extra stress period with a perlen=365 days and nstp=12
perioddata = [(1.0, 1, 1.0), 
                (365, 12, 1.0)] # an additional stress period has been added

# the number of periods (nper) should match the number of tuples in the perioddata list
nper = len(perioddata)

# tdis
time_units = 'days'
tdis = flopy.mf6.ModflowTdis(sim, pname="tdis",
                                  nper=nper, 
                                  perioddata=perioddata, 
                                  time_units=time_units)


# With the exception of the RIV, WEL and STO packages, the rest remains the same for now. 
# Because we are adding a transient stress period we also need to include the storage (STO) package.
# We also add the WEL package, with two wells pumping during the transient stress peroid. This is so that we see some change in our observations. Otherwise it would be boring.
# 

# In[21]:


# model
model_name = 'symp06'
gwf = flopy.mf6.ModflowGwf(sim,
                            modelname=model_name,
                            save_flows=True, print_flows=True)
# ims pacakge
ims = flopy.mf6.ModflowIms(sim,
                            pname="ims",
                            complexity="SIMPLE",
                            linear_acceleration="BICGSTAB",)
sim.register_ims_package(ims, [gwf.name])

# dis package
length_units = "METERS"
nlay = 3
Lx = 1000
Ly = 1500
delr = 100 #row length
delc = 100 #column length
ncol = int(Lx/delc)
nrow = int(Ly/delr)
top = 50
botm = [40, 35, 0]

dis = flopy.mf6.ModflowGwfdis(
                            gwf,
                            nlay=nlay,
                            nrow=nrow,
                            ncol=ncol,
                            delr=delr,
                            delc=delc,
                            top=top,
                            botm=botm)

# IC package
strt = np.full((nlay, nrow, ncol), top)
ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=strt)

# NPF package
k = [5, 0.1, 10]
icelltype = [1, 0, 0]

npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=icelltype, k=k,
                              save_flows=True, 
                              save_specific_discharge=True)

# RCH package
recharge = 50/1000/365
rcha = flopy.mf6.ModflowGwfrcha(gwf, pname='rch', recharge=recharge)

# construct the WEL package
wel_spd = {
            0:[],
            1:[((2, 5, 3), -100), ((2, 10, 6), -50)]
            }
wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_spd,
                                    print_input=True, 
                                    print_flows=True,
                                    save_flows=True)


# OC package
# the name of the binary head file
headfile = f"{gwf.name}.hds"
head_filerecord = [headfile]
# the name of the binary budget file
budgetfile = f"{gwf.name}.cbb"
budget_filerecord = [budgetfile]

# which outputs are crecored to the binary files
saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
# which outputs are printed in the list file
printrecord = [("HEAD", "LAST")]
oc = flopy.mf6.ModflowGwfoc(gwf,
                            saverecord=saverecord,
                            head_filerecord=head_filerecord,
                            budget_filerecord=budget_filerecord,
                            printrecord=printrecord)


# ## **Adding the STO package**
# 
# We will take the oportunity to introduce how to specify transient stress periods with the STO package. For this exercise we will simulate the frst stress period as steady state, followed by a transient stress period. 
# 
# These are defined using the "steady_state" and "transient" arguments when constructing the STO packge. These arguments take a list or dictionary of booleans (True/False). If a dictionary is passed, the dictionary keys refer to the stress period number (zero-based). Either steadystate or transient conditions will apply until a subsequent stress period is specified as bineg of the other type.

# In[22]:


ss = nlay * [1e-5]
sy = nlay * [0.2]

sto = flopy.mf6.ModflowGwfsto(gwf,
                                steady_state={0:True}, 
                                transient={1:True},
                                iconvert = [1, 0, 0],
                                ss=ss, 
                                sy=sy,
                                save_flows=True)


# ## **Adding Observations to a Stress Package**
# 
# Observations can be set for any package using the package.obs object (for example: riv.obs). 
# 
# Observations can be specifed on a cell-by-cell basis or for groups of cells using "boundnames" when specifying list data. For things such as boundary conditions, the latter is likley to be a more common use case (i.e. you are morel likley to want to record the flow through all RIV cells rather than for each individual cell). 
# 
# Each observation also represents a unique column of data recorded in the output CSV file. So if you are monitoring every cell, such a file can get very large very quickly. Writting the file also slows down model run-times. 
# 
# We will go through both options in this exercise.

# In[23]:


# Start by specifying the same inputs as in previous exercises
# RIV package
riv_row = 7
stage = top - 5
rbot = botm[0]
cond = 0.1 * delr*delc/1

# Now, when specifyin the list data a new value "river_bc" is added to the tuple. This is a string defining the "boundname". 
# Think of it as a tag for all the cells which make up this river boundary condition.
riv_spd = []
for col in range(ncol):
    riv_spd.append(((0, riv_row, col), stage, cond, rbot, 'river_bc'))


# note that to use boundamens, the "boundnames" argument in the riv package is set to True.
riv = flopy.mf6.ModflowGwfriv(gwf,
                                stress_period_data=riv_spd,
                                boundnames=True)


# The next step is to build the observation data dictionary. The dictionary key is the filename of the output file. We shall record our observations to the file "riv_obs.csv". The dictionary value is a list of tuples with the contents of the OBS package's continuous block (see MF6 manual). Each tuple in the list is comprised of (the output file column header, the observation type, the boundname or cellid).
# 
# We will record RIV obsservations assocaited to the "river_bc" boundname to "riv_obs.csv". We will add an additional observation to the same output file for a RIV observations at a single cell. Then we will add a second output file ("riv_obs2.csv") with observations from another cell.

# In[24]:


# build the observation data dictionary
riv_obs = {
            "riv_obs.csv": [("river", "RIV", "river_bc"), ("riv_7_9", "RIV", (0, 7, 9))],
            "riv_obs2.csv": [("riv_7_0", "RIV", (0, 7, 0))]
            }


# In[25]:


# we can then initialize the observations
riv.obs.initialize(digits=10, 
                    print_input=False,
                    continuous=riv_obs)


# ## **Adding Observations of State Variables**
# 
# Model outputs of heads, concentrations and flows between cells are not associated to any specific package. These are assigned in the same manner, but using *flopy.mf6.ModflowUtlobs()* for the OBS Utility.
# 
# In the example below we will specify observations at two cells, one in each layer. We will record both head and drawdown.

# In[26]:


# as before, first we construct the observation data dictionary and lists
hd_obs_list = [
                ('h_0_7_4', 'HEAD', (0, 7, 4)), # head in the upper aquifer
                ('h_2_7_4', 'HEAD', (2, 7, 4)), # head in the lower aquifer
                ]

dd_obs_list = [
                ('dd_0_7_4', 'DRAWDOWN', (0, 7, 4)), # drawdown in the upper aquifer
                ('dd_2_7_4', 'DRAWDOWN', (2, 7, 4)), # drawdown in the lower aquifer
                ]


obs_data = {
            'head_obs.csv':hd_obs_list,
            'drawdown_obs.csv':dd_obs_list,
            }


# In[27]:


# then we initialize the OBS utility pacakge
# initialize obs package
obs_package = flopy.mf6.ModflowUtlobs(gwf, 
                                      digits=10, 
                                      print_input=False,
                                      continuous=obs_data)


# ## **Write the Model Files and Run**
# Write the model files. You can compare them to those in the exercise 01 folder to see how they have changed.

# In[28]:


sim.write_simulation()
sim.run_simulation()


# # **Access Output Observations**
# 
# Model outputs have been written to CSV's in the model workspace folder. These can be accessed as you would any CSV file. 
# 
# Alternatively you can use FloPy's .output method as shown below.

# In[29]:


# check how many obs pacakges are in the model
print(f'Number of obs packages: {len(gwf.obs)}')


# In[30]:


# access a list of observation ouput file names of the firs OBS package
gwf.obs[0].output.obs_names


# In[31]:


# access a list of observation ouput file names of the second OBS package
gwf.obs[1].output.obs_names


# In[32]:


# load the output obsevration csv by referencing the file name
riv_obs_csv = gwf.obs[0].output.obs(f='riv_obs.csv')

# access a recarray of the observation data with
riv_obs_csv.data


# In[33]:


# you can then manipulate or plot that data as desired. Personaly I find Pandas dataframes easier to handle
import pandas  as pd
obs_df = pd.DataFrame(riv_obs_csv.data)

obs_df.head()


# In[34]:


# a quick and dirty plot
obs_df.plot(x='totim')


# In[ ]:




____________________________________________________________________________________________________________________________________________________07
#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This exercise will introduce how to assign time series to stress pacakges. 
# 
# **From the MODFLOW 6 Manual:**
# 
# Any package that reads data as a list of cells and associated time-dependent input values can obtain
# those values from time series. For example, flow rates for a well or stage for a river boundary can
# be extracted from time series. During a simulation, values used for time-varying stresses (or auxiliary
# values) are based on the values provided in the time series and are updated each time step (or each
# subtime step, as appropriate).

# In[1]:


# Import necessary libraries
# for the purposes of this course we are using frozen versions of flopy to avoid depenecy failures.  
import os 
import sys
sys.path.append('../dependencies/')
import flopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Load a Model
# 
# In this exercise we will modify the model constructed in exercise 06. This will give us some practice in loading and modifying existing models. 
# 
# In this case, we will load the existing model, remove the existing WEL package and then construct a new WEL package using time-series to specify pumping rates during the second (transient) stress period.

# In[2]:


# As in the previous exercise, if you do not have MODFLOW 6 in yout system path, you must provide the location of the executable file
# The MODFLOW 6 executable is named "mf6.exe":
exe_name = os.path.join('..','bin', 'mf6.exe')

# define the location of the model working directory;  this is where the existing model is currently stored. We are going to load the model constructed during exercise 01.
org_workspace = os.path.join('..','models','symple_ex06')

# define a new model working directory
workspace = os.path.join('..','models','symple_ex07')

# load the Simulation
sim = flopy.mf6.MFSimulation.load(sim_name='symple_ex07', 
                                    exe_name=exe_name, 
                                    sim_ws=org_workspace)
# change the model workspace to a new folder
sim.set_sim_path(workspace)

# access the Model Object using the model name
gwf = sim.get_model(list(sim.model_names)[0])

# check the package names
print(gwf.get_package_list())


# In[3]:


# remove the existing WEL package
gwf.remove_package("WEL_0")

print(gwf.get_package_list())


# In[8]:


sim.ims


# ## **Construct a New WEL Package**
# 
# Now we will reconstrcut the WEL pacakge using time series data. First we need to have some time series data.
# 
# Time series data are constructed as a list of tuples. Each tuple contains the time and one or more values specified at that time. For example: (time, val1, val2)
# 
# Additionaly, a "time series namerecord" (i.e. the time series name) and a "interpolation_methodrecord" are required for each value in the ts data tuple. The namerecord is used to assign the time series to cells when p=constructing the stress package. The interpolation_methodrecord specifies how the time series values are interpolated between listed times.
# 
# For this exercise we will assign two diferent time series to two wells. Each well (i.e. each time series) will pump at a diferent rate. Both wells will pump for 182.5 days at full capacity, then recude to 50% for the remainder of the stress period. We will name these tie series "well1" and "well2" and use STEPWISE interpolation (see the MF6 manual for more info on available interpolation methods).

# In[4]:


# ts data
val1 = -100
val2 = -5

ts_data = [
            (1, val1, val2), #the transeint stress period starts on day 1. The time series could begin at day 0, it wouldnt matter as it is not assigned during the first sress period
            (183.5, val1*0.5, val2*0.5),
            (366, val1*0.5, val2*0.5) # we must asign a value at the end of the model run. 
            ]

#ts names
ts_names = ['well1', 'well2']

# interpolation methods
ts_methods = ['stepwise', 'stepwise']


# In[5]:


# Now when we construct the WEL pakcage list data, instead of a pumping rate, we assign a time series namerecord 
wel_spd={}
# set no pumping in the first stress period
wel_spd[0] = []
# set pumping in the second (transient) stress period. Let's also add some observations so that we can record the pumping rate.
wel_spd[1] = [ ((2, 5, 3), 'well1', 'w1'), 
                ((2, 10, 6), 'well2', 'w2')]

# construct the WEL pacakge
wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_spd,
                                    print_input=True, 
                                    print_flows=True,
                                    save_flows=True,
                                    boundnames=True)

# Now initialize the pacakge's time series 
wel.ts.initialize(filename='wel.ts', # the filename which MODFLOW will use to read the time series
                    timeseries=ts_data,
                    time_series_namerecord=ts_names,
                    interpolation_methodrecord=ts_methods
                    )

# Add the observations
wel_obs = {'wel_obs.csv': [('wel1', 'WEL', 'w1') , ('wel2', 'WEL', 'w2') ]}
wel.obs.initialize(digits=10, 
                    print_input=False,
                    continuous=wel_obs)


# In[6]:


sim.write_simulation()
sim.run_simulation()


# In[7]:


# access a list of observation ouput file names of the well OBS package
gwf.obs[2].output.obs_names


# In[8]:


# plot the the series of pumping rate
welobs_data = gwf.obs[2].output.obs(f='wel_obs.csv').data

welobs_df = pd.DataFrame(welobs_data).replace(3e30, np.nan)

welobs_df.plot(x='totim')


# In[ ]:





____________________________________________________________________________________________________________________________________________________08

