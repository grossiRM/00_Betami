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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


# to view the option in the package simply acces it using the npf package's attribute with the same name.


# ## Editting Scalar Data

# In[ ]:


# To change the scalar data value simply


# and then check it again


# In[ ]:


# the same applies for single string or integer scalar data


# In[ ]:


# alter the IMS solver settings


# ## Write the model files 
# Write the model files. You can compare them to those in the exercise 01 folder to see how they have changed.

# In[ ]:





# In[ ]:




