#!/usr/bin/env python
# coding: utf-8

# # Sneak-peak at `PstFrom` - a recipe
# 
# ### Recap
# In the last notebook, we learned that to unleash the model-independent power of PEST and PEST++ we need three vital ingredients.
# * **Template** files -- for interfacing with model **input** files
# * **Instruction** files -- for extracting simulated outputs from model **output** files
# * **PEST Control** file -- well, for controlling the whole thing
# 
# We also learned that there is a bit of "structure" to these files!
# Constructing well formatted template and instruction files, for a few parameters, or simulated outputs, in a couple of files, may not be too much of a biggie. But when we move to high dimensions it can get a little... painful.
# 
# ### Help is at hand
# `PstFrom` is a helpful class and set of methods, within `pyemu` that can assist us with constructing these PEST ingredients for our models (sometimes we call this our **PEST interface**). This tutorial aims to give us a quick look at the main steps (methods) that we call upon when setting up a PEST interface with `PstFrom`.
# 
# ### Prerequisite
# Model input and output files can be expressed as nice ascii array- or list(table)-like files. There is support for adding your own pre- and post-processing methods to execute automatically in your forward model run (`add_py_function()`). 
# 
# ### An interesting concept -- parameters as multipliers 
# Parameter values can be setup as "direct", "multiplier" or "addend". This means the "parameter value" which PEST(++) sees can be (1) the same value the model sees, (2) a multiplier on the value in the existing/original model input file, or (3) a value which is added to the value in the existing/original model input file. This is very nifty and allows for some pretty advanced parameterization schemes by allowing mixtures of different types of parameters. In `PstFrom` the default is designed to setup parameters as multipliers. This lets us preserve the existing model inputs and treat them as the mean of the prior parameter distribution.
# 
# <img src="./mult.png" style="float: center">

# ______________
# ## The `PstFrom` Basic Recipe
# 
# ### Ingredients
# ____
# #### Main
# ```
# 0. python
# 1. pyemu.utils.PstFrom()
# 2. add_parameters() [PstFrom method]
# 3. add_observations() [PstFrom method
# 4. mod_sys_cmds [PstFrom method]
# 5. build_pst() [PstFrom method]
# 6. os_utils.run() [pyemu.os_utils method]
# ```
# #### Supplementary
# ```
# * add_py_function() [PstFrom method]
# * build_prior() [PstFrom method]
# * draw() [PstFrom method]
# * Pst.write() [pyemu.Pst method]
# ```

# ### Method
# _____
# #### 0. Import packages
# Import python packages as usual:
# ```
# import pyemu
# ```

# #### 1. Instantiate a PstFrom object:
# ```
# pf = pyemu.utils.PstFrom(original_d, new_d)
# ```
# In the simplest form, PstFrom just needs to know what directory your model files are in (`original_d`). And where you want your PEST interface to be housed (`new_d`). Note: this step will attempt to replace the content of `new_d` with the content of `original_d`; as a default, it will fail if `new_d` already exists (see `remove_existing` argument for different behavior).
# 
# There are a few other options for a bit of fine tuning, see:
# ```
# pyemu.utils.PstFrom?
# ```

# #### 2. Add Parameters
# ```
# pf.add_parameters(filename[s], par_type)
# ```
# Again, in the simplest form, just need to pass the name of the file (`filename[s]`) that contains the model inputs that we want to parameterize and the type of parameter (`par_type`). `par_type` options are `"constant"`, `"zone"`, `"pilotpoint"` or `"grid"`. Some additional args are probably worth passing though. E.g. `lower_bound` and `upper_bound` can be used to define our anticipated minimum and maximum of the parameter values (Note: this is one of the access points for defining our prior parameter uncertainty).
# 
# Use `index_cols` (and `use_cols`) arguments to work with list-like (tabular) input files.
# 
# More options:
# ```
# pyemu.utils.PstFrom.add_parameters?
# ```

# #### 3. Add Observations
# ```
# pf.add_observations(filename)
# ```
# Bare minimum, just pass a (nicely formatted) model output filename.
# Note: Step 3. can come before Step 2.
# 
# Use `index_cols` (and `use_cols`) arguments to work with list-like (tabular) output files.
# 
# More options:
# ```
# pf.add_observations?
# ```

# #### 4. Add forward model run command
# ```
# pf.mod_sys_cmds.append(command)
# ```
# Just a list attribute of the `PstFrom` object -- we can just add our forward run command here (e.g. `mf6`), `pyemu` will deal with the rest.

# #### 5. Finalize interface and build PEST Control File
# ```
# pst = pf.build_pst()
# ```
# Worth passing the `filename` argument, or it'll be interpreted from `original_d` and could get pretty long!

# #### 6. Run PEST(++)
# What!?! already?
# ```
# pyemu.os_utils.run(run_cmd, cwd)
# ```

# ## A schematic look at this setup

# In[ ]:


from IPython.display import Image, display
display(Image(url='./pest_flow2.gif'))


# # #tenlinesofpython

# In[ ]:


# 0. Import packages
import os, sys, pyemu
sys.path.insert(0,"..")
import herebedragons as hbd

# 1. Instantiate a PstFrom object:
org_d = os.path.join('..', '..', 'models', 'monthly_model_files_1lyr_newstress')
pf = pyemu.utils.PstFrom(org_d, 'template', remove_existing=True)
   # ------- these lines don't count! haha! -------->
hbd.prep_bins("template")  # just a bit of admin
pyemu.os_utils.run("mf6", "template") # so we have up2date outputs
   # <---------------------------------------------

# 2. Add Parameters
pf.add_parameters("freyberg6.npf_k_layer1.txt", 'grid')

# 3. Add Observations
pf.add_observations("sfr.csv", index_cols='time', use_cols='TAILWATER')

# 4. Add forward model run command
pf.mod_sys_cmds.append("mf6")

# 5. Finalize interface and build PEST Control File
pst = pf.build_pst('eg.pst')

# 6. Run PEST(++)
pyemu.os_utils.run("pestpp-glm eg.pst", "template")


# In[ ]:




