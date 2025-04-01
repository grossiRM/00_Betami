#!/usr/bin/env python
# coding: utf-8

# ## 01: Introduction to FloPy
# Environment mbase.py
# 
# 

# In[6]:


import flopy

model = flopy.modflow.Modflow(modelname='test', version='mfnwt')  # or 'mf2005' for MODFLOW 2005 (the default)


# For MODFLOW 6, we first have to create a simulation, which we then assign the model to:

# In[7]:


sim = flopy.mf6.MFSimulation(sim_name='test')
gwf = flopy.mf6.ModflowGwf(sim, modelname='test')


# As you can see, in addition to the different FloPy subpackages, the syntax is different between the two versions. In general, the syntax for the `flopy.modflow` and `flopy.mf6` subpackages follows the respective MODFLOW versions they support. For example, `ModflowGwf` is simply a [CamelCase](https://en.wikipedia.org/wiki/Camel_case) representation of the MODFLOW 6 Groundwater Flow (GWF) model. Similarly, the Discretization Package (for the GWF model) class is named `ModflowGwfdis`. Arguments to the `ModflowGwfdis` constructor follow the input variables to MODFLOW 6.

# In[5]:


dis = flopy.mf6.ModflowGwfdis(
    gwf,
    nlay=3,
    nrow=21,
    ncol=20,
    delr=500.,
    delc=500.,
    top=400.0,
    botm=[220, 200, 0],
)


# The [MODFLOW 6 Input and Output Guide](https://modflow6.readthedocs.io/en/latest/mf6io.html) ([individual releases](https://github.com/MODFLOW-USGS/modflow6/releases) also contain a PDF version) can therefore be a valuable resource for understanding how to use Flopy.
# 
# Other Flopy subpackages (`flopy.modpath`, `flopy.mt3d`, etc) provide varying levels of support for previous versions of MODFLOW or related software such as MODPATH and MT3D; or general (Flopy-wide) support for discretization, exporting or other processing (`flopy.discretization`, `flopy.export`, `flopy.utils`, etc.).
# 
# This class will focus almost exclusively on FloPy support for MODFLOW 6. Examples and documentation for using FloPy in other contexts is available in the [FloPy online documentation](https://flopy.readthedocs.io/en/stable/index.html). The [Examples gallery](https://flopy.readthedocs.io/en/stable/examples.html) are a resource that can be used to quickly learn the underlying capabilities of FloPy.

# 
