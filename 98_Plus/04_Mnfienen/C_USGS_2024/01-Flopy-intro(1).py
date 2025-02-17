#!/usr/bin/env python
# coding: utf-8

# ![](data/flopylogo_sm.png)
# 
# # 01: Introduction to FloPy
# 
# The second part of this course focuses on [FloPy](https://github.com/modflowpy/flopy), a Python package for creating, running, and post-processing MODFLOW-based groundwater flow and transport models. Why would we want this? MODFLOW—especially older versions—has idiosyncratic input and output that can be difficult to work with directly. FloPy translates MODFLOW input and output into the general Python data structures we explored in the first part of the course, making it easier to script groundwater modeling workflows with the entire scientific Python ecosysem.
# 
#  FloPy was originally developed by [Mark Bakker and others (2016)](https://ngwa.onlinelibrary.wiley.com/doi/10.1111/gwat.12413) for working with MODFLOW 2005 (including MODFLOW-NWT) and earlier versions of MODFLOW. [FloPy has since been expanded](https://doi.org/10.1111/gwat.13327) to support [MODFLOW 6](https://github.com/MODFLOW-USGS/modflow6), the current version of MODFLOW that includes general support for structured and unstructured grids, tight coupling of multiple models within a simulation, and a redesigned input structure that is meant to be more intuitive and human readable. FloPy support for MODFLOW 6 is tightly coupled to the MODFLOW 6 code, in that the relevant FloPy code is auto-generated from text files (_definition files_) that describe MODFLOW 6 models and packages. As a result, MODFLOW 2005-style and MODFLOW 6 functionality within FloPy is accessed through different subpackages—`flopy.modflow` and `flopy.mf6` respectively. 
# 
# For example, to instantiate a MODFLOW-NWT model instance, one would enter:
# 

# In[1]:


import flopy

model = flopy.modflow.Modflow(modelname='test', version='mfnwt')  # or 'mf2005' for MODFLOW 2005 (the default)


# For MODFLOW 6, we first have to create a simulation, which we then assign the model to:

# In[2]:


sim = flopy.mf6.MFSimulation(sim_name='test')
gwf = flopy.mf6.ModflowGwf(sim, modelname='test')


# As you can see, in addition to the different FloPy subpackages, the syntax is different between the two versions. In general, the syntax for the `flopy.modflow` and `flopy.mf6` subpackages follows the respective MODFLOW versions they support. For example, `ModflowGwf` is simply a [CamelCase](https://en.wikipedia.org/wiki/Camel_case) representation of the MODFLOW 6 Groundwater Flow (GWF) model. Similarly, the Discretization Package (for the GWF model) class is named `ModflowGwfdis`. Arguments to the `ModflowGwfdis` constructor follow the input variables to MODFLOW 6.

# In[3]:


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
