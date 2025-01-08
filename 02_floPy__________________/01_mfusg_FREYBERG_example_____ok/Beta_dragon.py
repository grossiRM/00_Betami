#!/usr/bin/env python
# coding: utf-8

# # __Beta Dragon__

# In[ ]:


import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import shutil
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import shutil
import sys
# sys.path.insert(0,os.path.join("..","..","dependencies"))                               
import pyemu
import flopy


# In[ ]:


if "linux" in platform.platform().lower():
    bin_path = os.path.join("..","..", "bin_new", "linux")
elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower():
    bin_path = os.path.join("..","..", "bin_new", "mac")
else:
    bin_path = os.path.join("..", "..", "bin_new", "win")


# In[ ]:


def prep_bins(dest_path):
    files = os.listdir(bin_path)
    for f in files:
        if os.path.exists(os.path.join(dest_path,f)):
            os.remove(os.path.join(dest_path,f))
        shutil.copy2(os.path.join(bin_path,f),os.path.join(dest_path,f))


# In[ ]:


def dir_cleancopy(org_d, new_d, delete_orgdir=False):
    # remove existing folder
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    # copy the original model folder across
    shutil.copytree(org_d, new_d)
    print(f'Files copied from:{org_d}\nFiles copied to:{new_d}')

    if delete_orgdir==True:
        shutil.rmtree(org_d)
        print(f'Hope you did that on purpose. {org_d} has been deleted.')
    #prep_bins(new_d)
    return

