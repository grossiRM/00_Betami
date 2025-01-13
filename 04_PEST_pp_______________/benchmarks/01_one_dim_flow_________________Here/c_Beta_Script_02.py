#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os, sys
import numpy as np
import pandas as pd
import flopy


# In[4]:


def forward_model():
    try:
        os.remove("Beta_02/heads_out.csv")
    except:
        pass
    
    indx = pd.read_csv('Beta_02/index.csv')['idx'].values.astype(int)   
    
    
    x = np.loadtxt('Beta_02/inputx.dat')
    x = np.power(10.0, x)
    mf = flopy.modflow.Modflow.load(r'flow_1d.nam', model_ws = 'Beta_02/Beta_03' )
    
    hk = mf.upw.hk.array.copy()
    hk = x[np.newaxis, np.newaxis, :]
    mf.upw.hk = hk
    mf.upw.write_file()
    
    basefolder = os.getcwd()
    os.chdir("Beta_02")
    os.system("mfnwt.exe flow_1d.nam")                          # pending a little path directory ajustment
    os.chdir(basefolder)

    hds = flopy.utils.HeadFile(os.path.join('Beta_02/Beta_03', 'flow_1d.hds'))
    wl = hds.get_data(totim=1.0)
    wl = wl.squeeze()
    y =wl[indx]                                                 # model maping  

    out = pd.DataFrame()
    out['y'] = y
    out.to_csv('Beta_02/heads_out.csv', index_label = 'id')             # write model output
    
if __name__ == "__main__":
    forward_model()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script c_Beta_Script_02.ipynb')


# In[ ]:




