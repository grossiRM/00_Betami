#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 
import pyemu
import pandas as pd


# In[ ]:


pst=pyemu.Pst('template/freyberg.pst')


# In[ ]:


obs=pst.observation_data


# In[ ]:


obs


# In[ ]:


truth = pd.read_csv('data/truth.obs_data.csv')


# In[ ]:


truth


# In[ ]:


dfs = []
for i in range(len(obs)):
    print (truth.loc[((truth.time == float(obs.iloc[i,7])) &
           (truth.usecol ==  obs.iloc[i,6]) & 
           (truth.oname=='hds')), 'obsval'].values[0])
    obs.iloc[i,1] = truth.loc[((truth.time == float(obs.iloc[i,7])) &
           (truth.usecol ==  obs.iloc[i,6]) & 
           (truth.oname=='hds')), 'obsval'].values[0]
    dfs.append(pd.DataFrame(data={'time':[float(obs.iloc[i,7])],
                                      'location':[obs.iloc[i,6]],
                                        'observation': [truth.loc[((truth.time == float(obs.iloc[i,7])) &
                                       (truth.usecol ==  obs.iloc[i,6]) & 
                                       (truth.oname=='hds')), 'obsval'].values[0]]}))
                                


# In[ ]:


obstrue = pd.concat(dfs)


# In[ ]:


obstrue


# In[ ]:


obstrue.to_csv('data/obstrue.csv')


# In[ ]:




