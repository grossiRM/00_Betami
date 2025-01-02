#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import pyemu
import pandas as pd


# # Part 0 - `PstFrom`

# In[2]:


model_ws = os.path.join('monthly_model_files_1lyr_newstress')


# In[3]:


pyemu.os_utils.run('mf6',cwd=model_ws)


# In[4]:


pf = pyemu.utils.PstFrom(model_ws,'template',remove_existing=True)


# In[5]:


pf.add_parameters('freyberg6.npf_k_layer1.txt',par_type='grid')


# In[6]:


h = pd.read_csv(os.path.join(model_ws , 'heads.csv'))
h


# In[7]:


pf.add_observations('heads.csv',index_cols='time',use_cols=h.columns.tolist()[1:])


# In[8]:


pf.mod_sys_cmds.append('mf6')


# In[9]:


pst = pf.build_pst(filename=os.path.join(pf.new_d,'freyberg.pst'),version=2)


# In[10]:


pyemu.os_utils.run('pestpp-ies freyberg.pst',cwd='template')


# # PART 1 - obs, weights, and prior MC

# In[11]:


pst = pyemu.Pst('template/freyberg.pst')


# In[12]:


trueobs = pd.read_csv('./data/obstrue.csv',index_col=0)
trueobs


# In[13]:


obs = pst.observation_data


# In[14]:


obs


# In[15]:


for cob in obs.index:
    tmp = obs.loc[cob]
    print(trueobs.loc[(trueobs.time==float(tmp.time))&(trueobs.location==tmp.usecol), 'observation'].values[0])
    obs.loc[cob,'obsval'] = trueobs.loc[(trueobs.time==float(tmp.time))&(trueobs.location==tmp.usecol), 'observation'].values[0]


# In[16]:


obs.weight=0.5


# In[17]:


obs


# In[18]:


pst.plot(kind='phi_pie')


# In[19]:


newbalance = {grp:1/len(pst.obs_groups)*pst.phi for grp in pst.obs_groups}
pst.adjust_weights(obsgrp_dict=newbalance)


# In[20]:


pst.plot(kind='phi_pie')


# In[21]:


pst.control_data.noptmax=-1


# In[22]:


pst.pestpp_options['ies_num_reals']=100


# In[23]:


pst.parameter_data.parlbnd=0.5
pst.parameter_data.parubnd=1.5


# In[24]:


pst.write('template/freyberg_prior.pst')


# In[25]:


import shutil
shutil.copy2('pestpp-ies','template/pestpp-ies')


# In[26]:


pyemu.utils.start_workers('./template',master_dir='cm_prior', exe_rel_path='pestpp-ies',pst_rel_path='freyberg_prior.pst',worker_root='.', num_workers=10)


# In[27]:


phi = pd.read_csv('cm_prior/freyberg_prior.phi.actual.csv').T[6:]
phi


# In[28]:


phi.hist(bins=20)


# In[ ]:




