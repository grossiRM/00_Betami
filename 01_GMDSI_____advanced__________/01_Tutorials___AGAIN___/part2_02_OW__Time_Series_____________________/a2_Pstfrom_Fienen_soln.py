#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import os 
import pyemu
import pandas as pd


# # Part 0 - `PstFrom`

# %%


model_ws = os.path.join('monthly_model_files_1lyr_newstress')


# %%


pyemu.os_utils.run('mf6',cwd=model_ws)


# %%


pf = pyemu.utils.PstFrom(model_ws,'template',remove_existing=True)


# %%


pf.add_parameters('freyberg6.npf_k_layer1.txt',par_type='grid')


# %%


h = pd.read_csv(os.path.join(model_ws , 'heads.csv'))
h


# %%


pf.add_observations('heads.csv',index_cols='time',use_cols=h.columns.tolist()[1:])


# %%


pf.mod_sys_cmds.append('mf6')


# %%


pst = pf.build_pst(filename=os.path.join(pf.new_d,'freyberg.pst'),version=2)


# %%


pyemu.os_utils.run('pestpp-ies freyberg.pst',cwd='template')


# # PART 1 - obs, weights, and prior MC

# %%


pst = pyemu.Pst('template/freyberg.pst')


# %%


trueobs = pd.read_csv('./data/obstrue.csv',index_col=0)
trueobs


# %%


obs = pst.observation_data


# %%


obs


# %%


for cob in obs.index:
    tmp = obs.loc[cob]
    print(trueobs.loc[(trueobs.time==float(tmp.time))&(trueobs.location==tmp.usecol), 'observation'].values[0])
    obs.loc[cob,'obsval'] = trueobs.loc[(trueobs.time==float(tmp.time))&(trueobs.location==tmp.usecol), 'observation'].values[0]


# %%


obs.weight=0.5


# %%


obs


# %%


pst.plot(kind='phi_pie')


# %%


newbalance = {grp:1/len(pst.obs_groups)*pst.phi for grp in pst.obs_groups}
pst.adjust_weights(obsgrp_dict=newbalance)


# %%


pst.plot(kind='phi_pie')


# %%


pst.control_data.noptmax=-1


# %%


pst.pestpp_options['ies_num_reals']=100


# %%


pst.parameter_data.parlbnd=0.5
pst.parameter_data.parubnd=1.5


# %%


pst.write('template/freyberg_prior.pst')


# %%


import shutil
shutil.copy2('pestpp-ies','template/pestpp-ies')


# %%


pyemu.utils.start_workers('./template',master_dir='cm_prior', exe_rel_path='pestpp-ies',pst_rel_path='freyberg_prior.pst',worker_root='.', num_workers=10)


# %%


phi = pd.read_csv('cm_prior/freyberg_prior.phi.actual.csv').T[6:]
phi


# %%


phi.hist(bins=20)


# %%




