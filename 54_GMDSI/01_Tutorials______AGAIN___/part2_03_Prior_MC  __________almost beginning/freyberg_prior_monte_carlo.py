#!/usr/bin/env python
# coding: utf-8

# # Prior Monte Carlo
# 
# Prior-based (or "unconstrained") Monte Carlo is a useful, but quite often underused, analysis. It is conceptually simple, does not require much in terms of algorithmic controls and forces the modeller to think about the prior parameter probability distribution - both the mean vector (i.e. the initial parameter values) and the prior parameter covariance matrix. 
# 
# The idea is simple: sample many sets of parameters (i.e. an ensemble) from a prior probability distribution and run the model forward for each realization in this ensemble and collate the results. Do not try and fit historical data (yet!). Do not throw any of the simulations out because they "do not represent historical data well". This allows us to explore the entire range of model outcomes across the (prior) range of parameter values. It let's us investigate model stability (e.g. can the model setup handle the parameters we are throwing at it?). It also let's us start to think critically about what observations the model will be able to match. If we can't match observations we can explore deficiencies in the parameters, the model, or even the observation values themselves.
# 
# Sometimes, it shows us that history matching is not required - saving us a whole lot of time and effort!
# 
# In this notebook we will demonstrate:
#  - how to use `pyemu` to run `pestpp` in parallel locally (that is on your machine only)
#  - using `pestpp-ies` to undertake prior monte carlo with an existing geostatistically correlated prior parameter ensemble
#  - using `pestpp-ies` to undertake prior monte carlo with an uncorrelated prior parameter ensemble 
#  - post-processing stochastic model outputs
# 
# ### The modified Freyberg PEST dataset
# 
# The modified Freyberg model is introduced in another tutorial notebook (see ["intro to freyberg model"](../part0_02_intro_to_freyberg_model/intro_freyberg_model.ipynb)). The current notebook picks up following the ["freyberg psfrom pest setup"](../part2_01_pstfrom_pest_setup/freyberg_pstfrom_pest_setup.ipynb) notebook, in which a high-dimensional PEST dataset was constructed using `pyemu.PstFrom`. You may also wish to go through the ["intro to pyemu"](../part0_intro_to_pyemu/intro_to_pyemu.ipynb) and ["pstfrom sneakpeak"](../part1_02_pest_setup/pstfrom_sneakpeak.ipynb) notebooks beforehand.
# 
# The next couple of cells load necessary dependencies and call a convenience function to prepare the PEST dataset folder for you. This is the same dataset that was constructed during the ["freyberg psfrom pest setup"](../part2_01_pstfrom_pest_setup/freyberg_pstfrom_pest_setup.ipynb) tutorial. Simply press `shift+enter` to run the cells.

# In[ ]:


import os
import shutil
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import matplotlib.pyplot as plt;
import psutil 

import sys
import pyemu
import flopy
assert "dependencies" in flopy.__file__
assert "dependencies" in pyemu.__file__
sys.path.insert(0,"..")
import herebedragons as hbd


# In[ ]:


# specify the temporary working folder
t_d = os.path.join('freyberg6_template')
if os.path.exists(t_d):
    shutil.rmtree(t_d)

org_t_d = os.path.join("..","part2_02_obs_and_weights","freyberg6_template")
if not os.path.exists(org_t_d):
    raise Exception("you need to run the '/part2_02_obs_and_weights/freyberg_obs_and_weights.ipynb' notebook")
shutil.copytree(org_t_d,t_d)


# Load the PEST control file as a `Pst` object.

# In[ ]:


pst_path = os.path.join(t_d, 'freyberg_mf6.pst')
pst = pyemu.Pst(os.path.join(t_d, 'freyberg_mf6.pst'))


# In[ ]:


# check to see if obs&weights notebook has been run
if not pst.observation_data.observed.sum()>0:
    raise Exception("You need to run the '/part2_02_obs_and_weights/freyberg_obs_and_weights.ipynb' notebook")


# Load the prior parameter ensemble we generated previously:

# In[ ]:


[f for f in os.listdir(t_d) if f.endswith(".jcb")]


# In[ ]:


pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(t_d,"prior_pe.jcb"))
pe.shape


# ### Run the Ensemble in Parallel
# 
# Here we are going to make use of the `pestpp-ies` executable to execute the prior monte carlo in parallel. It may seem we are jumping ahead by using `pestpp-ies` (this is the iterative Ensemble Smoother version of PEST++) already, but the prior Monte Carlo is such an important aspect of the parameter estimation workflow that it is built in to the `pestpp-ies` executable. 
# 
# To run the prior Monte Carlo in this way, we simply need to:
# 1. specify a `pestpp_options` argument, identifying `ies_parameter_ensemble` to point to a prior parameter ensemble and
# 2. specify `noptmax` to be `-1`. Recall from GLM that `noptmax` has special values for things like calculating the Jacobian, or running the model only once. For `pestpp-ies`, `noptmax=-1` runs the prior Monte Carlo.
# 
# > **Pro Tip**: you can run a subset of a prior ensemble by also specifying `ies_num_reals` to be a number smaller than the number of parameter realizations in the prior ensemble. 
# 
# Also note there is a general purpose, simple parametric sweep utility called `pestpp-swp`  that can run a collection of parameter sets in parallel and collate the results, but we will stick with the simpler approach here.
# 
# So let's start by specifying the name of the prior parameter ensemble file that we generated previously:

# In[ ]:


pst.pestpp_options['ies_parameter_ensemble'] = 'prior_pe.jcb'


# Then, re-write the PEST control file. If you open `freyberg_mf6.pst` in a text editor, you'll see a new PEST++ control variable has been added.

# In[ ]:


pst.control_data.noptmax = 0 
pst.write(os.path.join(t_d, 'freyberg_mf6.pst'),version=2)


# Always good to do the 'ole `noptmax=0` test:

# In[ ]:


pyemu.os_utils.run("pestpp-ies freyberg_mf6.pst",cwd=t_d)


# Now, we are going to run `pestpp-ies` in parallel with `noptmax=-1` to simulate the prior Monte Carlo. 

# In[ ]:


pst.control_data.noptmax = -1
pst.write(os.path.join(t_d, 'freyberg_mf6.pst'),version=2)


# To speed up the process, you will want to distribute the workload across as many parallel agents as possible. Normally, you will want to use the same number of agents (or less) as you have available CPU cores. Most personal computers (i.e. desktops or laptops) these days have between 4 and 10 cores. Servers or HPCs may have many more cores than this. Another limitation to keep in mind is the read/write speed of your machines disk (e.g. your hard drive). PEST and the model software are going to be reading and writting lots of files. This often slows things down if agents are competing for the same resources to read/write to disk.
# 
# The first thing we will do is specify the number of agents we are going to use.
# 
# # Attention!
# 
# You must specify the number which is adequate for ***your*** machine! Make sure to assign an appropriate value for the following `num_workers` variable:

# In[ ]:


num_workers = psutil.cpu_count(logical=False) #update this according to your resources


# Next, we shall specify the PEST run-manager/master directory folder as `m_d`. This is where outcomes of the PEST run will be recorded. It should be different from the `t_d` folder, which contains the "template" of the PEST dataset. This keeps everything separate and avoids silly mistakes.

# In[ ]:


m_d = os.path.join('master_priormc')


# The following cell deploys the PEST agents and manager and then starts the run using `pestpp-ies` (using `pestpp-ies freyberg_mf6.pst /h localhost:4004` on the agents, and `pestpp-ies freyberg_mf6.pst /h :4004` on the manager).
# 
# Run it by pressing `shift+enter`.
# 
# If you wish to see the outputs in real-time, switch over to the terminal window (the one which you used to launch the `jupyter notebook`). There you should see `pestpp-ies`'s progress written to the terminal window in real-time. 
# 
# If you open the tutorial folder, you should also see a bunch of new folders there named `worker_0`, `worker_1`, etc. These are the agent folders. The `master_priormc` folder is where the manager is running. 
# 
# This run should take several minutes to complete (depending on the number of workers and the speed of your machine). If you get an error, make sure that your firewall or antivirus software is not blocking `pestpp-ies` from communicating with the agents (this is a common problem!).
# 
# > **Pro Tip**: Running PEST from within a `jupyter notebook` has a tendency to slow things down and hog alot of RAM. When modelling in the "real world" it is more efficient to implement workflows in scripts which you can call from the command line.

# In[ ]:


pyemu.os_utils.start_workers(t_d, # the folder which contains the "template" PEST dataset
                            'pestpp-ies', #the PEST software version we want to run
                            'freyberg_mf6.pst', # the control file to use with PEST
                            num_workers=num_workers, #how many agents to deploy
                            worker_root='.', #where to deploy the agent directories; relative to where python is running
                            master_dir=m_d, #the manager directory
                            )


# ### Explore the Outcomes
# 
# `pestpp-swp` writes the results of the the prior Monte Carlo to a csv file called `freyberg_mf6.0.obs.csv`. Note the naming convention - this is the base `pst` file name (`freyberg_mf6`) followed by the iteration number (`0` indicating this is at the beginning of the process - e.g. _prior_ Monte Carlo) and `.obs.csv` indicating this file contains observation values. This file has columns for each observation listed in the control file, plus an index column with the realization name. 
# 

# In[ ]:


obs_df = pd.read_csv(os.path.join(m_d,"freyberg_mf6.0.obs.csv"),index_col=0)
print('number of realizations in the ensemble: ' + str(obs_df.shape[0]))


# We can take a look at the distribution of Phi obtained for the ensemble. These are in another file with the same root, in this case called `freyberg_mf6.phi.actual.csv`. let's read in the file.

# In[ ]:


phi_df = pd.read_csv(os.path.join(m_d,"freyberg_mf6.phi.actual.csv"),index_col=0)
phi_df


# Note there are a few summary columns here, but for now we want to check out a quick histogram so we can use some quick pandas trickery to skip those and plot a histogram:

# In[ ]:


phi_df = phi_df.T.iloc[5:].rename(columns={0:'phi'})
phi_df.phi.hist(bins=50)


# Some pretty high values there. But that's fine. We are not concerned with getting a "good fit" in prior MC.

# More important is to inspect whether the ***distribution*** of simulated observations encompass measured values. Our first concern is to ensure that the model is ***able*** to captured observed behaviour. If measured values do not fall within the range of simualted values, this is a sign that something ain't right and we should revisit our model or prior parameter distributions.
# 

# A quick check is to plot stochastic (ensemble-based) 1-to-1 plots. We can plot 1to1 plots for obsvervation groups using the `pyemu.plot_utils.ensemble_res_1to1()` method. However, in our case that will result in lots of plots (we have many obs groups!). 

# In[ ]:


# pyemu.plot_utils.ensemble_res_1to1(obs_df, pst);


# Feel free to uncomment the previous cell and see what happens. This can be usefull for a quick review, but for the purposes of this tutorial, let's just look at four observation groups (recall, each group is made up of a time series of observations from a single location).
# 
# Now, this plot does not look particularily pretty...but we aren't here for pretty, we are here for results! What are we concerned with? Whether the range of ensemble simulated outcomes form the prior covers the measured values. Recall that plots on the left are 1to1 plots and on the right the residuals ar edisplayed.  In both cases, a grey line represents the range of simulated values for a given observation
# 
# In plots on the left, each grey line should interesect the 1-to-1 line. In the plots on the right, each grey line should intersect the "zero-residual" line. 

# In[ ]:


zero_weighted_obs_groups = [i for i in pst.obs_groups if i not in pst.nnz_obs_groups]
len(zero_weighted_obs_groups)


# In[ ]:


pyemu.plot_utils.ensemble_res_1to1(obs_df, pst, skip_groups=zero_weighted_obs_groups); 


# As we can see above, the prior covers the "measured" values (which is good).
# 
# But hold on a second! What about measurement noise? If we are saying that it is *possible* that our measurements are wrong by a certain amount, shouldn't we make sure our model can represent conditions in which they are? Yes, of course!
# 
# No worries, `pyemu` has you covered. Let's quickly cook up an ensemble of observations with noise. (Recall we recorded a covariance matrix of observation noise during the "freyberg pstfrom pest setup" notebook; this has also been discussed in the "observation and weights" notebook.)

# In[ ]:


obs_cov = pyemu.Cov.from_binary(os.path.join(t_d, 'obs_cov.jcb'))
obs_plus_noise = pyemu.ObservationEnsemble.from_gaussian_draw(pst=pst, cov=obs_cov);


# OK, now let's plot that again but with observation noise. 
# 
# Aha! Good, not only do our ensemble of model outcomes cover the measured values, but they also entirely cover the range of measured values with noise (red shaded area in the plot below). 

# In[ ]:


pyemu.plot_utils.ensemble_res_1to1(obs_df,
                                    pst, 
                                    skip_groups=zero_weighted_obs_groups,
                                    base_ensemble=obs_plus_noise); 


# Another, perhaps coarser, method to quickly explore outcomes is to look at histograms of observations. 
# 
# The following figure groups observations according to type (just to lump them together and make a smaller plot) and then plots histograms of observation values. Grey shaded columns represent simulated values from the prior. Red shaded columns represent the ensemble of measured values + noise. The grey columns should ideally be spread wider than the red columns.

# In[ ]:


plot_cols = pst.observation_data.loc[pst.nnz_obs_names].apply(lambda x: x.usecol + " "+x.oname,axis=1).to_dict()
plot_cols = {v: [k] for k, v in plot_cols.items()}
pyemu.plot_utils.ensemble_helper({"r":obs_plus_noise,"0.5":obs_df}, 
                                  plot_cols=plot_cols,bins=20,sync_bins=True,
                                  )
plt.show();


# Finally, let's plot the obs vs sim timeseries - everyone's fav!

# In[ ]:


pst.try_parse_name_metadata()
obs = pst.observation_data.copy()
obs = obs.loc[obs.oname.apply(lambda x: x in ["hds","sfr"])]
obs = obs.loc[obs.obgnme.apply(lambda x: x in pst.nnz_obs_groups),:]
obs.obgnme.unique()


# ## First let's see the entire ensemble compared with the observed values

# In[ ]:


ogs = obs.obgnme.unique()
fig,axes = plt.subplots(len(ogs),1,figsize=(10,5*len(ogs)))
ogs.sort()
for ax,og in zip(axes,ogs):
    oobs = obs.loc[obs.obgnme==og,:].copy()
    oobs.loc[:,"time"] = oobs.time.astype(float)
    oobs.sort_values(by="time",inplace=True)
    tvals = oobs.time.values
    onames = oobs.obsnme.values
    [ax.plot(tvals,obs_df.loc[i,onames].values,"0.5",lw=0.5,alpha=0.5) for i in obs_df.index]
    oobs = oobs.loc[oobs.weight>0,:]
    ax.plot(oobs.time,oobs.obsval,"r-",lw=2)
    ax.set_title(og,loc="left")


# ## Next we can inspect a single realization

# In[ ]:


ogs = obs.obgnme.unique()
fig,axes = plt.subplots(len(ogs),1,figsize=(10,5*len(ogs)))
ogs.sort()
for ax,og in zip(axes,ogs):
    oobs = obs.loc[obs.obgnme==og,:].copy()
    oobs.loc[:,"time"] = oobs.time.astype(float)
    oobs.sort_values(by="time",inplace=True)
    tvals = oobs.time.values
    onames = oobs.obsnme.values
    i = obs_df.index[1]
    ax.plot(tvals,obs_df.loc[i,onames].values,"0.5",lw=0.5,alpha=0.5)
    oobs = oobs.loc[oobs.weight>0,:]
    ax.plot(oobs.time,oobs.obsval,"r-",lw=2)
    ax.set_title(og,loc="left")


# ### Forecasts
# 
# As usual, we bring this story back to the forecasts - after all they are why we are modelling.

# In[ ]:


pst.forecast_names


# The following cell will plot the distribution of each forecast obtained by running the prior parameter ensemble. Because we are using a synthetic model, we also have the privilege of being able to plot the "truth" (in the real world we don't know the truth of course). 
# 
# Many modelling analyses could stop here. If outcomes from a prior MC analysis show that the simulated distribution of forecasts *does not* cause some "bad-thing" to happen within an "acceptable" confidence, then you are done. No need to go and do expensive and time-consuming history-matching! 
# 
# On the other hand, if the uncertainty (e.g. variance) is unacceptably wide, then it *may* be justifiable to try to reduce forecast uncertainty through history matching. But only if you have forecast-sensitive observation data, and if the model is amenable to assimilating these data! How do I know that you ask? Worry not, we will get to this in subsequent tutorials.
# 

# In[ ]:


for forecast in pst.forecast_names:
    plt.figure()
    ax = obs_df.loc[:,forecast].plot(kind="hist",color="0.5",alpha=0.5, bins=20)
    ax.set_title(forecast)
    fval = pst.observation_data.loc[forecast,"obsval"]
    ax.plot([fval,fval],ax.get_ylim(),"r-")


# In[ ]:




