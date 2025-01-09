#!/usr/bin/env python
# coding: utf-8
# %%

# # Formulating the Objective Function and The Dark Art of Weighting
# 
# The objective function expresses model-to-measurement misfit for use in the solution of an inverse problem through the *weighted squared difference between measured and simulated observations*. There is no formula for a universally "best approach" to formulate an objective function. However, the universal underlying principle is to ensure that as much information as possible is transferred to the parameters of a model, in as "safe" a way as possible (Doherty, 2015). 
# 
# **Observation Types**
# 
# In most history matching contexts a “multicomponent” objective function is recommended. Each component of this objective function is calculated on the basis of different groups of observations or of the same group of observations processed in different ways. In a nut-shell, this means as many (useful!) types of observation as are available should be included in the parameter-estimation process. This does not **just** mean different "measurement types"! It also means teasing out components *within* a given measurement type. These "secondary" components often contain information that is otherwise lost or overshadowed. 
# 
# **Observation Grouping** 
# 
# Using a multicomponent approach can extract as much information from an observation dataset as possible and transfer this information to estimated parameters. When constructing a PEST dataset, it is often useful (and convenient) to group observations by type. This makes it easier to customize objective function design and track the flow of information from data to parameters (and subsequently to predictions). Ideally, each observation grouping should illuminate and constrain parameter estimation related to a separate aspect of the system being modelled (Doherty and Hunt, 2010). For example, absolute values of heads may inform parameters that control horizontal flow patterns, whilst vertical differences between heads in different aquifers may inform parameters that control vertical flow patterns. 
# 
# **Observation Weighting**
# 
# A user must decide how to weight observations before estimating parameters with PEST(++). In some cases, it is prudent to strictly weight observations based on the inverse of the standard deviation of measurement noise. Observations with higher credibility should, without a doubt, be given more weight than those with lower credibility. However, in many history-matching contexts, model defects are just as important as noise in inducing model-to-measurement misfit as field measurements. Some types of model outputs are more affected by model imperfections than others. (Notably, the effects of imperfections on model output _differences_ are frequently less than their effects on raw model outputs.) In some cases, accommodating the "structural" nature of model-to-measurement misfit (i.e. the model is inherently better at fitting some measurements than others) is preferable rather than attempting to strictly respect the measurement credibility. 
# 
# The PEST Book (Doherty, 2015) and the USGS published report "A Guide to Using PEST for Groundwater-Model Calibration" (Doherty et al 2010) go into detail on formulating an objective function and discuss common issues with certain data-types. 
# 
# **References and Recommended Reading:**
# >  - Doherty, J., (2015). Calibration and Uncertainty Analysis for Complex Environmental Models. Watermark Numerical Computing, Brisbane, Australia. ISBN: 978-0-9943786-0-6.
# >  - <div class="csl-entry">White, J. T., Doherty, J. E., &#38; Hughes, J. D. (2014). Quantifying the predictive consequences of model error with linear subspace analysis. <i>Water Resources Research</i>, <i>50</i>(2), 1152–1173. https://doi.org/10.1002/2013WR014767</div>
# >  - <div class="csl-entry">Doherty, J., &#38; Hunt, R. (2010). <i>Approaches to Highly Parameterized Inversion: A Guide to Using PEST for Groundwater-Model Calibration: U.S. Geological Survey Scientific Investigations Report 2010–5169</i>. https://doi.org/https://doi.org/10.3133/sir20105169</div>

# 
# ### Recap: the modified-Freyberg PEST dataset
# 
# The modified Freyberg model is introduced in another tutorial notebook (see "intro to freyberg model"). The current notebook picks up following the ["pstfrom pest setup"](../part2_01_pstfrom_pest_setup/freyberg_pstfrom_pest_setup.ipynb) notebook, in which a high-dimensional PEST dataset was constructed using `pyemu.PstFrom`. You may also wish to go through the ["intro to pyemu"](../part0_intro_to_pyemu/intro_to_pyemu.ipynb) and ["pstfrom sneakpeak"](../part1_02_pest_setup/pstfrom_sneakpeak.ipynb) notebooks beforehand.
# 
# We will now address assigning observation target values from "measured" data and observation weighting prior to history matching. 
# 
# Recall from the ["pstfrom pest setup"](../part2_01_pstfrom_pest_setup/freyberg_pstfrom_pest_setup.ipynb) notebook that we included several observation types in the history matching dataset, namely:
#  - head time-series
#  - river gage time-series
#  - temporal differences between both heads and gage time-series
# 
# We also included many observations of model outputs for which we do not have measured data. We kept these to make it easier to keep track of model outputs (this becomes a necessity when working with ensembles). We also included observations of "forecasts", i.e. model outputs of management interest.
# 
# The next couple of cells load necessary dependencies and call a convenience function to prepare the PEST dataset folder for you. This is the same dataset that was constructed during the ["pstfrom pest setup"](../part2_01_pstfrom_pest_setup/freyberg_pstfrom_pest_setup.ipynb) tutorial. 
# 
# Simply press `shift+enter` to run the cells.

# %%


import os
import shutil
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import matplotlib

import sys
import pyemu
import flopy
assert "dependencies" in flopy.__file__
assert "dependencies" in pyemu.__file__
sys.path.insert(0,"..")
import herebedragons as hbd

plt.rcParams.update({'font.size': 10})


# %%


# specify the temporary working folder
t_d = os.path.join('freyberg6_template')


# %%


# use the conveninece function to get the pre-preprepared PEST dataset;
# this is the same dataset constructed in the "pstfrom" tutorial
if os.path.exists(t_d):
        shutil.rmtree(t_d)
org_t_d = os.path.join("..","part2_01_pstfrom_pest_setup",t_d)
assert os.path.exists(org_t_d),"you need to run the '/part2_01_pstfrom_pest_setup/freyberg_pstfrom_pest_setup.ipynb' notebook"
print("using files at ",org_t_d)
shutil.copytree(org_t_d,t_d)


# Let's load in the `Pst` control file we constructed during the "pstfrom" tutorial:

# %%


pst_file = "freyberg_mf6.pst"
pst = pyemu.Pst(os.path.join(t_d, pst_file))


# When we constructed the PEST dataset (in the "pstfrom pest setup" tutorial) we simply identified what model outputs we wanted PEST to "observe". In doing so, `pyemu.PstFrom` assigned observation target values that it found in the existing model output files. (Which conveniently allowed us to test whether out PEST setup was working correctly). All observation weights were assigned a default value of 1.0. 
# 
# As a reminder:

# %%


obs = pst.observation_data
obs.head()


# As mentioned above, we need to do several things:
#  - replace observation target values (the `obsval` column) with corresponding values from "measured data";
#  - assign meaningful weights to history matching target observations (the `weight` column);
#  - assign zero weight to observations that should not affect history matching.

# Let's start off with the basics. First set all weights to zero. We will then go through and assign meaningful weights only to relevant target observations. 

# %%


#check for nonzero weights
obs.weight.value_counts()


# %%


# assign all weight zero
obs.loc[:, 'weight'] = 0

# check for non zero weights
obs.weight.unique()


# ### Measured Data
# 
# In most data assimilation contexts you will have some relevant measured data (e.g. water levels, river flow rates, etc.) which correspond to simulated model outputs. These will probably not coincide exactly with your model outputs. Are the wells at the same coordinate as the center of the model cell? Do measurement times line up nicely with model output times? Doubt it. And if they do, are single measurements that match model output times biased? And so on... 
# 
# A modeller needs to ensure that the observation values assigned in the PEST control file are aligned with simulated model outputs. This will usually require some case-specific pre-processing. Here we are going to demonstrate __an example__ - but remember, every case is different!

# First, let's access our dataset of "measured" observations.

# %%


obs_csv = os.path.join('..', '..', 'models', 'daily_freyberg_mf6_truth',"obs_data.csv")
assert os.path.exists(obs_csv)
obs_data = pd.read_csv(obs_csv)
obs_data.set_index('site', inplace=True)
obs_data.iloc[:5]


# As you can see, we have measured data at daily intervals. But our model simulates monthly stress periods. So what observation value do we use? 
# 
# One option is to simply sample measured values from the data closest to our simulated output. The next cell does this, with a few checks along the way:

# %%


#just pick the nearest to the sp end
model_times = pst.observation_data.time.dropna().astype(float).unique()
# get the list of osb names for which we have data
obs_sites =  obs_data.index.unique().tolist()

# restructure the obsevration data 
es_obs_data = []
for site in obs_sites:
    site_obs_data = obs_data.loc[site,:].copy()
    if isinstance(site_obs_data, pd.Series):
        site_obs_data.loc["site"] = site_obs_data.index.values
    elif isinstance(site_obs_data, pd.DataFrame):
        site_obs_data.loc[:,"site"] = site_obs_data.index.values
        site_obs_data.index = site_obs_data.time
        site_obs_data = site_obs_data.reindex(model_times,method="nearest")

    if site_obs_data.shape != site_obs_data.dropna().shape:
        print("broke",site)
    es_obs_data.append(site_obs_data)
es_obs_data = pd.concat(es_obs_data,axis=0,ignore_index=True)
es_obs_data.shape


# Right then...let's plot our down-sampled measurement data and compare it to the original high-frequency time series.
# 
# The next cell generates plots for each time series of measured data. Blue lines are the original high-frequency data. The marked red line is the down-sampled data. What do you think? Does sampling to the "closest date" capture the behaviour of the time series? Doesn't look too good...It does not seem to capture the general trend very well.
# 
# Let's try something else instead.

# %%


for site in obs_sites:
    #print(site)
    site_obs_data = obs_data.loc[site,:]
    es_site_obs_data = es_obs_data.loc[es_obs_data.site==site,:].copy()
    es_site_obs_data.sort_values(by="time",inplace=True)
    #print(site,site_obs_data.shape)
    fig,ax = plt.subplots(1,1,figsize=(10,2))
    ax.plot(site_obs_data.time,site_obs_data.value,"b-",lw=0.5)
    #ax.plot(es_site_obs_data.datetime,es_site_obs_data.value,'r-',lw=2)
    ax.plot(es_site_obs_data.time,es_site_obs_data.value,'r-',lw=1,marker='.',ms=10)
    ax.set_title(site)
plt.show()


# This time, let's try using a moving-average instead. Effectively this is applying a low-pass filter to the time-series, smooting out some of the spiky noise. 
# 
# The next cell re-samples the data and then plots it. Measured data sampled using a low-pass filter is shown by the marked green line. What do you think? Better? It certainly does a better job at capturing the trends in the original data! Let's go with that.

# %%


ess_obs_data = {}
for site in obs_sites:
    #print(site)
    site_obs_data = obs_data.loc[site,:].copy()
    if isinstance(site_obs_data, pd.Series):
        site_obs_data.loc["site"] = site_obs_data.index.values
    if isinstance(site_obs_data, pd.DataFrame):
        site_obs_data.loc[:,"site"] = site_obs_data.index.values
        site_obs_data.index = site_obs_data.time
        sm = site_obs_data.value.rolling(window=20,center=True,min_periods=1).mean()
        sm_site_obs_data = sm.reindex(model_times,method="nearest")
    #ess_obs_data.append(pd.DataFrame9sm_site_obs_data)
    ess_obs_data[site] = sm_site_obs_data
    
    es_site_obs_data = es_obs_data.loc[es_obs_data.site==site,:].copy()
    es_site_obs_data.sort_values(by="time",inplace=True)
    fig,ax = plt.subplots(1,1,figsize=(10,2))
    ax.plot(site_obs_data.time,site_obs_data.value,"b-",lw=0.25)
    ax.plot(es_site_obs_data.time,es_site_obs_data.value,'r-',lw=1,marker='.',ms=10)
    ax.plot(sm_site_obs_data.index,sm_site_obs_data.values,'g-',lw=0.5,marker='.',ms=10)
    ax.set_title(site)
plt.show()
ess_obs_data = pd.DataFrame(ess_obs_data)
ess_obs_data.shape


# ### Update Target Observation Values in the Control File
# 
# Right then - so, these are our smoothed-sampled observation values:

# %%


ess_obs_data.head()


# Now we are confronted with the task of getting these _processed_ measured observation values into the `Pst` control file. Once again, how you do this will end up being somewhat case-specific and will depend on how your obsveration names were constructed. For example, in our case we can use the following function (we made it a function because we are going to repeat it a few times):

# %%


def update_pst_obsvals(obs_names, obs_data):
    """obs_names: list of selected obs names
       obs_data: dataframe with obs values to use in pst"""
    # for checking
    org_nnzobs = pst.nnz_obs
    # get list of times for obs name sufixes
    time_str = obs_data.index.map(lambda x: f"time:{x}").values
    # empyt list to keep track of misssing observation names
    missing=[]
    for col in obs_data.columns:
        # get obs list sufix for each column of data
        obs_sufix = col.lower()+"_"+time_str
        for string, oval, time in zip(obs_sufix,obs_data.loc[:,col].values, obs_data.index.values):
                
                if not any(string in obsnme for obsnme in obs_names):
                    if string.startswith("trgw-2"):
                        pass
                    else:
                        missing.append(string)
                # if not, then update the pst.observation_data
                else:
                    # get a list of obsnames
                    obsnme = [ks for ks in obs_names if string in ks] 
                    assert len(obsnme) == 1,string
                    obsnme = obsnme[0]
                    # assign the obsvals
                    obs.loc[obsnme,"obsval"] = oval
                    # assign a generic weight
                    if time > 3652.5 and time <=4018.5:
                        obs.loc[obsnme,"weight"] = 1.0
    # checks
    #if (pst.nnz_obs-org_nnzobs)!=0:
    #    assert (pst.nnz_obs-org_nnzobs)==obs_data.count().sum()
    if len(missing)==0:
        print('All good.')
        print('Number of new nonzero obs:' ,pst.nnz_obs-org_nnzobs) 
        print('Number of nonzero obs:' ,pst.nnz_obs)  
    else:
        raise ValueError('The following obs are missing:\n',missing)
    return


# %%


pst.nnz_obs_groups


# %%


# subselection of observaton names; this is because several groups share the same obs name sufix
obs_names = obs.loc[obs.oname.isin(['hds', 'sfr']), 'obsnme']

# run the function
update_pst_obsvals(obs_names, ess_obs_data)


# %%


pst.nnz_obs_groups


# %%


pst.observation_data.oname.unique()


# So that has sorted out the absolute observation groups. But remember the 'sfrtd' and 'hdstd' observation groups? Yeah thats right, we also added in a bunch of other "secondary observations" (the time difference between obsevrations) as well as postprocessing functions to get them from model outputs. We need to get target values for these observations into our control file as well!
# 
# Let's start by calculating the secondary values from the absolute measured values. In our case, the easiest is to populate the model output files with measured values and then call our postprocessing function.
# 
# Let's first read in the SFR model output file, just so we can see what is happening:

# %%


obs_sfr = pd.read_csv(os.path.join(t_d,"sfr.csv"),
                    index_col=0)

obs_sfr.head()


# Now update the model output csv files with the smooth-sampled measured values:

# %%


def update_obs_csv(obs_csv):
    obsdf = pd.read_csv(obs_csv, index_col=0)
    check = obsdf.copy()
    # update values in reelvant cols
    for col in ess_obs_data.columns:
        if col in obsdf.columns:
            obsdf.loc[:,col] = ess_obs_data.loc[:,col]
    # keep only measured data columns; helps for vdiff and tdiff obs later on
    #obsdf = obsdf.loc[:,[col for col in ess_obs_data.columns if col in obsdf.columns]]
    # rewrite the model output file
    obsdf.to_csv(obs_csv)
    # check 
    obsdf = pd.read_csv(obs_csv, index_col=0)
    assert (obsdf.index==check.index).all()
    return obsdf

# update the SFR obs csv
obs_srf = update_obs_csv(os.path.join(t_d,"sfr.csv"))
# update the heads obs csv
obs_hds = update_obs_csv(os.path.join(t_d,"heads.csv"))


# OK...now we can run the postprocessing function to update the "tdiff" model output csv's. Copy across the `helpers.py` we used during the `PstFrom` tutorial. Then import it and run the `process_secondary_obs()` function.

# %%


shutil.copy2(os.path.join('..','part2_01_pstfrom_pest_setup', 'helpers.py'),
            os.path.join('helpers.py'))

import helpers
helpers.process_secondary_obs(ws=t_d)


# %%


# the oname column in the pst.observation_data provides a useful way to select observations in this case
obs.oname.unique()


# %%


org_nnzobs = pst.nnz_obs
    #if (pst.nnz_obs-org_nnzobs)!=0:
    #    assert (pst.nnz_obs-org_nnzobs)==obs_data.count().sum()


# %%


print('Number of nonzero obs:', pst.nnz_obs)

diff_obsdict = {'sfrtd': "sfr.tdiff.csv", 
                'hdstd': "heads.tdiff.csv",
                }

for keys, value in diff_obsdict.items():
    print(keys)
    # get subselct of obs names
    obs_names = obs.loc[obs.oname.isin([keys]), 'obsnme']
    # get df
    obs_csv = pd.read_csv(os.path.join(t_d,value),index_col=0)
    # specify cols to use; make use of info recorded in pst.observation_data to only select cols with measured data
    usecols = list(set((map(str.upper, obs.loc[pst.nnz_obs_names,'usecol'].unique()))) & set(obs_csv.columns.tolist()))
    obs_csv = obs_csv.loc[:, usecols]
    # for checking
    org_nnz_obs_names = pst.nnz_obs_names
    # run the function
    update_pst_obsvals(obs_names,
                        obs_csv)
    # verify num of new nnz obs
    print(pst.nnz_obs)
    print(len(org_nnz_obs_names))
    print(len(usecols))
    assert (pst.nnz_obs-len(org_nnz_obs_names))==12*len(usecols), [i for i in pst.nnz_obs_names if i not in org_nnz_obs_names]


# %%


pst.nnz_obs_groups


# %%


pst.nnz_obs_names


# The next cell does some sneaky things in the background to populate `obsvals` for forecast observations just so that we can keep track of the truth. In real-world applications you might assign values that reflect decision-criteria (such as limits at which "bad things" happen, for example) simply as a convenience. For the purposes of history matching, these values have no impact because they are assigned zero weight. They can play a role in specifying constraints when undertaking optimisation problems.  

# %%


pst.observation_data.loc[pst.forecast_names]


# %%


hbd.prep_forecasts(pst)


# %%


pst.observation_data.loc[pst.forecast_names]


# ## Observation Weights
# 
# Of all the issues that we have seen over the years, none is greater than (in)appropriate weighting strategies.  It is a critical and fundamental component of any inverse problem, but is especially important in settings where the model is imperfect simulator and the observation data are noisy and there are diverse types of data.  Goundwater modeling anyone? 
# 
# In essence the weights will change the shape of the objective function surface in parameter space, moving the minimum around and altering the path to the minimum (this can be seen visually in the response surface notebooks).  Given the important role weights play in the outcome of a history-matching/data assimilation analysis, rarely is a weighting strategy "one and done", instead it is continuously revisited during a modeling analysis, based on what happened during the previous history-matching attempt.  
# 
# We are going to start off by taking a look at our current objective function value and the relative contributions from the various observation groups - these relative contributions are a function of the residuals and weights in each group. Recall that this is the objective function value with **initial parameter values** and the default observations weights.
# 
# First off, we need to get PEST to run the model once so that the objective function can be calculated. Let's do that now. Start by reading the control file and checking that NOPTMAX is set to zero:
# 
# 

# %%


# check noptmax
pst.control_data.noptmax


# You got a zero? Alrighty then! Let's write the uprated control file and run PEST again and see what that has done to our Phi:

# %%


pst.write(os.path.join(t_d,pst_file),version=2)


# %%


pyemu.os_utils.run("pestpp-ies.exe {0}".format(pst_file),cwd=t_d)


# Now we need to reload the `Pst` control file so that the residuals are updated:

# %%


pst = pyemu.Pst(os.path.join(t_d, pst_file))
pst.phi


# Jeepers - that's large! Before we race off and start running PEST to lower it we should compare simualted and measured values and take a look at the components of Phi. 
# 
# Let's start with taking a closer look. The `pst.phi_components` attribute returns a dictionary of the observation group names and their contribution to the overal value of Phi. 

# %%


pst.phi_components


# 
# Unfortunately, in this case we have too many observation groups to easily display (we assigned each individual time series to its own observation group; this is a default setting in `pyemu.PstFrom`). 
# 
# So let's use `Pandas` to help us sumamrize this information (note: `pyemu.plot_utils.res_phi_pie()` does the same thing, but it looks a bit ugly because of the large number of observation groups). To make it easier, we are going to just look at the nonzero observation groups:

# %%


nnz_phi_components = {k:pst.phi_components[k] for k in pst.nnz_obs_groups} # that's a dictionary comprehension there y'all
nnz_phi_components


# And while we are at it, plot these in a pie chart. 
# 
# If you wish, try displaying this with `pyemu.plot_utils.res_phi_pie()` instead. Because of the large number of columns it's not going to be pretty, but it gets the job done.

# %%


phicomp = pd.Series(nnz_phi_components)
plt.pie(phicomp, labels=phicomp.index.values);
#pyemu.plot_utils.res_phi_pie(pst,);


# Well that is certainly not ideal - phi is dominated by the SFR observation groups. Why? Because the magnitude of these observation values are much larger than groundwater-level based observations, so we can expect the residuals in SFR observations to be yuge compared to groundwater level residuals...and we assigned the same weight to all of them...
# 
# Now we have some choices to make.  In many settings, there are certain observations (or observation groups) that are of increased importance, whether its for predictive reasons (like some data are more similar to the predictive outputs from the modeling) or political - "show me obs vs sim for well XXX"...If this is the case, then it is probably important to give those observations a larger portion of the composite objective function so that the results of the history matching better reproduce those important observations.  
# 
# In this set of notebooks, we will use another very common approach: give all observations groups an equal portion of the composite objective function.  This basically says "all of the different observation groups are important, so do your best with all of them"
# 
# The `Pst.adjust_weights()` method provides a mechanism to fine tune observation weights according to their contribution to the objective function. (*Side note: the PWTADJ1 utility from the PEST-suite automates this same process of "weighting for visibility".*) 
# 
# We start by creating a dictionary of non-zero weighted observation group names and their respective contributions to the objective function. Herein, we will use the existing composite phi value as the target composite phi...
# 

# %%


# create a dictionary of group names and weights
contrib_per_group = pst.phi / float(len(pst.nnz_obs_groups))
balanced_groups = {grp:contrib_per_group for grp in pst.nnz_obs_groups}
balanced_groups


# %%


# make all non-zero weighted groups have a contribution of 100.0
pst.adjust_weights(obsgrp_dict=balanced_groups,)


# Let's take a look at how that has affected the contributions to Phi:

# %%


plt.figure(figsize=(7,7))
phicomp = pd.Series({k:pst.phi_components[k] for k in pst.nnz_obs_groups})
plt.pie(phicomp, labels=phicomp.index.values);
plt.tight_layout()


# Better! Now each observation group contributes equally to the objective function

# The next cell adds in a column to the `pst.observation_data` for checking purposes in subsequent tutorials. In practice, when you have lots of model outputs treated as "obserations" in the pest control file, setting a flag to indicate exactly which observation quantities correspond to actual real-world information can be important for tracking things through your workflow...

# %%


pst.observation_data.loc[pst.nnz_obs_names,'observed'] = 1


# ### Understanding Observation Weights and Measurement Noise
# 
# Let's have a look at what weight values were assigned to our observation groups:

# %%


obs = pst.observation_data
for group in pst.nnz_obs_groups:
    print(group,obs.loc[obs.obgnme==group,"weight"].unique())


# Ok, some variability there, and, as expected, the sfr flowout observations have been given a very low weight and the groundwater level obs have been given a very high weight - this is simply to overcome the difference in magnitudes between these two different data types.  All good...or is it?
# 
# In standard deterministic parameter estimation, only the relative difference between the weights matters, so we are fine there...but in uncertainty analysis, we often want to account for the contribution from measurement noise and we havent told any of the pest++ tools not to use the inverse of the weights to approximate measurement noise, and this is a problem because those weights we assigned have no relation to measurement noise!  This can cause massive problems later, especially is you are using explicit noise realizations in uncertainty analysis - Imagine how much SFR flow noise is implied by that tiny weight?  It's easy to see how negative SFR flow noise values might be drawn with that small of a weight (high of a standard deviation) #badtimes.   
# 
# So what can we do?  Well there are options.  An easy way is to simply supply a "standard_deviation" column in the `pst.observation_data` dataframe that will cause these values to be used to represent measurement noise.  

# %%


obs = pst.observation_data
obs.loc[:,"standard_deviation"] = np.nan
hds_obs = [o for o in pst.nnz_obs_names if "oname:hds_" in o]
assert len(hds_obs) > 0
obs.loc[hds_obs,"standard_deviation"] = 0.5
hdstd_obs = [o for o in pst.nnz_obs_names if "oname:hdstd_" in o]
assert len(hdstd_obs) > 0
obs.loc[hdstd_obs,"standard_deviation"] = 0.01

sfr_obs = [o for o in pst.nnz_obs_names if "oname:sfr_" in o]
assert len(sfr_obs) > 0
# here we will used noise that is a function of the observed flow value so that 
# when flow is high, noise is high.
obs.loc[sfr_obs,"standard_deviation"] = abs(obs.loc[sfr_obs,"obsval"] * 0.1)
sfrtd_obs = [o for o in pst.nnz_obs_names if "oname:sfrtd_" in o]
assert len(sfrtd_obs) > 0
obs.loc[sfrtd_obs,"standard_deviation"] = abs(obs.loc[sfrtd_obs,"obsval"] * 0.1)


# %%


pst.write(os.path.join(t_d,pst_file),version=2)


# Some **caution** is required here. Observation weights and how these pertain to history-matching *versus* how they pertain to generating an observation ensemble for use with `pestpp-ies` or FOSM is a frequent source of confusion.
# 
#  - when **history-matching**, observation weights listed in the control file determine their contribution to the objective function, and therefore to the parameter estimation process. Here, observation weights may be assigned to reflect observation uncertainty, the balance required for equal "visibility", or other modeller-defined (and perhaps subjective...) measures of observation worth.  
#  - when undertaking **uncertainty analysis**, weights should reflect ideally the inverse of observation error (or the standard deviation of measurement noise). Keep this in mind when using `pestpp-glm` or `pestpp-ies`. If the observations in the control file do not have error-based weighting then (1) care is required if using `pestpp-glm` for FOSM and (2) make sure to provide an adequately prepared observation ensemble to `pestpp-ies`.
# 
# 

# Made it this far? Congratulations! Have a cookie :)
