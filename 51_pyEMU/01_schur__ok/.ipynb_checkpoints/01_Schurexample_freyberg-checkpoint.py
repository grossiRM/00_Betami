#!/usr/bin/env python
# coding: utf-8

# # Freyberg Model Schur Complement Example
# This example uses a synthetic model (described below) to illustrate the `pyemu` capabilities of the Schur complement for calculating posterior covariance.
# 
# Note that, in addition to `pyemu`, this notebook relies on `flopy`. `flopy` can be obtained (along with installation instructions) at https://github.com/modflowpy/flopy.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle as rect
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
newparams = {'legend.fontsize':10, 'axes.labelsize':10,
             'xtick.labelsize':10, 'ytick.labelsize':10,
             'font.family':'Univers 57 Condensed', 
             'pdf.fonttype':42}
plt.rcParams.update(newparams)


# ## Model background
# This example is based on the synthetic classroom model of Freyberg(1988).  The  model is a 2-dimensional MODFLOW model with 1 layer,  40 rows, and 20 columns.  The model has 2 stress periods: an initial steady-state stress period used for calibration, and a 5-year transient stress period.  The calibration period uses the recharge and well flux of Freyberg(1988); the last stress period use 25% less recharge and 25% more pumping to represent future conditions for a forecast period.
# 
# The inverse problem has 761 parameters: hydraulic conductivity of each active model cell, calibration and forecast period recharge multipliers, storage and specific yield, calibration and forecast well flux for each of the six wells, and river bed conductance for each 40 cells with river-type boundary conditions.  The inverse problem has 12 head observations, measured at the end of the steady-state calibration period.  The forecasts of interest include the sw-gw exchange flux during both stress periods (observations named ```sw_gw_0``` and ``sw_gw_1``), and the water level in well cell 6 located in at row 28 column 5 at the end of the stress periods (observations named ```or28c05_0``` and ```or28c05_1```).  The forecasts are included in the Jacobian matrix as zero-weight observations. The model files, pest control file and previously-calculated jacobian matrix are in the `freyberg/` folder
# 
# 
# Freyberg, David L. "AN EXERCISE IN GROUND‐WATER MODEL CALIBRATION AND PREDICTION." Groundwater 26.3 (1988): 350-360.

# In[ ]:


import flopy

# load the model
model_ws = os.path.join("Freyberg","extra_crispy")
ml = flopy.modflow.Modflow.load("freyberg.nam",model_ws=model_ws)


# In[ ]:


# Because this model is old -- it predates flopy's modelgrid implementation. 
# And because modelgrid has been implemented without backward compatibility 
# the modelgrid object is not constructed properly. 
# - We will use some sneaky pyemu to get things to how they should be 
import pyemu
sr = pyemu.helpers.SpatialReference.from_namfile(
    os.path.join(model_ws, ml.namefile), 
    delc=ml.dis.delc, 
    delr=ml.dis.delr
)
ml.modelgrid.set_coord_info(
    xoff=sr.xll,
    yoff=sr.yll,
    angrot=sr.rotation,
    proj4=sr.proj4_str,
    merge_coord_info=True,
)


# In[ ]:


# plot some model attributes
fig = plt.figure(figsize=(6.5,6.5))
ax = plt.subplot(111,aspect="equal")
#mm = flopy.plot.ModelMap(model=ml)
mm = flopy.plot.PlotMapView(model=ml)
mm.plot_grid()
ax = mm.ax
#ml.upw.hk.plot(axes=[ax],colorbar="K m/d",alpha=0.0)
ml.wel.stress_period_data.plot(axes=[ax])
ml.riv.stress_period_data.plot(axes=[ax])

# plot obs locations
obs = pd.read_csv(os.path.join("Freyberg","misc","obs_rowcol.dat"),
                  delim_whitespace=True)

obs_x = [ml.modelgrid.xcellcenters[r-1,c-1] for r,c in obs.loc[:,["row","col"]].values]
obs_y = [ml.modelgrid.ycellcenters[r-1,c-1] for r,c in obs.loc[:,["row","col"]].values]
ax.scatter(obs_x,obs_y,marker='.',label="water-level obs")

#plot names on the pumping well locations
wel_data = ml.wel.stress_period_data[0]
wel_x = ml.modelgrid.xcellcenters[wel_data["i"],wel_data["j"]]
wel_y = ml.modelgrid.ycellcenters[wel_data["i"],wel_data["j"]]
for i,(x,y) in enumerate(zip(wel_x,wel_y)):
    ax.text(x,y,"{0}".format(i+1),ha="center",va="center")

ax.set_ylabel("y(m)")
ax.set_xlabel("x(m)")

ax.add_patch(rect((0,0),0,0,label="well",ec="none",fc="r"))
ax.add_patch(rect((0,0),0,0,label="river",ec="none",fc="g"))

ax.legend(bbox_to_anchor=(1.75,1.0),frameon=False)
plt.savefig("domain.pdf")
plt.savefig("domain.png")


# The plot shows the Freyberg (1988) model domain.  The colorflood is the hydraulic conductivity $\left(\frac{m}{d}\right)$.  Red and green cells correspond to well-type and river-type boundary conditions. Blue dots show the locations of water levels used for calibration.

# ## Using `pyemu`

# In[ ]:


import pyemu


# First we need to create a linear_analysis object of the `schur`  derived type, which replicates the behavior of the `PREDUNC` suite of PEST for calculating posterior parameter covariance.  We pass it the name of the Jacobian matrix file.  Since we don't pass an explicit argument for `parcov` or `obscov`, `pyemu` attempts to build them from the parameter bounds and observation weights in a pest control file (.pst) with the same base case name as the Jacobian.  This assumes that the bounds represent the mean value + and - 2 times the standard deviation.
# 
# Since we are interested in forecast uncertainty as well as parameter uncertainty, we also pass the names of the forecast sensitivity vectors we are interested in, which are stored in the Jacobian as well.  Note that the `forecasts` argument can be a mixed list of observation names, other Jacobian files or PEST-compatible ASCII matrix files.

# In[ ]:


# just set the path and filename for the jco file
jco = os.path.join("Freyberg","freyberg.jcb") 
# use the jco name with extension "pst" for the control file
pst = pyemu.Pst(jco.replace(".jcb",".pst"))     
# get the list of forecast names from the pest++ argument
la = pyemu.Schur(jco=jco, pst=pst, verbose="schur_example.log")
print("observations,parameters in Jacobian:",la.jco.shape)
print(la.forecast_names)


# ##  General parameter uncertainty analysis--evaluating posterior parameter covariance
# Let's calculate and save the posterior parameter covariance matrix. In this linear analysis, the posterior covariance represents the updated covariance following notional calibration as represented by the Jacobian matrix and both prior parameter and epistemic observation covariance matrices using the Schur complement:

# In[ ]:


#writes posterior covariance to a text file
la.posterior_parameter.to_ascii(jco+"_post.cov") 


# You can open this file (it will be called `freyberg.jcb_post.cov`) in a text editor to examine.  The diagonal of this matrix is the posterior variance of each parameter. Since we already calculated the posterior parameter covariance matrix, additional calls to the `posterior_parameter` decorated methods only require access--they do not recalculate the matrix:
# 

# In[ ]:


#easy to read in the notebook
la.posterior_parameter.to_dataframe().sort_index().\
sort_index(axis=1).iloc[0:3,0:3] 


# We can see the posterior variance of each parameter along the diagonal of this matrix. Now, let's make a simple plot of prior vs posterior uncertainty for the 761 parameters. The ``.get_parameters_summary()`` method is the easy way:

# In[ ]:


#get the parameter uncertainty dataframe and sort it
par_sum = la.get_parameter_summary().\
   sort_values("percent_reduction",ascending=False)
#make a bar plot of the percent reduction 
par_sum.loc[par_sum.index[:20],"percent_reduction"].\
   plot(kind="bar",figsize=(10,4),edgecolor="none")
#echo the first 10 entries
par_sum.iloc[0:10,:]


# In[ ]:


# we can plot up the top 10 uncertainty reductions
par_sum.iloc[0:10,:]['percent_reduction'].plot(kind='bar')
plt.title('Percent Reduction')


# In[ ]:


# we can plot up the prior and posterior variance 
# of the top 10 percent reductions
par_sum.iloc[0:10,:][['prior_var','post_var']].plot(kind='bar')


# We can see that calibrating the model to the 12 water levels reduces the uncertainty of the calibration period recharge parameter (`rch_0`) by 43%.  Additionally, the hydraulic conductivity of many model cells is also reduced.  

# Now lets look at the other end of the parameter uncertainty summary -- the values with the _least_ amount of uncertainty reduction.  Note that calling ``get_parameter_summary()`` again results in no additional computation but is just accessing information already calculated

# In[ ]:


# sort in increasing order without 'ascending=False'
par_sum = la.get_parameter_summary().sort_values("percent_reduction") 
# plot the first 20
par_sum.loc[par_sum.index[:20],"percent_reduction"].\
   plot(kind="bar",figsize=(10,4),edgecolor="none")
#echo the first 10 
par_sum.iloc[0:20,:]


# We see that several parameters are unaffected by calibration - these are mostly parameters that represent forecast period uncertainty (parameters that end with ```_2```), but there are also some hydraulic conductivities that are uninformed by the 12 water level observations.
# 
# The naming conventions for the hydraulic conductivity parameters include their row and column location (starting at 0 rather than 1) so, for example, `hkr39r14` indicates hydraulic conductivity in row 39, column 14. This location is in a constant head cell, so it makes sense it would be uninformed by head values anywhere in the model. The other hydraulic conductivity values are in the upper right hand corner of the model, far from the observations in a stagnant area with limited groundwater flow.

# We can also make a map of uncertainty reduction for the hydraulic conductivity parameters using some ```flopy``` action

# In[ ]:


ml.modelgrid.extent, ml.modelgrid.xoffset, ml.modelgrid.xvertices


# In[ ]:


hk_pars = par_sum.loc[par_sum.groupby(lambda x:"hk" in x).groups[True],:]
hk_pars.loc[:,"names"] = hk_pars.index
names = hk_pars.names
# use the parameter names to parse out row and column locations
hk_pars.loc[:,"i"] = names.apply(lambda x: int(x[3:5]))
hk_pars.loc[:,"j"] = names.apply(lambda x: int(x[6:8]))
# set up an array of the value -1 the same shape 
# as the HK array in the UPW package
unc_array = np.zeros_like(ml.upw.hk[0].array) - 1
# fill the array with percent reduction values
for i,j,unc in zip(hk_pars.i,hk_pars.j,hk_pars.percent_reduction):
    unc_array[i,j] = unc 
# setting the array values that are still -1 
# (e.g. no percent reduction value in the cell)
# to np.NaN so that they don't get displayed on the plot
unc_array[unc_array == -1] = np.nan

# plot some model attributes
extent=ml.modelgrid.extent

fig = plt.figure(figsize=(10,10))
ax = plt.subplot(111,aspect="equal")

ml.wel.stress_period_data.plot(axes=[ax])
ml.riv.stress_period_data.plot(axes=[ax])
# plot obs locations
obs = pd.read_csv(os.path.join("Freyberg","misc","obs_rowcol.dat"),
                  delim_whitespace=True)
obs_x = [ml.modelgrid.xcellcenters[r-1,c-1] for r,c in obs.loc[:,["row","col"]].values]
obs_y = [ml.modelgrid.ycellcenters[r-1,c-1] for r,c in obs.loc[:,["row","col"]].values]
modelmap = flopy.plot.PlotMapView(model=ml)
modelmap.plot_grid()
cb = modelmap.plot_array(unc_array, alpha=0.5)
plt.colorbar(cb,label="percent uncertainty reduction")
ax.scatter(obs_x,obs_y,marker='d')

plt.savefig("par_unc_map.pdf")


# As expected, most of the information in the observations is reduces uncertainty for the hydraulic conductivity parameters near observations themselves. Areas farther from the observations experience less reduction in uncertainty due to calibration. 

# ## Forecast uncertainty
# Now let's examine the prior and posterior variance of the forecasts. The uncertainty in parameters directly impacts the uncertainty of forecasts made with the model. Four forecasts were identified for analysis, as described above:
# 
# 1. `sw_gw_0`: the surface water/groundwater exchange during the calibration stress period
# 2. `sw_gw_1`: the surface water/groundwater exchange during the prediction stress period
# 3. `or28c05_0`: the head in well cell 6 (row 28, column 5) at the end of the calibration stress period
# 4. `or28c05_1`: the head in well cell 6 (row 28, column 5) at the end of the prediction stress period
# 

# In[ ]:


# get the forecast summary then make a bar chart of the percent_reduction column
fig = plt.figure(figsize=(4,4))
ax = plt.subplot(111)
ax = la.get_forecast_summary().percent_reduction.plot(kind='bar',
                                                      ax=ax,grid=True)
ax.set_ylabel("percent uncertainy\nreduction from calibration")
ax.set_xlabel("forecast")
plt.tight_layout()
plt.savefig("forecast_sum.pdf")
la.get_forecast_summary()


# Notice the spread on the uncertainty reduction: some forecasts benefit more from calibration than others.  For example, ```or28c05_0```, the calibration-period water level forecast, benefits from calibration since its uncertainty is reduced by about 75%, while ```sw_gw_1```, the forecast-period surface-water groundwater exchange forecast does not benefit from calibration - its uncertainty is unchanged by calibration

# 
# 
# ## Parameter contribution to forecast uncertainty
# 
# ### Overview
# As we observed above, information cascades from observations to parameters and then out to forecasts. With specific forecasts of interest, we can evaluate which information contributes most to forecast uncertainty. This is accomplished by assuming a parameter (or group of parameters) is perfectly known and assessing the forecast uncertainty under that assumption. Of course, this is a pretty serious approximation because perfect knowledge of a parameter can never be obtained in reality. In fact, it is difficult to calculate what, for example, a pumping test will provide in terms of uncertainty reduction for a parameter. Nonetheless, this metric can still provide important insights into model dynamics and help guide future data collection efforts.
# 
# ### Evaluating parameters by groups
# With the Freyberg example, we can evaluate parameter contributions to forecast uncertainty with groups of parameters by type. 

# In[ ]:


df = la.get_par_group_contribution()
df


# In[ ]:


#calc the percent reduction in posterior
df_percent = 100.0 * (df.loc["base",:]-df)/\
                      df.loc["base",:]
#drop the base column
df_percent = df_percent.iloc[1:,:]
#transpose and plot
ax = df_percent.T.plot(kind="bar", ylim=[0,100],figsize=(8,5))
ax.grid()
plt.tight_layout()
plt.savefig('indiv_pars_certain_future.pdf')


# 
# We see some interesting results here.  The sw-gw flux during calibration (```sw_gw_0```) is influenced by both recharge and hk uncertainty, but the forecast period sw-gw flux is influenced most by recharge uncertainty. For the water level forecasts (```or28c05_0 and or28c05_1```), the results are similar: the forecast of water level at the end of the calibration period benefits most from hk knowledge, while the forecast period water level is most informed by recharge and storage. 
# 
# As expected, in both cases `rcond` has no impact on forecast uncertainty (typically, river conductance is insensitive and noninfluential across a wide range of values) and `storage` plays no role in the steady-state calibration period but is important for the transient forecast period. Uncertainty in `welflux` plays a small role but is eclipsed by `rch` which is responsible for a much more substantial amount of flux.

# ### Evaluating an alternative grouping 
# Let's repeat the analysis, but now group the parameters differently:
# 
# The forcings in the model are the well pumping rates (`welflux`) and recharge (`rch`). The suffix `_1` indicates the calibration period while `_2` indicates the forecast period. Based on this, we can create two groupings for the forcings in the two periods. Other parameters (hydraulic conductivity, storage, and river conductance) are relegated to a third group of properties.
# 
# If we create a dictionary identifying groups as keys with lists of parameter names as values, we can pass that to the `get_par_contribution()` method. The dataframe returned will group results by the keys of the dictionary.

# In[ ]:


pnames = la.pst.par_names
fore_names = [pname for pname in pnames if pname.endswith("_2")]
props = [pname for pname in pnames if pname[:2] in ["hk","ss","sy","rc"] and\
         "rch" not in pname]
cal_names = [pname for pname in pnames if pname.endswith("_1")]
pdict = {'forecast forcing':fore_names,"properties":props,
         "calibration forcing":cal_names}
df = la.get_par_contribution(pdict)


# In[ ]:


#calc the percent reduction in posterior
df_percent_alt = 100.0 * (df.loc["base",:]-df)/\
                          df.loc["base",:]
#drop the base column
df_percent_alt = df_percent_alt.iloc[1:,:]
df_percent_alt


# In[ ]:


#transpose and plot
df_percent_alt.T.plot(kind="bar", ylim=[0, 100], figsize=(8,5))
plt.tight_layout()
plt.grid()
plt.savefig('certain_future.pdf')


# In[ ]:


df_percent_alt


# These results are also intuitive. For both forecasts originating from the second model stress period (the "forecast" period), the forecast-period forcings (which represent future recharge and future water use) play a role in reducing forecast uncertainty for the forecast period. Calibration forcings (current recharge and water use) are important for the calibration-period `sw_gw` exchange forecast (``sw_gw_0``), but are dwarfed by properties for the calibration-period head forecast (``or28c05_0``). Properties are important across the board, but in both cases their importance is reduced in the forecast period due to the increasing importance of forcing.
# 
# Evaluation of these dynamics is useful to understand the dynamics of the model, but it is difficult to quantify just how uncertainty can be reduced directly on parameters. On the other hand, we know that observations provide information on parameters through the calibration process. In a sense it is more straightforward to quantify how observation information impacts forecast uncertainty, so we can explore the worth of observation data.

# # Data worth analysis
# 
# ## Overview
# Data worth can be broken into two main categories: the worth of data pertaining directly to parameters, and the data pertaining to observations.
# 
# There are two main applications of data worth analysis. One is to evaluate the worth of observations in an existing network of observations, and the other is to evaluate the value of potential new observations. 

# ## Data worth--evaluating the value of existing observations
# Now, let's try to identify which observations are most important to reducing the posterior uncertainty (e.g.the forecast worth of every observation).  We simply recalculate Schur's complement without some observations and see how the posterior forecast uncertainty increases
# 
# ```get_removed_obs_group_importance()``` is a thin wrapper that calls the underlying ```get_removed_obs_importance()``` method using the observation groups in the pest control file and stacks the results into a ```pandas DataFrame```.  This method tests how important non-zero weight observations are for reducing forecast uncertainty. The metric provided is the forecast uncertainty that can be attributed to each observation group. 
# 
# This call will test all of the non-forecast, non-zero weight observations in the PEST data set to see which ones are most important. 
# 

# In[ ]:


df_worth = la.get_removed_obs_importance()
df_worth


# The ```base``` row contains the results of the Schur's complement calculation using all observations.  The increase in posterior forecast uncertainty for each of the 12 water level observations (e.g. or17c17 is the observation in row 18 column 18) show how much forecast uncertainty increases when that particular observation is not used in history matching.  So we see again that each forecast depends on the observations differently.
# 
# We can normalize the importance to the maximum importance value to create a metric of data worth which will be between 0 and 100%. Then we can also determine which observation has the highest data worth with respect to each forecast and also report how much reduction in uncertainty it is responsible for (e.g. how much does forecast uncertainty increase if that data point is not used for history matching)
# 
# 

# In[ ]:


# a little processing of df_worth
df_base = df_worth.loc["base",:].copy()
df_imax = df_worth.apply(lambda x:x-df_base,axis=1).idxmax()
df_max = 100.0 * (df_worth.apply(lambda x:x-df_base,axis=1).max() / df_base)
df_par = pd.DataFrame([df_imax,df_max],
                      index=["most important observation",
                             "percent increase when left out"])
df_par


# We see that observation ```or27c07_0``` is the most important for the water level forecasts (```or28c05_0``` and ```or28c05_1```), while observation ```or10c02_0``` is the most important for the surface water groundwater exchange forecasts (```sw_gw_0``` and ```sw_gw_1```). Also, observation ```or10c02_0```) results in a much greater increase in uncertainty for forecast ```sw_gw_0``` than it does for ```sw_gw_1```.

# ## Data worth--evaluating the potential value of new observations
# A potential water-level observation for each active model cell was also "carried" in the PEST control file.  This means we can run this same analysis to find the best next place to collect a new water level.  This takes a little longer because it is rerunning the schur's complement calculations many times, so this section can be skipped.

# ### Define the potential observation locations
# First we need a list of the observations with zero weight and that start with `"or"`--- (these are the synthetic proposed locations)

# In[ ]:


pst.observation_data.index = pst.observation_data.obsnme
new_obs_list = [n for n in pst.observation_data.obsnme.tolist() if n not in la.forecast_names \
                and n not in la.pst.nnz_obs_names]
print ("number of potential new obs locations:",len(new_obs_list))


# This takes a while since we are evaluating forecast uncertainty for each of the potential obs locations...

# In[ ]:


from datetime import datetime
start = datetime.now()
df_worth_new_0= la.get_added_obs_importance(base_obslist=la.pst.nnz_obs_names,
                            obslist_dict=new_obs_list,reset_zero_weight=1.0)
print("took:",datetime.now() - start)


# In[ ]:


df_worth_new_0.head()


# Similar to the value of existing data, these results are specific to the forecast of interest. However, when adding potential new observation data, we are looking at how uncertainty will _decrease_ if a proposed observation is _added_  to the 12 water level observations already being used for calibration(this is opposite of looking for the _increase_ in forecast uncertainty if an existing observation is _removed_). 
# 
# For each forecast, we can first determine which proposed new observation is most valuable.

# ### Make a function to postprocess the new data worth

# In[ ]:


def postproc_newworth(df_worth_new):
    # a little processing of df_worth
    df_new_base = df_worth_new.loc["base",:].copy()
    df_new_imax = df_worth_new.apply(lambda x:df_base-x,axis=1).idxmax()
    df_new_worth = 100.0 * (df_worth_new.apply(lambda x:df_base-x,axis=1) /\
                            df_new_base)
    df_new_max = df_new_worth.max()
    df_par_new = pd.DataFrame([df_new_imax,df_new_max],
                              index=["most important observation",
                                     "percent decrease when added"])
    df_par_new

    df_new_base1 = df_worth_new.loc["base",:].copy()
    df_new_imax1 = df_worth_new.apply(lambda x:df_new_base1-x,axis=1).\
                                      idxmax()
    df_new_worth1 = 100.0 * (df_worth_new.apply(
            lambda x:df_new_base1-x,axis=1) / df_new_base1)

    df_new_worth_plot1 = df_new_worth1[df_new_worth1.index != 'base'].copy()
    df_new_worth_plot1.loc[:,'names'] = df_new_worth_plot1.index
    names = df_new_worth_plot1.names
    df_new_worth_plot1.loc[:,"i"] = names.apply(lambda x: int(x[2:4]))
    df_new_worth_plot1.loc[:,"j"] = names.apply(lambda x: int(x[5:7]))
    df_new_worth_plot1.loc[:,'SP'] = names.apply(lambda x: int(x[-1]))
    df_new_worth_plot1.head()
    return df_new_worth_plot1, df_par_new


# In[ ]:


df_new_worth_plot_0, df_par_new_0 = postproc_newworth(df_worth_new_0)
df_par_new_0


# ### Make a function that can display data worth for added observations
# 

# In[ ]:


def plot_added_importance(df_worth_plot, ml, forecast_name=None, 
                          newlox = None,figsize=(20,15)):

    vmax = df_worth_plot[forecast_name].max()
    
    #fig = plt.figure(figsize=(20,15))
    fig = plt.figure(figsize=figsize)
    axlist = []
    # if new locations provided, plot them with their numbers
    if newlox:
        currx = []
        curry = []
        for i,clox in enumerate(newlox):
            crow = int(clox[2:4])
            ccol = int(clox[5:7])
            currx.append(ml.modelgrid.xcellcenters[crow,ccol])
            curry.append(ml.modelgrid.ycellcenters[crow,ccol])

    
    for SP in range(1):
        
        unc_array = np.zeros_like(ml.upw.hk[0].array) - 1
        df_worth_csp = df_worth_plot.groupby('SP').get_group(SP)
        for i,j,unc in zip(df_worth_csp.i,df_worth_csp.j,
                           df_worth_csp[forecast_name]):
            unc_array[i,j] = unc 
        unc_array[unc_array == -1] = np.nan
        axlist.append(plt.subplot(111,aspect="equal"))
#         cb = axlist[-1].imshow(unc_array,interpolation="nearest",
#                                alpha=0.5,extent=ml.modelgrid.extent, 
#                                vmin=0, vmax=vmax)
        ml.riv.stress_period_data.plot(axes=[axlist[-1]])

        # plot obs locations
        obs = pd.read_csv(os.path.join("Freyberg","misc","obs_rowcol.dat"),
                          delim_whitespace=True)
        obs_x = [ml.modelgrid.xcellcenters[r-1,c-1] for r,c \
                 in obs.loc[:,["row","col"]].values]
        obs_y = [ml.modelgrid.ycellcenters[r-1,c-1] for r,c \
                 in obs.loc[:,["row","col"]].values]
        axlist[-1].scatter(obs_x,obs_y,marker='d')

        # add the heads
        headsp = int(forecast_name[-1])
        kstpkper = (0,headsp)
        fname = os.path.join(ml.model_ws,'freyberg.hds')
        hdobj = flopy.utils.HeadFile(fname)
        head = hdobj.get_data(kstpkper=kstpkper)
        levels = np.arange(10, 30, .5)
        modelmap = flopy.plot.PlotMapView(model=ml)
        cb = modelmap.plot_array(unc_array, alpha=0.5, vmin=0, vmax=vmax)
        contour_set = modelmap.contour_array(head, masked_values=[999.], 
                                             levels=levels,axes=axlist[-1])
        if SP==0:
            plt.colorbar(cb,label="percent uncertainty reduction")

        # plot the pumping wells
        ml.wel.stress_period_data.plot(axes=axlist[-1],color='k')
        
        # add discharge vectors
        fname = os.path.join(ml.model_ws, 'freyberg.cbc')
        cbb = flopy.utils.CellBudgetFile(fname)
        frf = cbb.get_data(kstpkper=kstpkper, text='FLOW RIGHT FACE')[0]
        fff = cbb.get_data(kstpkper=kstpkper, text='FLOW FRONT FACE')[0]
        #quiver = modelmap.plot_discharge(frf, fff, head=head, axes=axlist[-1])
        linecollection = modelmap.plot_grid(axes=axlist[-1])

        if newlox:
            for i,(cx,cy,cobs) in enumerate(zip(currx, curry, newlox)):
                csp = int(cobs[-1])
                if csp == SP:
                    axlist[-1].plot(cx, cy, 'rd', mfc=None, ms=18, alpha=0.8)
                    axlist[-1].text(cx-50,cy-50,i, size='medium')
                
        # finally, plot the location of the forecast if possible
        if forecast_name.startswith('or'):
            i = int(forecast_name[2:4])
            j = int(forecast_name[5:7])
            forecast_x = ml.modelgrid.xcellcenters[i,j]
            forecast_y = ml.modelgrid.ycellcenters[i,j]
            axlist[-1].scatter(forecast_x, forecast_y, marker='o', s=600, 
                               alpha=0.5)
            axlist[-1].scatter(forecast_x, forecast_y, marker='x', s=600)

        plt.title('Added Data Worth for {0}'.format(forecast_name))
    return fig


# ### We can look at the results for each forecast and for each stress period

# In[ ]:


fig0 = plot_added_importance(df_new_worth_plot_0, ml, 'or28c05_0')


# In[ ]:


fig1 = plot_added_importance(df_new_worth_plot_0, ml, 'or28c05_1')


# In[ ]:


fig2 = plot_added_importance(df_new_worth_plot_0, ml, 'sw_gw_0')


# In[ ]:


fig3 = plot_added_importance(df_new_worth_plot_0, ml, 'sw_gw_1')

