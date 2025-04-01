#!/usr/bin/env python
# coding: utf-8

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


# ## Model background
# Here is an example based on the model of Freyberg, 1988.  The synthetic model is a 2-dimensional MODFLOW model with 1 layer,  40 rows, and 20 columns.  The model has 2 stress periods: an initial steady-state stress period used for calibration, and a 5-year transient stress period.  The calibration period uses the recharge and well flux of Freyberg, 1988; the last stress period use 25% less recharge and 25% more pumping.
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
fig = plt.figure(figsize=(10,10))
ax = plt.subplot(111,aspect="equal")
ml.upw.hk.plot(axes=ax,colorbar="K m/d",alpha=0.3)
ml.wel.plot(axes=ax)  # flopy possibly now only plots BCs in black
ml.riv.plot(axes=ax)

# plot obs locations
obs = pd.read_csv(os.path.join("Freyberg","misc","obs_rowcol.dat"),
                  delim_whitespace=True)
obs_x = [ml.modelgrid.xcellcenters[r-1,c-1] for r,c in obs.loc[:,["row","col"]].values]
obs_y = [ml.modelgrid.ycellcenters[r-1,c-1] for r,c in obs.loc[:,["row","col"]].values]
ax.scatter(obs_x,obs_y,marker='.',label="obs" )

#plot names on the pumping well locations
wel_data = ml.wel.stress_period_data[0]
wel_x = ml.modelgrid.xcellcenters[wel_data["i"],wel_data["j"]]
wel_y = ml.modelgrid.ycellcenters[wel_data["i"],wel_data["j"]]
for i,(x,y) in enumerate(zip(wel_x,wel_y)):
    ax.text(x,y,"{0}  ".format(i+1),ha="right",va="center", font=dict(size=15), color='r')

ax.set_ylabel("y")
ax.set_xlabel("x")

ax.add_patch(rect((0,0),0,0,label="well",ec="none",fc="r"))
ax.add_patch(rect((0,0),0,0,label="river",ec="none",fc="g"))

ax.legend(bbox_to_anchor=(1.5,1.0),frameon=False)
plt.savefig("domain.pdf")


# ## Using `pyemu`

# In[ ]:


import pyemu


# First create a linear_analysis object.  We will use `ErrVar`  derived type, which replicates the behavior of the `PREDVAR` suite of PEST as well as `ident_par` utility.  We pass it the name of the jacobian matrix file.  Since we don't pass an explicit argument for `parcov` or `obscov`, `pyemu` attempts to build them from the parameter bounds and observation weights in a pest control file (.pst) with the same base case name as the jacobian.  Since we are interested in forecast uncertainty as well as parameter uncertainty, we also pass the names of the forecast sensitivity vectors we are interested in, which are stored in the jacobian as well.  Note that the `forecasts` argument can be a mixed list of observation names, other jacobian files or PEST-compatible ASCII matrix files.  Remember you can pass a filename to the `verbose` argument to write log file.
# 
# Since most groundwater model history-matching analyses focus on adjusting heterogeneous hydraulic properties and not boundary condition elements, let's identify the well flux and recharge parameters as `omitted` in the error variance analysis.  We can conceptually think of this action as excluding these parameters from the history-matching process. Later we will explicitly calculate the penalty for not adjusting these parameters.

# In[ ]:


# get the list of forecast names from the pest++ argument
# in the pest control file
jco = os.path.join("Freyberg","freyberg.jcb")
pst = pyemu.Pst(jco.replace("jcb","pst"))
omitted = [pname for pname in pst.par_names if \
           pname.startswith("wf") or pname.startswith("rch")]
forecasts = pst.pestpp_options["forecasts"].split(',')
la = pyemu.ErrVar(jco=jco,verbose="errvar_freyberg.log",
                  omitted_parameters=omitted)
print("observations, parameters found in jacobian:",la.jco.shape)


# # Parameter identifiability
# The `errvar` derived type exposes a method to get a `pandas` dataframe of parameter identifiability information.  Recall that parameter identifiability is expressed as $d_i = \Sigma(\mathbf{V}_{1i})^2$, where $d_i$ is the parameter identifiability, which ranges from 0 (not identified by the data) to 1 (full identified by the data), and $\mathbf{V}_1$ are the right singular vectors corresponding to non-(numerically) zero singular values.  First let's look at the singular spectrum of $\mathbf{Q}^{\frac{1}{2}}\mathbf{J}$, where $\mathbf{Q}$ is the cofactor matrix and $\mathbf{J}$ is the jacobian:

# In[ ]:


s = la.qhalfx.s


# In[ ]:


import pylab as plt
figure = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
ax.plot(s.x)
ax.set_title("singular spectrum")
ax.set_ylabel("power")
ax.set_xlabel("singular value")
ax.set_xlim(0,20)
plt.show()


# We see that the singular spectrum decays rapidly (not uncommon) and that we can really only support about 12 right singular vectors even though we have 700+ parameters in the inverse problem.  
# 
# Let's get the identifiability dataframe at 12 singular vectors:

# In[ ]:


# the method is passed the number of singular vectors to include in V_1
ident_df = la.get_identifiability_dataframe(12) 
ident_df.sort_values(by="ident",ascending=False).iloc[0:10]


# Plot the indentifiability:

# In[ ]:


ax = ident_df.sort_values(by="ident",ascending=False).iloc[0:20].\
     loc[:,"ident"].plot(kind="bar",figsize=(10,10))
ax.set_ylabel("identifiability")


# # Forecast error variance 
# 
# Now let's explore the error variance of the forecasts we are interested in.  We will use an extended version of the forecast error variance equation:   
# 
# $\sigma_{s - \hat{s}}^2 = \underbrace{\textbf{y}_i^T({\bf{I}} - {\textbf{R}})\boldsymbol{\Sigma}_{{\boldsymbol{\theta}}_i}({\textbf{I}} - {\textbf{R}})^T\textbf{y}_i}_{1} + \underbrace{{\textbf{y}}_i^T{\bf{G}}\boldsymbol{\Sigma}_{\mathbf{\epsilon}}{\textbf{G}}^T{\textbf{y}}_i}_{2} + \underbrace{{\bf{p}}\boldsymbol{\Sigma}_{{\boldsymbol{\theta}}_o}{\bf{p}}^T}_{3}$
# 
# Where term 1 is the null-space contribution, term 2 is the solution space contribution and term 3 is the model error term (the penalty for not adjusting uncertain parameters).  Remember the well flux and recharge parameters that we marked as omitted?  The consequences of that action can now be explicitly evaluated.  See Moore and Doherty (2005) and White and other (2014) for more explanation of these terms.  Note that if you don't have any `omitted_parameters`, the only terms 1 and 2 contribute to error variance
# 
# First we need to create a list (or numpy ndarray) of the singular values we want to test.  Since we have 12 data, we only need to test up to $13$ singular values because that is where the action is:

# In[ ]:


sing_vals = np.arange(13)


# The `ErrVar` derived type exposes a method to get a multi-index pandas dataframe with each of the terms of the error variance equation:

# In[ ]:


errvar_df = la.get_errvar_dataframe(sing_vals)
errvar_df.iloc[0:10]


# In[ ]:


errvar_df[["first"]].to_latex("sw_gw_0.tex")


# plot the error variance components for each forecast:

# In[ ]:


colors = {"first": 'g', "second": 'b', "third": 'c'}
max_idx = 19
idx = sing_vals[:max_idx]
for ipred, pred in enumerate(forecasts):
    pred = pred.lower()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    ax.set_title(pred)
    first = errvar_df[("first", pred)][:max_idx]
    second = errvar_df[("second", pred)][:max_idx]
    third = errvar_df[("third", pred)][:max_idx]
    ax.bar(idx, first, width=1.0, edgecolor="none", 
           facecolor=colors["first"], label="first",bottom=0.0)
    ax.bar(idx, second, width=1.0, edgecolor="none", 
           facecolor=colors["second"], label="second", bottom=first)
    ax.bar(idx, third, width=1.0, edgecolor="none", 
           facecolor=colors["third"], label="third", 
           bottom=second+first)
    ax.set_xlim(-1,max_idx+1)
    ax.set_xticks(idx+0.5)
    ax.set_xticklabels(idx)
    #if ipred == 2:
    ax.set_xlabel("singular value")
    ax.set_ylabel("error variance")
    ax.legend(loc="upper right")
plt.show()


# Here we see the trade off between getting a good fit to push down the null-space (1st) term and the penalty for overfitting (the rise of the solution space (2nd) term)).  The sum of the first two terms in the "apparent" error variance (e.g. the uncertainty that standard analyses would yield) without considering the contribution from the omitted parameters.  You can verify this by checking prior uncertainty from the Schur's complement notebook against the zero singular value result using only terms 1 and 2. Note that the top of the green bar is the limit of traditional uncertainty/error variance analysis: accounting for parameter and observation
# 
# We also see the added penalty for not adjusting the well flux and recharge parameters.  For the water level at the end of the calibration period forecast (``or28c05_0``), the fact the we have left parameters out doesn't matter - the parameter compensation associated with fixing uncertain model inputs can be "calibrated out" beyond 2 singular values.  For the water level forecast during forecast period (``or28c05_1``), the penalty for fixed parameters persists -it s nearly constant over the range of singular values.  
# 
# For ``sw_gw_0``, the situation is much worse: not only are we greatly underestimating uncertainty by omitting parameters, worse, calibration increases the uncertainty for this forecast because the adjustable parameters are compensating for the omitted, uncertaint parameters in ways that are damanaging to the forecast. 
# 
# For the forecast period sw-gw exchange (``sw_gw_1``), calibration doesn't help or hurt - this forecast depend entirely on null space parameter components.  But treating the recharge and well pumpage as "fixed" (omitted) results in greatly underestimated uncertainty.     
# 
# Let's check the ```errvar``` results against the results from ```schur```. This is simple with ```pyemu```, we simply  cast the ```errvar``` type to a ```schur``` type:

# In[ ]:


schur = la.get(astype=pyemu.Schur)
schur_prior = schur.prior_forecast
schur_post = schur.posterior_forecast
print("{0:10s} {1:>12s} {2:>12s} {3:>12s} {4:>12s}"
      .format("forecast","errvar prior","errvar min",
              "schur prior", "schur post"))
for ipred, pred in enumerate(forecasts):
    first = errvar_df[("first", pred)][:max_idx]
    second = errvar_df[("second", pred)][:max_idx]  
    min_ev = np.min(first + second)
    prior_ev = first[0] + second[0]
    prior_sh = schur_prior[pred]
    post_sh = schur_post[pred]
    print("{0:12s} {1:12.6f} {2:12.6f} {3:12.6} {4:12.6f}"
          .format(pred,prior_ev,min_ev,prior_sh,post_sh))


# We see that the prior from ```schur``` class matches the two-term ```errvar``` result at zero singular values.  We also see, as expected, the posterior from ```schur``` is slightly lower than the two-term ```errvar``` result.  This shows us that the "apparent" uncertainty in these predictions, as found through application of Bayes equation, is being under estimated because if the ill effects of the omitted parameters.
