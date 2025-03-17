#!/usr/bin/env python
# coding: utf-8

# # 03: Loading and visualizing groundwater models
# 
# This exercise, we will load an existing model into Flopy, run the model and then use [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/) and [numpy](https://www.numpy.org/) to look at the results and compare them to observed data. We will also export model input and output to shapefiles and rasters.
# 
# #### Required executables
# * MODFLOW-6; available here: https://github.com/MODFLOW-USGS/executables
# 
# #### Operations
# * reading tabular data from a file or url using the powerful [pandas.read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) method
# * getting `pandas.DataFrame`s of Hydmod, SFR, and global mass balance output
# * converting model times to real date-times to allow plotting against other temporally-referenced data
# * quickly subsetting data by category, attribute values, times, index position, etc.
# * computing quantiles and other basic statistics
# * making plots using `matplotlib` and the built-in hooks to it in `pandas`
# 
# #### The Pleasant Lake example
# The example model is a simplified version of the MODFLOW-6 model published by Fienen et al (2022, 2021; Figure 1), who used a multi-scale modeling approach to evaluate the effects of agricultural groundwater abstraction on the ecology of Pleasant Lake in central Wisconsin, USA. The original report and model files are available at the links below.
# 
# ##### Example model details:
# 
# * Transient MODFLOW-6 simulation with monthly stress periods for calendar year 2012
# * units of meters and days
# * 4 layers; 200 meter uniform grid spacing
#     * layers 1-3 represent surficial deposits
#     * layer 4 represents Paleozoic bedrock (sandstone)
# * Transient specified head perimeter boundary (CHD package) from a regional model solution
# * Recharge specified with RCHa
# * Streams specified with SFR
# * Pleasant Lake simulated with the Lake Package
# * Head observations specified with the OBS utility

# In[ ]:


from pathlib import Path
import flopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


# make an output folder
output_folder = Path('03-output')
output_folder.mkdir(exist_ok=True)


# ### load a preexisting MODFLOW 6 model
# Because this is MODFLOW 6, we need to load the simulation first, and then get the model.
# 
# **Note:** To avoid loading certain packages (that may be too slow) use the ``load_only`` argument to specify the packages that should be loaded.   
# e.g. ``load_only=['dis']``

# In[ ]:


get_ipython().run_cell_magic('capture', '', "sim_ws = Path('../data/pleasant-lake/')\nsim = flopy.mf6.MFSimulation.load('pleasant', sim_ws=str(sim_ws), exe_name='mf6',\n                                  #load_only=['dis']\n                         )\nsim.model_names\n")


# In[ ]:


m = sim.get_model('pleasant')


# ## Visualizing the model
# 
# First let's check that the model grid is correctly located. It is, in this case, because the model has the origin and rotation specified in the DIS package.

# In[ ]:


m.modelgrid


# In[ ]:


m.get_package_list()


# However, in order to write shapefiles with a ``.prj`` file that specifies the coordinate references system (CRS), we need to assign one to the grid (there currently is no CRS input for MODFLOW 6). We can do this by simply specifying an [EPSG code](https://epsg.io/) to the ``epsg`` attribute (in this case 3070 for Wisconsin Transverse Mercator).

# In[ ]:


m.modelgrid.crs = 3070


# ### On a map
# We can plot the model in the CRS using the ``PlotMapView`` object. More examples in the Flopy demo here (for unstructured grids too!): https://github.com/modflowpy/flopy/blob/develop/examples/Notebooks/flopy3.3_PlotMapView.ipynb

# In[ ]:


fig, ax = plt.subplots(figsize=(6, 6))
pmv = flopy.plot.PlotMapView(m, ax=ax)
lc = pmv.plot_grid()
pmv.plot_bc("WEL", plotAll=True)
pmv.plot_bc("LAK", plotAll=True)
pmv.plot_bc("SFR", plotAll=True)
pmv.plot_bc("CHD", plotAll=True)
ax.set_xlabel(f'{m.modelgrid.units.capitalize()} easting, {m.modelgrid.crs.name}')
ax.set_ylabel(f'{m.modelgrid.units.capitalize()} northing, {m.modelgrid.crs.name}')


# In[ ]:


fig, ax = plt.subplots(figsize=(6, 6))
pmv = flopy.plot.PlotMapView(m, ax=ax)
lc = pmv.plot_grid()
top = pmv.plot_array(m.dis.top.array)
ax.set_xlabel(f'{m.modelgrid.units.capitalize()} easting, {m.modelgrid.crs.name}')
ax.set_ylabel(f'{m.modelgrid.units.capitalize()} northing, {m.modelgrid.crs.name}')


# ### Exporting the model grid to a shapefile

# In[ ]:


m.modelgrid.write_shapefile(str(output_folder / 'pleasant_grid.shp'))


# ### Making a cross section through the model
# 
# more examples in the Flopy demo here: https://github.com/modflowpy/flopy/blob/develop/examples/Notebooks/flopy3.3_PlotCrossSection.ipynb

# #### By row or column

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 3))
xs = flopy.plot.PlotCrossSection(model=m, line={"row": 30}, ax=ax)
lc = xs.plot_grid()
xs.plot_bc("LAK")
xs.plot_bc("SFR")
ax.set_xlabel(f'Distance, in {m.modelgrid.units.capitalize()}')
ax.set_ylabel(f'Elevation, in {m.modelgrid.units.capitalize()}')


# #### Along an arbitrary line
# (and in Geographic Coordinates)

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 3))
xs_line = [(552400, 393000), (552400 + 5000, 393000 - 4000)]
xs = flopy.plot.PlotCrossSection(model=m, 
                                 line={"line": xs_line}, ax=ax,
                                 geographic_coords=True)
lc = xs.plot_grid(zorder=4)

pc = xs.plot_array(m.npf.k.array)
fig.colorbar(pc, label='Hydraulic Conductivity, in m/day')


# #### What if we want to look at cross sections for each row or column?
# This code allows for every row or column to be visualized in cross section within the Jupyter Notebook session.

# In[ ]:


fig, ax = plt.subplots(figsize=(14, 5))
frames = m.modelgrid.shape[1] # set frames to number of rows

def update(i):
    ax.cla()
    xs = flopy.plot.PlotCrossSection(model=m, line={"row": i}, ax=ax)
    lc = xs.plot_grid()
    xs.plot_bc("LAK")
    xs.plot_bc("SFR")
    ax.set_title(f"row: {i}")
    ax.set_xlabel(f'Distance, in {m.modelgrid.units.capitalize()}')
    ax.set_ylabel(f'Elevation, in {m.modelgrid.units.capitalize()}')
    return

import matplotlib.animation as animation
ani = animation.FuncAnimation(fig=fig, func=update, frames=frames)
plt.close()

from IPython.display import HTML
HTML(ani.to_jshtml())


# ### An aside on working with model input as numpy arrays
# Every input to MODFOW is attached to a Flopy object (with the attribute name of the variable) as a numpy ``ndarray`` (for ndarray-type data) or a ``recarray`` for tabular or list-type data. For example, we can access the recharge array (4D-- nper x nlay x nrow x ncol) with:
# 
# ```
# m.rcha.recharge.array
# ```

# #### ``ndarray`` example: plot spatial average recharge by stress period
# To minimize extra typing, it often makes sense to reassign the numpy array to a new variable to work with it further.

# In[ ]:


rch_inches = m.rcha.recharge.array[:, 0, :, :].mean(axis=(1, 2)) * 12 * 30.4 / .3048 
fig, ax = plt.subplots()
ax.plot(rch_inches)
ax.axhline(rch_inches.mean(), c='C1')
ax.set_ylabel(f"Average recharge, in monthly inches")
ax.set_xlabel("Model stress period")


# In[ ]:


m.rcha.recharge.array[:, 0, :, :].sum(axis=(1, 2)) * 100**2


# #### Tabular data example: plot pumping by stress period
# Most tabular input for the 'basic' stress packages (Constant Head, Drain, General Head, RIV, WEL, etc) are accessible via a ``stress_period_data`` attribute.   
# * To access the data, we have to call another ``.data`` attribute, which gives us a dictionary of ``recarray``s by stress period.  
# * Any one of these can be converted to a ``pandas.DataFrame`` individually, or we can make a dataframe of all of them with a simple loop.

# In[ ]:


dfs = []
for kper, df in m.wel.stress_period_data.get_dataframe().items():
    df['per'] = kper
    dfs.append(df)
df = pd.concat(dfs)
df.head()


# Now we can sum by stress period, or plot individual wells across stress periods

# In[ ]:


df.groupby('per').sum()['q'].plot()
plt.title('Total pumpage by Stress period')
plt.ylabel('$m^3$/day')
plt.xlabel('Model stress period')


# ### Exercise: plot pumping for the well at cellid: 2, 24, 2
# About many gallons did this well pump in stress periods 1 through 12?

# In[ ]:


ax = df.groupby('boundname').get_group('pleasant_2-13-2').plot(x='per')
ax.set_ylabel('$m^3$/day')


# #### Solution:
# 
# 1) get the pumping rates by stress period

# In[ ]:


rates = df.groupby('boundname').get_group('pleasant_2-13-2')[1:]
rates


# 2) Google "python get days in month" or similar (simply using 30.4 would be fine too for an approximate answer)

# In[ ]:


import calendar
rates['days'] = [calendar.monthrange(2012,m)[1] for m in rates['per']]


# 3) multiply the days by the daily pumping rate to get the totals; convert units and sum

# In[ ]:


rates


# In[ ]:


rates['gallons'] = rates['q'] * rates['days'] * 264.172
print(f"{rates['gallons'].sum():,}")


# ## Visualizing model output
# 
# #### Run the model first to get the output

# In[ ]:


sim.run_simulation()


# ### Getting the output
# With MODFLOW 6 models, we can get the output from the model object, without having to reference additional files. Sometimes though, it may be easier to read the file directly.
# 
# The head solution is reported for each layer. 
# * We can use the ``get_water_table`` utility to get a 2D surface of the water table position in each cell. 
# * To accurately portray the water table around the lake, we can read the lake stage from the observation file and assign it to the relevant cells in the water table array. 
# * Otherwise, depending on how the lake is constructed, the lake area would be shown as a nodata/no-flow area, or as the heads in the groundwater system below the lakebed.
# * In this case, we are getting the solution from the initial steady-state period

# In[ ]:


from flopy.utils.postprocessing import get_water_table
from mfexport.utils import get_water_table

hds = m.output.head().get_data(kstpkper=(0, 0))
wt = get_water_table(hds, nodata=-1e30)

# add the lake stage to the water table
lak_output = pd.read_csv(sim_ws / 'lake1.obs.csv')
stage = lak_output['STAGE'][0]
cnd = pd.DataFrame(m.lak.connectiondata.array)
k, i, j = zip(*cnd['cellid'])
wt[i, j] = stage
# add the SFR stage as well
sfr_stage = m.sfr.output.stage().get_data()[0, 0, :]
# get the SFR cell i, j locations
# by unpacking the cellid tuples in the packagedata
sfr_k, sfr_i, sfr_j = zip(*m.sfr.packagedata.array['cellid'])
wt[sfr_i, sfr_j] = sfr_stage

cbc = m.output.budget()
lak = cbc.get_data(text='lak', full3D=True)[0].sum(axis=0)
sfr = cbc.get_data(text='sfr', full3D=True)[0]


# ### Plot head and surface water flux results
# We can add output to a PlotMapView instance as arrays

# In[ ]:


levels=np.arange(280, 315, 2)

fig, ax = plt.subplots(figsize=(6, 6))
pmv = flopy.plot.PlotMapView(m, ax=ax)
ctr = pmv.contour_array(wt, levels=levels, 
                        linewidths=1, colors='b')
labels = pmv.ax.clabel(ctr, inline=True, fontsize=8, inline_spacing=1)
vmin, vmax = -100, 100
im = pmv.plot_array(lak, cmap='coolwarm', vmin=vmin, vmax=vmax)
im = pmv.plot_array(sfr.sum(axis=0), cmap='coolwarm', vmin=vmin, vmax=vmax)
cb = fig.colorbar(im, shrink=0.7, label='Leakage, in m$^3$/day')
ax.set_ylabel("Northing, WTM meters")
ax.set_xlabel("Easting, WTM meters")
ax.set_aspect(1)


# #### Zoom in on the lake

# In[ ]:


levels=np.arange(280, 315, 1)

fig, ax = plt.subplots(figsize=(6, 6))
pmv = flopy.plot.PlotMapView(m, ax=ax, extent=(554500, 557500, 388500, 392000))
ctr = pmv.contour_array(wt, levels=levels, 
                        linewidths=1, colors='b')
labels = pmv.ax.clabel(ctr, inline=True, fontsize=8, inline_spacing=1)
vmin, vmax = -100, 100
im = pmv.plot_array(lak, cmap='coolwarm', vmin=vmin, vmax=vmax)
im = pmv.plot_array(sfr.sum(axis=0), cmap='coolwarm', vmin=vmin, vmax=vmax)
cb = fig.colorbar(im, shrink=0.7, label='Leakage, in m$^3$/day')
ax.set_ylabel("Northing, WTM meters")
ax.set_xlabel("Easting, WTM meters")
ax.set_aspect(1)


# ### Exporting Rasters
# We can use the ``export_array`` utility to make a GeoTIFF of any 2D array on a structured grid. For example, make a raster of the simulated water table.

# In[ ]:


from flopy.export.utils import export_array

export_array(m.modelgrid, str(output_folder / 'water_table.tif'), wt)


# ### Exercise: evaluating overpressurization
# A common issue with groundwater flow models is overpressurization- where heads above the land surface are simulated. Sometimes, these indicate natural wetlands that aren't explicitly simulated in the model, but other times they are a sign of unrealistic parameters. Use the information in this lesson to answer the following questions:
# 
# 1) Does this model solution have any overpressiuzation? If so, where? Is it appropriate?
# 
# 2) What is the maximum value of overpressurization?
# 
# 3) What is the maximum depth to water simulated? Where are the greatest depths to water? Do they look appropriate?

# #### Solution
# 
# 1) Make a numpy array of overpressurization and get the max and min values

# In[ ]:


op = wt - m.dis.top.array
op.max(), op.min()


# 2) Plot it

# In[ ]:


plt.imshow(op); plt.colorbar()


# 3) Export a raster of overpressurization so we can compare it against air photos (or mapped wetlands if we have them!)

# In[ ]:


export_array(m.modelgrid, str(output_folder / 'op.tif'), op)


# The highest levels of overpressurization correspond to Pleasant Lake, where the model top represents the lake bottom. Other areas of *OP* appear to correspond to lakes or wetlands, especially the spring complex south of Pleasant Lake, where Tagatz Creek originates.
# 
# The greatest depths to water correspond to a topographic high in the southwest part of the model domain. A cross section through the area confirms that it is a bedrock high that rises more than 50 meters above the surrounding topography, so a depth to water of 76 meters in this area seems reasonable.

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 3))
xs = flopy.plot.PlotCrossSection(model=m, line={"row": 62}, ax=ax)
lc = xs.plot_grid()


# ### Plot a cross section of the head solution with the water table
# We can also view output in cross section. In this case, ``PlotMatView`` plots the head solution where the model is saturated. We can add the water table we created above that includes the lake surface.

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 5))
xs_line = [(552400, 393000), (552400 + 5000, 393000 - 4000)]
xs = flopy.plot.PlotCrossSection(model=m, 
                                 line={"line": xs_line}, 
                                 #line={"row": 32}, 
                                 ax=ax,
                                 geographic_coords=True)
lc = xs.plot_grid()
pc = xs.plot_array(hds, head=hds, alpha=0.5, masked_values=[1e30])
ctr = xs.contour_array(hds, head=hds, levels=levels, colors="b", masked_values=[1e30])
surf = xs.plot_surface(wt, masked_values=[1e30], color="blue", lw=2)

labels = pmv.ax.clabel(
    ctr, inline=True, 
    fontsize=8, inline_spacing=5)


# ## Getting observation output two ways
# In this model, head "observations" were specified at various locations using the MODFLOW 6 Observation Utility. MODFLOW reports simulated values of head at these locations, which can then be compared with equivalent field observations for model parameter estimation.
# 
# Earlier we obtained a DataFrame of Lake Package observation output for pleasant lake by reading in 'lake1.obs.csv' with pandas.
# We can read the head observation output with pandas too.

# 

# 

# In[ ]:


headobs = pd.read_csv(sim_ws / 'pleasant.head.obs')
headobs.head()


# Head observations can also be accessed via the ``.output`` attribute for their respective package. First we have to find the name associated with that package though. We can get this by calling ``get_package_list()``. Looks like ``"OBS_3"`` is it (since the only `OBS` packages in the model is for heads).

# In[ ]:


m.get_package_list()


# Next let's query the available output methods:

# In[ ]:


m.obs_3.output.methods()


# Now get the output. It comes back as a Numpy recarray be default, but we can easily cast it to a DataFrame.

# In[ ]:


pd.DataFrame(m.obs_3.output.obs().get_data())


# ### Using boundnames to define observations
# In MODFLOW 6, we can use boundnames to create observations for groups of cells. For example, in this model, each head value specified in the Constant Head Package has a ``boundname`` of east, west, north or south, to indicate the side of the model perimeter it's on. 
# 
# Example of boundnames specified in an external input file for the CHD Package:

# In[ ]:


pd.read_csv(sim_ws / 'external/chd_001.dat', delim_whitespace=True)


# The Constant Head Observation Utility input is then set up like so:
# ```
# BEGIN options
# END options
# 
# BEGIN continuous  FILEOUT  pleasant.chd.obs.output.csv
# # obsname obstype ID
#   east  chd  east
#   west  chd  west
#   north  chd  north
#   south  chd  south
# END continuous  FILEOUT  pleasant.chd.obs.output.csv
# ```
# 
# The resulting observation output (net flow across the boundary faces in model units of cubic meters per day) can be found in ``pleasant.chd.obs.output.csv``:

# In[ ]:


df = pd.read_csv(sim_ws / 'pleasant.chd.obs.output.csv')
df.index = df['time']
df.head()


# ## Plotting global mass balance from the listing file
# The ``Mf6ListBudget`` and ``MfListBudget`` (for earlier MODFLOW versions) utilities can assemble the global mass balance output (printed in the Listing file) into a DataFrame. A ``start_datetime`` can be added to convert the MODFLOW time to actual dates.
# 
# **Note:** The ``start_datetime`` functionality is unaware of steady-state periods, so if we put in the actual model start date of 2012-01-01, the 1-day initial steady-state will be included, resulting in the stress periods being offset by one day. Also note that the dates here represent the *end* of each stress period.

# In[ ]:


from flopy.utils import Mf6ListBudget 


# In[ ]:


mfl = Mf6ListBudget(sim_ws / 'pleasant.list')
flux, vol = mfl.get_dataframes(start_datetime='2011-12-30')
flux.head()


# In[ ]:


flux.columns


# ### Check the model mass balance error

# In[ ]:


flux['PERCENT_DISCREPANCY'].plot()
ax.set_ylabel('Percent mass balance error')


# ### Make a stacked bar plot of the global mass balance
# 
# Note: This works best if the in and out columns are aligned, such that ``STO-SY_IN`` and ``STO-SY_OUT`` are both colored orange, for example.

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 5))
in_cols = ['STO-SS_IN', 'STO-SY_IN', 'WEL_IN', 'RCHA_IN', 'CHD_IN', 'SFR_IN', 'LAK_IN']
out_cols = [c.replace('_IN', '_OUT') for c in in_cols]
flux[in_cols].plot.bar(stacked=True, ax=ax)
(-flux[out_cols]).plot.bar(stacked=True, ax=ax)
ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
ax.axhline(0, lw=0.5, c='k')
ax.set_ylabel('Simulated Flux, in $m^3/d$')


# ### References
# 
# Fienen, M. N., Haserodt, M. J., Leaf, A. T., and Westenbroek, S. M. (2022). Simulation of regional groundwater flow and groundwater/lake interactions in the central Sands, Wisconsin. U.S. Geological Survey Scientific Investigations Report 2022-5046. doi:10.3133/sir20225046
# 
# Fienen, M. N., Haserodt, M. J., and Leaf, A. T. (2021). MODFLOW models used to simulate groundwater flow in the Wisconsin Central Sands Study Area, 2012-2018. New York: U.S. Geological Survey Data Release. doi:10.5066/P9BVFSGJ

# In[ ]:




