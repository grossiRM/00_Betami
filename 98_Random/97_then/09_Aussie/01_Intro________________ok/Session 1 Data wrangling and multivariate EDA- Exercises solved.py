#!/usr/bin/env python
# coding: utf-8

# # Session 1: Data wrangling and multivariate exploratory data analysis
# 
# This material is prepared for the Australian Water School by Luk Peeters (luk.peeters@csiro.au) and will be presented on June 3rd 2021.
# 
# The first session of the Python course focusses on data wrangling, importing and exporting various data sets and manipulating data within a Python environment. We'll illustrate this with hydrological time series data and hydrochemistry data sets. The next session will look in greater detail into time series analysis and visualisation. This session will use hydrochemistry data to showcase Python to do multivariate analysis and visualisation.
# 
# ## 0. preamble
# In the great Python slicing tradition, we start counting at 0. Before you can use any of the functionality of a package after you installed it, you need to load it. I prefer to have one code-block at the start of a notebook where I list all the packages I'm going to use. This is not absolutely necessary, but it makes it easier for others when sharing a notebook to quickly see which packages need to be installed.
# There are two way to load a package:
# 1. import *package* as *short name*
# 2. from *package* import *function*

# In[1]:


# preamble
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import getcwd
# import functions we'll need later
from scipy.stats import rankdata
from sklearn.manifold import TSNE
# allow interactive figures in notebook
get_ipython().run_line_magic('matplotlib', 'notebook')


# The codeblock above imports 3 packages we'll use a lot: numpy, pandas and matplotlib. To call any function in any of these packages, use the short name followed by a . and the name of the function. With the from *package* import *function* you can import a single function without prefix. The *getcwd* is a function of the os package that gives you the current working directory:

# In[2]:


print(getcwd())


# The last line in the preamble ensures that matplotlib figures are displayed interactively in a notebook:

# In[3]:


# create an array with 10 equally spaced points between 0 and 2pi
x = np.linspace(0, 2*np.pi, 25) 
# cosine of x
y = np.cos(x) 
# create an empty figure
fig, ax = plt.subplots() 
# plot x vs y, with blue line, with label for the legend
ax.plot(x,y,'-ob',label='Data') 
# title for the plot
ax.set_title('Random numbers') 
# x-label
ax.set_xlabel('X-axis') 
# y-label
ax.set_ylabel('Y-axis') 
# set x limits
ax.set_xlim(x.min(),x.max()) 
# set y limits
ax.set_ylim(-1.05,1.05) 
# add grid
ax.grid() 
# add legend
ax.legend() 


# The plot display is interactive, i.e. you can zoom in and out, resize, pan and save the plot. The codeblock shows the basic elements to make a simple plot with matplotlib. A great resource is the [matplotlib cheatsheet](https://github.com/matplotlib/cheatsheets#cheatsheets). I often check the [matplotlib gallery](https://matplotlib.org/stable/gallery/index.html) for inspiration to visualise data. We'll use this basic template throughout the session and tweak it where needed.
# 
# ## 1. Importing and cleaning data - Streamflow time series
# Most hydrological data is still stored in spreadsheets, either in ASCII-text files, csv-files or excel spreadsheets.
# We've downloaded a csv file with streamflow discharge from Mt Barker in South Australia from the [Bureau of Meteorology website](http://www.bom.gov.au/waterdata/): **Q_A4260557.csv**. The file has over 40 years of daily data and is too large to open fully in excel.
# 
# We can open the file with numpy function [numpy.loadtxt](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html) and [numpy.genfromtxt](https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt) to create a numpy array:

# In[14]:


Q = np.loadtxt('Q_A4260557.csv')


# well, that didn't work - we need to use some more of the optional arguments

# In[16]:


Q = np.loadtxt('Q_A4260557.csv',skiprows=11, delimiter=',', usecols = 1)


# Still no luck. It seems like there is a missing value in the very last row. There are two ways around this; skip the last row or specify how to handle missing values. When you need more control on importing data, it is better to switch to `np.genfromtxt`:

# In[17]:


Q = np.genfromtxt('Q_A4260557.csv',
                  delimiter=',',
                  skip_header=11,
                  skip_footer=1,
                  usecols = 1)


# In[18]:


Q = np.genfromtxt('Q_A4260557.csv',
                  delimiter=',',
                  skip_header=11,
                  missing_values = '',
                  usecols = 1)


# To check the data we've loaded, we can make a quick plot:

# In[19]:


fig, ax = plt.subplots() # create an empty figure
ax.plot(Q,'-k')


# This is obviously not yet a decent hydrograph. We haven't imported the dates or the additional information on the quality of each data point. It is possible to do this directly in numpy, but it is much easier to do that with pandas [pd.read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html):

# In[4]:


Q = pd.read_csv('Q_A4260557.csv',
                header = 8, #which row to use for the headers
                index_col = 0, #which row to use as index
                parse_dates = True) # index is a data, parse it into a time series


# In[21]:


Q.head() #quick look at dataframe


# In[22]:


# quick and dirty plot
plt.figure()
Q['Value'].plot()


# This looks better, at least this is a hydrograph plot. As you zoom in on the plot, you'll see that the time axis updates as well. From the plot it is clear that there is a gap in the data in 2010, with negative values assigned to it.
# There might be information in the quality code column. The quality codes are:
# 
# | Code | Label | Description |
# |------|-------|:-------------|
# |10 | quality-A | The record set is the best available given the technologies, techniques and monitoring objectives at the time of classification|
# |90 | quality-B | The record set is compromised in its ability to truly represent the parameter |
# |110 | quality-C | The record set is an estimate |
# |140 | quality-E | The record set's ability to truly represent the monitored parameter is not known |
# |210 | quality-F | The record set is not of release quality or contains missing data |
# 
# We can use the [groupby](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html) function to quickly cross-tabulate the percentage of data points in each category:

# In[23]:


Q.groupby('Quality Code')['Value'].count()


# **Excercise: the dataset has a column with different codes used for the interpolation of data. Count the number of records for each class**

# In[5]:


Q.groupby('Interpolation Type')['Value'].count()


# To gain more insight, we can plot the hydrograph, color-coded by its quality label:

# In[24]:


# list the unique quality code values
codes = Q['Quality Code'].unique()
# create new figure
fig,ax = plt.subplots()
# for loop, looping through the values in list codes
for code in codes:
    # find the indices of the records for quality code 'code'
    inds = Q['Quality Code']==code 
    # plot the selected values
    ax.plot(Q.index[inds],Q['Value'][inds],'.',label=code) 
# add legend, outside for loop
l = ax.legend()
# set title, xlabel and ylabel
ax.set_title('Station A4260557')
ax.set_xlabel('Time')
ax.set_ylabel('Q (m3/s)')


# There are a couple of ways that we can clean this dataset:
# 1. Set all values below zero to NaN
# 2. Set all measurements with Quality Code 210 to NaN

# In[25]:


# set all values below zero to NaN
Q.loc[Q['Value']<0,'Value'] = np.nan
# set all values with quality code 210 to NaN
Q.loc[Q['Quality Code']==210,'Value'] = np.nan
# plot the result
fig,ax = plt.subplots()
ax.plot(Q.index,Q['Value'],'-k') 
ax.set_title('Station A4260557')
ax.set_xlabel('Time')
ax.set_ylabel('Q (m3/s)')


# Now that we've cleaned the data, we can do some analysis, like for instance making a flow duration curve. This code block uses a lot of functions associated with the pandas dataframe: [pd.DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)

# In[26]:


#create an array with values from 0 to 1, in increments of 0.01
percs = np.arange(0,1.01,0.01)
# calculate percentiles, but remove NaN first
discharge = Q['Value'].dropna().quantile(percs)
# create a new data frame
flowduration = pd.DataFrame(index=percs,columns=['Exceedance Probability','Discharge']) 
# reverse order of quantiles
flowduration['Exceedance Probability'] = 100*(1-percs)
flowduration['Discharge'] = discharge[-1::]
#plot with log y axis
fig,ax = plt.subplots()
ax.semilogy(flowduration['Exceedance Probability'],flowduration['Discharge'],'-k')
ax.grid()
ax.set_xlabel('Exceedance probability')
# use Latex formatting
ax.set_ylabel(u'Q ($m^3/s$)') 


# The code block above is pretty concise and gives you full control on calculating and plotting the flow duration curve. You might want to repeat this a lot. One way to do this without having to copy the code block over and over again is by creating a function:

# In[27]:


def flowduration(Q):
    #create an array with values from 0 to 1, in increments of 0.01
    percs = np.arange(0,1.01,0.01)
    # calculate percentiles, but remove NaN first
    discharge = Q['Value'].dropna().quantile(percs)
    # create a new data frame
    flowduration = pd.DataFrame(index=percs,columns=['Exceedance Probability','Discharge']) 
    # reverse order of quantiles
    flowduration['Exceedance Probability'] = 100*(1-percs)
    flowduration['Discharge'] = discharge[-1::]
    #plot with log y axis
    fig,ax = plt.subplots()
    ax.semilogy(flowduration['Exceedance Probability'],flowduration['Discharge'],'-k')
    ax.grid()
    ax.set_xlabel('Exceedance probability')
    # use Latex formatting
    ax.set_ylabel(u'Q ($m^3/s$)') 
    return(flowduration,fig)


# We'll use this function to illustrate how to save both the figure and the dataframe

# In[28]:


fd,fig = flowduration(Q)
# save figure as png
fig.savefig('Flowdurationcurve.png')
# save figure as csv file - index column by default has no label
fd.to_csv('Flowdurationcurve.csv', index_label='Percentile')


# ## 2 Intermezzo: inserting variables in strings
# A great feature of using scripts is to automate tedious tasks, like loading different files or labeling figures with values from your data-set. Automating such tasks often requires inserting values from variables or arrays into strings. This section gives a quick overview of a couple of different ways to achieve that. A detailed overview can be found on the website [PyFormat](https://pyformat.info/)
# 
# The 'old' method you'll still often find is based on the '%' operator, while the new method is based on the format function of a string.

# In[29]:


# create a string with the station name
name = 'A4260557'
# create a variable with a numeric value
Qmax = fd['Discharge'].max()


# In[30]:


# insert the station name in a file name
# old
filename = 'Flow_duration_%s.csv' % (name)
print(filename)


# In[31]:


# new, all version of Python
filename = 'Flow_duration_{}.csv'.format(name)
print(filename)


# In[32]:


# new, Python >=3.6
filename = f'Flow_duration_{name}.csv'
print(filename)


# When you're adding numbers to a string, you can specify the format

# In[33]:


# old
title = '%s: Qmax = %f m3/s' % (name,Qmax) # no formatting, just float
print(title)


# In[34]:


# new
title = '{}: Qmax = {:4.2f} m3/s'.format(name,Qmax) # 4 characters, 2 decimal places
print(title)


# In[35]:


# new, Python >=3.6
title = f'{name}:\nQmax = {Qmax:04.0f} m3/s' # 4 characters, 0 decimal places, padding with zeros, \n for new line
print(title)


# ## 3 Multivariate data: reading from excel
# Pandas can also read data directly from excel files, using [pd.read_excel](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html).
# The dataset we are using is a dataset of groundwater chemistry of South Australia
# 
# [Gray, David J. and Bardwell, Nicole (2016) Hydrogeochemistry of South Australia: Data Release: Accompanying Notes. CSIRO, Australia. EP156116 34p](https://data.csiro.au/collections/collection/CIcsiro:17862v1)
# 

# In[2]:


fname = 'CSH_SA.xlsx'
sheet = 'Data'
chem = pd.read_excel(fname,sheet)
chem


# The dataset has over 30.000 entries and 77 columns. Pandas has recognised the date column and converted these into dates. All the blanks in the dataset are converted to NaN, Not a Number.
# 
# The function pd.columns prints a list of all the columns in the dataframe:

# In[3]:


chem.columns


# There are a lot of missing values in this dataframe. The next codeblock summarizes the percentage of records with missing values for each variable.

# In[4]:


# number of records
nsample = len(chem)
# number of complete records
ncomplete = len(chem.dropna()) # dropna removes any row that contains at least one NaN
print('Number of records without missing values = {}'.format(ncomplete))


# This dataset has no single record with entries for all variables. The first 7 variables are meta-data. We want to check if there are any records that only have data for the metadata, not for the chemistry. We can use the dropna function again. The next line of code removes any rows that are all NaN for all columns except the first 7.

# In[5]:


chem.dropna(axis=0,how='all',subset=chem.columns[7::],inplace=True)
print('Number of records removed = {}'.format(nsample-len(chem)))


# The next step is to cross-tabulate the number of missing values for each variable

# In[6]:


chem_var_na = chem.isna().sum(axis=0).T
chem_var_na


# An easier way to inspect that dataset is to plot it as a bar chart:

# In[7]:


fig, ax = plt.subplots()
# modify the default figure size
fig.set_size_inches(7,10)
# create the ticks for a horizontal bar plot
ticks = np.arange(len(chem_var_na))
# bar plot of percent not NaN
ax.barh(ticks,
        100-100*(chem_var_na/float(nsample)))
# set labels for y-axis
ax.set_yticks(ticks)
ax.set_yticklabels(chem_var_na.index,fontdict={'fontsize':8})
# adjust y-axis limits
ax.set_ylim(-1,len(chem.columns))
# invert y-axis so meta-data is at the bottom
ax.invert_yaxis
# change x-axis limits
ax.set_xlim(0,100)
# set title
ax.set_title('GW chemistry SA:\n{} samples'.format(nsample))
# set x axis label
ax.set_xlabel('Percent samples with data')
# add grid
ax.grid()


# We can also make a plot of how the number of samples with missing values changes through time

# In[8]:


# same command as before, but now summing per row 
chem_sample_na = chem.isna().sum(axis=1)


# In[9]:


fig, ax = plt.subplots()
# time series plot, using transparency as many points are overlapping
ax.plot(chem['Date'],(len(chem.columns)-chem_sample_na),'.k',alpha=0.02)
ax.grid()
ax.set_xlabel('Year')
ax.set_ylabel('Number of variables measured')
ax.set_title('GW chemistry SA')
ax.set_ylim(bottom=0)


# **Excercise 2: recreate the plot above, but without counting the 7 meta-data variables**

# In[10]:


# recalculate missing values without counting the 7 meta-data variables (which are the 1st 7 variables)
chem_sample_na = chem[chem.columns[7::]].isna().sum(axis=1)
# use the same plotting command but change the y-value in the plot command
fig, ax = plt.subplots()
# time series plot, using transparency as many points are overlapping
ax.plot(chem['Date'],(len(chem.columns[7::])-chem_sample_na),'.k',alpha=0.02)
ax.grid()
ax.set_xlabel('Year')
ax.set_ylabel('Number of variables measured')
ax.set_title('GW chemistry SA')
ax.set_ylim(bottom=0)


# ## 4. Exploratory Data Analysis
# Exploratory data analysis is the process of visualising a dataset to formulate hypothesis of the processes that resulted in the data set.
# Histograms are a standard visualisation to understand the distribution of variables. Python makes it very easy to generate multi-panel figures, so that you can summarise a lot of information in a single composite figure. Below is a code block to make histograms of the major ions:

# In[11]:


majorions = ['HCO3_mgL', 'Na_mgL','K_mgL', 'Mg_mgL', 'Ca_mgL', 'Cl_mgL', 'SO4_mgL','NO3N_mgL']
abc = 'abcdefghijklmnopqrstuvwxyz'
plt.figure()
# enumerate iterates a list and gives both the index (i) and the value (ion)
for i,ion in enumerate(majorions):
    # use the index here to select a subplot (it is the only function in python that is not 0 based :-( )
    plt.subplot(3,3,i+1) 
    # use the value here to select a column in the data frame
    plt.hist(chem[ion].dropna())
    # use the index to select letter and value here as title
    plt.title('{}) {}'.format(abc[i],ion),loc='left',fontsize='small')
# optional command to optimise the layout and size of labels of a multipanel plot
plt.tight_layout()


# The histograms show that all data are very skewed. One quick way to deal with left-skewed data is to do a log transform first.

# In[12]:


plt.figure()
for i,ion in enumerate(majorions):
    plt.subplot(3,3,i+1)
    # 0 values lead to error in log10
    plt.hist(np.log10(chem[ion][chem[ion]>0].dropna()))
    plt.title('{}) Log10 {}'.format(abc[i],ion),loc='left',fontsize='small')
plt.tight_layout()


# This can be a misleading plot since all y-axis are different. It is possible to make all the y-axes the same, using `plt.subplots`. This however requires a rewrite of the `for` loop:

# In[13]:


fig,axs = plt.subplots(nrows=3,ncols=3,sharey='all')
# axs is a 2D array, which is difficult to iterate over. ravel() turns it into a 1D array
axs = axs.ravel()
# we've created 9 subplots for 8 variables, so we have to delete the last one
axs[-1].remove()
# now we can iterate by directly calling the axis in which we want the subplot
for i,ion in enumerate(majorions):
    axs[i].hist(np.log10(chem[ion][chem[ion]>0].dropna()))
    # when calling plt.title on an axis, use the prefix 'set_' (this works for most plt. commands)
    axs[i].set_title('{}) Log10 {}'.format(abc[i],ion),loc='left',fontsize='small')
fig.set_tight_layout(True)


# A more compact way to show distributions, especially if their scale is similar, is to use violinplots. Below is a violin plot for some of the minor ions:

# In[14]:


minorions = ['Li_mgL', 'Fe_mgL', 'Mn_mgL', 'Al_mgL', 'Cu_mgL', 'Zn_mgL', 'Pb_mgL', 'As_mgL', 'Cr_mgL','Cd_mgL', 'Ni_mgL']
fig,ax = plt.subplots()
d = [chem[a].dropna() for a in minorions]
ticks = np.arange(0,len(minorions))
ax.violinplot(d,positions=ticks,vert=False,showmedians=True)
ax.set_yticks(ticks)
ax.set_yticklabels(minorions)
ax.set_xscale('log')
ax.set_title('Minor ions')
ax.set_xlabel('Concentration (mg/L)')


# Exploratory Data Analysis is also looking for correlations in the dataset. We might look into how some of the variables correlate with pH or TDS.

# In[15]:


plt.figure()
for i,major in enumerate(majorions):
    plt.subplot(3,3,i+1)
    plt.loglog(chem['TDSc_mgL'],chem[major],'.k',alpha=0.01)
    plt.title(major)
    plt.xlabel('TDS (mg/L)')
plt.tight_layout()


# **Excercise: make the same plot, but with pH on the x-axis**

# In[16]:


# same as above, with column pH
# changed to semilogy as pH is already on a log scale
plt.figure()
for i,major in enumerate(majorions):
    plt.subplot(3,3,i+1)
    plt.semilogy(chem['pH'],chem[major],'.k',alpha=0.01)
    plt.title(major)
    plt.xlabel('pH')
plt.tight_layout()


# A more formal way of exploring correlations is through a correlation matrix.

# In[17]:


chemcorr = chem[chem.columns[7::]].corr()
chemcorr_sp = chem[chem.columns[7::]].corr('spearman')
chemcorr


# For large dataset, visualising this as a colored matrix works well

# In[18]:


fig,ax = plt.subplots(figsize=(10,8))
q = ax.pcolor(chemcorr_sp,
          vmin=-1,vmax=1,
          cmap='coolwarm')
ax.set_title('Correlation matrix')
cbar = plt.colorbar(q,shrink=0.5)
a = ax.set_xticks(np.arange(len(chem.columns[7::]))+.5)
b = ax.set_xticklabels(chem.columns[7::],rotation=90,fontsize='xx-small')
c = ax.set_yticks(np.arange(len(chem.columns[7::]))+.5)
d = ax.set_yticklabels(chem.columns[7::],fontsize='xx-small')


# ## 5. Bringing in raster data
# The package rasterio implements a lot of functionality importing and manipulating raster data. For this session however, we'll stick with numpy and pandas to showcase some general functionality on importing data and interacting with arrays.
# 
# The raster dataset we're using is a raster with estimated chloride deposition across Australia at a 0.05 degree grid.
# 
# [Davies, Phil; Crosbie, Russell. Mapping the spatial distribution of chloride deposition across Australia. Journal of Hydrology. 2018; 561:76-88. https://doi.org/10.1016/j.jhydrol.2018.03.051](https://data.csiro.au/collections/collection/CIcsiro:11068v4)
# 
# The chloride deposition rate in kg/ha/year (D) can be used together with the chloride concentration in groundwater (Cl) to estimate the recharge to groundwater (R) with the following equation:
# 
# R=100(D/Cl)
# 
# The raster is an ASCII grid file with following structure
# 
#     ncols         4
#     nrows         6
#     xllcorner     0.0
#     yllcorner     0.0
#     cellsize      50.0
#     NODATA_value  -9999
#     -9999 -9999 5 2
#     -9999 20 100 36
#     3 8 35 10
#     32 42 50 6
#     88 75 27 9
#     13 5 1 -9999
#     
# We'll read the grid in with `np.loadtxt`. For the metadata (the 1st 6 line), we'll use the more generic `open` command together with `readline`:

# In[19]:


# Chloride deposition grid
fname = 'cl_deposition_final.txt'
# read data in numpy array, skip metadata
cl_depo = np.loadtxt(fname,skiprows=6)
# create an empty dictionary for the meta-data
cl_depo_meta = {}
# open the grid file
with open(fname) as f:
    # while loop, as long as the dict has less than 6 entries
    while len(cl_depo_meta)<6:
        # readline reads 1 line in the file, the split function splits in a list based on white space
        tmp = f.readline().split()
        # first item in list is the name for the meta data, second is the value
        cl_depo_meta[tmp[0]] = float(tmp[1]) # use float to convert string to float


# In[20]:


cl_depo_meta


# In[21]:


cl_depo_meta['ncols']


# Replace the NODATA_values with NaN and plot the map with `imshow`

# In[22]:


cl_depo[cl_depo<(cl_depo_meta['NODATA_value']+1)] = np.nan
fig,ax = plt.subplots()
# modify the default figure size
fig.set_size_inches(10,10)
p = ax.imshow(cl_depo,
          origin = 'upper',
          cmap = 'Reds',
          extent = (cl_depo_meta['xllcorner'],
                   cl_depo_meta['xllcorner']+(cl_depo_meta['ncols']*cl_depo_meta['cellsize']),
                   cl_depo_meta['yllcorner'],
                   cl_depo_meta['yllcorner']+(cl_depo_meta['nrows']*cl_depo_meta['cellsize'])))
cbar = plt.colorbar(p)
cbar.set_label('kg/ha/year')
ax.set_title('Chloride deposition')


# We can add the points from the chemistry dataset to the map and zoom in on the chemistry dataset

# In[23]:


# zoom in to SA data
ax.set_xlim(chem['Long'].min(),chem['Long'].max())
ax.set_ylim(chem['Lat'].min(),chem['Lat'].max())
# use scatter to plot point with color based on log10 Cl
s = ax.scatter(chem['Long'],chem['Lat'],.2,np.log10(chem['Cl_mgL']),cmap='viridis')
# add second colorbar
cbars = plt.colorbar(s)
cbars.set_label('Log10 Cl (mg/L)')


# To calculate the recharge rate at the sampling locations, we need to extract the values of the chloride deposition from the grid. For this we need to convert the coordinates of the samples into indices of the numpy array:

# In[24]:


# setting up the coordinates of the grid based on the meta data
xmin = cl_depo_meta['xllcorner']
xmax = cl_depo_meta['xllcorner']+cl_depo_meta['cellsize']*cl_depo_meta['ncols']
ymin = cl_depo_meta['yllcorner']
ymax = cl_depo_meta['yllcorner']+cl_depo_meta['cellsize']*cl_depo_meta['nrows']
cell = cl_depo_meta['cellsize']
# the floor command rounds numbers down to the nearest integer. The astype('int') ensures the result is an integer
yind = (np.floor((chem['Long'] - xmin)/cell)).astype('int')
# the x-index of 0 is at the top of the grid, so we need to reverse the values
xind = (cl_depo_meta['nrows']-np.floor((chem['Lat'] - ymin)/cell)).astype('int')
chem['Cl_depo'] = cl_depo[xind,yind]


# How do we know that our code is doing what we expect from it? One simple test is to visualise a copy of the grid and mark all the cells that have samples in, at least according to our calculations:

# In[25]:


# create an empty array with same dimensions as cl_depo
a = np.zeros_like(cl_depo)*np.nan
# give all cells > 0 the value 1
a[cl_depo>0] = 1
# give all cells with a sample the value 5
a[xind,yind] = 5
# quick and dirty visualisation
plt.figure()
# use no interpolation to avoid pixels being affected by neighbours
plt.imshow(a,interpolation='none',cmap='Reds')
plt.colorbar()


# **excercise: try to figure out what went wrong in selecting grid cells**
# 
# The x and y coordinates were swapped - they are the right way now.

# We now have all the information to calculate the recharge rate from the chloride concentration:

# In[26]:


chem['Recharge'] = 100*(chem['Cl_depo']/chem['Cl_mgL'])

fig,ax = plt.subplots()
# modify the default figure size
fig.set_size_inches(10,10)
p = ax.imshow(cl_depo,
          origin = 'upper',
          cmap = 'Reds',
          extent = (cl_depo_meta['xllcorner'],
                   cl_depo_meta['xllcorner']+(cl_depo_meta['ncols']*cl_depo_meta['cellsize']),
                   cl_depo_meta['yllcorner'],
                   cl_depo_meta['yllcorner']+(cl_depo_meta['nrows']*cl_depo_meta['cellsize'])))
cbar = plt.colorbar(p, shrink=0.5)
cbar.set_label('kg/ha/year')
ax.set_title('Chloride mass balance')
# zoom in to SA data
ax.set_xlim(chem['Long'].min(),chem['Long'].max())
ax.set_ylim(chem['Lat'].min(),chem['Lat'].max())
# use scatter to plot point with color based on recharge
s = ax.scatter(chem['Long'],chem['Lat'],.2,chem['Recharge'],cmap='viridis',vmax=250)
# add second colorbar
cbars = plt.colorbar(s,orientation='horizontal')
cbars.set_label('Recharge (mm/yr)')


# ## 6 Intermezzo: Colors
# Color is crucial in data visualisation, especially when using color to visualize a continuous range of data. Matplotlib has some excellent colormaps that are perceptually uniform and provides a great discussion and comparison of different [colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)
# 
# Related to choosing colormaps is choosing colors for categorical data or in plotting. [ColorBrewer](https://colorbrewer2.org/#type=sequential&scheme=BuGn&n=3) is a great tool to create color pallettes. Most plotting functions can use RGB or HexRGB values, but there is a range of named colors available. An overview is provided [here](https://matplotlib.org/stable/gallery/color/named_colors.html). The codeblock below is copied from that site.

# In[27]:


from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_colortable(colors, title, sort_colors=True, emptycols=0):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24, loc="left", pad=10)

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig

plot_colortable(mcolors.BASE_COLORS, "Base Colors",
                sort_colors=False, emptycols=1)
plot_colortable(mcolors.TABLEAU_COLORS, "Tableau Palette",
                sort_colors=False, emptycols=2)

plot_colortable(mcolors.CSS4_COLORS, "CSS Colors")

# Optionally plot the XKCD colors (Caution: will produce large figure)
#xkcd_fig = plot_colortable(mcolors.XKCD_COLORS, "XKCD Colors")
#xkcd_fig.savefig("XKCD_Colors.png")

plt.show()


# **Excercise: make a plot of TDS vs pH with the datapoints colored 'darkseagreen'**

# In[30]:


# 
f,s = plt.subplots()
s.semilogx(chem['TDSc_mgL'],chem['pH'],'.',color='darkseagreen',alpha=0.1)
s.set_xlabel('TDS (mg/L)')
s.set_ylabel('pH')


# ## 7. Multivariate data analysis

# In[60]:


cols = ['Lat','Long','TDSc_mgL', 'pH', 'HCO3_mgL', 'Na_mgL', 'K_mgL', 'Mg_mgL', 'Ca_mgL', 'Cl_mgL', 'SO4_mgL','NO3N_mgL']
dat = chem[cols].dropna()
dat


# As we've seen before, the data are very skewed. Before doing any multivariate data analysis it is therefore recommended to normalise data. This is often done by rescaling all variables so their range falls between 0 and 1 or by transforming the variables by subtracting the mean and dividing by the standard deviation. We used log transform earlier for skewed data. This has some drawbacks, especially if there are 0s in the dataset.
# 
# Another way to normalise data is to calculate the rank, i.e. the position if you were to rank them from smallest to largest. This can be easily done wiith the `rankdata` function from [scipy.stats](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html).

# In[61]:


# calculate rank of data
dr = rankdata(np.array(dat[cols[2::]]),axis=0)
dr


# There are a wide range of approaches for multivariate data analysis. A great resource is [scikit-learn](https://scikit-learn.org/stable/), which is an easy to use package for machine learning.
# 
# What we'll look into is dimensionality reduction and manifold learning. The goal is to find a representation of the data in 2D such that samples that are similar are plotted close to each other and samples that are very different are plotted far apart. The [manifold learning page](https://scikit-learn.org/stable/modules/manifold.html#manifold) gives an overview of methods you can use. The method we'll be using is [t-distributed Stochastic Neighbor Embedding](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE).
# 
# We need to import the function from scikit learn, specify the parameters and then train the algorithm with our dataset:

# In[62]:


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init='pca')
X = tsne.fit_transform(dr)


# The result is a 2D numpy array with an x and y coordinate for each sample. We can visualise this with `plt.scatter` and color the plot with the rank of each of the variables:

# In[63]:


fig = plt.figure()
for i,col in enumerate(cols[2::]):
    ax = plt.subplot(4,3,i+1, aspect=1)
    ax.scatter(X[:,0],X[:,1],0.2,dr[:,i],cmap='viridis')
    ax.set_title(col)
    ax.set_axis_off()
plt.tight_layout()


# This is a spatial dataset, so we want to know show this information on a map. I've developed a 2D perceptually colormap that can be used to assign a unique color to each sample, based on the coordinates of the TSNE projection:

# In[64]:


def percuniform_rgb(x,y):
    '''
    Create RGB values for x,y positions from perceptually uniform colour scheme
    IN:
        x: [nx1] array of x values
        y: [nx1] array of y values
    OUT:
        rgb: [nx3] array of rgb values
    '''
    # rescale cartesian coordinates into range [-1,1]
    # normalise based on max(range(x),range(y))
    # multiply by 2 and subtract 1 to have data 
    # - centered on [0,0] 
    # - x and y each in range [-1,1]
    range_x = x.max()-x.min()
    range_y = y.max()-y.min()
    range_m = max(range_x,range_y)
    x_s = 2*((x-x.min())/range_m)-1
    y_s = 2*((y-y.min())/range_m)-1
    # load spline interpolant of colour scheme
    rgb_interp = np.load('BivariateColourScheme.npy', allow_pickle=True, encoding='latin1').item()
    # interpolate rgb values
    rgb = np.zeros((len(x),3))
    for i,col in enumerate(['R','G','B']):
        rgb[:,i] = np.clip(rgb_interp[col].ev(x_s,y_s),0,1)
    return(rgb)


# In[65]:


tsnergb = percuniform_rgb(X[:,0],X[:,1])


# In[66]:


fig,ax = plt.subplots()
ax.scatter(X[:,0],X[:,1],5,tsnergb)
ax.set_aspect('equal')
ax.set_title('Color based on position')
ax.set_axis_off()


# We can now make a map of the samples, where each sample is colored based on its location in the TSNE plot

# In[67]:


a = np.zeros_like(cl_depo)*np.nan
# give all cells > 0 the value 1
a[cl_depo>0] = 1

fig,ax = plt.subplots()
# modify the default figure size
fig.set_size_inches(10,10)

p = ax.imshow(a,
              origin = 'upper',
              cmap = 'gray',
              vmax = 2,
              extent = (cl_depo_meta['xllcorner'],
                        cl_depo_meta['xllcorner']+(cl_depo_meta['ncols']*cl_depo_meta['cellsize']),
                        cl_depo_meta['yllcorner'],
                        cl_depo_meta['yllcorner']+(cl_depo_meta['nrows']*cl_depo_meta['cellsize'])))
ax.set_title('Color based on TSNE projection')
# zoom in to SA data
ax.set_xlim(chem['Long'].min(),chem['Long'].max())
ax.set_ylim(chem['Lat'].min(),chem['Lat'].max())
# use scatter to plot point with color based on recharge
s = ax.scatter(dat['Long'],dat['Lat'],2,tsnergb)

