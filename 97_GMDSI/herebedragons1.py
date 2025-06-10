import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import shutil
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import shutil
import sys
import pyemu
import flopy

bin_path = os.path.join( "..", "bin", "win")

def prep_bins(dest_path):
    files = os.listdir(bin_path)
    for f in files:
        if os.path.exists(os.path.join(dest_path,f)):
            os.remove(os.path.join(dest_path,f))
        shutil.copy2(os.path.join(bin_path,f),os.path.join(dest_path,f))

def prep_forecasts(pst, model_times=False):
    pred_csv = os.path.join('models', 'daily_freyberg_mf6_truth',"pred_data.csv")
    assert os.path.exists(pred_csv)
    pred_data = pd.read_csv(pred_csv)
    pred_data.set_index('site', inplace=True)
    
    if type(model_times) == bool:
        model_times = [float(i) for i in pst.observation_data.time.unique()]
        
    ess_obs_data = {}
    for site in pred_data.index.unique().values:
        site_obs_data = pred_data.loc[site,:].copy()
        if isinstance(site_obs_data, pd.Series):
            site_obs_data.loc["site"] = site_obs_data.index.values
        if isinstance(site_obs_data, pd.DataFrame):
            site_obs_data.loc[:,"site"] = site_obs_data.index.values
            site_obs_data.index = site_obs_data.time
            sm = site_obs_data.value.rolling(window=20,center=True,min_periods=1).mean()
            sm_site_obs_data = sm.reindex(model_times,method="nearest")
        #ess_obs_data.append(pd.DataFrame9sm_site_obs_data)
        ess_obs_data[site] = sm_site_obs_data
    obs_data = pd.DataFrame(ess_obs_data)

    obs = pst.observation_data
    obs_names = [o for o in pst.obs_names if o not in pst.nnz_obs_names]

    # get list of times for obs name sufixes
    time_str = obs_data.index.map(lambda x: f"time:{x}").values
    # empyt list to keep track of misssing observation names
    missing=[]
    for col in obs_data.columns:
        if col.lower()=='part_time':
            obs_sufix = col.lower()
        else:
        # get obs list sufix for each column of data
            obs_sufix = col.lower()+"_"+time_str
        if type(obs_sufix)==str:
            obs_sufix=[obs_sufix]

        for string, oval, time in zip(obs_sufix,obs_data.loc[:,col].values, obs_data.index.values):
                if not any(string in obsnme for obsnme in obs_names):
                    missing.append(string)
                # if not, then update the pst.observation_data
                else:
                    # get a list of obsnames
                    obsnme = [ks for ks in obs_names if string in ks] 
                    if type(obsnme) == str:
                        obsnme=[obsnme]
                    obsnme = obsnme[0]
                    if obsnme=='part_time':
                        oval = pred_data.loc['part_time', 'value']
                    # assign the obsvals
                    obs.loc[obsnme,"obsval"] = oval
                        ## assign a generic weight
                        #if time > 3652.5 and time <=4018.5:
                        #    obs.loc[obsnme,"weight"] = 1.0      
    return 