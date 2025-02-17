import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyemu

# function added thru PstFrom.add_py_function()
def extract_hds_arrays_and_list_dfs():
    import flopy
    hds = flopy.utils.HeadFile("freyberg6_freyberg.hds")
    for it,t in enumerate(hds.get_times()):
        d = hds.get_data(totim=t)
        for k,dlay in enumerate(d):
            np.savetxt("hdslay{0}_t{1}.txt".format(k+1,it+1),d[k,:,:],fmt="%15.6E")

    lst = flopy.utils.Mf6ListBudget("freyberg6.lst")
    inc,cum = lst.get_dataframes(diff=True,start_datetime=None)
    inc.columns = inc.columns.map(lambda x: x.lower().replace("_","-"))
    cum.columns = cum.columns.map(lambda x: x.lower().replace("_", "-"))
    inc.index.name = "totim"
    cum.index.name = "totim"
    inc.to_csv("inc.csv")
    cum.to_csv("cum.csv")
    return




# function added thru PstFrom.add_py_function()
def process_secondary_obs(ws='.'):
    # load dependencies insde the function so that they get carried over to forward_run.py by PstFrom
    import os
    import pandas as pd

    def write_tdif_obs(orgf, newf, ws='.'):
        df = pd.read_csv(os.path.join(ws,orgf), index_col='time')
        df = df - df.iloc[0, :]
        df.to_csv(os.path.join(ws,newf))
        return

    # write the tdiff observation csv's
    write_tdif_obs('heads.csv', 'heads.tdiff.csv', ws)
    write_tdif_obs('sfr.csv', 'sfr.tdiff.csv', ws)

    print('Secondary observation files processed.')
    return



def main():

    try:
       os.remove(r'heads.csv')
    except Exception as e:
       print(r'error removing tmp file:heads.csv')
    try:
       os.remove(r'sfr.csv')
    except Exception as e:
       print(r'error removing tmp file:sfr.csv')
    try:
       os.remove(r'freyberg6.ic_strt_layer1.txt')
    except Exception as e:
       print(r'error removing tmp file:freyberg6.ic_strt_layer1.txt')
    try:
       os.remove(r'inc.csv')
    except Exception as e:
       print(r'error removing tmp file:inc.csv')
    try:
       os.remove(r'cum.csv')
    except Exception as e:
       print(r'error removing tmp file:cum.csv')
    try:
       os.remove(r'sfr.tdiff.csv')
    except Exception as e:
       print(r'error removing tmp file:sfr.tdiff.csv')
    try:
       os.remove(r'heads.tdiff.csv')
    except Exception as e:
       print(r'error removing tmp file:heads.tdiff.csv')
    pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv',chunk_len=50)
    pyemu.os_utils.run(r'mf6')

    pyemu.os_utils.run(r'mp7 freyberg_mp.mpsim')

    extract_hds_arrays_and_list_dfs()
    process_secondary_obs(ws='.')

if __name__ == '__main__':
    mp.freeze_support()
    main()

