import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyemu
import flopy

# function added thru PstFrom.add_py_function()
def replace_time_with_datetime(csv_file):
    import os
    import numpy as np
    import pandas as pd
    start_datetime = pd.to_datetime("1-1-2020")
    df = pd.read_csv(csv_file,index_col=0)
    df.loc[:,"datetime"] = start_datetime + pd.to_timedelta(np.round(df.index.values),unit='d')
    df.index = df.pop("datetime")
    df = df.loc[~df.index.duplicated(keep="last"),:]
    raw = os.path.split(csv_file)
    new_file = os.path.join(raw[0],"datetime_" + raw[1])
    df.to_csv(new_file)
    return new_file,df



def main():

    try:
       os.remove(r'datetime_sfr.csv')
    except Exception as e:
       print(r'error removing tmp file:datetime_sfr.csv')
    try:
       os.remove(r'datetime_heads.csv')
    except Exception as e:
       print(r'error removing tmp file:datetime_heads.csv')
    pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv',chunk_len=50)
    pyemu.os_utils.run(r'mf6')

    replace_time_with_datetime('sfr.csv')
    replace_time_with_datetime('heads.csv')

if __name__ == '__main__':
    mp.freeze_support()
    main()

