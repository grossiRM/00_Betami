def process_csub_obs(fpth):
    tdata = flopy.utils.Mf6Obs(fpth).data
    dtype = [("totim", float),("CUNIT", float) ,("AQUITARD", float),("NODELAY" , float),("DELAY", float) , ("SKELETAL", float),("TOTAL", float),]
    v = np.zeros(tdata.shape[0], dtype=dtype)  ;v["totim"] = tdata["totim"]    ; v["totim"] /= 365.25    ; v["totim"] += 1908.353182752
    for key in tdata.dtype.names[1:]: 
        if "SKC" in key[:3]: v["SKELETAL"]  += tdata[key]
    for key in tdata.dtype.names[1:]:
        if "TC" in key[:2]: v["TOTAL"]      += tdata[key]
    for key in pcomp: 
        if key != "TOTAL" and key != "SKELETAL": 
            for obs_key in tdata.dtype.names: 
                if key in obs_key: v[key]   += tdata[obs_key]
    return v

#ref_A = process_csub_obs(pth)
#ref_B = pd.read_csv     (pth)