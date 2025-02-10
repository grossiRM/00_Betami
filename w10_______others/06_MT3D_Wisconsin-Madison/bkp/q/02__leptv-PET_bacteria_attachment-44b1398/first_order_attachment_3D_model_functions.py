# -*- coding: utf-8 -*-
"""
flopy_first_order_3D_attachement_functions.py
Created on:

@author:Vy Le and Christopher Zahasky

This script contains the numerical and analytical model functions, ploting 
functions, and  ....
"""

# All packages called by functions should be imported
import sys
import os
import numpy as np
# for analytical solutions
from scipy.special import erfc as erfc
from math import pi

import time


# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join('..', '..'))
    sys.path.append(fpth)
    import flopy
    
# This function creates a cyclindrical mask that is hardcoded to be identical for all cores that are 20x20. 
def generate_standard_20x20mask(hk):
    hk_size = hk.shape
    if hk_size[0] == 20 and hk_size[1] == 20:
        mp = np.array([7, 5, 3, 2, 2, 1, 1])
        mask_corner = np.ones((10,10))
        for i in range(len(mp)):
            mask_corner[i, 0:mp[i]] = 0
            
        mask_top = np.concatenate((mask_corner, np.fliplr(mask_corner)), axis=1)
        core_mask = np.concatenate((mask_top, np.flipud(mask_top))) 
    else:
        print('Dimensions not 20x20, no core mask generated')
        core_mask = np.ones([hk_size[0], hk_size[1]])
    
    return core_mask
    
def apply_core_mask(hom_hk, core_mask):
    hk_size = hom_hk.shape
    # set permeability values outside of core to zero with core mask by multiplying mask of zeros and ones to every slice
    for col in range(hk_size[2]):
        hom_hk[:,:,col] = np.multiply(hom_hk[:,:,col], core_mask)
    return hom_hk    

# FloPy Model function   
def first_order_pulse_injection_sim(dirname, raw_hk, prsity, al, rc1, grid_size, v,
                             perlen_mt, nprs, mixelm, exe_name_mf, exe_name_mt, workdir):
    # Model workspace and new sub-directory
    model_ws = os.path.join(workdir, dirname)
    # Call function and time it
    start = time.time() # start a timer
# =============================================================================
#     UNIT INFORMATION
# =============================================================================
    # units must be set for both MODFLOW and MT3D, they have different variable names for each
    # time units (itmuni in MODFLOW discretization package)
    # 1 = seconds, 2 = minutes, 3 = hours, 4 = days, 5 = years
    itmuni = 2 # MODFLOW length units
    mt_tunit = 'M' # MT3D units
    # length units (lenuniint in MODFLOW discretization package)
    # 0 = undefined, 1 = feet, 2 = meters, 3 = centimeters
    lenuni = 3 # MODFLOW units
    mt_lunit = 'CM' # MT3D units
    
# =============================================================================
#     STRESS PERIOD INFO
# =============================================================================
    perlen_mf = [np.sum(perlen_mt)]
    # number of stress periods (MF input), calculated from period length input
    nper_mf = len(perlen_mf)
    # number of stress periods (MT input), calculated from period length input
    nper = len(perlen_mt)
    
# =============================================================================
#     MODEL DIMENSION AND MATERIAL PROPERTY INFORMATION
# =============================================================================
    # Make model dimensions the same size as the hydraulic conductivity field input 
    # NOTE: that the there are two additional columns added as dummy slices (representing coreholder faces)
    hk_size = raw_hk.shape
    # determine dummy slice perm based on maximum hydraulic conductivity
    dummy_slice_hk = raw_hk.max()*10
    # define area with hk values above zero
    core_mask = np.ones((hk_size[0], hk_size[1]))
    core_mask = np.multiply(core_mask, raw_hk[:,:,0])
    core_mask[np.nonzero(core_mask)] = 1
    # define hk in cells with nonzero hk to be equal to 10x the max hk
    # This represents the 'coreholder' slices
    dummy_ch = core_mask[:,:, np.newaxis]*dummy_slice_hk
    # option to uncomment to model porosity field
    # dummy_ch_por = core_mask[:,:, np.newaxis]*0.15
    
    # concantenate dummy slice on hydraulic conductivity array
    hk = np.concatenate((dummy_ch, raw_hk, dummy_ch), axis=2)
    # prsity = np.concatenate((dummy_ch_por, prsity_field, dummy_ch_por), axis=2)
        
    # hk = raw_hk
    # Model information (true in all models called by 'p01')
    nlay = int(hk_size[0]) # number of layers / grid cells
    nrow = int(hk_size[1]) # number of rows / grid cells
    # ncol = hk_size[2]+2+ndummy_in # number of columns (along to axis of core)
    ncol = int(hk_size[2]+2) # number of columns (along to axis of core)
    # ncol = hk_size[2] # number of columns (along to axis of core)
    delv = grid_size[0] # grid size in direction of Lx (nlay)
    delr = grid_size[2] # grid size in direction of Ly (nrow)
    delc = grid_size[1] # grid size in direction of Lz (ncol)
    
    laytyp = 0
    # cell elevations are specified in variable BOTM. A flat layer is indicated
    # by simply specifying the same value for the bottom elevation of all cells in the layer
    botm = [-delv * k for k in range(1, nlay + 1)]
    
    # ADDITIONAL MATERIAL PROPERTIES
    # prsity = 0.25 # porosity. float or array of floats (nlay, nrow, ncol)
    prsity = prsity
    al = al # longitudinal dispersivity
    trpt = 0.1 # ratio of horizontal transverse dispersivity to longitudinal dispersivity
    trpv = trpt # ratio of vertical transverse dispersivity to longitudinal dispersivity
    
# =============================================================================
#     BOUNDARY AND INTIAL CONDITIONS
# =============================================================================
    # backpressure, give this in kPa for conversion
    bp_kpa = 70
    
     # Initial concentration (MT input)
    c0 = 0.
    # Stress period 2 concentration
    c1 = 1.0
    
    # Core radius
    core_radius = 2.54 # [cm]
    # Calculation of core area
    core_area = 3.1415*core_radius**2
    # Calculation of mask area
    mask_area = np.sum(core_mask)*grid_size[0]*grid_size[1]
    # total specific discharge or injection flux (rate/area)
    # q = injection_rate*(mask_area/core_area)/np.sum(core_mask)
    # scale injection rate locally by inlet permeability
    q_total = v*mask_area*prsity
    # print(q_total)
    # q_total = injection_rate/core_area
    q = q_total/np.sum(dummy_ch)
    
    # MODFLOW head boundary conditions, <0 = specified head, 0 = no flow, >0 variable head
    # ibound = np.ones((nlay, nrow, ncol), dtype=np.int)
    ibound = np.repeat(core_mask[:, :, np.newaxis], ncol, axis=2)
    
    # inlet conditions (currently set with well so inlet is zero)
    # ibound[:, :, 0] = ibound[:, :, -1]*-1
    # outlet conditions
    # ibound[5:15, 5:15, -1] = -1
    ibound[:, :, -1] = ibound[:, :, -1]*-1
    
    # MODFLOW constant initial head conditions
    strt = np.zeros((nlay, nrow, ncol), dtype=float)
    # Lx = (hk_size[2]*delc)
    # Q = injection_rate/(core_area)
    # geo_mean_k = np.exp(np.sum(np.log(hk[hk>0]))/len(hk[hk>0]))
    # h1 = Q * Lx/geo_mean_k
    # print(h1)
    
    # convert backpressure to head units
    if lenuni == 3: # centimeters
        hout = bp_kpa*1000/(1000*9.81)*100
    else: # elseif meters
        if lenuni == 2: 
            hout = bp_kpa*1000/(1000*9.81)
    # assign outlet pressure as head converted from 'bp_kpa' input variable
    # index the inlet cell
    # strt[:, :, 0] = h1+hout
    strt[:, :, -1] = core_mask*hout
    # strt[:, :, -1] = hout
    
    # Stress period well data for MODFLOW. Each well is defined through defintition
    # of layer (int), row (int), column (int), flux (float). The first number corresponds to the stress period
    # Example for 1 stress period: spd_mf = {0:[[0, 0, 1, q],[0, 5, 1, q]]}
    well_info = np.zeros((int(np.sum(core_mask)), 4), dtype=float)
    # Nested loop to define every inlet face grid cell as a well
    index_n = 0
    for layer in range(0, nlay):
        for row in range(0, nrow):
            # index_n = layer*nrow + row
            # index_n +=1
            # print(index_n)
            if core_mask[layer, row] > 0:
                well_info[index_n] = [layer, row, 0, q*dummy_ch[layer,row]]   
                index_n +=1
    
                
    # Now insert well information into stress period data 
    # Generalize this for multiple stress periods (see oscillatory flow scripts)
    # This has the form: spd_mf = {0:[[0, 0, 0, q],[0, 5, 1, q]], 1:[[0, 1, 1, q]]}
    spd_mf={0:well_info}
    
    # MT3D stress period data, note that the indices between 'spd_mt' must exist in 'spd_mf' 
    # This is used as input for the source and sink mixing package
    # Itype is an integer indicating the type of point source, 2=well, 3=drain, -1=constant concentration
    itype = 2
    cwell_info = np.zeros((int(np.sum(core_mask)), 5), dtype=float)
    # cwell_info = np.zeros((nrow*nlay, 5), dtype=np.float)
    # Nested loop to define every inlet face grid cell as a well
    index_n = 0
    for layer in range(0, nlay):
        for row in range(0, nrow):
            # index_n = layer*nrow + row
            if core_mask[layer, row] > 0:
                cwell_info[index_n] = [layer, row, 0, c0, itype] 
                index_n +=1
            # cwell_info[index_n] = [layer, row, 0, c0, itype]
            
    # Second stress period        
    cwell_info2 = cwell_info.copy()   
    cwell_info2[:,3] = c1 
    # Second stress period        
    cwell_info3 = cwell_info.copy()   
    cwell_info3[:,3] = c0 
    # Now apply stress period info    
    spd_mt = {0:cwell_info, 1:cwell_info2, 2:cwell_info3}

    # Concentration boundary conditions, this is neccessary to indicate 
    # inactive concentration cells outside of the more
    # If icbund = 0, the cell is an inactive concentration cell; 
    # If icbund < 0, the cell is a constant-concentration cell; 
    # If icbund > 0, the cell is an active concentration cell where the 
    # concentration value will be calculated. (default is 1).
    icbund = np.repeat(core_mask[:, :, np.newaxis], ncol, axis=2)
    # icbund[0, 0, 0] = -1
    # Initial concentration conditions, currently set to zero everywhere
    # sconc = np.zeros((nlay, nrow, ncol), dtype=np.float)
    # sconc[0, 0, 0] = c0
    
# =============================================================================
# MT3D OUTPUT CONTROL 
# =============================================================================
    # nprs (int): A flag indicating (i) the frequency of the output and (ii) whether 
    #     the output frequency is specified in terms of total elapsed simulation 
    #     time or the transport step number. If nprs > 0 results will be saved at 
    #     the times as specified in timprs; if nprs = 0, results will not be saved 
    #     except at the end of simulation; if NPRS < 0, simulation results will be 
    #     saved whenever the number of transport steps is an even multiple of nprs. (default is 0).
    # nprs = 20
    
    # timprs (list of float): The total elapsed time at which the simulation 
    #     results are saved. The number of entries in timprs must equal nprs. (default is None).
    timprs = np.linspace(0, np.sum(perlen_mt), nprs, endpoint=False)
    # obs (array of int): An array with the cell indices (layer, row, column) 
    #     for which the concentration is to be printed at every transport step. 
    #     (default is None). obs indices must be entered as zero-based numbers as 
    #     a 1 is added to them before writing to the btn file.
    # nprobs (int): An integer indicating how frequently the concentration at 
    #     the specified observation points should be saved. (default is 1).
    
# =============================================================================
# START CALLING MODFLOW PACKAGES AND RUN MODEL
# =============================================================================
    # Start callingwriting files
    modelname_mf = dirname + '_mf'
    # same as 1D model
    mf = flopy.modflow.Modflow(modelname=modelname_mf, model_ws=model_ws, exe_name=exe_name_mf)
    dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper_mf,
                                   delr=delr, delc=delc, top=0., botm=botm,
                                   perlen=perlen_mf, itmuni=itmuni, lenuni=lenuni)
    
    # MODFLOW basic package class
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    # MODFLOW layer properties flow package class
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, laytyp=laytyp)
    # MODFLOW well package class
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=spd_mf)
    # MODFLOW preconditioned conjugate-gradient package class
    pcg = flopy.modflow.ModflowPcg(mf, mxiter=100, rclose=1e-5, relax=0.97)
    # MODFLOW Link-MT3DMS Package Class (this is the package for solute transport)
    lmt = flopy.modflow.ModflowLmt(mf)
    # # MODFLOW output control package
    oc = flopy.modflow.ModflowOc(mf)
    
    mf.write_input()
    # RUN MODFLOW MODEL, set to silent=False to see output in terminal
    mf.run_model(silent=True)
    
# =============================================================================
# START CALLING MT3D PACKAGES AND RUN MODEL
# =============================================================================
    # RUN MT3dms solute tranport 
    modelname_mt = dirname + '_mt'

    # MT3DMS Model Class
    # Input: modelname = 'string', namefile_ext = 'string' (Extension for the namefile (the default is 'nam'))
    # modflowmodelflopy.modflow.mf.Modflow = This is a flopy Modflow model object upon which this Mt3dms model is based. (the default is None)
    mt = flopy.mt3d.Mt3dms(modelname=modelname_mt, model_ws=model_ws, 
                           exe_name=exe_name_mt, modflowmodel=mf)
    
    # Basic transport package class
    btn = flopy.mt3d.Mt3dBtn(mt, icbund=icbund, prsity=prsity, sconc=0, 
                             tunit=mt_tunit, lunit=mt_lunit, nper=nper, perlen=perlen_mt, 
                             nprs=nprs, timprs=timprs)
    
    # mixelm is an integer flag for the advection solution option, 
    # mixelm = 0 is the standard finite difference method with upstream or 
    # central in space weighting.
    # mixelm = 1 is the forward tracking method of characteristics, this seems to result in minimal numerical dispersion.
    # mixelm = 2 is the backward tracking
    # mixelm = 3 is the hybrid method
    # mixelm = -1 is the third-ord TVD scheme (ULTIMATE)
    adv = flopy.mt3d.Mt3dAdv(mt, mixelm=mixelm)

    dsp = flopy.mt3d.Mt3dDsp(mt, al=al, trpt=trpt)
    # =============================================================================    
    ## Note this additional line to call the  (package MT3D react)
    
    # set check if rc1 is a single value
    if type(rc1)==np.ndarray: # if prsity is an array
        # if rc1 is an array add the dummy slices
        rc1_dummy_slice = np.zeros((hk_size[0], hk_size[1], 1))    
        # concantenate dummy slice on rc1 array
        rc1 = np.concatenate((rc1_dummy_slice, rc1, rc1_dummy_slice), axis=2)
    
    rct = flopy.mt3d.Mt3dRct(mt, isothm=0, ireact=1, igetsc=0, rc1=rc1)
    #if want to test for conservative tracer, input rc1 = 0.
    # =============================================================================
    # source and sink mixing package
    ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=spd_mt)
    gcg = flopy.mt3d.Mt3dGcg(mt)
    
    mt.write_input()
    fname = os.path.join(model_ws, 'MT3D001.UCN')
    if os.path.isfile(fname):
        os.remove(fname)
    mt.run_model(silent=True)
    
    # Extract concentration information
    fname = os.path.join(model_ws, 'MT3D001.UCN')
    ucnobj = flopy.utils.UcnFile(fname)
    timearray = np.array(ucnobj.get_times()) # convert to min
    # print(times)
    conc = ucnobj.get_alldata()
    
    # Extract head information
    fname = os.path.join(model_ws, modelname_mf+'.hds')
    hdobj = flopy.utils.HeadFile(fname)
    heads = hdobj.get_data()
    
    # set inactive cell pressures to zero, by default inactive cells have a pressure of -999
    # heads[heads < -990] = 0
    
    # convert heads to pascals
    if lenuni == 3: # centimeters
        pressures = heads/100*(1000*9.81) 
    else: # elseif meters
        if lenuni == 2: 
            pressures = heads*(1000*9.81)
            
    
    # crop off extra concentration slices
    conc = conc[:,:,:,1:-1]
    # MT3D sets the values of all concentrations in cells outside of the model 
    # to 1E30, this sets them to 0
    conc[conc>2]=0
    # extract breakthrough curve data
    c_btc = np.transpose(np.sum(np.sum(conc[:, :, :, -1], axis=1), axis=1)/core_mask.sum())
    
    # calculate pressure drop
    p_inlet = pressures[:, :, 1]*core_mask
    p_inlet = np.mean(p_inlet[p_inlet>1])
    # print(p_inlet)
    p_outlet = pressures[:, :, -1]*core_mask
    p_outlet = np.mean(p_outlet[p_outlet>1])
    dp = p_inlet-p_outlet
    # crop off extra pressure slices
    pressures = pressures[:,:,1:-1] 
    # print('Pressure drop: '+ str(dp/1000) + ' kPa')
    
    # calculate mean permeability from pressure drop
    # water viscosity
    mu_water = 0.00089 # Pa.s
    L = hk_size[2]*delc
    km2_mean = (q_total/mask_area)*L*mu_water/dp /(60*100**2)
    
    # print('Core average perm: '+ str(km2_mean/9.869233E-13*1000) + ' mD')
    
    # Option to plot and calculate geometric mean to double check that core average perm in close
    geo_mean_K = np.exp(np.sum(np.log(raw_hk[raw_hk>0]))/len(raw_hk[raw_hk>0]))
    geo_mean_km2 = geo_mean_K/(1000*9.81*100*60/8.9E-4)
    # print('Geometric mean perm: ' + str(geo_mean_km2/9.869233E-13*1000) + ' mD')

    # Print final run time
    end_time = time.time() # end timer
    # print('Model run time: ', end - start) # show run time
    print(f"Model run time: {end_time - start:0.4f} seconds")
    
    # Possible output: mf, mt, conc, timearray, km2_mean, pressures
    return mf, mt, conc, c_btc, timearray


#### Analytical models
# Retardation with 1st type BC (equation C5 of Analytical Solutions of the 1D Convective-Dispersive Solute Transport Equation)
    # Genuchten, M Th Van and Alves, W J (1982)
def ADEwReactions_type1_fun(x, t, v, D, mu, C0, t0, Ci):
    # We are not focused on sorption so R can be set to one (equivalent to kd = 0)
    R = 1
    # 'u' term identical in equation c5 and c6 (type 3 inlet)
    u = v*(1+(4*mu*D/v**2))**(1/2)
    
    # Note that the '\' means continued on the next line
    Atrf = np.exp(-mu*t/R)*(1- (1/2)* \
        erfc((R*x - v*t)/(2*(D*R*t)**(1/2))) - \
        (1/2)*np.exp(v*x/D)*erfc((R*x + v*t)/(2*(D*R*t)**(1/2))))
    
    # term with B(x, t)
    Btrf = 1/2*np.exp((v-u)*x/(2*D))* \
        erfc((R*x - u*t)/(2*(D*R*t)**(1/2))) \
        + 1/2*np.exp((v+u)*x/(2*D))* erfc((R*x + u*t)/ \
        (2*(D*R*t)**(1/2)))
    
    # if a pulse type injection
    if t0 > 0:
        tt0 = t - t0
        
        indices_below_zero = tt0 <= 0
        # set values equal to 1 (but this could be anything)
        tt0[indices_below_zero] = 1
    
        Bttrf = 1/2*np.exp((v-u)*x/(2*D))* \
            erfc((R*x - u*tt0)/(2*(D*R*tt0)**(1/2))) \
            + 1/2*np.exp((v+u)*x/(2*D))* erfc((R*x + u*tt0)/ \
            (2*(D*R*tt0)**(1/2)))
        
        # Now set concentration at those negative times equal to 0
        Bttrf[indices_below_zero] = 0
        
        C_out = Ci*Atrf + C0*Btrf - C0*Bttrf
        
    else: # if a continous injection then ignore the Bttrf term (no superposition)
        C_out = Ci*Atrf + C0*Btrf
        
    
    # Return the concentration (C) from this function
    return C_out

# Retardation with 3rd type BC (equation C6 of Analytical Solutions of the 1D Convective-Dispersive Solute Transport Equation)
#! To do: program in C8 for finite length columns 
def ADEwReactions_type3_fun(x, t, v, D, mu, C0, t0, Ci):
    # We are not focused on sorption so R can be set to one (equivalent to kd = 0)
    R = 1
    # 'u' term identical in equation c5 and c6 (type 3 inlet)
    u = v*(1+(4*mu*D/v**2))**(1/2)
    
    Atv = np.exp(-mu*t/R)*(1 - (1/2)*erfc((R*x - v*t)/(2*(D*R*t)**(1/2))) - \
        (v**2*t/(pi*D*R))**(1/2)*np.exp(-(R*x - v*t)**2/(4*(D*R*t))) + \
        (1/2)*(1 + (v*x/D) + (v**2*t/(D*R)))*np.exp(v*x/D)* \
        erfc((R*x + v*t)/(2*(D*R*t)**(1/2))))
    
    Btv = v/(v+u)*np.exp((v-u)*x/(2*D))* erfc((R*x - u*t)/(2*(D*R*t)**(1/2))) + \
        v/(v-u)*np.exp((v+u)*x/(2*D))* erfc((R*x + u*t)/(2*(D*R*t)**(1/2))) + \
        v**2/(2*mu*D)*np.exp((v*x/D)-(mu*t/R))* erfc((R*x + v*t)/(2*(D*R*t)**(1/2)))

    # if a pulse type injection
    if t0 > 0:
        tt0 = t - t0
        
        indices_below_zero = tt0 <= 0
        # set values equal to 1 (but this could be anything)
        tt0[indices_below_zero] = 1
    
        Bttv = v/(v+u)*np.exp((v-u)*x/(2*D))* erfc((R*x - u*tt0)/(2*(D*R*tt0)**(1/2))) + \
        v/(v-u)*np.exp((v+u)*x/(2*D))* erfc((R*x + u*tt0)/(2*(D*R*tt0)**(1/2))) + \
        v**2/(2*mu*D)*np.exp((v*x/D)-(mu*tt0/R))* erfc((R*x + v*tt0)/(2*(D*R*tt0)**(1/2)))
        
        # Now set concentration at those negative times equal to 0
        Bttv[indices_below_zero] = 0
        
        C_out = Ci*Atv + C0*Btv - C0*Bttv
        
    else: # if a continous injection then ignore the Bttrf term (no superposition)
        C_out = Ci*Atv + C0*Btv
    
    # Return the concentration (C) from this function
    return C_out

