
Model archive in support of USGS Cooperator publication:
Fienen, M.N., Haserodt, M.J., Leaf, A.T., Westenbroek, S.M., 2021. 
Appendix C: Central Sands Lakes Study Technical Report: Modeling 
Documentation in Wisconsin Department of Natural Resources, 
Central Sands Lake Study. Available at 
https://dnr.wisconsin.gov/topic/Wells/HighCap/CSLStudy.html

Data Release containing this model archive:

Fienen, M.N., Haserodt, M.J., and Leaf, A.T, 2021, 
MODFLOW models used to simulate groundwater flow in the Wisconsin Central Sands 
Study Area, 2012-2018, U.S. Geological Survey Data Release, 
https://doi.org/10.5066/P9BVFSGJ

MODEL ARCHIVE
-------------

Archive created: 2021-04-20

--------------------------------------------------------------------------
DISCLAIMER-
                                                                          
  THE FILES CONTAINED HEREIN ARE PROVIDED AS A CONVENIENCE TO THOSE
  WHO WISH TO REPLICATE SIMULATIONS OF RECHARGE THAT ARE
  DESCRIBED IN THE WISCONSIN DEPARTMENT OF NATURAL RESOURCES PUBLICATION
  THE CENTRAL SANDS LAKE STUDY TECHICAL REPORT: APPENDIX C, MODELING DOCUMENTATION.
  ANY CHANGES MADE TO THESE FILES COULD HAVE UNINTENDED, UNDESIRABLE      
  CONSEQUENCES.   THESE CONSEQUENCES COULD INCLUDE, BUT MAY NOT BE 
  NOT LIMITED TO: ERRONEOUS MODEL OUTPUT, NUMERICAL INSTABILITIES, 
  AND VIOLATIONS OF UNDERLYING ASSUMPTIONS ABOUT THE SUBJECT HYDROLOGIC       
  SYSTEM THAT ARE INHERENT IN RESULTS PRESENTED IN THE 
  WISCONSIN DEPARTMENT OF NATURAL RESOURCES PUBLICATION. 
  THE U.S. GEOLOGICAL SURVEY ASSUMES NO RESPONSIBILITY FOR THE            
  CONSEQUENCES OF ANY CHANGES MADE TO THESE FILES.  IF CHANGES ARE MADE
  TO THE MODEL, THE USER IS RESPONSIBLE FOR DOCUMENTING THE CHANGES AND
  JUSTIFYING THE RESULTS AND CONCLUSIONS.   

--------------------------------------------------------------------------

MODEL ARCHIVE:
    Description:
    ------------
    The directories in this archive contain input and output files for model runs,
    performed using MODFLOW-NWT and MODFLOW 6, history matching using PEST_HP and PEST++,
    and several scenarios. The goal of this modeling was to simulate and understand the
    potential effects of agricultural irrigation groundwater pumping on lake levels in the
    Wisconsin Central Sands. The Soil Water Balance (SWB) model was used to simulate
    irrigation and net infiltration, as documented in the main report and in the companion
    model archive/data release at https://doi.org/10.5066/P9SOJ01N.

    Descriptions of the data in each subdirectory are given to facilitate
    understanding of this model archive. Files descriptions are provided 
    for select files to provide additional information that may be of use
    for understanding this model archive. 

    Reconstructing the model archive from the online data release:
    --------------------------------------------------------------
    The model archive is available as a data release from:
    
        https://doi.org/10.5066/P9BVFSGJ

    
    The models will run successfully only if the correct directory 
    structure is correctly restored. The model archive is broken into 
    several pieces to reduce the likelihood of download timeouts. 
    Each zipfile contains a directory structure beneath it. The directories
    are self-contained (with the caveat that the bin directory contains
    all executable codes required to run the models), provided that each one 
    is unzipped properly and maintains the subfolder structure within it. To run the 
    PEST_HP or PEST++ files, a python installation is also required. The python.zip file
    contains self-contained python environments for linux, windows, and mac. To activate
    follow the platform-specific directions at 
    https://conda.github.io/conda-pack/#commandline-usage

    In the remainder of this document, each main folder is described, indicating
    which model or scenario is contained in the folder and highlighting the key
    filenames from which all other files are referred. 

    The highest level directory structure is:

    MODEL ARCHIVE
    ├── PLAINFIELD_TUNNEL_CHANNEL_LAKES_MODEL
    ├── PLEASANT_LAKE_MODEL
    ├── REGIONAL_MODEL
    ├── model_output
    ├── scenarios
    ├── scenarios_output
    ├── bin
    ├── python    
    ├── georef
    └── source

    The full directory structure of the model archive and the files  
    within each subdirectory are listed below. 

    Running the models:
    ------------------
    Each folder is self-contained with all files needed to run. Exceptions
    are the PEST++ and PEST_HP files which need both a python environment
    (chosen from the bin folder for the correct operating system). Each model
    folder is arranged so that running the model will generate output in the
    same folder. The model_output and scenarios_output folders contain output
    generated on the project for comparison. 
    For MODFLOW 6, execute by typing "mf6" in the folder
    For MODFLOW-NWT, exceute by typing "modflow_nwt <case>.nam" where <case> indicates
        the root of the name file
    For PEST++, execute by typing "pestpp-ies <case>.pst" where <case> indicates the
        root of the PEST control file
    For PEST_HP, execute by typing "pesthp <case>.pst" where <case> indicates the
        root of the PEST control file
    Special note for PEST++ and PEST_HP. To run, this will require python to be in the execution path.
        The packages in the source/python directory must be available to the python distribution. The 
        python dependencies used in this work can be accessed either by building with Anaconda using 
        the environment file: in bin/python/geoproc.yml or by unzipping the packed python environment 
        in bin/python for the appropriate operating system.


    ├── PLAINFIELD_TUNNEL_CHANNEL_LAKES_MODEL
    │    │    Description: 
    │    │    -----------
    │    │    Model files for the Plainfield Tunnel Channel Lakes MODFLOW6 inset model
    │    ├── PEST_setup
    │    │    │    Description: 
    │    │    │    -----------
    │    │    │    MODFLOW6 and PEST++ files for Plainfield Tunnel Channel Lakes inset model
    │    │    │    iterative ensemble smoother (iES) parameter estimation
    │    │    │    Files:
    │    │    │    -----------
    │    │    │    plainfield_ies.loc.pst           PEST++ Control File for history matching at starting parameter values
    │    │    │    plainfield_ies.loc.best.3.pst    PEST++ Control File for history matching at 
    │    │    │                                     best parameter values from optimal 3rd iteration
    │    │    │    all other files are referenced from the *.pst files
    │    │    ├── pyemu
    │    │    │       Description:
    │    │    │       -----------
    │    │    │       python files required to execute the model by PEST++
    │    │    ├── mult
    │    │    │    Description:
    │    │    │    -----------
    │    │    │    Empty directory into which multiplier arrays are written in the PEST++ process
    │    │    ├── org
    │    │    │    Description:
    │    │    │    -----------
    │    │    │    Original model array and list files against which multipliers will be applied
    │    │    └── tables
    │    │          Description:
    │    │          -----------
    │    │          Single file stress_period_data.csv that maps MODFLOW6 timing to calendar timing
    │    │
    │    └── final_model_files
    │            Description: 
    │             -----------
    │             MODFLOW 6 model files with parameters assigned from the base realization of the iterative ensemble smoother
    │            iteration 3, which was selected as optimal. 
    │            Files:
    │            -----------
    │            mfsim.nam            MODFLOW6 name file that controls the MODFLOW6 execution
    │            pfl_lgr_parent.nam   MODFLOW6 name file for the main inset submodel 
    │            pfl_lgr_inset.nam    MODFLOW6 name file for the lake-focused inset submodel
    │            all other files are referenced in the *.nam files
    │         
    ├── PLEASANT_LAKE_MODEL
    │    │    Description: 
    │    │    -----------
    │    │    Model files for the Pleasant Lake MODFLOW6 inset model
    │    ├── PEST_setup
    │    │    │        Description: 
    │    │    │        -----------
    │    │    │        MODFLOW6 and PEST++ files for Pleasant Lake inset model
    │    │    │        iterative ensemble smoother (iES) parameter estimation
    │    │    │        Files:
    │    │    │        -----------
    │    │    │        pleasant_ies.loc.pst             PEST++ Control File for history matching at starting parameter values
    │    │    │        pleasant_ies.loc.best.3.pst      PEST++ Control File for history matching at 
    │    │    │                                         best parameter values from optimal 3rd iteration
    │    │    │        all other files are referenced from the *.pst files
    │    │    ├── mult
    │    │    │       Description:
    │    │    │       -----------
    │    │    │       Empty directory into which multiplier arrays are written in the PEST++ process
    │    │    ├── pyemu
    │    │    │       Description:
    │    │    │       -----------
    │    │    │       python files required to execute the model by PEST++
    │    │    ├── org
    │    │    │       Description:
    │    │    │       -----------
    │    │    │       Original model array and list files against which multipliers will be applied
    │    │    └── tables
    │    │            Description:
    │    │            -----------
    │    │            Single file stress_period_data.csv that maps MODFLOW6 timing to calendar timing
    │    │
    │    └── final_model_files
    │            Description: 
    │            -----------
    │            MODFLOW 6 model files with parameters assigned from the base realization of the iterative ensemble smoother
    │            iteration 3, which was selected as optimal. 
    │            Files:
    │            -----------
    │            mfsim.nam                MODFLOW6 name file that controls the MODFLOW6 execution
    │            plsnt_lgr_parent.nam     MODFLOW6 name file for the main inset submodel 
    │            plsnt_lgr_inset.nam      MODFLOW6 name file for the lake-focused inset submodel
    │            all other files are referenced in the *.nam files
    │
    ├── REGIONAL_MODEL
    │    │    Description: 
    │    │    -----------
    │    │    Model files for the regional MODFLOW-NWT model
    │    ├── PEST_setup
    │    │    │        Description: 
    │    │    │        -----------
    │    │    │        MODFLOW-NWT and PEST_HP files for the regional model
    │    │    │        Files:
    │    │    │        -----------
    │    │    │        parent_transient_200_polish_par2_reg_noptmax0_sy_lay4.pst  PEST_HP Control file for final optimal parameters
    │    │    │        parent_transient_200_polish_par2_reg_noptmax0_sy_lay4.hp   PEST_HP initial conditions file for final optimal parameters
    │    │    ├── arr_mlt
    │    │    │       Description:
    │    │    │       -----------
    │    │    │       Empty directory into which multiplier arrays are written in the PEST++ process
    │    │    ├── arr_org
    │    │    │          Description:
    │    │    │          -----------
    │    │    │          Original model array files against which multipliers will be applied
    │    │    ├── list_org
    │    │    │       Description:
    │    │    │       -----------
    │    │    │       Original model list files against which multipliers will be applied
    │    │    ├── nwis_flux
    │    │    │   │    Description:
    │    │    │   │    -----------
    │    │    │   │    USGS streamflow data files used by the forward run python script in PEST
    │    │    │   └── monthly_dv_bf
    │    │    │           Description:
    │    │    │           -----------
    │    │    │           USGS monthly streamflow data files used by the forward run python script in PEST
    │    │    └── wdnr_fluxes
    │    │            Description:
    │    │            -----------
    │    │            Wisconsin DNR streamflow data files used by the forward run python script in PEST
    │    │
    │    └── final_model_files
    │            Description: 
    │            -----------
    │            MODFLOW NWT model files with parameters assigned from the optimal PEST_HP history matching results
    │            Files:
    │            -----------
    │            mfsim.nam                MODFLOW6 name file that controls the MODFLOW6 execution
    │            plsnt_lgr_parent.nam     MODFLOW6 name file for the main inset submodel 
    │            plsnt_lgr_inset.name     MODFLOW6 name file for the lake-focused inset submodel
    │
    ├── model_output    
    │   Description:
    │   -----------
    │   Model output files from running the models and from the optimal PEST files for each of the three
    │   major models. These files are provided for comparison with files generated upon rerunning the models
    │   or PEST setups in the model files directories.
    │   │
    │   ├── PLAINFIELD_TUNNEL_CHANNEL_LAKES_MODEL
    │   │   ├──  model
    │   │   │       Description:
    │   │   │       ------------
    │   │   │       Model output from the PLAINFIELD_TUNNEL_CHANNEL_LAKES_MODEL/final_model_files MODFLOW 6 run
    │   │   │       Files:
    │   │   │       ------------
    │   │   │       ├── lake1.obs.csv                         observation output for lake stage in Plainfield Lake
    │   │   │       ├── lake2.obs.csv                         observation output for lake stage in Second Lake
    │   │   │       ├── lake3.obs.csv                         observation output for lake stage in Sherman Lake
    │   │   │       ├── lake4.obs.csv                         observation output for lake stage in Long Lake
    │   │   │       ├── mfsim.lst                             listing file for entire simulation
    │   │   │       ├── pfl_lgr_inset.cbc                     cell budget file for the focused inset LGR model
    │   │   │       ├── pfl_lgr_inset.hds                     head file for the focused inset LGR model
    │   │   │       ├── pfl_lgr_inset.head.obs                head observation file for the focused inset LGR model
    │   │   │       ├── pfl_lgr_inset.list                    listing file for the focused inset LGR model
    │   │   │       ├── pfl_lgr_parent.cbc                    cell budget file for the inset model
    │   │   │       ├── pfl_lgr_parent.hds                    head file for the inset model
    │   │   │       ├── pfl_lgr_parent.head.obs               head observation file for the inset model
    │   │   │       ├── pfl_lgr_parent.list                   listing file for the inset model
    │   │   │       ├── pfl_lgr_parent.sfr.cbc                streamflow routing cell budget file for the inset model
    │   │   │       ├── pfl_lgr_parent.sfr.obs.output.csv     streamflow routing observations file for the inset model
    │   │   │       └── pfl_lgr_parent.sfr.stage.bin          streamflow routing stage file for the inset 
    │   │   │
    │   │   └── PEST++
    │   │           Description:
    │   │           ------------
    │   │           PEST++ output both from the ensemble parameter estimation effort and from running the 
    │   │           "best" base parameter set parameter 
    │   │           Files:
    │   │           ------------
    │   │           ├── plainfield_ies.loc.3.obs.csv             ensemble of observation values from 3rd iES iteration
    │   │           ├── plainfield_ies.loc.3.par.csv             ensemble of parameter values from 3rd iES iteration
    │   │           ├── plainfield_ies.loc.best.3.base.par       base parameter values from 3rd iES iteration
    │   │           ├── plainfield_ies.loc.best.3.base.rei       base observation values from 3rd iES iteration
    │   │           └── plainfield_ies.loc.best.3.phi.group.csv  base objective function value from 3rd iES iteration
    │   │
    │   ├── PLEASANT_LAKE_MODEL
    │   │   ├──  model
    │   │   │       Description:
    │   │   │       ------------
    │   │   │       Model output from the PLEASANT_LAKE_MODEL/final_model_files MODFLOW 6 run
    │   │   │       Files:
    │   │   │       ------------
    │   │   │       ├── lake1.obs.csv                           observation output for lake stage in Pleasant Lake
    │   │   │       ├── mfsim.lst                               listing file for entire simulation
    │   │   │       ├── plsnt_lgr_inset.cbc                     cell budget file for the focused inset LGR model
    │   │   │       ├── plsnt_lgr_inset.hds                     head file for the focused inset LGR model
    │   │   │       ├── plsnt_lgr_inset.head.obs                head observation file for the focused inset LGR model
    │   │   │       ├── plsnt_lgr_inset.list                    listing file for the focused inset LGR model
    │   │   │       ├── plsnt_lgr_parent.cbc                    cell budget file for the inset model
    │   │   │       ├── plsnt_lgr_parent.hds                    head file for the inset model
    │   │   │       ├── plsnt_lgr_inset.sfr.stage.bin           streamflow routing stage file for the inset LGR model
    │   │   │       ├── plsnt_lgr_parent.head.obs               head observation file for the inset model
    │   │   │       ├── plsnt_lgr_parent.list                   listing file for the inset model
    │   │   │       ├── plsnt_lgr_parent.sfr.cbc                streamflow routing cell budget file for the inset model
    │   │   │       ├── plsnt_lgr_parent.sfr.obs.output.csv     streamflow routing observations file for the inset model
    │   │   │       └── plsnt_lgr_parent.sfr.stage.bin          streamflow routing stage file for the inset model
    │   │   │
    │   │   └── PEST++
    │   │           Description:
    │   │           ------------
    │   │           PEST++ output both from the ensemble parameter estimation effort and from running the 
    │   │           "best" base parameter set parameter 
    │   │           Files:
    │   │           ------------
    │   │           ├── pltsnt_ies.loc.3.obs.csv             ensemble of observation values from 3rd iES iteration
    │   │           ├── pltsnt_ies.loc.3.par.csv             ensemble of parameter values from 3rd iES iteration
    │   │           ├── pltsnt_ies.loc.best.3.base.par       base parameter values from 3rd iES iteration
    │   │           ├── pltsnt_ies.loc.best.3.base.rei       base observation values from 3rd iES iteration
    │   │           └── pltsnt_ies.loc.best.3.phi.group.csv  base objective function value from 3rd iES iteration
    │   │
    │   └── REGIONAL_MODEL
    │       ├──  model
    │       │       Description:
    │       │       ------------
    │       │       Model output from the REGIONAL_MODEL/final_model_files MODFLOW NWT run
    │       │       Files:
    │       │       ------------
    │       │       ├── *.ggo                gage package streamflow observation files
    │       │       ├── cs200_nwt.cbc        cell budget file for the regional model
    │       │       ├── cs200_nwt.hds        head file for the regional model
    │       │       ├── cs200_nwt.hyd.bin    hydmod head observations file for the regional model
    │       │       ├── cs200_nwt.list       listing for for the regional model
    │       │       └── cs200_nwt.sfr.out    streamflow routing output fil for the regional model
    │       │
    │       └── PEST_HP
    │               Description:
    │               ------------
    │               PEST_HP output from the optimal parameter set
    │               Files:
    │               ------------
    │               Files:
    │               ------------
    │               └── parent_transient_200_polish_par2_reg_noptmax0_sy_lay4.res  model outputs from PEST_HP
    │                                                                                run at optimal parameter values
    │
    ├── scenarios
    │       ├── PLAINFIELD_TUNNEL_CHANNEL_LAKES_MODEL
    │       │   │
    │       │   ├── full_buildout_plainfield
    │       │   │   Description:
    │       │   │   ------------
    │       │   │   Model files for the potential future agricultural irrigation scenario
    │       │   │   Files:
    │       │   │   ------------
    │       │   │   ├── mfsim.nam            MODFLOW6 name file that controls the MODFLOW6 execution
    │       │   │   ├── pfl_lgr_parent.nam   MODFLOW6 name file for the main inset submodel 
    │       │   │   └── pfl_lgr_inset.nam    MODFLOW6 name file for the lake-focused inset submodel
    │       │   │       all other files are referenced in the *.nam files
    │       │   │
    │       │   ├── plainfield_irr
    │       │   │   Description:
    │       │   │   ------------
    │       │   │   Model files for the current agricultural irrigation scenario
    │       │   │   Files:
    │       │   │   ------------
    │       │   │   ├── mfsim.nam            MODFLOW6 name file that controls the MODFLOW6 execution
    │       │   │   ├── pfl_lgr_parent.nam   MODFLOW6 name file for the main inset submodel 
    │       │   │   └── pfl_lgr_inset.nam    MODFLOW6 name file for the lake-focused inset submodel
    │       │   │       all other files are referenced in the *.nam files
    │       │   │
    │       │   └── plainfield_noirr
    │       │       Description:
    │       │       ------------
    │       │       Model files for the no agricultural irrigation scenario
    │       │       Files:
    │       │       ------------
    │       │       ├── mfsim.nam            MODFLOW6 name file that controls the MODFLOW6 execution
    │       │       ├── pfl_lgr_parent.nam   MODFLOW6 name file for the main inset submodel 
    │       │       └── pfl_lgr_inset.nam    MODFLOW6 name file for the lake-focused inset submodel
    │       │           all other files are referenced in the *.nam files
    │       │   
    │       ├── PLEASANT_LAKE_MODEL
    │       │   │
    │       │   ├── full_buildout_pleasant
    │       │   │   Description:
    │       │   │   ------------
    │       │   │   Model files for the potential future agricultural irrigation scenario
    │       │   │   Files:
    │       │   │   ------------
    │       │   │   ├── mfsim.nam              MODFLOW6 name file that controls the MODFLOW6 execution
    │       │   │   ├── plsnt_lgr_parent.nam   MODFLOW6 name file for the main inset submodel 
    │       │   │   └── plsnt_lgr_inset.nam    MODFLOW6 name file for the lake-focused inset submodel
    │       │   │       all other files are referenced in the *.nam files
    │       │   │
    │       │   ├── pleasant_irr
    │       │   │   Description:
    │       │   │   ------------
    │       │   │   Model files for the current agricultural irrigation scenario
    │       │   │   Files:
    │       │   │   ------------
    │       │   │   ├── mfsim.nam              MODFLOW6 name file that controls the MODFLOW6 execution
    │       │   │   ├── plsnt_lgr_parent.nam   MODFLOW6 name file for the main inset submodel 
    │       │   │   └── plsnt_lgr_inset.nam    MODFLOW6 name file for the lake-focused inset submodel
    │       │   │       all other files are referenced in the *.nam files
    │       │   │
    │       │   └── pleasant_noirr
    │       │       Description:
    │       │       ------------
    │       │       Model files for the no agricultural irrigation scenario
    │       │       Files:
    │       │       ------------
    │       │       ├── mfsim.nam              MODFLOW6 name file that controls the MODFLOW6 execution
    │       │       ├── plsnt_lgr_parent.nam   MODFLOW6 name file for the main inset submodel 
    │       │       └── plsnt_lgr_inset.nam    MODFLOW6 name file for the lake-focused inset submodel
    │       │           all other files are referenced in the *.nam files
    │       │   
    │       └── REGIONAL_MODEL
    │               Description:
    │               ------------
    │               Model files for the no agricultural irrigation scenario
    │               Files:
    │               ------------
    │               └── noirr
    │                   ├── cs200_nwt.nam      MODFLOW-NWT name file that controls MODFLOW-NWT execution
    │                   └── external           directory of external array files for MODFLOW-NWT
    │                       all other files are referenced from the .nam file, 
    │
    ├── scenarios_output
    │       ├── PLAINFIELD_TUNNEL_CHANNEL_LAKES_MODEL
    │       │   │
    │       │   ├── full_buildout_plainfield
    │       │   │   Description:
    │       │   │   ------------
    │       │   │   Model output files for the potential future agricultural irrigation scenario
    │       │   │   Files:
    │       │   │   ------------
    │       │   │   ├── lake1.obs.csv                         observation output for lake stage in Plainfield Lake
    │       │   │   ├── lake2.obs.csv                         observation output for lake stage in Second Lake
    │       │   │   ├── lake3.obs.csv                         observation output for lake stage in Sherman Lake
    │       │   │   ├── lake4.obs.csv                         observation output for lake stage in Long Lake
    │       │   │   ├── MODFLOW_long_fluxes_irr.csv           flux output for Long Lake
    │       │   │   ├── MODFLOW_long_stages_irr.csv           stage output for Long Lake
    │       │   │   ├── MODFLOW_plainfield_fluxes_irr.csv     flux output for Plainfield Lake
    │       │   │   ├── MODFLOW_plainfield_stages_irr.csv     stage output for Plainfield Lake
    │       │   │   ├── mfsim.lst                             listing file for entire simulation
    │       │   │   ├── pfl_lgr_inset.head.obs                head observation file for the focused inset LGR model
    │       │   │   ├── pfl_lgr_inset.list                    listing file for the focused inset LGR model
    │       │   │   ├── pfl_lgr_parent.head.obs               head observation file for the inset model
    │       │   │   ├── pfl_lgr_parent.list                   listing file for the inset model
    │       │   │   ├── pfl_lgr_parent.sfr.obs.output.csv     streamflow routing observations file for the inset model
    │       │   │   ├── pfl_lgr_parent.sfr.stage.bin          streamflow routing stage file for the inset model
    │       │   │   └── stress_period_data.csv                mapping of datetime to stress periods in the MODFLOW output
    │       │   │
    │       │   ├── plainfield_irr
    │       │   │   Description:
    │       │   │   ------------
    │       │   │   Model output files for the current agricultural irrigation scenario
    │       │   │   Files:
    │       │   │   ------------
    │       │   │   ├── lake1.obs.csv                                observation output for lake stage in Plainfield Lake
    │       │   │   ├── lake2.obs.csv                                observation output for lake stage in Second Lake
    │       │   │   ├── lake3.obs.csv                                observation output for lake stage in Sherman Lake
    │       │   │   ├── lake4.obs.csv                                observation output for lake stage in Long Lake
    │       │   │   ├── mfsim.lst                                    listing file for entire simulation
    │       │   │   ├── plainfield_withirr_SFR_obs.csv               Monte Carlo SFR output for Long Lake
    │       │   │   ├── plainfield_withirr_long_fluxes_obs.csv       Monte Carlo flux output for Long Lake
    │       │   │   ├── plainfield_withirr_long_stages_obs.csv       Monte Carlo stage output for Long Lake
    │       │   │   ├── plainfield_withirr_plainfield_fluxes_obs.csv Monte Carlo flux output for Plainfield Lake
    │       │   │   ├── plainfield_withirr_plainfield_stages_obs.csv Monte Carlo stage output for Plainfield Lake
    │       │   │   ├── pfl_lgr_inset.head.obs                       head observation file for the focused inset LGR model
    │       │   │   ├── pfl_lgr_inset.list                           listing file for the focused inset LGR model
    │       │   │   ├── pfl_lgr_parent.head.obs                      head observation file for the inset model
    │       │   │   ├── pfl_lgr_parent.list                          listing file for the inset model
    │       │   │   ├── pfl_lgr_parent.sfr.obs.output.csv            streamflow routing observations file for the inset model
    │       │   │   ├── pfl_lgr_parent.sfr.stage.bin                 streamflow routing stage file for the inset model
    │       │   │   └── stress_period_data.csv                       mapping of datetime to stress periods in the MODFLOW output
    │       │   │ 
    │       │   ├── plainfield_noirr
    │       │   │   Description:
    │       │   │   ------------
    │       │   │   Model output files for the no agricultural irrigation scenario
    │       │   │   Files:
    │       │   │   ------------
    │       │   │   ├── lake1.obs.csv                                observation output for lake stage in Plainfield Lake
    │       │   │   ├── lake2.obs.csv                                observation output for lake stage in Second Lake
    │       │   │   ├── lake3.obs.csv                                observation output for lake stage in Sherman Lake
    │       │   │   ├── lake4.obs.csv                                observation output for lake stage in Long Lake
    │       │   │   ├── mfsim.lst                                    listing file for entire simulation
    │       │   │   ├── plainfield_noirr_SFR_obs.csv                 SFR output for Long Lake
    │       │   │   ├── plainfield_noirr_long_fluxes_obs.csv         flux output for Long Lake
    │       │   │   ├── plainfield_noirr_long_stages_obs.csv         stage output for Long Lake
    │       │   │   ├── plainfield_noirr_plainfield_fluxes_obs.csv   flux output for Plainfield Lake
    │       │   │   ├── plainfield_noirr_plainfield_stages_obs.csv   stage output for Plainfield Lake
    │       │   │   ├── pfl_lgr_inset.head.obs                       head observation file for the focused inset LGR model
    │       │   │   ├── pfl_lgr_inset.list                           listing file for the focused inset LGR model
    │       │   │   ├── pfl_lgr_parent.head.obs                      head observation file for the inset model
    │       │   │   ├── pfl_lgr_parent.list                          listing file for the inset model
    │       │   │   ├── pfl_lgr_parent.sfr.obs.output.csv            streamflow routing observations file for the inset model
    │       │   │   ├── pfl_lgr_parent.sfr.stage.bin                 streamflow routing stage file for the inset model
    │       │   │   └── stress_period_data.csv                       mapping of datetime to stress periods in the MODFLOW output
    │       │   │     
    │       │   └── quantiles
    │       │       Description:
    │       │       ------------
    │       │       Model output files for the quantile-based lag-time scenarior
    │       │       Files:
    │       │       ------------
    │       │       ├── plainfield_20_quantiles_SFR_obs.csv                lag-time SFR output for Long Lake
    │       │       ├── plainfield_long_20_quantiles_fluxes_obs.csv        lag-time flux output for Long Lake
    │       │       ├── plainfield_long_20_quantiles_stages_obs.csv        lag-time stage output for Long Lake
    │       │       ├── plainfield_plainfield_20_quantiles_fluxes_obs.csv  lag-time flux output for Plainfield Lake
    │       │       └── plainfield_plainfield_20_quantiles_stages_obs.csv  lag-time stage output for Plainfield Lake
    │       │ 
    │       ├── PLEASANT_LAKE_MODEL
    │       │   ├── full_buildout_pleasant
    │       │   │   ├── MODFLOW_pleasant_fluxes_irr.csv         Monte Carlo flux output for Pleasant Lake
    │       │   │   ├── MODFLOW_pleasant_stages_irr.csv         Monte Carlo stage output for Pleasant Lake
    │       │   │   ├── lake1.obs.csv                           observation output for lake stage in Pleasant Lake
    │       │   │   ├── mfsim.lst                               listing file for entire simulation
    │       │   │   ├── plsnt_lgr_inset.head.obs                head observation file for the focused inset LGR model
    │       │   │   ├── plsnt_lgr_inset.list                    listing file for the focused inset LGR model
    │       │   │   ├── plsnt_lgr_inset.sfr.stage.bin           streamflow routing stage file for the inset LGR model
    │       │   │   ├── plsnt_lgr_parent.head.obs               head observation file for the inset model
    │       │   │   ├── plsnt_lgr_parent.list                   listing file for the inset model
    │       │   │   ├── plsnt_lgr_parent.sfr.obs.output.csv     streamflow routing observations file for the inset model
    │       │   │   └── plsnt_lgr_parent.sfr.stage.bin          streamflow routing stage file for the inset model
    │       │   │
    │       │   ├── pleasant_irr
    │       │   │   ├── lake1.obs.csv                               observation output for lake stage in Pleasant Lake
    │       │   │   ├── mfsim.lst                                   listing file for entire simulation
    │       │   │   ├── pleasant_withirr_SFR_obs.csv                Monte Carlo SFR output for Pleasant Lake
    │       │   │   ├── pleasant_withirr_pleasant_fluxes_obs.csv    Monte Carlo flux output for Pleasant Lake
    │       │   │   ├── pleasant_withirr_pleasant_stages_obs.csv    Monte Carlo stage output for Pleasant Lake
    │       │   │   ├── plsnt_lgr_inset.head.obs                    head observation file for the focused inset LGR model
    │       │   │   ├── plsnt_lgr_inset.list                        listing file for the focused inset LGR model
    │       │   │   ├── plsnt_lgr_inset.sfr.stage.bin               streamflow routing stage file for the inset LGR model
    │       │   │   ├── plsnt_lgr_parent.head.obs                   head observation file for the inset model
    │       │   │   ├── plsnt_lgr_parent.list                       listing file for the inset model
    │       │   │   ├── plsnt_lgr_parent.sfr.obs.output.csv         streamflow routing observations file for the inset model
    │       │   │   ├── plsnt_lgr_parent.sfr.stage.bin              streamflow routing stage file for the inset model
    │       │   │   └── stress_period_data.csv                      mapping of datetime to stress periods in the MODFLOW output
    │       │   │
    │       │   ├── pleasant_noirr
    │       │   │   ├── lake1.obs.csv                               observation output for lake stage in Pleasant Lake
    │       │   │   ├── mfsim.lst                                   listing file for entire simulation
    │       │   │   ├── pleasant_withirr_SFR_obs.csv                Monte Carlo SFR output for Pleasant Lake
    │       │   │   ├── pleasant_withirr_pleasant_fluxes_obs.csv    Monte Carlo flux output for Pleasant Lake
    │       │   │   ├── pleasant_withirr_pleasant_stages_obs.csv    Monte Carlo stage output for Pleasant Lake
    │       │   │   ├── plsnt_lgr_inset.head.obs                    head observation file for the focused inset LGR model
    │       │   │   ├── plsnt_lgr_inset.list                        listing file for the focused inset LGR model
    │       │   │   ├── plsnt_lgr_inset.sfr.stage.bin               streamflow routing stage file for the inset LGR model
    │       │   │   ├── plsnt_lgr_parent.head.obs                   head observation file for the inset model
    │       │   │   ├── plsnt_lgr_parent.list                       listing file for the inset model
    │       │   │   ├── plsnt_lgr_parent.sfr.obs.output.csv         streamflow routing observations file for the inset model
    │       │   │   ├── plsnt_lgr_parent.sfr.stage.bin              streamflow routing stage file for the inset model
    │       │   │   └── stress_period_data.csv                      mapping of datetime to stress periods in the MODFLOW output
    │       │   │
    │       │   └── quantiles
    │       │       ├── pleasant_20_quantiles_SFR_obs.csv               lag-time SFR output for Long Lake
    │       │       ├── pleasant_pleasant_20_quantiles_fluxes_obs.csv   lag-time flux output for Long Lakes
    │       │       └── pleasant_pleasant_20_quantiles_stages_obs.csv   lag-time stage output for Long Lake
    │       │   
    │       └── REGIONAL_MODEL
    │           └── noirr
    │               ├── *.ggo          gage package output files
    │               └── cs200_nwt.hds  head save file
    │
    ├── bin
    │   Description:
    │   ------------
    │   Executable files for running models. Each must be in the path locally or systematically
    │   Each folder contains binary executables for each platform.
    │   ├── MODFLOW6 version 6.2.1     
    │   │   ├── linux
    │   │   ├── mac
    │   │   └── win64
    │   ├── MODFLOW_NWT version 1.2.0
    │   │   ├── linux
    │   │   ├── mac
    │   │   └── win64
    │   ├── PEST++ version 5.0.0
    │   │   ├── linux
    │   │   └── win
    │   └── PEST_HP version 17.22
    │       └── win
    ├── python
    │   Description:
    │   ------------
    │   Platform-specific python distributions created with conda-pack, required to run PEST_HP
    │   of PEST++ versions of the models. 
    │   Files:
    │   ------
    │   geoproc.yml  optional configuration file containing python requirements for use with conda
    │   ├── linux
    │   ├── mac
    │   └── windows    
    │
    ├── georef
    │   Description:
    │   ------------
    │   shapefiles and a figure of the model domains
    │   Files:
    │   ------
    │   cs200_nwt_bbox.shp          Model domain of the regional model
    │   plsnt_lgr_parent_bbox.shp   Pleasant Lake inset model domain
    │   plsnt_lgr_inset_bbox.shp    Pleasant Lake area of local-grid refinement
    │   pfl_lgr_parent_bbox.shp     Plainfield Tunnel Channel Lakes inset model domain
    │   pfl_lgr_inset_bbox.shp      Plainfield Tunnel Channel Lakes area of local-grid refinement
    │
    └── source
        ├── MODFLOW6 version 6.2.1     
        │   └── src
        ├── MODFLOW_NWT version 1.2.0
        │   └── src
        ├── PEST++ version 5.0.0
        │   └── src
        ├── PEST_HP version 17.22
        │   └── src
        └── python
            Description:
            ------------
            These python folders must be in the python path to run the PEST++
            and PEST_HP runs
            ├── flopy
            ├── gisutils
            ├── mfexport
            ├── mfsetup
            ├── pyemu
            └── sfrmaker


   

