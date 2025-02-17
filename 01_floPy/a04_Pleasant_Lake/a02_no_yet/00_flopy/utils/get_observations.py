import json
import glob
import numpy as np
import pandas as pd
from affine import Affine
from flopy.utils import binaryfile as bf
import flopy as fp
import sys

def get_modelgrid_transform(grid_json_file, shift_to_cell_centers=False):
    """Create an affine.Affine that describes the model grid
    from a json file. The affine package comes with rasterio
    as a dependency or can be installed separately.

    Parameters
    ----------
    grid_json_file : str
        Model grid json file produced by modflow-setup
    shift_to_cell_centers : bool
        By default, transform reflects the upper left corner of
        the first cell in the model, and any conversions of x, y
        coordinates to pixels will be relative to upper left corners.
        If shift_to_cell_centers=True, x,y points will be referenced
        to the nearest cell centers.
    """
    with open(grid_json_file) as f:
        cfg = json.load(f)

    for dx in 'delr', 'delc':
        if not np.isscalar(cfg[dx]):
            cfg[dx] = cfg[dx][0]
    xul = cfg['xul']
    yul = cfg['yul']
    if shift_to_cell_centers:
        xul += 0.5 * cfg['delr']
        yul -= 0.5* cfg['delc']
        
    transform = Affine(cfg['delr'], 0., xul,
                      0., -cfg['delr'], yul) * \
               Affine.rotation(cfg['angrot'])
    return transform


def load_array(files):
    """Create a 3D numpy array from a list
    of ascii files containing 2D arrays.

    Parameters
    ----------
    files : sequence
        List of text files. Array layers will be
        ordered the same as the files.
        
    Returns
    -------
    array3d : 3D numpy array
    """
    if isinstance(files, str):
        return np.loadtxt(files)
    arrays = []
    for f in files:
        arrays.append(np.loadtxt(f))
    array3d = np.stack(arrays)
    return array3d
    

def read_mf6_lake_obs(f, perioddata, start_date='2012-01-01',
                      keep_only_last_timestep=True):
    df = pd.read_csv(f)
    df.columns = df.columns.str.lower()

    # convert modflow time to actual elapsed time
    # (by subtracting off the day for the initial steady-state period)
    if df.time.iloc[0] == 1:
        df['time'] -= 1

    # get stress period information for each timestep recorded
    kstp, kper = get_kstp_kper(perioddata.nstp)
    df['kstp'] = kstp
    df['kper'] = kper
    if len(df) == len(kstp) + 1:
        df = df.iloc[1:].copy()
    if keep_only_last_timestep:
        df = df.groupby('kper').last().reset_index()
    start_ts = pd.Timestamp(start_date)
    df['datetime'] = pd.to_timedelta(df.time, unit='D') + start_ts
    df.index = df.datetime
    return df.drop('datetime', axis=1)


def get_kstp_kper(nstp):
    """Given a sequence of the number of timesteps in each stress period,
    return a sequence of timestep, period tuples (kstp, kper) used
    by various flopy methods.
    """
    kstp = []
    kper = []
    for i, nstp in enumerate(nstp):
        for j in range(nstp):
            kstp.append(j)
            kper.append(i)
    return kstp, kper


def get_layer(column_name):
    """Pandas appends duplicate column names with a .*,
    where * is the number of duplicate names from left to right
    (e.g. obs, obs.1, obs.2, ...). Modflow-setup writes observation input
    for each model layer, at each site location, with the same observation prefix (site identifier). 
    MODFLOW-6 reports the result for each layer with duplicate column names, 
    with layers increasing from left to right (e.g. obs, obs, obs, ...).
    
    Parse the layer number from column_name, returning zero if there is no '.' separator.
    
    Notes
    -----
    The approach can't be used if the model includes inactive cells (including pinched layers)
    at the locations of observations, because it assumes that the layer numbers are consecutive,
    starting at 0. For example, a pinched layer 2 in a 4 layer model would result in the observation
    being in layers 0, 1, 3, which would be misinterpreted as 0, 1, 2.
    """
    if '.' not in column_name:
        return 0
    return int(column_name.split('.')[-1])


def get_gwf_obs_input(gwf_obs_input_file):
    """Read the first BEGIN continuous  FILEOUT block of an input
    file to the MODFLOW-6 GWF observation utility.

    Parameters
    ----------
    gwf_obs_input_file : str
        Input file to MODFLOW-6 observation utility (contains layer information).
        
    Note
    ----
    As-is, this only reads the first block. Modflow-setup writes all of the
    observation input to a single block, but observation input can
    be broken out into multiple blocks (one per file).
    
    This also doesn't work with open/close statements.
    """
    with open(gwf_obs_input_file) as src:
        for line in src:
            if 'BEGIN continuous' in line:
                df = pd.read_csv(src, delim_whitespace=True, header=None,
                                 error_bad_lines=False)
    df.dropna(axis=0, inplace=True)
    df.columns = ['obsname', 'obstype', 'k', 'i', 'j']
    # cast columns as ints and convert to zero-based
    for index_col in 'k', 'i', 'j':
        df[index_col] = df[index_col].astype(int) - 1
    return df


def get_mf6_single_variable_obs(perioddata,
                                model_output_file,
                                gwf_obs_input_file=None,
                                variable_name='values',
                                abs=True,
                                outfile='processed_head_obs.dat',
                                write_ins=False):
    """Read raw MODFLOW-6 observation output from csv table with
    times along the row axis and observations along the column axis. Reshape
    (stack) results to be n times x n sites rows, with a single observation value
    in each row. If an input file to the MODFLOW-6 observation utility is included,
    include the observation layer number in the output.

    Parameters
    ----------
    perioddata : str
        Path to csv file with start/end dates for stress periods. Must have columns
        'time' (modflow time, in days), 'start_datetime' (start date for the stress period)
        and 'end_datetime' (end date for the stress period).
    model_output_file : str
        Path to MODFLOW-6 observation csv output (shape: n times rows x n obs columns).
    gwf_obs_input_file : str
        Input file to MODFLOW-6 observation utility (contains layer information).
    variable_name : str, optional
        Column with simulated output will be named "sim_<variable_name", 
        by default 'head_m'
    abs : bool, optional
        Option to convert simulated values to absolute values
    outfile : str, optional
        [description], by default 'processed_head_obs.dat'
    write_ins : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """    
    perioddata = perioddata.copy()
    print('reading model output from {}...'.format(model_output_file))
    model_output = pd.read_csv(model_output_file)
    
    # convert all observation names to lower case
    model_output.columns = model_output.columns.str.lower()

    # add stress period information to model output
    # by having pandas match time floats in indices
    # the last time in each stress period is retained
    # (corresponding to the model end time listed in perioddata)
    model_output.index = model_output['time']
    perioddata.index = perioddata['time']
    model_output['per'] = perioddata['per']
    model_output['perioddata_time'] = perioddata['time']
    model_output.dropna(subset=['per'], axis=0, inplace=True)
    model_output['per'] = model_output['per'].astype(int)
    assert np.allclose(model_output.time.values, model_output.perioddata_time.values)
    model_output.index = model_output['per']

    # reshape the model output from (nper rows, nsites columns) to nper x nsites rows
    stacked = model_output.drop(['time', 'perioddata_time', 'per'], axis=1).stack(level=0).reset_index()
    simval_col = 'sim_{}'.format(variable_name)
    stacked.columns = ['per', 'obsprefix', simval_col]
    
    # optionally convert simulated values to absolute values
    if abs:
        stacked[simval_col] = stacked[simval_col].abs()
        
    # add dates
    period_start_dates = dict(zip(perioddata.loc[~perioddata.steady, 'per'], 
                                  perioddata.loc[~perioddata.steady, 'start_datetime']))
    stacked['datetime'] = pd.to_datetime([period_start_dates.get(per) for per in stacked.per])
    
    # parse the layers from the column positions (prior to stacking)
    if gwf_obs_input_file is not None:
        gwf_obs_input = get_gwf_obs_input(gwf_obs_input_file)
        # Assign layer to each observation, 
        # assuming that numbering in gwf_obs_input is repeated nper times
        nper = len(stacked.per.unique())
        stacked['layer'] = gwf_obs_input['k'].tolist() * nper
    
    # reset the obsprefixes to be the same for different layers at a location
    stacked['obsprefix'] = [prefix.split('.')[0] for prefix in stacked.obsprefix]

    # assign obsnames using the prefixes (location identifiers) and month
    steady = dict(zip(perioddata.per, perioddata.steady))
    obsnames = []
    for prefix, per, dt in zip(stacked.obsprefix, stacked.per, stacked.datetime):
        if not pd.isnull(dt):
            name = '{}_{}'.format(prefix, dt.strftime('%Y%m'))
        elif steady[per]:
            name = '{}_ss'.format(prefix)
        else:
            name = prefix
        obsnames.append(name)
    stacked['obsnme'] = obsnames                            

    stacked.index = stacked['obsnme']
    sort_cols = [c for c in ['obsprefix', 'per', 'layer'] if c in stacked.columns]
    stacked.sort_values(by=sort_cols, inplace=True)
    return stacked


def get_ij(transform, x, y):
    """Return the row and column of a point or sequence of points
    in real-world coordinates. Uses the affine package. Basically, 
    
    * to get an x, y: transform * (col, row)
    * the inverse, ~transform * (x, y) returns a fractional i, j location on the grid

    Parameters
    ----------
    transform : affine.Affine instance describing a regular grid
        see https://github.com/sgillies/affine
    x : scalar or sequence of x coordinates
    y : scalar or sequence of y coordinates
    local : bool
        Flag for returning real-world or model (local) coordinates.
        (default False)
    chunksize : int
        Because this function compares each x, y location to a vector
        of model grid cell locations, memory usage can quickly get
        out of hand, as it increases as the square of the number of locations.
        This can be avoided by breaking the x, y location vectors into
        chunks. Experimentation with approx. 5M points suggests
        that a chunksize of 100 provides close to optimal
        performance in terms of execution time. (default 100)

    Returns
    -------
    i : row or sequence of rows (zero-based)
    j : column or sequence of columns (zero-based)
    """

    j, i = ~transform * (x, y)
    i = np.round(i, 0).astype(int)
    j = np.round(j, 0).astype(int)
    return i, j


def get_transmissivities(heads, hk, top, botm,
                         r=None, c=None, x=None, y=None, modelgrid=None,
                         sctop=None, scbot=None, nodata=-999):
    """
    Computes transmissivity in each model layer at specified locations and
    open intervals. A saturated thickness is determined for each row, column
    or x, y location supplied, based on the open interval (sctop, scbot),
    if supplied, otherwise the layer tops and bottoms and the water table
    are used.

    Parameters
    ----------
    heads : 2D array OR 3D array
        numpy array of shape nlay by n locations (2D) OR complete heads array
        of the model for one time (3D)
    hk : 3D numpy array
        horizontal hydraulic conductivity values.
    top : 2D numpy array
        model top elevations.
    botm : 3D numpy array
        layer botm elevations.
    r : 1D array-like of ints, of length n locations
        row indices (optional; alternately specify x, y)
    c : 1D array-like of ints, of length n locations
        column indices (optional; alternately specify x, y)
    x : 1D array-like of floats, of length n locations
        x locations in real world coordinates (optional).
        If x and y are specified, a modelgrid must also be provided.
    y : 1D array-like of floats, of length n locations
        y locations in real world coordinates (optional)
        If x and y are specified, a modelgrid must also be provided.
    modelgrid_transform : affine.Affine instance, optional
        Only required for getting i, j if x and y are specified.
    sctop : 1D array-like of floats, of length n locations
        open interval tops (optional; default is model top)
    scbot : 1D array-like of floats, of length n locations
        open interval bottoms (optional; default is model bottom)
    nodata : numeric
        optional; locations where heads=nodata will be assigned T=0

    Returns
    -------
    T : 2D array of same shape as heads (nlay x n locations)
        Transmissivities in each layer at each location

    """
    if r is not None and c is not None:
        pass
    elif x is not None and y is not None:
        # get row, col for observation locations
        r, c = get_ij(modelgrid, x, y)
    else:
        raise ValueError('Must specify row, column or x, y locations.')

    # get k-values and botms at those locations
    # (make nlayer x n sites arrays)
    hk2d = hk[:, r, c]
    botm2d = botm[:, r, c]

    if len(heads.shape) == 3:
        heads = heads[:, r, c]

    msg = 'Shape of heads array must be nlay x nhyd'
    assert heads.shape == botm2d.shape, msg

    # set open interval tops/bottoms to model top/bottom if None
    if sctop is None:
        sctop = top[r, c]
    if scbot is None:
        scbot = botm[-1, r, c]

    # make an nlayers x n sites array of layer tops
    tops = np.empty_like(botm2d, dtype=float)
    tops[0, :] = top[r, c]
    tops[1:, :] = botm2d[:-1]

    # expand top and bottom arrays to be same shape as botm, thickness, etc.
    # (so we have an open interval value for each layer)
    sctoparr = np.zeros(botm2d.shape)
    sctoparr[:] = sctop
    scbotarr = np.zeros(botm2d.shape)
    scbotarr[:] = scbot

    # start with layer tops
    # set tops above heads to heads
    # set tops above screen top to screen top
    # (we only care about the saturated open interval)
    openinvtop = tops.copy()
    openinvtop[openinvtop > heads] = heads[openinvtop > heads]
    openinvtop[openinvtop > sctoparr] = sctoparr[openinvtop > sctop]

    # start with layer bottoms
    # set bottoms below screened interval to screened interval bottom
    # set screen bottoms below bottoms to layer bottoms
    openinvbotm = botm2d.copy()
    openinvbotm[openinvbotm < scbotarr] = scbotarr[openinvbotm < scbot]
    openinvbotm[scbotarr < botm2d] = botm2d[scbotarr < botm2d]

    # compute thickness of open interval in each layer
    thick = openinvtop - openinvbotm

    # assign open intervals above or below model to closest cell in column
    not_in_layer = np.sum(thick < 0, axis=0)
    not_in_any_layer = not_in_layer == thick.shape[0]
    for i, n in enumerate(not_in_any_layer):
        if n:
            closest = np.argmax(thick[:, i])
            thick[closest, i] = 1.
    thick[thick < 0] = 0
    thick[heads == nodata] = 0  # exclude nodata cells
    thick[np.isnan(heads)] = 0  # exclude cells with no head value (inactive cells)

    # compute transmissivities
    T = thick * hk2d
    return T


def fill_nats(df, perioddata):
    """Fill in NaT (not a time) values with
    corresponding date for that stress period.

    Parameters
    ----------
    df : DataFrame
        Observation data. Must have 'datetime' column
        with date and 'per' column with stress period.
    perioddata : DataFrame
        Perioddata table produced by modflow-setup. Must have
        'per' column and 'start_datetime' column.
    """
    period_start_datetimes = pd.to_datetime(perioddata['start_datetime'])
    start_datetimes = dict(zip(perioddata['per'], period_start_datetimes))
    datetime = [start_datetimes[per] if pd.isnull(dt) else dt 
                for per, dt in zip(df['per'], df['datetime'])]
    df['datetime'] = datetime
    j=2
    
    
def get_head_obs(perioddata, modelgrid_transform, model_output_file, observed_values_file,
                 gwf_obs_input_file,
                 variable_name='head_m',
                 outfile=None,
                 observed_values_site_id_col='obsprefix',
                 observed_values_obsval_col='measured',
                 observed_values_x_col='x_utm',
                 observed_values_y_col='y_utm',
                 observed_values_screen_top_col='sctop',
                 observed_values_screen_botm_col='scbot',
                 drop_groups={'wdnrlks_tr', 'wdnr_lakes', 'wdnr_wells'},
                 hk_arrays=None, top_array=None, botm_arrays=None,
                 write_ins=False):
    """Post-processes model output to be read by PEST, and optionally,
    writes a corresponding PEST instruction file. Reads model output
    using get_mf6_single_variable_obs().

    Parameters
    ----------
    perioddata : str
        Path to csv file with start/end dates for stress periods. Must have columns
        'time' (modflow time, in days), 'start_datetime' (start date for the stress period)
        and 'end_datetime' (end date for the stress period).
    model_output_file : str, optional
        [description], by default 'plsnt_lgr_parent.head.obs'
    observed_values_file : str, optional
        [description], by default 'tables/tables/head_obs.csv'
    gwf_obs_input_file : str
        Input file to MODFLOW-6 observation utility (contains layer information).
    variable_name : str, optional
        Column with simulated output will be named "sim_<variable_name", 
        by default 'head_m'
    outfile : str, optional
        Output file of values to be read by PEST, by default 'processed_head_obs.dat'
    write_ins : bool, optional
        Option to write instruction file, by default False

    Returns
    -------
    [type]
        [description]
    """    
    obs_values_column = 'obs_' + variable_name  # output column with observed values
    sim_values_column = 'sim_' + variable_name  # output column with simulated equivalents to observed values

    perioddata = perioddata.copy()
    results = get_mf6_single_variable_obs(perioddata, model_output_file=model_output_file,
                                          gwf_obs_input_file=gwf_obs_input_file,
                                          variable_name=variable_name)
    
    # read in the observed values and site locations
    if not isinstance(observed_values_file, pd.DataFrame):
        observed = pd.read_csv(observed_values_file)
    else:
        observed = observed_values_file
    observed.index = observed['obsnme']
    
    
    # drop model results that aren't in the obs information file
    # these are probably observations that aren't in the model time period
    # (and therefore weren't included in the parent model calibration; 
    # but modflow-setup would include them in the MODFLOW observation input)
    # also drop sites that are in the obs information file, but not in the model results
    # these include sites outside of the model (i.e. in the inset when looking at the parent)
    no_info_sites = set(results.obsprefix).symmetric_difference(observed.obsprefix)
    # dump these out to a csv
    print('Dropping {} sites with no information'.format(len(no_info_sites)))
    no_info = results.loc[results.obsprefix.isin(no_info_sites)].to_csv('dropped_head_observation_sites.csv', index=False)
    results = results.loc[~results.obsprefix.isin(no_info_sites)].copy()
    observed = observed.loc[~observed.index.isin(no_info_sites)].copy()
    
    # get_mf6_single_variable_obs returns values for each layer
    # collapse these into one value for each location, time
    # by taking the transmissivity-weighted average
    hk = load_array(hk_arrays)
    top = load_array(top_array)
    botm = load_array(botm_arrays)
    
    # get the x and y location and open interval corresponding to each head observation
    x = dict(zip(observed[observed_values_site_id_col], 
                            observed[observed_values_x_col]))
    y = dict(zip(observed[observed_values_site_id_col], 
                            observed[observed_values_y_col]))
    sctop = dict(zip(observed[observed_values_site_id_col], 
                            observed[observed_values_screen_top_col]))
    scbot = dict(zip(observed[observed_values_site_id_col], 
                            observed[observed_values_screen_botm_col]))
    results['x'] = [x[obsprefix] for obsprefix in results.obsprefix]
    results['y'] = [y[obsprefix] for obsprefix in results.obsprefix]
    results['sctop'] = [sctop[obsprefix] for obsprefix in results.obsprefix]
    results['scbot'] = [scbot[obsprefix] for obsprefix in results.obsprefix]
    periods = results.groupby('per')
    simulated_heads = []
    for per, data in periods:
        
        # get a n layers x n sites array of simulated head observations
        data = data.reset_index(drop=True)
        heads_2d = data.pivot(columns='layer', values='sim_head_m', index='obsnme').T.values
        obsnme = data.pivot(columns='layer', values='obsnme', index='obsnme').index.tolist()
        
        # x, y, sctop and scbot have one value for each site
        kwargs = {}
        for arg in 'x', 'y', 'sctop', 'scbot':
            # pivot data to nsites rows x nlay columns
            # positions without data are filled with nans
            pivoted = data.pivot(columns='layer', values=arg, index='obsnme')
            # reduce pivoted data to just one value per site by taking the mean
            # (values should be the same across columns, which represent layers)
            kwargs[arg] = pivoted.mean(axis=1).values
        
        # get the transmissivity associated with each head obs
        T = get_transmissivities(heads_2d, hk, top, botm, 
                                 modelgrid=modelgrid_transform, **kwargs
                                 )
        
        # compute transmissivity-weighted average heads
        Tr_frac = T / T.sum(axis=0)
        Tr_frac_df = pd.DataFrame(Tr_frac.transpose())
        Tr_frac_df['obsnme'] = obsnme
        Tr_frac_df.to_csv('obs_layer_transmissivities.csv', float_format='%.2f')
        mean_t_weighted_heads = np.nansum((heads_2d * Tr_frac), axis=0)
        
        # in some cases, the open interval might be mis-matched with the layering
        # for example, an open interval might be primarily in layer 4,
        # in a location where layer 5 is the only active layer
        # this would result in a mean_t_weighted_heads value of 0
        # (from the zero transmissivity in that layer)
        # fill these instances with the mean of any valid heads at those locations
        mean_heads = np.nanmean(heads_2d, axis=0)
        misaligned = mean_t_weighted_heads == 0
        mean_t_weighted_heads[misaligned] = mean_heads[misaligned]
        
        # verify that there are no nans in the extracted head values (one per obs)
        assert not np.any(np.isnan(mean_t_weighted_heads))
        
        # add the simulated heads onto the list for all periods
        mean_t_weighted_heads_df = pd.DataFrame({sim_values_column: mean_t_weighted_heads}, index=obsnme)
        simulated_heads.append(mean_t_weighted_heads_df)
    all_simulated_heads = pd.concat(simulated_heads)

    # reduce results dataframe from nobs x nlay x nper to just nobs x nper
    head_obs = results.reset_index(drop=True).groupby('obsnme').first()
    
    # replace the simulated heads column with the transmissivity-weighted heads computed above
    head_obs[obs_values_column] = observed[observed_values_obsval_col]
    head_obs[sim_values_column] = all_simulated_heads[sim_values_column]
    for column in ['group', 'uncertainty']:
        head_obs[column] = observed[column]
    
    # drop any observations that are lake stages
    # (need to compare output from the lake package instead of any head values at these locations)
    head_obs = head_obs.loc[~head_obs.group.isin(drop_groups)].copy()
    
    # nans are where sites don't have observation values for that period
    # or sites that are in other model (inset or parent)
    head_obs.dropna(subset=[obs_values_column], axis=0, inplace=True)

    # reorder the columns
    head_obs['obsnme'] = head_obs.index
    head_obs = head_obs[['datetime', 'per', 'obsprefix', 'obsnme', obs_values_column, sim_values_column, 
                         'group', 'uncertainty', 'sctop', 'scbot']].copy()
    
    # fill NaT (not a time) datetimes
    fill_nats(head_obs, perioddata)
    
    if outfile is not None:
        head_obs.fillna(-9999).to_csv(outfile, sep=' ', index=False)

        # write the instruction file
        if write_ins:
            write_insfile(head_obs, outfile + '.ins', obsnme_column='obsnme',
                          simulated_obsval_column=sim_values_column, index=False)
    return head_obs


def get_flux_obs(perioddata,
                 model_output_file='meras3_1L.sfr.obs.output.csv',
                 observed_values_file='../tables/flux_obs.csv',
                 observed_values_column='measured',
                 variable_name='flux_m3',
                 outfile=None,
                 write_ins=False):
    """[summary]

    Parameters
    ----------
    perioddata : str
        Path to csv file with start/end dates for stress periods. Must have columns
        'time' (modflow time, in days), 'start_datetime' (start date for the stress period)
        and 'end_datetime' (end date for the stress period).
    model_output_file : str, optional
        [description], by default 'meras3_1L.sfr.obs.output.csv'
    observed_values_file : str, optional
        [description], by default '../tables/flow_obs_by_stress_period.csv'
    observed_values_column : str, optional
        Column in obs_values_file with measured flux values
    variable_name : str, optional
        [description], by default 'measured'
    outfile : str, optional
        [description], by default 'processed_flux_obs.dat'
    write_ins : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """    
    results = get_mf6_single_variable_obs(perioddata, model_output_file=model_output_file,
                          variable_name=variable_name)
    observed = pd.read_csv(observed_values_file)
    observed.index = observed['obsnme']

    sim_values_column = 'sim_' + variable_name
    obs_values_column = 'obs_' + variable_name
    results[obs_values_column] = observed[observed_values_column]
    for column in ['uncertainty', 'group']:
         results[column] = observed[column]

    # nans are where sites don't have observation values for that period
    results.dropna(subset=[obs_values_column], axis=0, inplace=True)

    # reorder the columns
    results = results[['per', 'obsprefix', 'obsnme', obs_values_column, sim_values_column, 'group', 'uncertainty']].copy()
    if outfile is not None:
        results.to_csv(outfile, sep=' ', index=False)

        # write the instruction file
        if write_ins:
            write_insfile(results, outfile + '.ins', obsnme_column='obsnme',
                  simulated_obsval_column=sim_values_column, index=False)
    return results


def get_lake_stage_obs(lake_obs_files, perioddata, observed_values_file, 
                       lake_site_numbers,
                       outfile=None,
                       variable_name='stage_m',
                       observed_values_site_id_col='obsprefix',
                       observed_values_obsval_col='measured',
                       write_ins=True):

    # read in the observed values and site locations
    if not isinstance(observed_values_file, pd.DataFrame):
        observed = pd.read_csv(observed_values_file)
    else:
        observed = observed_values_file
    observed.index = observed['obsnme']
    
    dfs = []
    for name, f in lake_obs_files.items():
        df = read_mf6_lake_obs(f, perioddata)
        df.reset_index(inplace=True)  # put index into datetime column
        
        # add obsnames
        steady = perioddata.steady.values
        prefix = '{}_lk'.format(lake_site_numbers[name])
        obsnme = []
        group = []
        for per, dt in zip(df.kper, df.datetime):
            if steady[per]:
                obsnme.append(prefix + '_ss')
            else:
                obsnme.append('{}_{}'.format(prefix, dt.strftime('%Y%m')))
        df['obsnme'] = obsnme
        df.index = df.obsnme
        df['obsprefix'] = prefix
        df['name'] = name
        sim_values_column = 'sim_' + variable_name
        obs_values_column = 'obs_' + variable_name
        df[obs_values_column] = observed[observed_values_obsval_col]
        df['group'] = observed['group']
        # rename columns for consistency with other obs
        renames = {'stage': sim_values_column,
                   'kper': 'per'}
        df.rename(columns=renames, inplace=True)
        # drop values that don't have an observation
        df.dropna(subset=[obs_values_column], axis=0, inplace=True)
        dfs.append(df)
    df = pd.concat(dfs)
        
    # write output
    if outfile is not None:
        df.to_csv(outfile, sep=' ', index=False)
        
        # write the instruction file
        if write_ins:
            write_insfile(df, outfile + '.ins', obsnme_column='obsnme',
                          simulated_obsval_column=sim_values_column, index=False)
    return df


def get_lake_in_out_gw_fluxes(cell_budget_file, perioddata, precision='double',
                              lakenames=None,
                              outfile=None):
    """Read Lake/groundwater flux information from MODFLOW-6 cell-by-cell budget output;
    sum in and out fluxes by lake, for each time recorded.

    Parameters
    ----------
    cell_budget_file : str
        Path to MODFLOW-6 binary cell budget output.
    perioddata : str
        Path to csv file with start/end dates for stress periods. Must have columns
        'time' (modflow time, in days), 'start_datetime' (start date for the stress period)
        and 'end_datetime' (end date for the stress period).
    precision : str, {'double', 'single'}
        Precision of data in cell_budget_fil. MODFLOW-6 output is written in double precision
        (by default 'double')
    lakenames : dict, optional
        Dictionary of lake name for each lake number, by default None
    outfile : str, optional
        Output file for saving table of summed lake fluxes, by default 'lake_gw_fluxes.csv'

    Returns
    -------
    df : DataFrame
        Table of summed lake fluxes.
        
    """    
    # open the cell budget file
    cbbobj = bf.CellBudgetFile(cell_budget_file, precision=precision)
    
    # get the times corresponding to each record
    kstpkper = cbbobj.get_kstpkper()
    times = cbbobj.get_times()
    times = [np.round(i) for i in times]
    
    # index perioddata by modflow time
    perioddata.index = perioddata.time
    
    # for each record summarize the in and out fluxes by lake
    sums = []
    for i, (kstp, kper) in enumerate(kstpkper):
        
        # results come out as a list of recarrays
        # cast the first item to a DataFrame
        results = cbbobj.get_data(text='LAK', kstpkper=(kstp, kper))
        df = pd.DataFrame(results[0])
        
        # categorize as in or out
        df['in_out'] = 'in'  # gw inflow to lake
        df.loc[df.q > 0, 'in_out'] = 'out'  # lake leakage to gw
        
        # groupby lake and by in/out; results in 2 levels of columns
        sums_by_lake = df.groupby(['node2', 'in_out']).sum().unstack(level=-1)
        # flatten the columns
        cols = ['_'.join(col).strip() for col in sums_by_lake.columns.values]
        sums_by_lake.columns = cols
        
        # check for only in or only out possibility
        if 'q_in' not in cols:
            sums_by_lake['q_in'] = 0
        if 'q_out' not in cols:
            sums_by_lake['q_out'] = 0
            
        # just keep the fluxes
        sums_by_lake = sums_by_lake[['q_in', 'q_out']].copy()
        sums_by_lake.rename(columns={'q_in': 'gw_in',
                                     'q_out': 'gw_out',
                                     }, inplace=True)
        sums_by_lake['gw_net'] = sums_by_lake['gw_in'] + sums_by_lake['gw_out']
        sums_by_lake.index.name = 'lakeno'
        if lakenames is not None:
            sums_by_lake['lake_name'] = [lakenames.get(no, '') for no in sums_by_lake.index]
        
        # add time information
        sums_by_lake['kstp'] = kstp
        sums_by_lake['kper'] = kper
        sums_by_lake['mf_time'] = times[i]
        sums_by_lake['start_datetime'] = perioddata.loc[times[i], 'start_datetime']
        sums_by_lake['end_datetime'] = perioddata.loc[times[i], 'end_datetime']
        
        sums.append(sums_by_lake)
    
    df = pd.concat(sums)
    df.reset_index(inplace=True)
    df.index = df['start_datetime']
    df.sort_values(by=['lakeno', 'mf_time'], inplace=True)
    
    if outfile is not None:
        df.to_csv(outfile, index=False)
    return df
    

def get_temporal_head_difference_obs(head_obs, perioddata,
                                     head_obs_values_col='obs_head_m',
                                     head_sim_values_col='sim_head_m',
                                     obs_diff_value_col='obsval',
                                     sim_diff_values_col='sim_obsval', 
                                     outfile=None,
                                     write_ins=False):
    """Takes the head_obs dataframe output by get_head_obs, 
    creates temporal head difference observations.

    Parameters
    ----------
    head_obs : DataFrame
    head_obs_values_col : str
        Column in head_obs with observed values to difference.
    head_sim_values_col : str
        Column in head_obs with simulated values to difference.
    obs_diff_value_col : str
        Name of column with computed observed differences.
    sim_diff_values_col : str
        Name of column with computed simulated differences. 
    outfile : str
        CSV file to write output to. By default, None (no output written)
    outfile : str, optional
        Output file of values to be read by PEST, by default 'processed_head_obs_diffs.dat'
    write_ins : bool, optional
        Option to write instruction file, by default False
        
    Returns
    -------
    period_diffs : DataFrame
    """
    # only compute differences on transient obs
    tr_groups = {g for g in head_obs.group.unique() if 'tr' in g}
    # group observations by site (prefix)
    sites = head_obs.loc[head_obs.group.isin(tr_groups)].groupby('obsprefix')
    period_diffs = []
    for site_no, values in sites:
        values = values.sort_values(by=['per'])
        
        # compute the differences
        values[obs_diff_value_col] = values[head_obs_values_col].diff()
        values[sim_diff_values_col] = values[head_sim_values_col].diff()
        
        # base the uncertainty on the amount of time that has elapsed
        # assume 1 meter of annual variability; leaving periods > 1 year at 1 m
        values['uncertainty'] = values.per.diff()/12
        values.loc[values.uncertainty > 1, 'uncertainty'] = 1.
        
        period_diffs.append(values)
    period_diffs = pd.concat(period_diffs).reset_index(drop=True)
    period_diffs['datetime'] = pd.to_datetime(period_diffs['datetime'])
    
    # name the temporal head difference obs as
    # <obsprefix>_<obsname a suffix>d<obsname b suffix>
    # where the obsval = obsname b - obsname a
    obsnme = []
    for i, r in period_diffs.iterrows():
        obs_b_suffix = ''
        if i > 0:
            obs_b_suffix = period_diffs.loc[i-1, 'datetime'].strftime('%Y%m')
        obsnme.append('{}d{}'.format(r.obsnme, obs_b_suffix))
    period_diffs['obsnme'] = obsnme
    period_diffs['group'] = ['{}_tdiff'.format(g) for g in period_diffs['group']]
    
    # drop some columns that aren't really valid; if they exist
    period_diffs.drop(['n', 'obsnme_in_parent'], axis=1, inplace=True, errors='ignore')
    
    # drop observations with no difference (first observations at each site)
    period_diffs.dropna(axis=0, subset=[obs_diff_value_col, sim_diff_values_col], inplace=True)
    period_diffs['type'] = ['lake stage change' if 'lk' in prefix else 'head change' 
                            for prefix in period_diffs.obsprefix]
    
    # fill NaT (not a time) datetimes
    fill_nats(period_diffs, perioddata)

    if outfile is not None:
        period_diffs.fillna(-9999).to_csv(outfile, sep=' ', index=False)

        # write the instruction file
        if write_ins:
            write_insfile(period_diffs, outfile + '.ins',
                          obsnme_column='obsnme', simulated_obsval_column=sim_diff_values_col, 
                          index=False)
    return period_diffs
    

def get_spatial_head_differences(head_obs, perioddata,
                                 lake_head_difference_sites,
                                 head_obs_values_col='obs_head_m',
                                 head_sim_values_col='sim_head_m',
                                 obs_diff_value_col='obsval',
                                 sim_diff_values_col='sim_obsval', 
                                 use_gradients=False, 
                                 write_ins=False, outfile=None):
    """Takes the head_obs dataframe output by get_head_obs_near_lakes, and
    maybe some other input, and creates spatial head difference observations 
    at locations where there are vertical head difference, and writes them 
    to a csv file in tables/.
    
    Parameters
    ----------
    head_obs : DataFrame
        Table of preprocessed head observations
    lake_head_difference_sites : dict
        Dictionary of lake site numbers (keys) and gw level sites (values) to compare.
        Values is list of strings; observations containing these strings will be compared
        to lake stage on the date of measurement.
    use_gradients : bool
        If True, use hydraulic gradients, if False, use vertical head differences.
        By default False.
    """
    
    # get subset of head_obs sites to compare to each lake in lake_head_difference_sites
    groups = head_obs.groupby('obsprefix')
    spatial_head_differences = []
    for lake_site_no, patterns in lake_head_difference_sites.items():
        compare = []
        for pattern in patterns:
            matches = [True if pattern in site_name else False 
                       for site_name in head_obs.obsprefix]
            compare.append(matches)
        compare = np.any(compare, axis=0)
        sites = set(head_obs.loc[compare, 'obsprefix'])
        
        # for each site in the subset, compare the values to the lake
        # index by stress period
        lake_values = groups.get_group(lake_site_no).copy()
        lake_values.index = lake_values.per
        
        for obsprefix, site_observations in groups:
            if obsprefix in sites:
                site_obs = site_observations.copy()
                site_obs.index = site_obs.per
                site_obs['other_obsnme'] = lake_values['obsnme']
                site_obs['obs_lake_stage'] = lake_values[head_obs_values_col]
                site_obs['sim_lake_stage'] = lake_values[head_sim_values_col]
                # negative values indicate discharge to the lake
                # (lake stage < head)
                site_obs['obs_dh'] = site_obs['obs_lake_stage'] - site_obs[head_obs_values_col]
                site_obs['sim_dh'] = site_obs['sim_lake_stage'] - site_obs[head_sim_values_col]
    
                # get a screen midpoint and add gradient
                # assume 1 meter between midpoint and lake if there is no open interval info
                screen_midpoint = site_obs[['sctop', 'scbot']].mean(axis=1).fillna(1)
                site_obs['dz'] = (site_obs['obs_lake_stage'] - screen_midpoint)
                site_obs['obs_grad'] = site_obs['obs_dh']/site_obs['dz']
                site_obs['sim_grad'] = site_obs['sim_dh']/site_obs['dz']
                spatial_head_differences.append(site_obs)
    spatial_head_differences = pd.concat(spatial_head_differences)
    
    # name the spatial head difference obs as
    # <obsprefix>_<obsname suffix>dlake
    obsnme = []
    for i, r in spatial_head_differences.iterrows():
        obs_b_suffix = r.other_obsnme
        obsnme.append('{}d{}'.format(r.obsnme, obs_b_suffix))
    spatial_head_differences['obsnme'] = obsnme
    spatial_head_differences['group'] = ['{}_sdiff'.format(g) 
                                         for g in spatial_head_differences['group']]
    
    # drop some columns that aren't really valid
    spatial_head_differences.drop(['n', 'obsnme_in_parent'], axis=1, inplace=True, errors='ignore')
    
    # whether to use gradients for the obsvals, or just head differences
    if use_gradients:
        spatial_head_differences['obsval'] = spatial_head_differences['obs_grad']
        spatial_head_differences[sim_diff_values_col] = spatial_head_differences['sim_grad']
        obstype = 'vertical head gradients'
    else:
        spatial_head_differences['obsval'] = spatial_head_differences['obs_dh']
        spatial_head_differences[sim_diff_values_col] = spatial_head_differences['sim_dh']
        obstype = 'vertical head difference'
    spatial_head_differences.dropna(axis=0, subset=['obsval'], inplace=True)
    spatial_head_differences['type'] = obstype
    
    # uncertainty column is from head_obs; 
    # assume that spatial head differences have double the uncertainty
    # (two wells/two measurements per obs)
    spatial_head_differences['uncertainty'] *= 2
    
    # check for duplicates
    assert not spatial_head_differences['obsnme'].duplicated().any()
    
    # fill NaT (not a time) datetimes
    fill_nats(spatial_head_differences, perioddata)  
      
    if outfile is not None:
        spatial_head_differences.fillna(-9999).to_csv(outfile, sep=' ', index=False)

        # write the instruction file
        if write_ins:
            write_insfile(spatial_head_differences, outfile + '.ins',
                          obsnme_column='obsnme',
                          simulated_obsval_column=sim_diff_values_col, index=False)
    return spatial_head_differences

    # todo: compute differences and gradients for any paired monitoring wells
    # (check WGNHS report)
def get_modflow_mass_balance(run_dir, modroot, outfile=None, write_ins=True):
    """
    read in the percent discrepancy for inset and parent models
    
    Parameters
    ----------
    run_dir: string pointing to the file locations
    modroot: root name of the model scenario
    outfile: filepath for output
    write_ins: bool. whether or not to write instruction file
    """    
    print('reading in the mass balance files')
    # make a list with which to concatenate results
    dfs = []
    # pull in the stress period dates
    sps = pd.read_csv('../tables/stress_period_data.csv')
    # read in both inset and parent list files
    for cmod in ['inset','parent']:
        # read in the list files
        mfl6 = fp.utils.Mf6ListBudget("{0}{1}_{2}.list".format(rundir, modroot, cmod))
        # get all the budget information
        df,  _ = mfl6.get_dataframes(start_datetime="12-31-2011")
        # construct the obsname with the date etc.
        df['obsnme'] = ['{0}_discrep_{1:d}{2:02d}'.format(cmod, i.year, i.month) for i in df.index]
        # trim only to the stress period end dates
        df.index = pd.to_datetime(df.index)
        df = df.loc[pd.to_datetime(sps.end_datetime.values)]
        
        # append on the max absolute percent discrepancy
        df = df.append({'obsnme':'{}_discrep_max'.format(cmod),
                        'PERCENT_DISCREPANCY':df.PERCENT_DISCREPANCY.abs().max()}, 
                       ignore_index=True)
        dfs.append(df[['obsnme','PERCENT_DISCREPANCY']])
    outdf = pd.concat(dfs)
    outdf['group'] = 'percent_discrep'
    outdf['obsval']= 0
    outdf.to_csv(outfile, index=False, sep=' ')
    if write_ins:
        write_insfile(outdf, outfile + '.ins', obsnme_column='obsnme',
            simulated_obsval_column='PERCENT_DISCREPANCY', index=False)    
    
def write_insfile(results_dataframe, outfile, obsnme_column='obsnme',
                  simulated_obsval_column='modelled', index=True):
    """Write instruction file for PEST. Assumes that
    observations names are in an obsnme_column and
    that the observation values an obsval_column. The values in obsval_column
    will be replaced in the instruction file with the names in obsnme_column.

    Parameters
    ----------
    results_dataframe : pandas dataframe
        Processed model output, in same structure/format
        as the processed output file.
    outfile : filepath
        Name of instruction file.
    obsnme_column : str
        Column in results_dataframe with observation names
    simulated_obsval_column : str
        Column in results_dataframe with the simulated observation equivalents
    index : bool
        Whether or not the index should be included; needs to be the same as the
        actual results file.
    """
    ins = results_dataframe.copy()    
    # if the index is included, move it to the columns
    if index:
        ins.reset_index(inplace=True)
    # fill the index with the 'l1' (line advance) flag for PEST ins file reader
    ins.index = ['l1'] * len(ins)
    cols = ins.columns.tolist()
    
    # replace the observation values with the obsnames
    ins[simulated_obsval_column] = ['!{}!'.format(s) for s in results_dataframe[obsnme_column]]
    
    # fill the remaining columns with whitespace flags
    for c in cols:
        if c != simulated_obsval_column:
            ins[c] = 'w'
            
    # write the output
    with open(outfile, 'w', newline="") as dest:
        dest.write('pif @\n@{}@\n'.format(obsnme_column))
        ins.to_csv(dest, sep=' ', index=True, header=False)
        print('wrote {}'.format(outfile))


if __name__ == '__main__':

    # select directory for the modflow files
    if len(sys.argv) > 1:
        rundir = sys.argv[1]
    else:
        rundir = '../pst_setup/'
        #rundir = '../run_data/'
        #rundir = '.'
    if not rundir.endswith('/'):
        rundir += '/'
    print('running in directory: {}'.format(rundir))

    # set context for making instruction files or not -- boolean. True if setting up, False if running in forward run
        
    if len(sys.argv) > 2:
        make_all_ins = sys.argv[2]
        if 'true' in make_all_ins.lower():
            make_all_ins = True
        else:
            make_all_ins = False
    else:
        make_all_ins = True        
    
    if make_all_ins is True:
        print('Making all instruction files')
    else:
        print('Not making instruction files -> just processing results')
    # parent-specific inputs
    parent_modelgrid_trans = get_modelgrid_transform('{}plsnt_lgr_parent_grid.json'.format(rundir), shift_to_cell_centers=True)
    parent_hk_arrays=sorted(glob.glob('{}plsnt_lgr_parent_k_*.dat'.format(rundir)))
    parent_top_array='{}plsnt_lgr_parent_top.dat'.format(rundir)
    parent_botm_arrays=sorted(glob.glob('{}plsnt_lgr_parent_botm*.dat'.format(rundir)))
    parent_heads_input_file = '{}plsnt_lgr_parent.obs'.format(rundir)
    parent_heads_output_file = '{}plsnt_lgr_parent.head.obs'.format(rundir)
    parent_flux_output_file = '{}plsnt_lgr_parent.sfr.obs.output.csv'.format(rundir)
    
    # inset-specific inputs
    # WDNR obs has longer record of misc values
    # USGS obs has continuous record but only for 2018
    lakenames = {1: 'Pleasant Lake'}
    lake_obs_files = {'Pleasant Lake (WDNR)': '{}lake1.obs.csv'.format(rundir),
                      'Pleasant Lake (USGS)': '{}lake1.obs.csv'.format(rundir),
                      }
    inset_modelgrid_trans = get_modelgrid_transform('{}plsnt_lgr_inset_grid.json'.format(rundir), shift_to_cell_centers=True)
    inset_hk_arrays=sorted(glob.glob('{}plsnt_lgr_inset_k_*.dat'.format(rundir)))
    inset_top_array='{}plsnt_lgr_inset_top.dat'.format(rundir)
    inset_botm_arrays=sorted(glob.glob('{}plsnt_lgr_inset_botm*.dat'.format(rundir)))
    inset_heads_input_file = '{}plsnt_lgr_inset.obs'.format(rundir)
    inset_heads_output_file = '{}plsnt_lgr_inset.head.obs'.format(rundir)
    inset_cell_budget_output = '{}plsnt_lgr_inset.cbc'.format(rundir)
    
    # site that are lake stage obs
    lake_site_numbers = {'Pleasant Lake (WDNR)': '10019209',
                         'Pleasant Lake (USGS)': '4358570893',
              
    }
    
    # sites to difference (values) with lakes (keys)
    # values are patterns to look for in the obsnames
    lake_head_difference_sites = {'4358570893_lk': ['ps_lb', 'psnt']}
    
    # general inputs
    perioddata = pd.read_csv('{}tables/stress_period_data.csv'.format(rundir))
    head_observation_info_file = '../tables/head_obs.csv'
    lakebed_head_observation_info_file = '../tables/lakebed_head_obs.csv'
    flux_observation_inf_file = '../tables/flux_obs.csv'
    
    # outputs
    lake_flux_obs_outfile = '{}Pleasant_lake_gw_fluxes.csv'.format(rundir)
    sfr_flux_outfile = '{}processed_flux_obs.dat'.format(rundir)
    lake_stage_outfile = '{}processed_lake_stages.dat'.format(rundir)
    parent_head_obs_outfile = '{}processed_head_obs_parent.dat'.format(rundir)
    inset_head_obs_outfile = '{}processed_head_obs_inset.dat'.format(rundir)
    parent_thead_diff_obs_outfile = '{}processed_thead_diff_obs_parent.dat'.format(rundir)
    inset_thead_diff_obs_outfile = '{}processed_thead_diff_obs_inset.dat'.format(rundir)
    inset_shead_diff_obs_outfile = '{}processed_shead_diff_obs_inset.dat'.format(rundir)

    # combine the head obs and lakebed head obs info
    head_obs_info = pd.read_csv(head_observation_info_file)
    lakebed_head_observation_info = pd.read_csv(lakebed_head_observation_info_file)
    head_obs_info = head_obs_info.append(lakebed_head_observation_info)
    

    lake_stage_obs = get_lake_stage_obs(lake_obs_files, perioddata, head_obs_info, lake_site_numbers,
                                        outfile=lake_stage_outfile, write_ins=make_all_ins)
    lake_flux_obs = get_lake_in_out_gw_fluxes(inset_cell_budget_output, perioddata,
                                              lakenames=lakenames, outfile=lake_flux_obs_outfile)
    
    # function to get the flux observations and write an ins file
    flux_obs = get_flux_obs(perioddata, model_output_file=parent_flux_output_file, 
                            observed_values_file=flux_observation_inf_file, write_ins=make_all_ins, 
                            outfile=sfr_flux_outfile)
    parent_head_obs = get_head_obs(perioddata, parent_modelgrid_trans, parent_heads_output_file, 
                                   head_obs_info, parent_heads_input_file,
                                   hk_arrays=parent_hk_arrays, top_array=parent_top_array, botm_arrays=parent_botm_arrays,
                                   write_ins=make_all_ins, outfile=parent_head_obs_outfile)
    inset_head_obs = get_head_obs(perioddata, inset_modelgrid_trans, inset_heads_output_file, 
                                  head_obs_info, inset_heads_input_file,
                                   hk_arrays=inset_hk_arrays, top_array=inset_top_array, botm_arrays=inset_botm_arrays,
                                   write_ins=make_all_ins, outfile=inset_head_obs_outfile)
    
    # get temporal head differences (and write ins)
    get_temporal_head_difference_obs(parent_head_obs, perioddata, write_ins=make_all_ins, outfile=parent_thead_diff_obs_outfile)
    get_temporal_head_difference_obs(inset_head_obs, perioddata, write_ins=make_all_ins, outfile=inset_thead_diff_obs_outfile)
    
    # get spatial head differences (and write ins)
    # combine the head and lake stage dataframes to do this
    # (to keep get_spatial_head_differences more general)
    lake_stage_obs_renamed = lake_stage_obs.rename(columns={'obs_stage_m': 'obs_head_m', 'sim_stage_m': 'sim_head_m'})
    cols = [c for c in lake_stage_obs_renamed.columns if c in inset_head_obs.columns]
    combined_obs = inset_head_obs.append(lake_stage_obs_renamed[cols])
    get_spatial_head_differences(combined_obs, perioddata, lake_head_difference_sites,
                                 use_gradients=False,
                                 write_ins=make_all_ins, outfile=inset_shead_diff_obs_outfile)
    # process the list files for mass balance/percent discrepancy
    get_modflow_mass_balance(rundir, 'plsnt_lgr', outfile='{}processed.mass.balance.dat'.format(rundir), write_ins=make_all_ins)
