#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Load supported data files."""

import datetime as dt
import glob
import numpy as np
import os
import pandas as pd

import cdflib
from netCDF4 import Dataset

from pyValEIA import logger
from pyValEIA.eia.types import clean_type
from pyValEIA.io.download import download_and_unzip_swarm
from pyValEIA.io.write import build_daily_stats_filename
from pyValEIA.stats import skill_score
from pyValEIA.utils import coords


def load_cdf_data(file_path, variable_names):
    """Load a CDF file.

    Parameters
    ----------
    file_path : str
        file path
    variable_names : array-like
        variable names for file to be extracted

    Returns
    -------
    var_dict : dict
        CDF file data with desired variables extracted
    cdf_data : cdflib.cdfread.CDF
        Loaded CDF data from `cdflib`

    See Also
    --------
    cdflib.CDF

    """
    # Load the desired file
    cdf_data = cdflib.CDF(file_path)

    # Extract the desired variable names
    var_dict = dict()
    for var in variable_names:
        try:
            var_dict[var] = cdf_data.varget(var)
        except ValueError:
            logger.warning('Unknown variable {:} in {:}'.format(var, file_path))

    return var_dict, cdf_data


def extract_cdf_time(cdf_data, time_var='Timestamp'):
    """Extract common coordinate data from a CDF file object.

    Parameters
    ----------
    cdf_data : cdflib.cdfread.CDF
        CDF data object from loaded file
    time_var : str
        Time variable (default='Timestamp')

    Returns
    -------
    epoch : array-like
        UT as datetime objects

    Raises
    ------
    ValueError
        If an incorrect variable name is requested

    """
    epoch = cdflib.cdfepoch.to_datetime(cdf_data.varget(time_var))
    epoch = pd.to_datetime(epoch).to_pydatetime()

    return epoch


def load_swarm(start_date, end_date, sat_id, file_dir, instrument='EFI',
               dataset='LP', f_end='0602'):
    """Load Swarm data, downloading any missing files.

    Parameters
    ----------
    start_date : dt.datetime
        Starting time
    end_date : dt.datetime
        Ending time
    sat_id : str
        Swarm satellite ID, one of 'A', 'B', or 'C'
    file_dir : str
        File directory where the instrument directory is located. Files will be
        located in a directory tree specified by `download_and_unzip_swarm`
    instrument : str
        Swarm instrument (default='EFI')
    dataset : str
        Desired dataset acronym from instrument, e.g. 'LP' is Langmuir Probe
        (default='LP')
    f_end : str
        For different data products there are different numbers at the end
        The most common for EFIxLP is '0602' where '0602' represents
        the file version. Other datasets may also have a string that represents
        the record type (default='0602')

    Returns
    -------
    swarm_data : pd.DataFrame
        DataFrame of Swarm data for the desired instrument and satellite

    Raises
    ------
    ValueError
        If an unknown dataset is requested (currently only supports 'LP')

    """
    # Test the input after assigning variables where first variable is the
    # time stamp, the second variable is geodetic latitude, and the third
    # variable is geographic longitude
    variables = {'LP': ["Timestamp", "Latitude", "Longitude", "Ne", "Ne_error",
                        "Te", "Te_error", "Flags_Ne", "Flags_Te", "Flags_LP",
                        "Radius"]}

    if dataset not in variables.keys():
        raise ValueError('unknown Swarm dataset.')

    time_var = variables[dataset][0]
    lat_var = variables[dataset][1]
    lon_var = variables[dataset][2]

    # Set variables to be renamed
    rename = {'LP': {'Flags_Ne': 'Ne_flag', 'Flags_Te': 'Te_flag',
                     'Flags_LP': 'LP_flag', 'Radius': 'Altitude'}}

    # Initalize the output
    swarm_data = pd.DataFrame()

    # Set the base directory
    base_path = os.path.join(file_dir, instrument,
                             'Sat_{:s}'.format(sat_id.upper()))

    # Cycle through the requested times
    file_date = dt.datetime(start_date.year, start_date.month, start_date.day)
    while file_date < end_date:
        search_pattern = os.path.join(base_path, file_date.strftime('%Y'),
                                      file_date.strftime('%Y%m%d'), "*.cdf")

        # Find the desired file
        filename = glob.glob(search_pattern)
        if len(filename) > 0:
            if len(filename) > 1:
                logger.warning(''.join([
                    'found multiple Swarm', sat_id, ' ', instrument,
                    ' files on ', file_date.strftime('%d-%b-%Y'),
                    ' disgarding: {:}'.format(filename[1:])]))
            filename = filename[0]  # There should only be one file per day
        else:
            # Download the missing file
            download_and_unzip_swarm(file_date, sat_id, file_dir,
                                     instrument=instrument, dataset=dataset,
                                     f_end=f_end)

            # Get the downloaded file
            filename = glob.glob(search_pattern)
            if len(filename) > 0:
                filename = filename[0]
            else:
                logger.warning(''.join(['unable to obtain Swarm', sat_id,
                                        ' ', instrument, ' file on ',
                                        file_date.strftime('%d-%b-%Y')]))
                filename = ''

        # Load data if the file exists
        if os.path.isfile(filename):
            # Load all the available, desired data but the time
            data, cdf_data = load_cdf_data(filename, variables[dataset][1:])

            # Load the time as an array of datetime objects
            data['Time'] = extract_cdf_time(cdf_data, time_var=time_var)

            # Get the additional coordinates
            data['Mag_Lat'], data['Mag_Lon'] = coords.compute_magnetic_coords(
                data[lat_var], data[lon_var], data['Time'][0])
            data['LT'] = coords.longitude_to_local_time(data[lon_var],
                                                        data['Time'])

            # Convert altitude to km from meters
            data['Radius'] -= coords.earth_radius(data[lat_var])
            data['Radius'] = data['Radius'] / 1000

            if swarm_data.empty:
                swarm_data = pd.DataFrame(data)
            else:
                swarm_data = pd.concat([swarm_data, pd.DataFrame(data)])

        # Cycle to the next day
        file_date += dt.timedelta(days=1)

    # Trim the DataFrame to the desired time range
    if not swarm_data.empty:
        swarm_data = swarm_data[(swarm_data['Time'] >= start_date)
                                & (swarm_data['Time'] < end_date)]

        if dataset in rename.keys():
            swarm_data = swarm_data.rename(columns=rename[dataset])

    return swarm_data


def load_madrigal(stime, fdir):
    """Load Madrigal TEC data from given time.

    Parameters
    ----------
    stime: datetime object
        Universal time for the desired madrigal output
    fdir : str kwarg
        directory where file is located

    Returns
    -------
    mad_dict : dict
        Dictionary of the madrigal data including: TEC, geographic latitude,
        geographic longitude, TEC error (dTEC), timestamp, and date in the
        datetime format

    Raises
    ------
    ValueError
        If no file was found for the desired date

    Notes
    -----
    This takes in madrgial files of format gps%y%m%dg.002.netCDF4
    5 minute cadence

    """
    # If Time input is not at midnight, convert it
    sday = stime.replace(hour=0, minute=0, second=0, microsecond=0)
    search_pattern = os.path.join(fdir, 'gps{:s}g.00*.netCDF4'.format(
        sday.strftime("%y%m%d")))

    if len(glob.glob(search_pattern)) > 0:
        fname = glob.glob(search_pattern)[0]
    else:
        raise ValueError('No Madrigal File Found for {:}'.format(sday))

    # Load the data file
    file_id = Dataset(fname)

    # Extract the file data
    mad_tec = file_id.variables['tec'][:]
    mad_gdlat = file_id.variables['gdlat'][:]
    mad_glon = file_id.variables['glon'][:]
    mad_dtec = file_id.variables['dtec'][:]
    mad_time = file_id.variables['timestamps'][:]  # every 5 minutes
    mad_date_list = np.array([sday + dt.timedelta(minutes=x * 5)
                              for x in range(288)])

    # Format data into a dict
    mad_dict = {'time': mad_date_list, 'timestamp': mad_time, 'glon': mad_glon,
                'glat': mad_gdlat, 'tec': mad_tec, 'dtec': mad_dtec}

    return mad_dict


def load_nimo(stime, file_dir, name_format='NIMO_AQ_%Y%j', ne_var='dene',
              lon_var='lon', lat_var='lat', alt_var='alt', hr_var='hour',
              min_var='minute', tec_var='tec', hmf2_var='hmf2', nmf2_var='nmf2',
              time_cadence=15):
    """Load daily NIMO model files.

    Parameters
    ----------
    stime : dt.datetime
        Day of desired NIMO run
    file_dir : str
        File directory, wildcards will be resolved but should only result
        in one file per day for the specified `name_format`
    name_format : str
        Format of NIMO file name including date format before .nc
        (default='NIMO_AQ_%Y%j')
    ne_var : str
        Electron density variable name (default='dene')
    lon_var : str
        Geographic longitude variable name (default='lon')
    lat_var : str
        Geodetic latitude variable name (default='lat')
    alt_var : str
        Altitude variable name (default='alt')
    hr_var : str
        UT hour variable name (default='hour')
    min_var : str
        UT minute variable name, or '' if not present (default='minute')
    tec_var : str
        TEC variable name (default='tec')
    hmf2_var : str
        hmF2 variable name (default='hmf2')
    nmf2_var : str
        NmF2 variable name (default='nmf2')
    time_cadence : int
        Model UT output time cadence of data in minutes (default=15)

    Returns
    -------
    nimo_dc : dict
        Dictionary with variables dene, glon, glat, alt, hour, minute, date,
        tec, nmf2, and hmf2

    Raises
    ------
    ValueError
        If no NIMO file could be found at the specified location and time
    KeyError
        If an unexpected variable is supplied

    """
    # Use the time to format the file name
    name_str = "{:s}.nc".format(stime.strftime(name_format))

    # Construct the file path and use glob to resolve any fill values
    fname = os.path.join(file_dir, name_str)
    fil = glob.glob(fname)

    # Ensure only one file was returned, warn user if not
    if len(fil) > 0:
        nimo_id = Dataset(fil[0])

        if len(fil) > 1:
            logger.warning(''.join(['multiple NIMO file identified for ',
                                    stime.strftime('%d-%b-%Y'), ', disgarding ',
                                    '{:}'.format(fil[1:])]))
    else:
        raise ValueError('No NIMO file found for {:} at {:}'.format(
            stime, fname))

    # Test the input variable keys
    for var in [ne_var, tec_var, hmf2_var, hmf2_var, lon_var, lat_var, alt_var,
                hr_var]:
        if var not in nimo_id.variables.keys():
            raise KeyError('Bad input variable {:} not in {:}'.format(
                repr(var), nimo_id.variables.keys()))

    # Retrieve the desired density variables
    nimo_ne = nimo_id.variables[ne_var][:]
    nimo_tec = nimo_id.variables[tec_var][:]
    nimo_hmf2 = nimo_id.variables[hmf2_var][:]
    nimo_nmf2 = nimo_id.variables[nmf2_var][:]

    # Get the desired location variables and test for both hemispheres
    nimo_lon = nimo_id.variables[lon_var][:]
    nimo_lat = nimo_id.variables[lat_var][:]
    nimo_alt = nimo_id.variables[alt_var][:]

    if np.sign(min(nimo_lat)) != -1:
        logger.warning("No Southern latitudes")
    elif np.sign(max(nimo_lat)) != 1:
        logger.warning("No Northern latitudes")

    # Retrieve the desired time variables
    nimo_hr = nimo_id.variables[hr_var][:]
    if min_var in nimo_id.variables.keys():
        nimo_mins = nimo_id.variables[min_var][:]
    else:
        logger.info('No minute variable, Treating hour as fractional hours')
        nimo_mins = np.array([(h % 1) * 60 for h in nimo_hr]).astype(int)
        nimo_hr = np.array([int(h) for h in nimo_hr])

    # Format the time
    sday = stime.replace(hour=nimo_hr[0], minute=nimo_mins[0],
                         second=0, microsecond=0)
    nimo_date_list = np.array([sday + dt.timedelta(minutes=(x - 1)
                                                   * time_cadence)
                               for x in range(len(nimo_ne))])

    # Format the output dictionary
    nimo_dc = {'time': nimo_date_list, 'dene': nimo_ne, 'glon': nimo_lon,
               'glat': nimo_lat, 'alt': nimo_alt, 'hour': nimo_hr,
               'minute': nimo_mins, 'tec': nimo_tec, 'hmf2': nimo_hmf2,
               'nmf2': nimo_nmf2}

    return nimo_dc


def load_daily_stats(stime, model, obs, file_dir, **kwargs):
    """Load the daily statistics file with model-data comparisons.

    Parameters
    ----------
    stime : datetime
        day of desired file
    model : str
        Case-sensitive name of model requested (e.g., 'NIMO', 'PyIRI').
    obs : str
        Name of data set requested (e.g., 'SWARM', 'MADRIGAL')
    file_dir : str
        File directory
    kwargs : dict
        Optional kwargs by data type.  Includes 'mad_lon', which expects
        longitudes of either -90 deg E or 60 deg E for Madrigal data.

    Returns
    -------
    stat_data : pd.DataFrame
        Dataframe that includes all information from type file

    Raises
    ------
    ValueError
        If expected file does not exist

    """
    # Build the year and date strings
    date_dir, fname = build_daily_stats_filename(stime, model, obs, file_dir,
                                                 **kwargs)

    # Combine the directory and filename
    fname = os.path.join(date_dir, fname)

    # Test that the file exists
    if not os.path.isfile(fname):
        raise ValueError('Could not file file: {:s}'.format(fname))

    # Open the file
    dat = np.genfromtxt(fname, delimiter=None, dtype=None, skip_header=0,
                        names=True, encoding='utf-8',
                        missing_values='NaN', filling_values=np.nan)

    # If an entire column is np.nan, then genfromtxt cannot interpret the
    # dtype, so it assigns it to True. The following code replaces True with
    # The original np.nan values
    for name in dat.dtype.names:
        col = dat[name]
        if (col.dtype == bool):
            nan_col = np.full(col.shape, np.nan, dtype='float64')
            dat = dat.astype([(n, 'float64',) if n == name else
                              (n, dat.dtype[n]) for n in dat.dtype.names])
            dat[name] = nan_col

    # Save as a DataFrame
    stat_data = pd.DataFrame(dat, columns=dat.dtype.names)

    return stat_data


def multiday_states_report(date_range, daily_dir, model_name, obs_name,
                           obs_constraint, comp_type='eia'):
    """Create a state report for a range of dates with a skill check.

    Parameters
    ----------
    date_range : pd.DateRange
        Date range of desired states files
    daily_dir : str
        Directory containing the daily files
    model_name : str
        Case-sensitive model name to load, expects the capitalization used in
        the daily stats file header. (e.g., 'Nimo', 'PyIRI')
    obs_name : str
        Observation name to load (e.g., 'SWARM', 'MADRIGAL')
    obs_constraint : str or int
        Additional constraint for observation type. For Madrigal, this is the
        longitude as a integer. For Swarm, this is the altitude string, which
        currently accepts 'swarm', 'hmf2', or '100'.
    comp_type : str
        Desired type to check against for orientation or EIA state, expecting
        one of 'eia', 'peak', 'flat', or 'trough' for EIA state comparison or
        one of 'north', 'south', or 'neither' for an orientation comparison
        (default='eia')

    Returns
    -------
    model_frame : pd.DataFrame
        Model longitude, local time, states, directions, and EIA types.  If
        Swarm observations are requested, will also contain a satellite list.
    obs_frame : pd.DataFrame
        Data longitude, local time, states, directions, and EIA types.  If
        Swarm observations are requested, will also contain a satellite list.

    Raises
    ------
    ValueError
        For incompatible input combinations

    See Also
    --------
    load_daily_stats
        For accepted models, observations, and Madrigal longitudes

    """
    # Check to see what we are comparing (states or directions)
    if comp_type.lower() in ['north', 'south', 'neither']:
        orientation = 'direction'
    else:
        orientation = 'state'

    # Initialize the parameter dicts
    mod_dict = {'state': list(), 'direction': list(), 'type': list(),
                'Glon': list(), 'LT': list()}
    obs_dict = {'state': list(), 'direction': list(), 'type': list(),
                'Glon': list(), 'LT': list()}

    # Determine the observation-specific inputs
    if obs_name.lower() == 'swarm':
        load_kwargs = {}
        obs_str = 'Swarm_EIA_Type'
        mod_dict['Sat'] = list()
        obs_dict['Sat'] = list()

        if model_name.lower() == 'pyiri':
            model_str = 'PyIRI_Type'
        else:
            if obs_constraint == 'swarm':
                model_str = '{:s}_Swarm_Type'.format(model_name)
            elif obs_constraint == 'hmf2':
                model_str = '{:s}_hmf2_Type'.format(model_name)
            elif obs_constraint == '100':
                model_str = '{:s}_Swarm100_Type'.format(model_name)
            else:
                raise ValueError(''.join([
                    'Unknown altitude type provided in `obs_constraint`: ',
                    repr(obs_constraint)]))
    elif obs_name.lower() == 'madrigal':
        obs_str = 'Mad_EIA_Type'
        load_kwargs = {'mad_lon': obs_constraint}

        if model_name.lower() == 'pyiri':
            raise ValueError('Have not yet implemented TEC retrieval in PyIRI')
        else:
            model_str = '{:s}_Type'.format(model_name)
    else:
        raise ValueError('Unknown observation source')

    # Cycle through each day
    for sday in date_range.to_pydatetime():
        # Load the desired file
        daily_stats = load_daily_stats(sday, model_name.upper(),
                                       obs_name.upper(), daily_dir,
                                       **load_kwargs)

        # Clean and save the model data
        clean_out = clean_type(daily_stats[model_str].values)
        mod_dict['state'].extend(clean_out[0])
        mod_dict['direction'].extend(clean_out[1])
        mod_dict['type'].extend(daily_stats[model_str].values)

        # Clean and save the observed data
        clean_out = clean_type(daily_stats[obs_str].values)
        obs_dict['state'].extend(clean_out[0])
        obs_dict['directon'].extend(clean_out[1])
        obs_dict['type'].extend(daily_stats[obs_str].values)

        # Save the location information
        mod_dict['Glon'].extend(
            daily_stats['{:s}_Glon'.format(model_name)].values)
        obs_dict['Glon'].extend(
            daily_stats['{:s}_Glon'.format(model_name)].values)
        mod_dict['Glon'].extend(daily_stats['LT_Hour'].values)
        obs_dict['Glon'].extend(daily_stats['LT_Hour'].values)

        if 'Sat' in mod_dict.keys():
            mod_dict['Sat'].extend(daily_stats['Satellite'].values)
            obs_dict['Sat'].extend(daily_stats['Satellite'].values)

    # Cast the dictionaries as pandas DataFrames
    model_frame = pd.DataFrame(mod_dict)
    obs_frame = pd.DataFrame(obs_dict)

    # Get the skill score of the model against the observations
    model_frame['skill'] = skill_score.state_check(
        obs_frame[orientation].values, model_frame[orientation].values,
        state=comp_type)

    return model_frame, obs_frame
