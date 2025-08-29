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
from pyValEIA.io.download import download_and_unzip_swarm
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
                     'Flags_LP': 'LP_flag'}}

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
