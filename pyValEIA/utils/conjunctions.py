#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Model-data conjunction functions."""

import datetime as dt
import numpy as np
import pandas as pd

from pyValEIA.utils import coords


def set_swarm_alt(sat_id):
    """Set the Swarm satellite altitude.

    Parameters
    ----------
    sat_id : str
        Satellite ID, expects one of 'A', 'B', or 'C'

    Returns
    -------
    sat_alt : float
        Satellite altitude in km

    Raises
    ------
    ValueError
        If an unknown satellite ID is entered
    
    """
    if sat_id.upper() not in ['A', 'B', 'C']:
        raise ValueError('unknown Swarm satellite: {:}'.format(sat_id))
    
    sat_alt = 511.0 if sat_id.upper() == 'B' else 462.0

    return sat_alt


def swarm_conjunction(mod_dc, swarm_check, alt_str='hmf2', inc=0, max_tdif=15,
                      offset=0, mk_time='time', mk_lon='glon', mk_lat='glat',
                      mk_alt='alt', mk_ne='dene', mk_hmf2='hmf2',
                      mk_nmf2='nmf2', mod_loc_type='geo'):
    """Find conjunctions between a model and Swarm.

    Parameters
    ----------
    mod_dc : dict
        Dictionary of model data
    swarm_check : pd.DataFrame
        DataFrame of Swarm data
    alt_str: str kwarg
        'A', 'B', 'C' or 'hmf2' for altitude (default='hmf2')
    inc : int
        Increase altitude by specified incriment in km (default=0)
    max_tdif : double nkwarg
        Maximum time distance (in minutes) between a NIMO and Swarm
        conjunction allowed (default=15)
    offset : int
        Number of days beyond the loaded Swarm period to check (default=0)
    mk_time : str
        Model time key (default='time')
    mk_lon : str
        Model longitude key (default='glon')
    mk_lat : str
        Model latitude key (default='glat')
    mk_alt : str
        Model altitude key (default='alt')
    mk_ne : str
        Model electron density key (default='dene')
    mk_hmf2 : str
        Model hmF2 key (default='hmf2')
    mk_nmf2 : str
        Model NmF2 key (default='nmf2')
    mod_loc_type : str
        Model latitude and longitude type, expects either 'geo' for
        geodetic/geographic or 'mag' for magnetic (default='geo')

    Returns
    -------
    mod_df : pd.DataFrame
        NIMO data at Swarm location/time
    mod_map : dict
        Dictionary of 2D arrays of NmF2, geo lon, and geo lat prepared for
        map plots

    Raises
    ------
    ValueError
        If NIMO time and starting Swarm time are more than `max_tdif` apart,
        if Swarm altitude is greater than 600 km, or
        if the model location type is not one of 'geo' or 'mag'

    Notes
    -----
    This returns the model data along the equatorial meridian Swarm intersects,
    in between the Swarm magnetic latitude limits.  This is not a pairing of
    model and satellite data, since using as many points as possible is the
    best strategy for evaluating the EIA morphology.

    The function expects time to be the first index in the model data.  For the
    hmf2, it expects latitude as the second index and longitude as the third
    index.

    """
    # Define the start and end times for Swarm during the conjunction
    sw_time1 = swarm_check["Time"].iloc[0] + dt.timedelta(days=offset)
    sw_time2 = swarm_check["Time"].iloc[-1] + dt.timedelta(days=offset)

    # Use mediam swarm altitude for model
    sw_alt = np.nanmedian(swarm_check['Altitude'])

    # Make sure that altitude provided is reasonable
    if sw_alt > 600:
        raise ValueError(f"Altitude of {sw_alt} not reasonable for Swarm")

    # Conjunction Longitude Range for Swarm
    sw_lon1 = min(swarm_check["Longitude"])
    sw_lon2 = max(swarm_check["Longitude"])
    sw_lon_check = ((sw_lon1 + sw_lon2) / 2)

    # Check longitudes and times for NIMO
    mod_lon_ch = mod_dc[mk_lon][(abs(mod_dc[mk_lon] - sw_lon_check)
                                 == min(abs(mod_dc[mk_lon] - sw_lon_check)))]
    mod_time = mod_dc[mk_time][((mod_dc[mk_time] >= sw_time1)
                                & (mod_dc[mk_time] <= sw_time2))]

    # If no time is between sw_time1 and sw_time2 look outside of range
    if len(mod_time) == 0:
        if max_tdif > 0:
            near_tdif = int(np.floor(max_tdif / 3))
            mod_time = mod_dc[mk_time][((mod_dc[mk_time] >= sw_time1
                                         - dt.timedelta(minutes=near_tdif))
                                        & (mod_dc[mk_time] <= sw_time2))]
            if len(mod_time) == 0:
                mod_time = mod_dc[mk_time][
                    ((mod_dc[mk_time] >= sw_time1)
                     & (mod_dc[mk_time] <= sw_time2
                        + dt.timedelta(minutes=near_tdif)))]

    if len(mod_time) > 1:
        mint = np.array([min([abs(mtime - sw_time1), abs(mtime - sw_time2)])
                         for mtime in mod_time])
        mod_time = [mod_time[mint.argmin()]]

    if len(mod_time) == 0:
        mod_time = min(mod_dc[mk_time], key=lambda t: abs(sw_time1 - t))

        if abs(mod_time - sw_time1) < dt.timedelta(minutes=max_tdif):
            mod_time = [mod_time]
        else:
            raise ValueError(
                f"Model {mod_time} - Swarm{sw_time1} > {max_tdif} min")

    # Find the time and place where NIMO coincides with SWARM. Start with the
    # time and lontitude indices
    n_t = np.where(mod_time == mod_dc[mk_time])[0][0]
    n_l = np.where(mod_lon_ch == mod_dc[mk_lon])[0][0]

    # Get the altitude from alt_str and inc
    if alt_str == 'hmf2':  # hmf2(time, lat, lon)
        alt = np.mean(mod_dc[mk_hmf2][n_t, :, n_l])
    else:
        alt = sw_alt

    # Incriment by user specified altitude in km
    alt += inc

    # Altitude index
    n_a = np.where(min(abs(mod_dc[mk_alt] - alt))
                   == abs(mod_dc[mk_alt] - alt))[0][0]

    # Extract the NIMO density and longitudes for the desired slice
    mod_ne_lat_all = mod_dc[mk_ne][n_t, n_a, :, n_l]
    mod_lon_ls = np.full(shape=len(mod_dc[mk_lat]), fill_value=mod_lon_ch[0])

    # Compute NIMO in magnetic coordinates
    if mod_loc_type.lower() == 'geo':
        glat = mod_dc[mk_lat]
        glon = mod_dc[mk_lon]
        mlat, mlon = coords.compute_magnetic_coords(mod_dc[mk_lat],
                                                    mod_lon_ls, mod_time[0])
    elif mod_loc_type.lower() == 'mag':
        mlat = mod_dc[mk_lat]
        mlon = np.full(shape=mlat.shape, fill_value=mod_lon_ch[0])
        glat, glon = coords.compute_magnetic_coords(mod_dc[mk_lat],
                                                    mod_lon_ls, mod_time[0],
                                                    mag_type='geo')
        mod_lon_ch = glon[(abs(glon - sw_lon_check)
                           == min(abs(glon - sw_lon_check)))]
    else:
        raise ValueError('unknown coordinate type: {:}'.format(mod_loc_type))

    # Max and min of Swarm magnetic lats
    sw_mlat1 = min(swarm_check['Mag_Lat'])
    sw_mlat2 = max(swarm_check['Mag_Lat'])

    # Select the same range of magnetic latitudes from NIMO as are available
    # in the Swarm data
    mod_ne_return = mod_ne_lat_all[(mlat >= sw_mlat1) & (mlat <= sw_mlat2)]

    # Set a list of times for output; all are the conjugate time
    time_ls = [mod_time for i in range(len(mod_ne_return))]

    # Create Dataframe of the model data
    mlat_mask = (mlat >= sw_mlat1) & (mlat <= sw_mlat2)
    mod_df = pd.DataFrame()
    mod_df['Time'] = time_ls
    mod_df['Ne'] = mod_ne_return
    mod_df['Mag_Lat'] = mlat[mlat_mask]
    mod_df['Mag_Lon'] = mlon[mlat_mask]
    mod_df['alt'] = np.full(shape=mod_ne_return.shape,
                            fill_value=mod_dc['alt'][n_a])
    mod_df['Longitude'] = np.full(shape=mod_ne_return.shape,
                                  fill_value=mod_lon_ch[0])
    mod_df['Latitude'] = mod_dc[mk_lat][mlat_mask]

    # Save the model map dictionary
    mod_map = {'nmf2': mod_dc[mk_nmf2][n_t], 'glon': glon, 'glat': glat}

    return mod_df, mod_map


def mad_conjunction(mod_dc, mlat_val, lon_val, stime, max_tdif=20, mad_tres=5,
                    lon_type='geo'):
    """Find conjunctions between a model and Madrigal data.

    Parameters
    ----------
    mod_dc : dict
        Dictionary of model data
    mlat_val : double
        +/- magnetic latitude
    lon_val : double
        Geographic or magnetic longitude of conjunction
    stime : dt.datetime
        Datetime for conjunction
    max_tdif : int
        Maximum time difference in minutes (default=20)
    mad_tres : int
        Time resolution of the Madrigal TEC data in minutes (default=5)
    lon_type : str
        Specify whether to select by geographic, 'geo', or magnetic, 'mag',
        longitude (default='geo')

    Returns
    -------
    mod_df : pd.DataFrame
        NIMO data at Madrigal location/time
    mod_map : dict
        Dictionary of 2D arrays of TEC, geo lon, and geo lat for map plots

    Raises
    ------
    ValueError
        For unknown longitude type

    """
    if lon_type.lower() not in ['geo', 'mag']:
        raise ValueError('unknown longitude type: {:}'.format(repr(lon_type)))

    # 15 minute time range
    etime = stime + dt.timedelta(minutes=max_tdif)

    # Get NIMO time of conjunction
    mod_time = mod_dc['time'][((mod_dc['time'] >= stime)
                               & (mod_dc['time'] <= etime))]

    if len(mod_time) == 0:
        mod_time = mod_dc['time'][((mod_dc['time'] >= stime
                                    - dt.timedelta(minutes=mad_tres))
                                   & (mod_dc['time'] <= etime))]
        if len(mod_time) == 0:
            mod_time = mod_dc['time'][((mod_dc['time'] >= stime)
                                       & (mod_dc['time'] <= etime
                                          + dt.timedelta(minutes=mad_tres)))]
    elif len(mod_time) > 1:
        mod_time = [mod_time[0]]
    if len(mod_time) == 0:
        mod_time = min(mod_dc['time'], key=lambda t: abs(stime - t))
        if mod_time - stime < dt.timedelta(minutes=max_tdif):
            mod_time = [mod_time]
        else:
            raise ValueError(f"Model {mod_time} - Mad{stime} > {max_tdif} min")

    # Assign the time index
    n_t = np.where(mod_time == mod_dc['time'])[0][0]

    # Get the longitude of the conjunction
    lon_grid, lat_grid = np.meshgrid(mod_dc['glon'], mod_dc['glat'])
    if lon_type.lower() == 'geo':
        mod_lon_ch = mod_dc['glon'][(abs(mod_dc['glon'] - lon_val)
                                     == min(abs(mod_dc['glon'] - lon_val)))]
        lon_mask = (mod_lon_ch[0] == lon_grid)
        glat = mod_dc['glat']

        # Convert geo to mag coordinates
        mlat, mlon = coords.compute_magnetic_coords(
            mod_dc['glat'], np.full(shape=mod_dc['glat'].shape,
                                    fill_value=mod_lon_ch[0]), mod_time[0])
    else:
        # Convert geo to mag coordinates
        mlat, mlon = coords.compute_magnetic_coords(lat_grid, lon_grid,
                                                    mod_time[0])

        # Get the model longitude resolution
        lon_res = max(mod_dc['glon'][1:] - mod_dc['glon'][:-1])
        lon_mask = abs(mlon - lon_val) < lon_res
        mod_lon_ch = [np.mean(mlon[lon_mask])]
        mlat = mlat[lon_mask]
        mlon = mlon[lon_mask]
        glat = lat_grid[lon_mask]

    # Model COINCIDENCE selection
    mod_tec_lat_all = mod_dc['tec'][n_t, lon_mask]

    # Mask data to the desired magnetic latitude range and order by
    # magnetic latitude
    mlat1 = -1 * abs(mlat_val)
    mlat2 = abs(mlat_val)
    lat_mask = (mlat >= mlat1) & (mlat <= mlat2)
    isort = np.argsort(mlat[lat_mask])

    mod_tec_return = mod_tec_lat_all[(mlat >= mlat1) & (mlat <= mlat2)]
    time_ls = [mod_time for i in range(len(mod_tec_return))]

    mod_df = pd.DataFrame()
    mod_df['Time'] = time_ls
    mod_df['tec'] = mod_tec_return
    mod_df['Mag_Lat'] = mlat[lat_mask][isort]
    mod_df['Mag_Lon'] = mlon[lat_mask][isort]
    mod_df['Longitude'] = np.full(shape=mod_tec_return.shape,
                                  fill_value=mod_lon_ch[0])
    mod_df['Latitude'] = glat[lat_mask][isort]

    mod_map = {'tec': mod_dc['tec'][n_t, :, :], 'glon': mod_dc['glon'],
               'glat': mod_dc['glat']}

    return mod_df, mod_map
