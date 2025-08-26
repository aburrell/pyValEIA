#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""NIMO conjunction functions."""

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

    """
    sat_alt = 511.0 if sat_id == 'B' else 462.0

    return sat_alt


def nimo_conjunction(nimo_dc, swarm_check, alt_str='hmf2', inc=0, max_tdif=15):
    """Find conjunctions between NIMO and Swarm.

    Parameters
    ----------
    nimo_dc : dict
        Dictionary of NIMO data
    swarm_check : pd.DataFrame
        DataFrame of Swarm data
    alt_str: str kwarg
        'A', 'B', 'C' or 'hmf2' for altitude (default='hmf2')
    inc : int
        Increase altitude by specified incriment in km (default=0)
    max_tdif : double nkwarg
        Maximum time distance (in minutes) between a NIMO and Swarm
        conjunction allowed (default=15)

    Returns
    -------
    nimo_df : pd.DataFrame
        NIMO data at Swarm location/time
    nimo_map : dict
        Dictionary of 2D arrays of NmF2, geo lon, and geo lat prepared for
        map plots

    Raises
    ------
    ValueError
        If NIMO time and starting Swarm time are more than `max_tdif` apart

    """
    # Define the start and end times for Swarm during the conjunction
    sw_time1 = swarm_check["Time"].iloc[0]
    sw_time2 = swarm_check["Time"].iloc[-1]

    # Conjunction Longitude Range for Swarm
    sw_lon1 = min(swarm_check["Longitude"])
    sw_lon2 = max(swarm_check["Longitude"])
    sw_lon_check = ((sw_lon1 + sw_lon2) / 2)

    # Check longitudes and times for NIMO
    nimo_lon_ch = nimo_dc['glon'][(abs(nimo_dc['glon']
                                       - sw_lon_check)
                                   == min(abs(nimo_dc['glon']
                                              - sw_lon_check)))]
    nimo_time = nimo_dc['time'][((nimo_dc['time'] >= sw_time1)
                                 & (nimo_dc['time'] <= sw_time2))]

    # If no time is between sw_time1 and sw_time2 look outside of range
    if len(nimo_time) == 0:
        nimo_time = nimo_dc['time'][((nimo_dc['time'] >= sw_time1
                                      - dt.timedelta(minutes=5))
                                     & (nimo_dc['time'] <= sw_time2))]
        if len(nimo_time) == 0:
            nimo_time = nimo_dc['time'][((nimo_dc['time'] >= sw_time1)
                                         & (nimo_dc['time']
                                            <= sw_time2
                                            + dt.timedelta(minutes=5)))]
    elif len(nimo_time) > 1:
        nimo_time = [nimo_time[0]]

    if len(nimo_time) == 0:
        nimo_time = min(nimo_dc['time'], key=lambda t: abs(sw_time1 - t))
        if nimo_time - sw_time1 < dt.timedelta(minutes=max_tdif):
            nimo_time = [nimo_time]
        else:
            print(nimo_dc['time'][0])
            raise RuntimeError(
                f"NIMO {nimo_time} - Swarm{sw_time1} > {max_tdif} min")

    # Find the time and place where NIMO coincides with SWARM. Start with the
    # time and lontitude indices
    n_t = np.where(nimo_time == nimo_dc['time'])[0][0]
    n_l = np.where(nimo_lon_ch == nimo_dc['glon'])[0][0]

    # Get the altitude from alt_str and inc
    if (alt_str == 'A') or (alt_str == 'C'):
        alt = 462
    elif alt_str == 'B':
        alt = 511
    elif alt_str == 'hmf2':  # hmf2(time, lat, lon)
        alt = np.mean(nimo_dc['hmf2'][n_t, :, n_l])

    # Incriment by user specified altitude in km
    alt += inc

    # Altitude index
    n_a = np.where(min(abs(nimo_dc['alt'] - alt))
                   == abs(nimo_dc['alt'] - alt))[0][0]

    # Extract the NIMO density and longitudes for the desired slice
    nimo_ne_lat_all = nimo_dc['dene'][n_t, n_a, :, n_l]
    nimo_lon_ls = np.ones(len(nimo_dc['glat'])) * nimo_lon_ch[0]

    # Compute NIMO in magnetic coordinates
    mlat, mlon = coords.compute_magnetic_coords(nimo_dc['glat'],
                                                nimo_lon_ls, nimo_time)

    # Max and min of Swarm magnetic lats
    sw_mlat1 = min(swarm_check['Mag_Lat'])
    sw_mlat2 = max(swarm_check['Mag_Lat'])

    # Select the same range of magnetic latitudes from NIMO as are available
    # in the Swarm data
    nimo_ne_return = nimo_ne_lat_all[(mlat >= sw_mlat1) & (mlat <= sw_mlat2)]

    # Set a list of times for output; all are the conjugate time
    time_ls = []
    for i in range(len(nimo_ne_return)):
        time_ls.append(nimo_time)

    # Create Dataframe of NIMO data
    nimo_df = pd.DataFrame()
    nimo_df['Time'] = time_ls
    nimo_df['Ne'] = nimo_ne_return
    nimo_df['Mag_Lat'] = mlat[(mlat >= sw_mlat1) & (mlat <= sw_mlat2)]
    nimo_df['Mag_Lon'] = mlon[(mlat >= sw_mlat1) & (mlat <= sw_mlat2)]
    nimo_df['alt'] = np.ones(len(nimo_ne_return)) * nimo_dc['alt'][n_a]
    nimo_df['Longitude'] = np.ones(len(nimo_ne_return)) * nimo_lon_ch[0]
    nimo_df['Latitude'] = nimo_dc['glat'][((mlat >= sw_mlat1)
                                           & (mlat <= sw_mlat2))]

    nimo_nmf2 = nimo_dc['nmf2'][n_t, :, :]
    nimo_lat = nimo_dc['glat']
    nimo_lon = nimo_dc['glon']
    nimo_map = {
        'nmf2': nimo_nmf2, 'glon': nimo_lon, 'glat': nimo_lat
    }
    return nimo_df, nimo_map


def nimo_mad_conjunction(nimo_dc, mlat_val, glon_val, stime, max_tdif=20,
                         mad_tres=5):
    """Find conjunctions between NIMO and Madrigal data.

    Parameters
    ----------
    nimo_dc : dict
        Dictionary of NIMO data
    mlat_val : double
        +/- magnetic latitude
    glon_val : double
        Geographic longitude of conjunction
    stime : dt.datetime
        Datetime for conjunction
    max_tdif : int
        Maximum time difference in minutes (default=20)
    mad_tres : int
        Time resolution of the Madrigal TEC data in minutes (default=5)

    Returns
    -------
    nimo_df : pd.DataFrame
        NIMO data at Madrigal location/time
    nimo_map : dict
        Dictionary of 2D arrays of TEC, geo lon, and geo lat for map plots

    """
    # 15 minute time range
    etime = stime + dt.timedelta(minutes=max_tdif)

    # Get NIMO longitudes and time of conjunction
    nimo_lon_ch = nimo_dc['glon'][(abs(nimo_dc['glon'] - glon_val)
                                   == min(abs(nimo_dc['glon'] - glon_val)))]
    nimo_time = nimo_dc['time'][((nimo_dc['time'] >= stime)
                                 & (nimo_dc['time'] <= etime))]
    if len(nimo_time) == 0:
        nimo_time = nimo_dc['time'][((nimo_dc['time'] >= stime
                                      - dt.timedelta(minutes=mad_tres))
                                     & (nimo_dc['time'] <= etime))]
        if len(nimo_time) == 0:
            nimo_time = nimo_dc['time'][((nimo_dc['time'] >= stime)
                                         & (nimo_dc['time'] <= etime
                                            + dt.timedelta(minutes=mad_tres)))]
    elif len(nimo_time) > 1:
        nimo_time = [nimo_time[0]]
    if len(nimo_time) == 0:
        nimo_time = min(nimo_dc['time'], key=lambda t: abs(stime - t))
        if nimo_time - stime < dt.timedelta(minutes=max_tdif):
            nimo_time = [nimo_time]
        else:
            raise (ValueError
                   (f"NIMO {nimo_time} - Mad{stime} > {max_tdif} min"))

    # NIMO COINCIDENCE
    # time and longitude indices
    n_t = np.where(nimo_time == nimo_dc['time'])[0][0]
    n_l = np.where(nimo_lon_ch[0] == nimo_dc['glon'])[0][0]

    nimo_tec_lat_all = nimo_dc['tec'][n_t, :, n_l]
    # Convert geo to mag coor
    nimo_lon_ls = np.ones(len(nimo_dc['glat'])) * nimo_lon_ch[0]
    mlat, mlon = coords.compute_magnetic_coords(nimo_dc['glat'],
                                                nimo_lon_ls, nimo_time)

    mlat1 = -1 * abs(mlat_val)
    mlat2 = abs(mlat_val)

    nimo_tec_return = nimo_tec_lat_all[(mlat >= mlat1) & (mlat <= mlat2)]
    time_ls = []
    for i in range(len(nimo_tec_return)):
        time_ls.append(nimo_time)

    nimo_df = pd.DataFrame()
    nimo_df['Time'] = time_ls
    nimo_df['tec'] = nimo_tec_return
    nimo_df['Mag_Lat'] = mlat[(mlat >= mlat1) & (mlat <= mlat2)]
    nimo_df['Mag_Lon'] = mlon[(mlat >= mlat1) & (mlat <= mlat2)]
    nimo_df['Longitude'] = np.ones(len(nimo_tec_return)) * nimo_lon_ch[0]
    nimo_df['Latitude'] = nimo_dc['glat'][(mlat >= mlat1) & (mlat <= mlat2)]

    nimo_nmf2 = nimo_dc['tec'][n_t, :, :]
    nimo_lat = nimo_dc['glat']
    nimo_lon = nimo_dc['glon']
    nimo_map = {
        'tec': nimo_nmf2, 'glon': nimo_lon, 'glat': nimo_lat
    }

    return nimo_df, nimo_map
