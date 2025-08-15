#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
# NIMO Load
import datetime as dt
import glob
import numpy as np
import os
import pandas as pd

from apexpy import Apex
from netCDF4 import Dataset


def compute_magnetic_coords(lat, lon, time):
    """ compute magnetic coordinates from geographic

    Parameters
    ----------
    lat : array-like
        latitudes
    lon : array-like
        longitudes
    time : array-like
        time
    Returns
    -------
    mlat : array-like
        magnetic latitude
    mlon : array-like
        magnetic longitude
    """
    apex = Apex(date=time[0])
    mlat, mlon = apex.convert(lat, lon, 'geo', 'qd')
    return mlat, mlon


def load_nimo(stime, fdir='/Users/aotoole/Documents/Python_Code/data/NIMO/*',
              name_format='NIMO_AQ_%Y%j', ne_var='dene', lon_var='lon',
              lat_var='lat', alt_var='alt', hr_var='hour', min_var='minute',
              tec_var='tec', hmf2_var='hmf2', nmf2_var='nmf2',
              time_cadence=15):
    """ Load NIMO day
    Parameters
    ----------
    stime : datetime
        day of desired NIMO run
    fdir : str kwarg
        file directory
    name_format : str kwarg
        format of NIMO file name including date format before .nc
        Default: 'NIMO_AQ_%Y%j'
    *_var : str kwarg
        variable names to be opened in the file
        defaults
        --------
        electron density - 'dene'
        geo longitude - 'lon'
        geo latitude - 'lat'
        altitude - 'alt'
        hour - 'hour'
        minute - 'minute'
        TEC - 'tec'
        hmf2 - 'hmf2'
        nmf2 - 'nmf2'
    time_cadence: int kwarg
        time cadence of data in minutes
        default is 15 minutes
    Returns
    -------
    nimo_dc : dictionary
        dictionary with variables: dene,lon,lat,alt,hour,minute,date, tec,hmf2
    """
    fil_str = stime.strftime(name_format)
    name_str = f"{fil_str}.nc"
    fname = os.path.join(fdir, name_str)
    fil = glob.glob(fname)[0]
    nimo_id = Dataset(fil)
    nimo_ne = nimo_id.variables[ne_var][:]
    nimo_lon = nimo_id.variables[lon_var][:]
    nimo_lat = nimo_id.variables[lat_var][:]
    nimo_alt = nimo_id.variables[alt_var][:]
    nimo_hr = nimo_id.variables[hr_var][:]
    if min_var in nimo_id.variables.keys():
        nimo_mins = nimo_id.variables[min_var][:]
    else:
        print('Warning: No minute variable, Treating hour as fractional hours')
        nimo_mins = np.array([(h % 1) * 60 for h in nimo_hr]).astype(int)
        nimo_hr = np.array([int(h) for h in nimo_hr])
    nimo_tec = nimo_id.variables[tec_var][:]
    nimo_hmf2 = nimo_id.variables[hmf2_var][:]
    nimo_nmf2 = nimo_id.variables[nmf2_var][:]
    sday = stime.replace(hour=nimo_hr[0], minute=nimo_mins[0],
                         second=0, microsecond=0)
    nimo_date_list = np.array([sday + dt.timedelta(minutes=(x - 1)
                                                   * time_cadence)
                               for x in range(len(nimo_ne))])
    if np.sign(min(nimo_lat)) != -1:
        print("Warning: No Southern latitudes")
    elif np.sign(max(nimo_lat)) != 1:
        print("Warning: No Northern latitudes")

    nimo_dc = {
        'time': nimo_date_list, 'dene': nimo_ne, 'glon': nimo_lon,
        'glat': nimo_lat, 'alt': nimo_alt,
        'hour': nimo_hr, 'minute': nimo_mins, 'tec': nimo_tec,
        'hmf2': nimo_hmf2, 'nmf2': nimo_nmf2
    }
    return nimo_dc


def nimo_conjunction(nimo_dc, swarm_check, alt_str='hmf2', inc=0, max_tdif=15):
    """ Find conjunction between NIMO and swarm

    Parameters
    ----------
    nimo_dc : dictionary
        dictionary of NIMO data
    swarm_check : dataframe
        dataframe of swarm data
    alt_str: str kwarg
        'A', 'B', 'C' or 'hmf2' (defualt) for altitude
    inc : int kwarg
        increase altitude by inc defulat is 0
    max_tdif : double nkwarg
        maximum time distance (in minutes) between a NIMO and Swarm
        conjunction allowed (default 15)

    Returns
    -------
    nimo_df : DataFrame
        NIMO data at Swarm location/time
    nimo_map : Dictionary
        Dictionary of NmF2, geo lon, and geo lat
        All 2D arrays for a map plot
    Raises
    ------
    Value error if NIMO time and starting Swarm time are more than 15 minutes
        apart
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
    mlat, mlon = compute_magnetic_coords(nimo_dc['glat'],
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


def nimo_mad_conjunction(nimo_dc, mlat_val, glon_val, stime, max_tdif=20):
    """ Find conjunction between NIMO and Madrigal
    Parameters
    ----------
    nimo_dc : dictionary
        dictionary of NIMO data
    mlat_val : double
        +/- magnetic latitude
    glon_val : double
        geographic longitude of conjunction
    stime : Datetime
        datetime for conjunction

    Returns
    -------
    nimo_df : DataFrame
        NIMO data at Madrigal location/time
    nimo_map : Dictionary
        Dictionary of TEC, geo lon, and geo lat
        All 2D arrays for a map plot
    """
    # 15 minute time range
    etime = stime + dt.timedelta(minutes=15)

    # Get NIMO longitudes and time of conjunction
    nimo_lon_ch = nimo_dc['glon'][(abs(nimo_dc['glon'] - glon_val)
                                   == min(abs(nimo_dc['glon'] - glon_val)))]
    nimo_time = nimo_dc['time'][((nimo_dc['time'] >= stime)
                                 & (nimo_dc['time'] <= etime))]
    if len(nimo_time) == 0:
        nimo_time = nimo_dc['time'][((nimo_dc['time'] >= stime
                                      - dt.timedelta(minutes=5))
                                     & (nimo_dc['time'] <= etime))]
        if len(nimo_time) == 0:
            nimo_time = nimo_dc['time'][((nimo_dc['time'] >= stime)
                                         & (nimo_dc['time'] <= etime
                                            + dt.timedelta(minutes=5)))]
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
    mlat, mlon = compute_magnetic_coords(nimo_dc['glat'],
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
