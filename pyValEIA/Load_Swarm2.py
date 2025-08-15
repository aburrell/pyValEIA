#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
# Upgraded Swarm Loading Fxns

import pandas as pd
import numpy as np
import cdflib
from apexpy import Apex
import glob
import os
from download_swarm import download_and_unzip


def longitude_to_local_time(longitude, utc_time):
    """ Convert Longiutde to local time
    Parameters
    ----------
    longitude : array-like
        longitudes
    utc_time : array-like
        time in UT
    Returns
    -------
    local_times : array-like
        local times array
    """
    offset_sec = (3600 * np.array(longitude)) / 15
    offset = pd.to_timedelta(offset_sec, unit='s')
    local_times = pd.to_datetime(utc_time) + offset
    return local_times


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


def load_cdf_data(file_path, variable_names):
    """ Load CDF file
    Parameters
    ----------
    file_path : str
        file path
    variable_names : array-like
        variable names for file to be extracted
    Returns :
        cdf file data
    """
    cdf = cdflib.CDF(file_path)
    return {var: cdf.varget(var) for var in variable_names}, cdf


def extract_common_coords(cdf_file, satellite):
    """ Extract data from cdf_file
    Parameters
    ----------
    cdf_file : dictionary
        cdf file dictionary returned from load_cdf_data
    satellite : str
        'A', 'B', or 'C' for Swarm
    Returns :
        lat : array-like
            geo latitudes
        lon : array-like
            geo longitudes
        time : array-like
            time
        rad : array-like
            satellite altitude in km
    """
    lat = cdf_file.varget('Latitude')
    lon = cdf_file.varget('Longitude')
    epoch = cdf_file.varget('Timestamp')
    time = cdflib.cdfepoch.to_datetime(epoch)
    rad = 511 if satellite == 'B' else 462
    return lat, lon, time, rad


def load_EFI(start_date, end_date, satellite,
             fdir=''):
    """ Load Swarm EFI dataset
    Parameters
    ----------
    start_date : datetime
        starting time
    end_date : dateteim
        ending time
    satellite : str
        'A', 'B', or 'C'
    fdir : str
        File directory
    Returns
    -------
    df2 : Dataframe
        Dataframe of swarm data
    """
    base_path = f"{fdir}/EFI/Sat_{satellite}/{start_date.strftime('%Y')}/"

    date_str = start_date.strftime('%Y%m%d')

    dir_path = f"{base_path}/{date_str}/"
    search_pattern = os.path.join(dir_path, "*.cdf")
    filename = 'swarm_file'
    if len(glob.glob(search_pattern)) > 0:
        filename = glob.glob(search_pattern)[0]
    else:  # Download File
        download_and_unzip(start_date, satellite, fdir)
        # Get File
        if len(glob.glob(search_pattern)) > 0:
            filename = glob.glob(search_pattern)[0]

    if os.path.exists(filename):
        variables = [
            "Timestamp", "Latitude", "Longitude", "Ne", "Ne_error", "Te",
            "Te_error", "Flags_Ne", "Flags_Te", "Flags_LP", "Radius"
        ]

        data, cdf_file = load_cdf_data(filename, variables)
        lat, lon, time, rad_km = extract_common_coords(cdf_file, satellite)
        in_time = pd.DataFrame(time)
        glat, glon = compute_magnetic_coords(lat, lon, in_time[0])
        local_time = longitude_to_local_time(lon, time)

        df = pd.DataFrame({
            "Time": time, "Ne": data["Ne"], "Ne_error": data["Ne_error"],
            "Ne_flag": data["Flags_Ne"], "Te": data["Te"],
            "Te_error": data["Te_error"], "Te_flag": data["Flags_Te"],
            "LP_flag": data["Flags_LP"], "Latitude": lat, "Longitude": lon,
            "Mag_Lat": glat, "Mag_Lon": glon, "LT": local_time
        })
        df2 = df[(df['Time'] > start_date) & (df['Time'] < end_date)]
    else:
        df2 = pd.DataFrame()

    return df2
