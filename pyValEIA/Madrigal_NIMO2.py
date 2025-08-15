#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------

import datetime as dt
import glob
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import pydarn

from pyValEIA.EIA_type_detection import eia_complete
from pyValEIA import Load_NIMO2


def longitude_to_local_time(longitude, utc_time):
    """ Computes local time from longitude
    Parameters
    ----------
    longiutde: array-like
        array of longiudes
    utc_time : array-like datetime
        time (UT)
    """
    offset_sec = (3600 * np.array(longitude)) / 15
    offset = pd.to_timedelta(offset_sec, unit='s')
    return pd.to_datetime(utc_time) + offset


def load_madrigal(stime, fdir):

    """ Load madrigal tec data from given time

    Parameters
    ----------
    stime: datetime object
        Universal time for the desired madrigal output
    fdir : str kwarg
        directory where file is located
    Returns
    -------
    mad_dc : dictionary object
        dictionary of the madrigal data including:
        tec, geographic latitude, geographic longitude,
        dtec, timestamp, date (datetime format),
        magnetic latitude, magnetic longitude

    Notes
    -----
    This takes in madrgial files of format gps%y%m%dg.002.netCDF4
    5 minute cadence

    """
    # If Time input is not at midnight, convert it
    sday = stime.replace(hour=0, minute=0, second=0, microsecond=0)
    dt_str = sday.strftime("%y%m%d")
    search_pattern = os.path.join(fdir, 'gps' + dt_str + 'g.00*.netCDF4')

    if len(glob.glob(search_pattern)) > 0:
        fname = glob.glob(search_pattern)[0]
    else:  # Download File
        raise RuntimeError(f'No Madrigal File Found for {dt_str}')
    file_id = Dataset(fname)

    mad_tec = file_id.variables['tec'][:]
    mad_gdlat = file_id.variables['gdlat'][:]
    mad_glon = file_id.variables['glon'][:]
    mad_dtec = file_id.variables['dtec'][:]
    mad_time = file_id.variables['timestamps'][:]  # every 5 minutes
    mad_date_list = np.array([sday + dt.timedelta(minutes=x * 5)
                              for x in range(288)])

    mad_dc = {
        'time': mad_date_list, 'timestamp': mad_time, 'glon': mad_glon,
        'glat': mad_gdlat, 'tec': mad_tec, 'dtec': mad_dtec}
    return mad_dc


def madrigal_nimo_world_maps(stime, mad_dc, nimo_map):
    """ Plot world maps for both nimo and madrigal tec

    Parameters
    ----------
    stime: datetime object
        Universal time for the tec data and solar terminator
    mad_dc : dictionary
        Madrigal data input
    nimo_map : dictionary
        NIMO data input
    Returns
    -------
    fig : figure
        matplotlib figure with 2 panels (Madrigal (top) NIMO (bottom))
        Not automatically saved
    """
    # Get antisolar position and the arc (terminator) at the given height
    antisolarpsn, arc, ang = pydarn.terminator(stime, 300)
    lat_antisolar = antisolarpsn[1]
    lon_antisolar = antisolarpsn[0]

    # Get positions along the terminator arc in geographic coordinates
    lats = []
    lons = []

    for b in range(-180, 180, 1):  # Iterate over longitudes from -180 to 180
        lat, lon = pydarn.GeneralUtils.new_coordinate(lat_antisolar,
                                                      lon_antisolar, arc, b,
                                                      R=pydarn.Re)
        lats.append(lat)
        lons.append(lon)
    lons = [(lon + 180) % 360 - 180 for lon in lons]

    time_remain = stime.minute % 5
    time_min = stime.minute
    if time_remain != 0:
        if time_remain < 3:
            stime = stime.replace(minute=time_min - time_remain)
        else:
            stime = stime.replace(minute=stime.minute + 5 - stime.minute % 5)

    m_t = np.where(stime == mad_dc['time'])[0][0]

    # Plot Madrigal
    fig = plt.figure(figsize=(15, 12))
    plt.rcParams.update({'font.size': 15})
    gs = gridspec.GridSpec(2, 1)
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax.set_global()

    # Add coaslines
    ax.add_feature(cfeature.COASTLINE)
    heatmap = ax.pcolormesh(mad_dc['glon'], mad_dc['glat'],
                            mad_dc['tec'][m_t, :, :], cmap='cividis', vmin=0,
                            vmax=25, transform=ccrs.PlateCarree())
    ax.scatter(lons, lats, color='orange', s=1, zorder=1.0, linewidth=1.0)
    ax.plot(lons[0], lats[0], color='orange', linestyle=':', zorder=2.0,
            linewidth=2.0, label='Terminator 300km')
    leg = ax.legend(framealpha=0, loc='upper right')
    for text in leg.get_texts():
        text.set_color('white')

    cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical',
                        pad=0.02, shrink=0.7)

    # Set gray facecolor with alpha = 0.7
    ax.set_facecolor((0.5, 0.5, 0.5, 0.7))

    cbar.set_label("Madrigal TEC")

    # x and y labels
    ax.text(-215, -40, 'Geographic Latitude', color='k', rotation=90)
    ax.text(-50, -110, 'Geographic Longitude', color='k')

    gl = ax.gridlines(draw_labels=True, linewidth=0, color='gray', alpha=0.5)
    gl.top_labels = False  # Optional: Turn off top labels
    gl.right_labels = False  # Optional: Turn off right labels

    # plot NIMO TEC
    ax = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax.set_global()

    # Add coastlines
    ax.add_feature(cfeature.COASTLINE)

    heatmap = ax.pcolormesh(nimo_map['glon'], nimo_map['glat'],
                            nimo_map['tec'], cmap='cividis',
                            transform=ccrs.PlateCarree())
    plt.rcParams.update({'font.size': 15})
    lons = np.squeeze(lons)
    lats = np.squeeze(lats)
    ax.scatter(lons, lats, color='orange', s=1, zorder=2.0, linewidth=1.0)
    ax.plot(lons[0], lats[0], color='orange', linestyle=':', zorder=2.0,
            linewidth=2.0, label='Terminator 300km')
    leg = ax.legend(framealpha=0, loc='upper right')
    for text in leg.get_texts():
        text.set_color('white')
    # force x and y labels
    ax.text(-215, -40, 'Geographic Latitude', color='k', rotation=90)
    ax.text(-50, -110, 'Geographic Longitude', color='k')

    cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', pad=0.02,
                        shrink=0.8)
    cbar.set_label("NIMO TEC")
    gl = ax.gridlines(draw_labels=True, linewidth=0, color='gray', alpha=0.5)
    gl.top_labels = False  # Turn off top labels
    gl.right_labels = False  # Turn off right labels

    plt.suptitle(str(mad_dc['time'][m_t]), x=0.5, y=0.92, fontsize=20)

    return fig


def detect_outliers(arr):
    """ Detect outliers in an array
    Parameters:
    -----------
    arr: numpy array object
        set of numbers
    Returns:
    --------
    outlier_indices: numpy array object
        array of indices where arr has outliers
    Notes:
    ------
    Uses InterQuartile Range (IQR)
    IQR = q3-q1
    outlier > q3 + 1.5*IQR
    outlier < q1 - 1.5*IQR
    """
    arr = np.array(arr)
    q1 = np.percentile(arr[np.isfinite(arr)], 25)
    q3 = np.percentile(arr[np.isfinite(arr)], 75)
    IQR = q3 - q1
    upper_lim = q3 + 1.5 * IQR
    lower_lim = q1 - 1.5 * IQR
    outlier_indices = np.where((arr > upper_lim) | (arr < lower_lim))[0]

    return outlier_indices


def mad_tec_clean(mad_tec_meas, mad_std_meas, mad_mlat, mlat_val, max_nan=20):
    """ clean madrigal tec data
    Parameters
    ----------
    mad_tec_meas : array-like
        averaged tec over longitude and time
    mad_std_meas : array-like
        standard deviation of mad_tec_meas
    mad_mlat : array-like
        magnetic laittude of mad_tec_meas
    mlat_val : int
        magnetic latitude cutoff
    max_nan : double
        Maximum acceptable percent nan values in a pass
    """
    # minimum is 20 degree cutoff on either side
    # filter by by magnetic latitude (start with given mlat_val)
    mad_tec_lat = mad_tec_meas[abs(mad_mlat) < mlat_val]
    mad_std_lat = mad_std_meas[abs(mad_mlat) < mlat_val]

    if np.all(mad_tec_lat[np.isfinite(mad_tec_lat)] < 5):
        mad_tec_lat[:] = np.nan
        mad_std_lat[:] = np.nan

    nan_perc = (np.isnan(mad_tec_lat).mean() * 100)

    if nan_perc != 100:

        # remove oultier tec values
        out_tec = detect_outliers(mad_tec_lat)
        mad_tec_lat[out_tec] = np.nan
        mad_std_lat[out_tec] = np.nan

    # calculate nan percent
    nan_perc = (np.isnan(mad_tec_lat).mean() * 100)
    mlat_try = mlat_val

    # if nan_perc is greater than max_nan,
    # we want to try to get it below 20 until we hit max_nan degrees mag lat
    if (nan_perc > max_nan) & (nan_perc < 80):
        while (nan_perc > max_nan) & (mlat_try >= max_nan) & (nan_perc < 80):
            mlat_try = mlat_try - 1
            mad_tec_lat = mad_tec_meas[abs(mad_mlat) < mlat_try]
            mad_std_lat = mad_std_meas[abs(mad_mlat) < mlat_try]

            # remove oultier tec values
            out_tec = detect_outliers(mad_tec_lat)
            mad_tec_lat[out_tec] = np.nan
            mad_std_lat[out_tec] = np.nan

            # calculate nan percent
            nan_perc = (np.isnan(mad_tec_lat).mean() * 100)

    # if all data is below 5, then remove completely
    if np.all(mad_tec_lat[np.isfinite(mad_tec_lat)] < 5):
        mad_tec_lat[:] = np.nan
        mad_std_lat[:] = np.nan

    # calculate nan percent one final time
    nan_perc = (np.isnan(mad_tec_lat).mean() * 100)

    return mad_tec_lat, mad_std_lat, nan_perc, mlat_try


def mad_nimo_single_plot(mad_dc, nimo_dc, lon_start, stime, mlat_val,
                         max_nan=20, fosi=14):
    """ Plot 1 madrigal nimo plot
    Parameters
    ----------
    mad_dc : dictionary
        dict of madrigal data
    nimo_dc : dictionary
        dict of nimo data
    lon_start : int
        starting longitude for plot. i.e. 90
    stime : datetime
        datetime for plot
    mlat_val : int
        magnetic latitude cutoff
    max_nan : double kwarg
        Maximum acceptable percent nan values in a pass
    fosi : int kwarg
        font size
    Returns
    -------
    single figure of madrigal and nimo not automatically saved
    """

    # get time index and adjust if minute is not a factor of 5 lik mad data
    time_remain = stime.minute % 5
    time_min = stime.minute
    if time_remain != 0:
        if time_remain < 3:
            stime = stime.replace(minute=time_min - time_remain)
        else:
            stime = stime.replace(minute=stime.minute + 5 - stime.minute % 5)

    # Get index closest to input time
    mt = np.where(stime == mad_dc['time'])[0][0]

    # Intialize figure
    j = 2
    fig = plt.figure(figsize=(25, 24))
    plt.rcParams.update({'font.size': fosi})
    mlat_val_og = mlat_val
    for i in range(12):
        mlat_val = mlat_val_og

        # longiutdinal range
        lon_min = lon_start + 5 * i
        lon_max = lon_start + 5 * (i + 1)

        # compute magnetic latitude
        mad_lon_ls = np.ones(len(mad_dc['glat'])) * (lon_min + lon_max) / 2
        mad_mlat, mad_mlon = Load_NIMO2.compute_magnetic_coords(
            mad_dc['glat'], mad_lon_ls, [mad_dc['time'][mt]])

        # tec and dtec values by time
        mad_tec_T = mad_dc['tec'][mt:mt + 3, :, :]
        mad_dtec_T = mad_dc['dtec'][mt:mt + 3, :, :]

        # by longitude
        mad_tec_lon = mad_tec_T[:, :, ((mad_dc['glon'] >= lon_min)
                                       & (mad_dc['glon'] < lon_max))]
        mad_dtec_lon = mad_dtec_T[:, :, ((mad_dc['glon'] >= lon_min)
                                         & (mad_dc['glon'] < lon_max))]
        mad_tec_lon[mad_dtec_lon > 2] = np.nan
        mad_dtec_lon[mad_dtec_lon > 2] = np.nan

        # calculate the mean of all tec values and
        # pick out the largest dtec value, for all latitudes
        mad_tec_meas = []
        mad_std_meas = []
        for r in range(np.shape(mad_tec_lon)[1]):
            rr = np.array(mad_tec_lon[:, r, :])
            if not np.all(np.isnan(rr)):
                mad_tec_meas.append(np.nanmean(rr))
                mad_std_meas.append(np.nanstd(rr))
            else:
                mad_tec_meas.append(np.nan)
                mad_std_meas.append(np.nan)
        mad_tec_meas = np.array(mad_tec_meas)
        mad_std_meas = np.array(mad_std_meas)

        # remove outliers and clean data
        mad_tec_meas, mad_std_meas, nan_perc, mlat_val = mad_tec_clean(
            mad_tec_meas, mad_std_meas, mad_mlat, mlat_val)

        # get nimo data ------------------------------------------------
        glon_val = (lon_max + lon_min) / 2
        nimo_df, nimo_map = Load_NIMO2.nimo_mad_conjunction(nimo_dc, mlat_val,
                                                            glon_val, stime)

        # Add legend as first panel
        if i == 0:
            ax = fig.add_subplot(4, 3, 1)
            ax.plot(mad_mlat[abs(mad_mlat) < mlat_val],
                    mad_tec_meas, linestyle='-.', label='Madrigal TEC')
            ax.plot(mad_mlat[abs(mad_mlat) < mlat_val],
                    mad_tec_meas, color='orange',
                    label='Madrigal Barrel Average')
            ax.fill_between(mad_mlat[abs(mad_mlat) < mlat_val],
                            mad_tec_meas - mad_std_meas,
                            mad_tec_meas + mad_std_meas, color='g', alpha=0.2,
                            label='Tec +/- dTec')
            ax.plot(mad_mlat[abs(mad_mlat) < mlat_val],
                    mad_tec_meas, linestyle='--', color='k', label='NIMO TEC')
            ax.set_ylim([-99, -89])
            ax.set_xlim([-100, -99])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.legend()
            ax.axis('off')

        mad_df = pd.DataFrame()
        if (nan_perc < max_nan):

            # make plots
            ax = fig.add_subplot(4, 3, j)
            ax.plot(mad_mlat[abs(mad_mlat) < mlat_val], mad_tec_meas)
            ax.scatter(mad_mlat[abs(mad_mlat) < mlat_val], mad_tec_meas)
            ax.fill_between(mad_mlat[abs(mad_mlat) < mlat_val],
                            mad_tec_meas - mad_std_meas,
                            mad_tec_meas + mad_std_meas, color='g', alpha=0.2,
                            label=None)

            nlat = nimo_df['Mag_Lat'].values
            nden = nimo_df['tec'].values
            (nimo_lat, nimo_filt, eia_type_slope, z_lat, plats,
             p3) = eia_complete(nlat, nden, 'tec', interpolate=2,
                                barrel_envelope=False)

            ax.plot(nimo_df['Mag_Lat'], nimo_df['tec'], linestyle='--',
                    color='k', label=eia_type_slope)
            if lon_min < 180:
                mad_df["tec"] = mad_tec_meas
                mad_df["Mag_Lat"] = mad_mlat[abs(mad_mlat) < mlat_val]
                time_ls = []
                for i in range(len(mad_tec_meas)):
                    time_ls.append(mad_dc['time'][mt])
                mad_df["Time"] = np.array(time_ls)

                filt = 'barrel_average'
                lat_use = mad_df["Mag_Lat"].values
                den_mad = mad_df["tec"].values

                (mad_lats, mad_filt,
                 eia_type_slope,
                 z_loc, plats, p3) = eia_complete(lat_use, den_mad, 'tec',
                                                  filt=filt,
                                                  interpolate=2,
                                                  barrel_envelope=False,
                                                  barrel_radius=3)

                ax.plot(mad_lats, mad_filt, color='orange',
                        label=eia_type_slope)
                for pi, p in enumerate(plats):
                    lat_loc = (abs(p - mad_df["Mag_Lat"]).argmin())
                    ax.vlines(mad_df["Mag_Lat"].iloc[lat_loc],
                              ymin=min(mad_df["tec"]),
                              ymax=mad_df["tec"].iloc[lat_loc],
                              alpha=0.5, color='black')
                ax.set_title(str(lon_min) + ' to ' + str(lon_max) + ' GeoLon')
                ax.set_xlim([-mlat_val, mlat_val])
            j = j + 1
        ax.legend()
    if mt + 3 < 288:
        plt.suptitle('Madrigal TEC from ' + str(mad_dc['time'][mt]) + ' to '
                     + str(mad_dc['time'][mt + 3]), x=0.5, y=0.93, fontsize=25)
    else:
        plt.suptitle('Madrigal TEC from ' + str(mad_dc['time'][mt]) + ' to '
                     + str(mad_dc['time'][mt + 2]), x=0.5, y=0.93, fontsize=25)

    return fig


def NIMO_MAD_DailyFile(
        start_day, mad_file_dir, nimo_file_dir, mlat_val=30,
        lon_start=-90, file_save_dir='', fig_on=True, fig_save_dir='',
        max_nan=20, mad_filt='barrel_average', mad_interpolate=2,
        mad_envelope=False, mad_barrel=3, mad_window=3, nimo_filt='',
        nimo_interpolate=2, nimo_envelope=False, nimo_barrel=3, nimo_window=3,
        fosi=15, nimo_name_format='NIMO_AQ_%Y%j', ne_var='dene', lon_var='lon',
        lat_var='lat', alt_var='alt', hr_var='hour', min_var='minute',
        tec_var='tec', hmf2_var='hmf2', nmf2_var='nmf2', nimo_cadence=15,
        max_tdif=20):

    """ Create daily files for Madrigal/NIMO and daily plots if fig_on is True
    Parameters
    ----------
    start_day: datetime
        day of desired files
    mad_file_dir : str kwarg
        Madrigal file directory
    nimo_file_dir : str kwarg
        NIMO file directory
    MLat: int kwarg
        magnetic latitude cutoff
        30 mlat is default
    lon_start : int kwarg
        longitude of desired region
        default is -90, which will span -90 to -30 degrees
        Another Recommended region is 60 to 120 degrees
    file_save_dir : str kwarg
        directory to save file to, default cwd
    fig_on: bool kwarg
        if true (default), plot will be made, if false, plot will not be made
    fig_save_dir : str kwarg
        directory to save figure, default cwd
    max_nan : double kwarg
        Maximum acceptable percent nan values in a pass
    mad_filt : str kwarg
        Desired Filter for madrigal data (default barrel_average)
    mad_interpolate : int kwarg
        int that determines the number of data points in interpolation
        new length will be len(density)xinterpolate
        default is 2 indicating double number of points
    mad_envelope : bool kwarg
        if True, barrel roll will include points inside an
        envelope, if False (default), no envelope will be used
    mad_barrel : double kwarg
        latitudinal radius of barrel for madrigal (default: 3 degrees maglat)
    mad_window : double kwarg
        latitudinal width of moving window (default: 3 degrees maglat)
    nimo_filt : str kwarg
        Desired Filter for nimo data (no filter default)
    nimo_interpolate : int kwarg
        int that determines the number of data points in interpolation
        new length will be len(density)xinterpolate
        default is 2
    nimo_envelope : bool kwarg
        if True, barrel roll will include points inside an
        envelope, if false (default), no envelope will be used
    nimo_barrel : double kwarg
        latitudinal radius of barrel for swarm (default: 3 degrees maglat)
    nimo_window : double kwarg
        latitudinal width of moving window (default: 3 degrees maglat)
    fosi : int kwarg
        fontsize for plot (default is 15)
        Exceptions:
            Super Title (fosi + 10)
    nimo_name_format : str kwarg
        prefix of NIMO file including date format before .nc
        Default: 'NIMO_AQ_%Y%j'
    *_var : str kwarg
        variable names to be opened in the NIMO file
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
    nimo_cadence: int kwarg
        time cadence of NIMO data in minutes
        default is 15 minutes
    max_tdif : double kwarg
        maximum time distance (in minutes) between a NIMO and Madrigal
        conjunction allowed (default 20)
    OUTPUT:
    df : dataframe
        columns ['Mad_Time_Start', 'Mad_MLat', 'Mad_GLon_Start',
                'Mad_GLat_Start', 'LT_Hour', 'Mad_Nan_Percent',
                'Mad_EIA_Type', 'Mad_Peak_MLat1', 'Mad_Peak_TEC1',
                'Mad_Peak_MLat2', 'Mad_Peak_TEC2', 'Mad_Peak_MLat3',
                'Mad_Peak_TEC3', 'Nimo_Time', 'Nimo_GLon', 'Nimo_Min_MLat',
                'Nimo_Max_MLat', 'Nimo_Type','Nimo_Peak_MLat1',
                'Nimo_Peak_TEC1', 'Nimo_Peak_MLat2', 'Nimo_Peak_TEC2',
                'Nimo_Peak_MLat3', 'Nimo_Peak_TEC3', 'Nimo_Third_Peak_MLat1',
                'Nimo_Third_Peak_TEC1']
    fig : figure
        Saved not opened

    """
    columns = ['Mad_Time_Start', 'Mad_MLat', 'Mad_GLon_Start',
               'Mad_GLat_Start', 'LT_Hour', 'Mad_Nan_Percent',
               'Mad_EIA_Type', 'Mad_Peak_MLat1', 'Mad_Peak_TEC1',
               'Mad_Peak_MLat2', 'Mad_Peak_TEC2', 'Mad_Peak_MLat3',
               'Mad_Peak_TEC3', 'Nimo_Time', 'Nimo_GLon', 'Nimo_Min_MLat',
               'Nimo_Max_MLat', 'Nimo_Type', 'Nimo_Peak_MLat1',
               'Nimo_Peak_TEC1', 'Nimo_Peak_MLat2', 'Nimo_Peak_TEC2',
               'Nimo_Peak_MLat3', 'Nimo_Peak_TEC3', 'Nimo_Third_Peak_MLat1',
               'Nimo_Third_Peak_TEC1']
    df = pd.DataFrame(columns=columns)
    sday = start_day.replace(hour=0, minute=0, second=0, microsecond=0)
    mad_dc = load_madrigal(sday, mad_file_dir)
    nimo_dc = Load_NIMO2.load_nimo(
        start_day, fdir=nimo_file_dir, name_format=nimo_name_format,
        ne_var=ne_var, lon_var=lon_var, lat_var=lat_var, alt_var=alt_var,
        hr_var=hr_var, min_var=min_var, tec_var=tec_var, hmf2_var=hmf2_var,
        nmf2_var=nmf2_var, time_cadence=nimo_cadence)  # get nimo data

    f = -1
    mlat_val_og = mlat_val
    for m in range(96):
        m_t = m * 3  # time range 5 minute cadence, 15 minute windows
        stime = sday + dt.timedelta(minutes=5 * m_t)
        mt = np.where(stime == mad_dc['time'])[0][0]
        j = 2
        panel1 = 0
        if fig_on:
            fig = plt.figure(figsize=(25, 24))
            plt.rcParams.update({'font.size': fosi})
        for i in range(12):
            mlat_val = mlat_val_og

            # longiutdinal range
            lon_min = lon_start + 5 * i
            lon_max = lon_start + 5 * (i + 1)

            # compute magnetic latitude
            mad_lon_ls = np.ones(len(mad_dc['glat'])) * (lon_min + lon_max) / 2
            mad_mlat, mad_mlon = Load_NIMO2.compute_magnetic_coords(
                mad_dc['glat'], mad_lon_ls, [mad_dc['time'][mt]])

            # tec and dtec values by time
            mad_tec_T = mad_dc['tec'][mt:mt + 3, :, :]
            mad_dtec_T = mad_dc['dtec'][mt:mt + 3, :, :]

            # by longitude
            mad_tec_lon = mad_tec_T[:, :, ((mad_dc['glon'] >= lon_min)
                                           & (mad_dc['glon'] < lon_max))]
            mad_dtec_lon = mad_dtec_T[:, :, ((mad_dc['glon'] >= lon_min)
                                             & (mad_dc['glon'] < lon_max))]
            mad_tec_lon[mad_dtec_lon > 2] = np.nan
            mad_dtec_lon[mad_dtec_lon > 2] = np.nan

            # calculate the mean of all tec values and
            # pick out the largest dtec value, for all latitudes
            mad_tec_meas = []
            mad_std_meas = []
            for r in range(np.shape(mad_tec_lon)[1]):
                rr = np.array(mad_tec_lon[:, r, :])
                if not np.all(np.isnan(rr)):
                    mad_tec_meas.append(np.nanmean(rr))  # Calculate mean
                    mad_std_meas.append(np.nanstd(rr))  # Calculate stdev
                else:
                    mad_tec_meas.append(np.nan)
                    mad_std_meas.append(np.nan)
            mad_tec_meas = np.array(mad_tec_meas)
            mad_std_meas = np.array(mad_std_meas)

            # remove outliers and clean data
            mad_tec_meas, mad_std_meas, nan_perc, mlat_val = mad_tec_clean(
                mad_tec_meas, mad_std_meas, mad_mlat, mlat_val,
                max_nan=max_nan)

            # get nimo and conjunction
            glon_val = (lon_max + lon_min) / 2
            try:
                nimo_df, nimo_map = Load_NIMO2.nimo_mad_conjunction(
                    nimo_dc, mlat_val, glon_val, stime, max_tdif=max_tdif)
            except ValueError:
                continue

            # create madrigal dataframe
            mad_df = pd.DataFrame()
            mad_df["tec"] = mad_tec_meas
            mad_df["Mag_Lat"] = mad_mlat[abs(mad_mlat) < mlat_val]
            mad_df["GLat"] = mad_dc['glat'][abs(mad_mlat) < mlat_val]
            if (nan_perc < 20):
                f += 1
                df.at[f, 'Mad_Time_Start'] = mad_dc['time'][mt].strftime(
                    '%Y/%m/%d_%H:%M:%S.%f')

                df.at[f, 'Mad_MLat'] = abs(mlat_val)
                df.at[f, 'Mad_GLon_Start'] = lon_min
                df.at[f, 'Mad_GLat_Start'] = max(mad_df["GLat"])

                # calculate Local Time
                # local time halfway between longitudes and between times
                mad_lt = longitude_to_local_time(lon_min, mad_dc['time'][mt])
                lt_hr = mad_lt.hour + mad_lt.minute / 60 + mad_lt.second / 3600
                df.at[f, 'LT_Hour'] = lt_hr
                df.at[f, 'Mad_Nan_Percent'] = nan_perc

                if fig_on:
                    if panel1 == 0:  # Use first panel for legend
                        ax = fig.add_subplot(4, 3, 1)
                        ax.plot(mad_mlat[abs(mad_mlat) < mlat_val],
                                mad_tec_meas, linestyle='-.',
                                label='Madrigal TEC')
                        ax.plot(mad_mlat[abs(mad_mlat) < mlat_val],
                                mad_tec_meas, color='orange',
                                label='Madrigal Barrel Average')
                        ax.fill_between(mad_mlat[abs(mad_mlat) < mlat_val],
                                        mad_tec_meas - mad_std_meas,
                                        mad_tec_meas + mad_std_meas,
                                        color='g', alpha=0.2, label='stdev')
                        ax.plot(mad_mlat[abs(mad_mlat) < mlat_val],
                                mad_tec_meas, linestyle='--', color='k',
                                label='NIMO TEC')
                        ax.set_ylim([-99, -89])
                        ax.set_xlim([-100, -99])
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.legend()
                        ax.axis('off')
                        panel1 = 1

                # Get nimo eia_type ------------------------------------------
                nlat = nimo_df['Mag_Lat'].values
                nden = nimo_df['tec'].values

                (nimo_lat, nimo_tecfilt, eia_type_slope, z_lat, plats,
                 p3) = eia_complete(nlat, nden, 'tec', filt=nimo_filt,
                                    interpolate=nimo_interpolate,
                                    barrel_envelope=nimo_envelope,
                                    barrel_radius=nimo_barrel,
                                    window_lat=nimo_window)

                df.at[f, 'Nimo_Time'] = nimo_df["Time"].iloc[0][0].strftime(
                    '%Y/%m/%d_%H:%M:%S.%f')

                df.at[f, 'Nimo_GLon'] = nimo_df["Longitude"].iloc[0]
                df.at[f, 'Nimo_Min_MLat'] = min(nimo_df["Mag_Lat"])
                df.at[f, 'Nimo_Max_MLat'] = max(nimo_df["Mag_Lat"])
                df.at[f, 'Nimo_Type'] = eia_type_slope
                if len(plats) > 0:  # plot peak latitudes
                    for pi, p in enumerate(plats):
                        lat_loc = (abs(p - nimo_df['Mag_Lat']).argmin())
                        df_strl = 'Nimo_Peak_MLat' + str(pi + 1)
                        df_strn = 'Nimo_Peak_TEC' + str(pi + 1)
                        df.at[f, df_strl] = nimo_df['Mag_Lat'].iloc[lat_loc]
                        df.at[f, df_strn] = nimo_df['tec'].iloc[lat_loc]

                if len(p3) > 0:  # plot third peak for nimo
                    for pi, p in enumerate(p3):
                        lat_loc = (abs(p - nimo_df['Mag_Lat']).argmin())
                        df_strl3 = 'Nimo_Third_Peak_MLat' + str(pi + 1)
                        df_strn3 = 'Nimo_Third_Peak_TEC' + str(pi + 1)
                        df.at[f, df_strl3] = nimo_df['Mag_Lat'].iloc[lat_loc]
                        df.at[f, df_strn3] = nimo_df['tec'].iloc[lat_loc]

                if fig_on:
                    ax = fig.add_subplot(4, 3, j)
                    ax.plot(mad_mlat[abs(mad_mlat) < mlat_val], mad_tec_meas)
                    ax.scatter(mad_mlat[abs(mad_mlat) < mlat_val],
                               mad_tec_meas)
                    ax.fill_between(mad_mlat[abs(mad_mlat) < mlat_val],
                                    mad_tec_meas - mad_std_meas,
                                    mad_tec_meas + mad_std_meas, color='g',
                                    alpha=0.2)
                    ax.plot(nimo_df['Mag_Lat'], nimo_df['tec'], linestyle='--',
                            color='k', label=eia_type_slope)
                if abs(lon_min) < 180:
                    time_ls = []
                    for i in range(len(mad_tec_meas)):
                        time_ls.append(mad_dc['time'][mt])
                    mad_df["Time"] = np.array(time_ls)

                    lats_mad = mad_df['Mag_Lat'].values
                    den_mad = mad_df["tec"].values

                    # Madrigal EIA Type --------------------------------------
                    (mad_lats, mad_tecfilt, eia_type_slope, z_loc, plats,
                     p3) = eia_complete(
                         lats_mad, den_mad, 'tec', filt=mad_filt,
                         interpolate=mad_interpolate,
                         barrel_envelope=mad_envelope,
                         barrel_radius=mad_barrel,
                         window_lat=mad_window)

                    df.at[f, 'Mad_EIA_Type'] = eia_type_slope

                    if fig_on:  # Plot MADRIGAL
                        ax.plot(mad_lats, mad_tecfilt, color='orange',
                                label=eia_type_slope)
                    for pi, p in enumerate(plats):  # Plot Madrigal peaks
                        lat_loc = (abs(p - mad_df["Mag_Lat"]).argmin())
                        df_strl = 'Mad_Peak_MLat' + str(pi + 1)
                        df_strn = 'Mad_Peak_TEC' + str(pi + 1)
                        df.at[f, df_strl] = mad_df["Mag_Lat"].iloc[lat_loc]
                        df.at[f, df_strn] = mad_df["tec"].iloc[lat_loc]
                        if fig_on:
                            ax.vlines(mad_df["Mag_Lat"].iloc[lat_loc],
                                      ymin=min(mad_df["tec"]),
                                      ymax=mad_df["tec"].iloc[lat_loc],
                                      alpha=0.5, color='black')
                    if fig_on:  # add local time
                        lt_plot = np.round(lt_hr, 2)
                        ax.set_title(str(lon_min) + ' to ' + str(lon_max)
                                     + ' GeoLon ' + str(lt_plot) + 'LT')
                        ax.set_xlim([-mlat_val, mlat_val])
                        ax.legend()
                j = j + 1
        if fig_on:
            t1 = mad_dc['time'][mt].strftime('%Y/%m/%d %H:%M')
            ts1 = mad_dc['time'][mt].strftime('%H%M')
            t2 = mad_dc['time'][mt] + dt.timedelta(minutes=15)
            ts2 = t2.strftime('%H%M')
            t2 = t2.strftime('%H:%M')
            plt.suptitle('Madrigal TEC from ' + t1 + '-' + t2, x=0.5, y=0.93,
                         fontsize=fosi + 10)
            ds = mad_dc['time'][mt].strftime('%Y%m%d')
            ys = mad_dc['time'][mt].strftime('%Y')

            # Save Directory
            if fig_save_dir == '':
                fig_save_dir = os.getcwd()
            save_dir = fig_save_dir + ys + '/' + ds
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            # Save Figures
            save_as = (save_dir + '/NIMO_MADRIGAL_' + ds + '_' + ts1 + '_'
                       + ts2 + '_' + str(lon_start) + '_'
                       + str(lon_start + 5 * 12) + 'glon.jpg')
            fig.savefig(save_as)
            plt.close()
            fig_map = madrigal_nimo_world_maps(stime, mad_dc=mad_dc,
                                               nimo_map=nimo_map)
            save_as = (save_dir + '/NIMO_MADRIGAL_MAP_' + ds + '_' + ts1
                       + '_' + ts2 + '.jpg')
            fig_map.savefig(save_as)
            plt.close()

    # Save File
    ds = mad_dc['time'][mt].strftime('%Y%m%d')
    ys = mad_dc['time'][mt].strftime('%Y')

    if file_save_dir == '':
        file_save_dir = os.getcwd()

    file_dir = file_save_dir + ys
    Path(file_dir).mkdir(parents=True, exist_ok=True)
    save_file = (file_dir + '/NIMO_MADRIGAL_EIA_type' + '_' + ds + '_'
                 + str(lon_start) + 'glon_ascii.txt')

    delimiter = '\t'  # Use '\t' for tab-separated text

    # Create the custom header row with a hashtag
    header_line = '#' + delimiter.join(df.columns) + '\n'

    # Write the header to the file
    with open(save_file, 'w') as f:
        f.write(header_line)

    # Append the DataFrame data without the header and index
    df.to_csv(save_file, sep=delimiter, index=False, na_rep='NaN',
              header=False, mode='a', encoding='ascii')

    return df
