#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Functions for plotting Madrigal TEC data and evaluating EIA detection."""

import datetime as dt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pydarn

from pyValEIA import logger
from pyValEIA.eia.detection import eia_complete
from pyValEIA import io
from pyValEIA.utils import coords
from pyValEIA.utils import conjunctions
from pyValEIA.utils.clean import mad_tec_clean


def madrigal_model_world_maps(stime, mad_dc, mod_map):
    """Plot world maps for both model data and Madrigal TEC.

    Parameters
    ----------
    stime : datetime object
        Universal time for the tec data and solar terminator
    mad_dc : dict
        Madrigal data input
    mod_map : dict
        NIMO data input

    Returns
    -------
    fig : figure
        matplotlib figure with 2 panels (Madrigal (top) NIMO (bottom))

    """
    # Get antisolar position and the arc (terminator) at the given height
    antisolarpsn, arc, ang = pydarn.terminator(stime, 300)
    lat_antisolar = antisolarpsn[1]
    lon_antisolar = antisolarpsn[0]

    # Get positions along the terminator arc in geographic coordinates
    lats = []
    lons = []

    # Iterate over longitudes from -180 to 180 with a one degree resolution
    for b in range(-180, 180, 1):
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

    cbar.set_label("Madrigal TEC (TECU)")

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

    heatmap = ax.pcolormesh(mod_map['glon'], mod_map['glat'],
                            mod_map['tec'], cmap='cividis',
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


def mad_model_single_plot(mad_dc, mod_dc, lon_start, stime, mlat_val,
                          model_name='NIMO', max_nan=20, fosi=14):
    """Create one Madrigal TEC vs model data plot.

    Parameters
    ----------
    mad_dc : dict
        dict of Madrigal TEC data
    mod_dc : dict
        dict of model data
    lon_start : int
        starting longitude for plot. i.e. 90
    stime : datetime
        datetime for plot
    mlat_val : int
        magnetic latitude cutoff
    model_name : str
        Name of model (default='NIMO')
    max_nan : float
        Maximum acceptable percent nan values in a pass (default=20)
    fosi : int
        font size (default=14)

    Returns
    -------
    fig : mpl.Figure
        Fingle figure of madrigal and input model, not automatically saved

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
        mad_mlat, mad_mlon = coords.compute_magnetic_coords(
            mad_dc['glat'], mad_lon_ls, mad_dc['time'][mt])

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
        mod_df, mod_map = conjunctions.mad_conjunction(
            mod_dc, mlat_val, glon_val, stime)

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
                    mad_tec_meas, linestyle='--', color='k',
                    label='{:s} TEC'.format(model_name))
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

            nlat = mod_df['Mag_Lat'].values
            nden = mod_df['tec'].values
            (mod_lat, mod_filt, eia_type_slope, z_lat, plats,
             p3) = eia_complete(nlat, nden, 'tec', interpolate=2,
                                barrel_envelope=False)

            ax.plot(mod_df['Mag_Lat'], mod_df['tec'], linestyle='--',
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

                # TODO: add kwarg options to figure input.
                (mad_lats, mad_filt, eia_type_slope, z_loc, plats,
                 p3) = eia_complete(lat_use, den_mad, 'tec', filt=filt,
                                    interpolate=2, barrel_envelope=False,
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


def model_mad_daily_file(start_day, mad_file_dir, mod_file_dir, mod_name_format,
                         model_name='NIMO', mod_load_func=io.load.load_nimo,
                         mlat_val=30, lon_start=-90, file_save_dir='',
                         fig_on=True, fig_save_dir='', max_nan=20,
                         mad_filt='barrel_average', mad_interpolate=2,
                         mad_envelope=False, mad_barrel=3, mad_window=3,
                         mod_filt='', mod_interpolate=2, mod_envelope=False,
                         mod_barrel=3, mod_window=3, fosi=15, ne_var='dene',
                         lon_var='lon', lat_var='lat', alt_var='alt',
                         hr_var='hour', min_var='minute', tec_var='tec',
                         hmf2_var='hmf2', nmf2_var='nmf2', mod_cadence=15,
                         max_tdif=20):
    """Create daily files for Madrigal/model and daily plots.

    Parameters
    ----------
    start_day : datetime
        day of desired files
    mad_file_dir : str
        Madrigal file directory
    mod_file_dir : str
        NIMO file directory
    mod_name_format : str
        prefix of NIMO file including date format before .nc extension, e.g.,
        'NIMO_AQ_%Y%j'
    model_name : str
        Model name (default='NIMO')
    mod_load_func : function
        Function for loading model data (default=`io.load.load_nimo`)
    mlat_val: int
        magnetic latitude cutoff (default=30)
    lon_start : int
        longitude of desired region, e.g., -90 will span -90 to -30 degrees
        (default=-90)
    file_save_dir : str
        directory to save file to (default='')
    fig_on: bool
        if True, plot will be made, if False, plot will not be made
        (default=True)
    fig_save_dir : str
        directory to save figure (default='')
    max_nan : int or float
        Maximum acceptable percent nan values in a pass (default=20)
    mad_filt : str
        Desired Filter for madrigal data (default='barrel_average')
    mad_interpolate : int
        int that determines the number of data points in interpolation
        new length will be len(density) x interpolate (default=2)
    mad_envelope : bool
        if True, barrel roll will include points inside an envelope, if False,
        no envelope will be used (default=False)
    mad_barrel : float
        latitudinal radius of barrel for madrigal (default=3)
    mad_window : float
        latitudinal width of moving window (default=3)
    mod_filt : str
        Desired Filter for nimo data (default='')
    mod_interpolate : int
        int that determines the number of data points in interpolation
        new length will be len(density) x interpolate (default=2)
    mod_envelope : bool
        if True, barrel roll will include points inside an envelope, if False,
        no envelope will be used (default=False)
    mod_barrel : float
        latitudinal radius of barrel for swarm (default=3)
    mod_window : float
        latitudinal width of moving window (default=3)
    fosi : int
        fontsize for plot, with title being `fosi` + 10 (default=15)
    ne_var : str
        Electron denstiy variable in the model file (default='dene')
    lon_var : str
        Longitude variable in the model file (default='lon')
    lat_var : str
        Latitude variable in the model file (default='lat')
    alt_var : str
        Altitude variable in the model file (default='alt')
    hr_var : str
        Hour of day variable in the model file (default='hour')
    min_var : str
        Minute of hour variable in the model file (default='minute')
    tec_var : str
        TEC variable in the model file (default='tec')
    hmf2_var : str
        hmF2 variable in the model file (default='hmf2')
    nmf2_var : str
        NmF2 variable in the model file (default='nmf2')
    mod_cadence : int
        Time cadence of Model data in minutes (default=15)
    max_tdif : float
        Maximum allowed time in minutes between a model and Swarm conjunction
        (default=20)

    Returns
    -------
    df : pd.DataFrame
        DataFrame with Madrigal and model data for the desired longitude sector
    fig : mpl.Figure
        Saved not opened

    """
    # Initialize column names
    col_mod_name = model_name.capitalize()
    columns = ['Mad_Time_Start', 'Mad_MLat', 'Mad_GLon_Start',
               'Mad_GLat_Start', 'LT_Hour', 'Mad_Nan_Percent',
               'Mad_EIA_Type', 'Mad_Peak_MLat1', 'Mad_Peak_TEC1',
               'Mad_Peak_MLat2', 'Mad_Peak_TEC2', 'Mad_Peak_MLat3',
               'Mad_Peak_TEC3', f'{col_mod_name}_Time', f'{col_mod_name}_GLon',
               f'{col_mod_name}_Min_MLat', f'{col_mod_name}_Max_MLat',
               f'{col_mod_name}_Type', f'{col_mod_name}_Peak_MLat1',
               f'{col_mod_name}_Peak_TEC1', f'{col_mod_name}_Peak_MLat2',
               f'{col_mod_name}_Peak_TEC2', f'{col_mod_name}_Peak_MLat3',
               f'{col_mod_name}_Peak_TEC3', f'{col_mod_name}_Third_Peak_MLat1',
               f'{col_mod_name}_Third_Peak_TEC1']
    df = pd.DataFrame(columns=columns)
    sday = start_day.replace(hour=0, minute=0, second=0, microsecond=0)
    mad_dc = io.load_madrigal(sday, mad_file_dir)

    # Load the model data
    mod_dc = mod_load_func(
        start_day, fdir=mod_file_dir, name_format=mod_name_format,
        ne_var=ne_var, lon_var=lon_var, lat_var=lat_var, alt_var=alt_var,
        hr_var=hr_var, min_var=min_var, tec_var=tec_var, hmf2_var=hmf2_var,
        nmf2_var=nmf2_var, time_cadence=mod_cadence)

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
            mad_mlat, mad_mlon = coords.compute_magnetic_coords(
                mad_dc['glat'], mad_lon_ls, mad_dc['time'][mt])

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
                mod_df, mod_map = conjunctions.mad_conjunction(
                    mod_dc, mlat_val, glon_val, stime, max_tdif=max_tdif)
            except ValueError:
                logger.info('no Madrigal/model conjunction at this time')
                continue

            # Create madrigal dataframe
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
                mad_lt = coords.longitude_to_local_time(lon_min,
                                                        mad_dc['time'][mt])
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
                nlat = mod_df['Mag_Lat'].values
                nden = mod_df['tec'].values

                (mod_lat, mod_tecfilt, eia_type_slope, z_lat, plats,
                 p3) = eia_complete(nlat, nden, 'tec', filt=mod_filt,
                                    interpolate=mod_interpolate,
                                    barrel_envelope=mod_envelope,
                                    barrel_radius=mod_barrel,
                                    window_lat=mod_window)

                df.at[f, f'{col_mod_name}_Time'] = mod_df["Time"].iloc[0][
                    0].strftime('%Y/%m/%d_%H:%M:%S.%f')

                df.at[f, f'{col_mod_name}_GLon'] = mod_df["Longitude"].iloc[0]
                df.at[f, f'{col_mod_name}_Min_MLat'] = min(mod_df["Mag_Lat"])
                df.at[f, f'{col_mod_name}_Max_MLat'] = max(mod_df["Mag_Lat"])
                df.at[f, f'{col_mod_name}_Type'] = eia_type_slope
                if len(plats) > 0:  # plot peak latitudes
                    for pi, p in enumerate(plats):
                        lat_loc = (abs(p - mod_df['Mag_Lat']).argmin())
                        df_strl = f'{col_mod_name}_Peak_MLat' + str(pi + 1)
                        df_strn = f'{col_mod_name}_Peak_TEC' + str(pi + 1)
                        df.at[f, df_strl] = mod_df['Mag_Lat'].iloc[lat_loc]
                        df.at[f, df_strn] = mod_df['tec'].iloc[lat_loc]

                if len(p3) > 0:  # plot third peak for nimo
                    for pi, p in enumerate(p3):
                        lat_loc = (abs(p - mod_df['Mag_Lat']).argmin())
                        df_strl3 = f'{col_mod_name}_Third_Peak_MLat' + str(
                            pi + 1)
                        df_strn3 = f'{col_mod_name}_Third_Peak_TEC' + str(
                            pi + 1)
                        df.at[f, df_strl3] = mod_df['Mag_Lat'].iloc[lat_loc]
                        df.at[f, df_strn3] = mod_df['tec'].iloc[lat_loc]

                if fig_on:
                    ax = fig.add_subplot(4, 3, j)
                    ax.plot(mad_mlat[abs(mad_mlat) < mlat_val], mad_tec_meas)
                    ax.scatter(mad_mlat[abs(mad_mlat) < mlat_val],
                               mad_tec_meas)
                    ax.fill_between(mad_mlat[abs(mad_mlat) < mlat_val],
                                    mad_tec_meas - mad_std_meas,
                                    mad_tec_meas + mad_std_meas, color='g',
                                    alpha=0.2)
                    ax.plot(mod_df['Mag_Lat'], mod_df['tec'], linestyle='--',
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
                    if fig_on:
                        # add local time
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
            fig_map = madrigal_model_world_maps(stime, mad_dc=mad_dc,
                                                mod_map=mod_map)
            save_as = os.path.join(save_dir, '_'.join([
                model_name.upper(), 'MADRIGAL', 'MAP', ds, ts1,
                '{:}.jpg'.format(ts2)]))
            fig_map.savefig(save_as)
            plt.close()

    # Save the statistics to a dialy stats file
    io.write.write_daily_stats(df, mad_dc['time'][mt], model_name.upper(),
                               'MADRIGAL', file_save_dir, mad_lon=lon_start)

    return df
