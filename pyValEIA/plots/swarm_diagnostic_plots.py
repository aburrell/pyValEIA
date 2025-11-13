#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Functions for plotting Swarm data and evaluating EIA detection."""

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import gridspec
from matplotlib import colormaps
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy import stats


import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pydarn
import PyIRI
import PyIRI.edp_update as ml

from pyValEIA import logger
from pyValEIA import io
from pyValEIA.utils import coords
from pyValEIA.utils import conjunctions
from pyValEIA.eia.detection import eia_complete
from pyValEIA.io import load
from pyValEIA.utils.filters import find_all_gaps
from pyValEIA.plots import utils as putils


def swarm_panel(axs, stime, satellite, swarm_file_dir, MLat=30,
                swarm_filt='barrel_average', swarm_interpolate=1,
                swarm_envelope=True, swarm_barrel=3, swarm_window=2, fosi=14,
                scale=False, scale_by='num', scale_num=10**5):
    """Plot a single Swarm panel without model data.

    Parameters
    ----------
    axs : matplotlib axis
        axis for the data to be plotted onto
    stime : datetime object
        time of desired plot, nearest time within mlatitudinal window will be
        plotted
    satellite: str
        'A', 'B', or 'C' for Swarm
    swarm_file_dir : str
        directory where swarm file can be found
    MLat : int
        magnetic latitude range +/-MLat (default=30)
    swarm_filt : str
        Desired Filter for swarm data (default='barrel_average')
    swarm_interpolate : int
        int that determines the number of data points in interpolation
        new length will be len(density) x interpolate; if 1 is specified there
        will not be any interpolation. (default=1)
    swarm_envelope : bool
        If True, barrel roll will include points inside an envelope and if
        False, no envelope will be used (default=True)
    swarm_barrel : double
        latitudinal radius of barrel for Swarm in degrees of magnetic latitude
        (default=3)
    swarm_window : float
        latitudinal width of moving window in degrees of magnetic latitude
        (default=2)
    fosi : int
        fontsize for the legend (default=14)
    scale : bool
        Specifies whether to scale the data, if True, or leave as-is, if False
        (default=False)
    scale_by : str
        If `scale` is True, scale the data by a number if 'num' is provided or
        scale the data by the maximum if 'max' is provided (default='num')
    scale_num : float
        If `scale` is True and `scale_by` is 'num', the density data will be
        divided by `scale_num` (default=10**5)

    Returns
    -------
    axs : matplotlib axis
        axis for the data to be plotted onto

    Notes
    -----
    filt options include: 'barrel', 'average', 'median', 'barrel_average'
    'barrel_median', 'average_barrel', and 'median_barrel'

    """
    # Convert to Day if not already
    sday = stime.replace(hour=0, minute=0, second=0, microsecond=0)
    eday = sday + dt.timedelta(days=1)

    # Get full day of Swarm Data
    swarm_df = load.load_swarm(sday, eday, satellite, swarm_file_dir)

    # Housekeeping
    swarm_df['LT_hr'] = swarm_df['LT'].dt.hour + swarm_df['LT'].dt.minute / 60
    swarm_df.loc[(swarm_df['Ne_flag'] > 20), 'Ne'] = np.nan

    sw_lat = swarm_df[(swarm_df["Mag_Lat"] < MLat) & (swarm_df["Mag_Lat"]
                                                      > -MLat)]
    lat_ind = sw_lat.index.values
    gap_all = find_all_gaps(lat_ind)
    start_val = [0]
    end_val = [len(lat_ind)]  # add the beginning and end to gap indices
    gaps = start_val + gap_all + end_val

    # Get closest time to Input
    tim_arg = abs(sw_lat["Time"] - stime).argmin()
    if abs(sw_lat["Time"].iloc[tim_arg] - stime) > dt.timedelta(minutes=10):
        logger.info(f'Selecting {sw_lat["Time"].iloc[tim_arg]}')

    # Choose latitudinally limited segment using gap indices
    gap_arg = abs(tim_arg - gaps).argmin()
    if gaps[gap_arg] <= tim_arg:
        g1 = gap_arg
        g2 = gap_arg + 1
    else:
        g1 = gap_arg - 1
        g2 = gap_arg

    # Desired Swarm Data Segment
    swarm_check = sw_lat[gaps[g1]:gaps[g2]]

    # Evaluate Swarm EIA-------------------------------------------------
    lat_use = swarm_check['Mag_Lat'].values
    density = swarm_check['Ne'].values
    den_str = 'Ne'
    sw_lat, sw_filt, eia_type_slope, z_lat, plats, p3 = eia_complete(
        lat_use, density, den_str, filt=swarm_filt,
        interpolate=swarm_interpolate, barrel_envelope=swarm_envelope,
        barrel_radius=swarm_barrel, window_lat=swarm_window)

    # Plot Findings

    # Ne scaling...
    if scale:
        if scale_by == 'max':
            Ne_sc = swarm_check['Ne'] / max(swarm_check['Ne'])
        elif scale_by == 'num':
            Ne_sc = swarm_check['Ne'] / scale_num
    else:
        Ne_sc = swarm_check['Ne']

    # Add legend labels with Satelltie and times
    d1 = swarm_check['Time'].iloc[0].strftime('%Y %b %d')
    t1 = swarm_check['Time'].iloc[0].strftime('%H:%M')
    t2 = swarm_check['Time'].iloc[-1].strftime('%H:%M')
    axs.plot(swarm_check['Mag_Lat'], Ne_sc, linestyle='-')
    axs.scatter(swarm_check['Mag_Lat'].iloc[0], Ne_sc.iloc[0],
                color='white', s=0, label=f'Swarm {satellite}')
    axs.scatter(swarm_check['Mag_Lat'].iloc[0], Ne_sc.iloc[0],
                color='white', s=0, label=f'{d1}')
    axs.scatter(swarm_check['Mag_Lat'].iloc[0], Ne_sc.iloc[0],
                color='white', s=0, label=f'{t1}-{t2}UT')

    # Add peak lat lines
    if len(plats) > 0:
        for pi, p in enumerate(plats):
            lat_loc = (abs(p - swarm_check['Mag_Lat']).argmin())
            axs.vlines(swarm_check['Mag_Lat'].iloc[lat_loc],
                       ymin=min(Ne_sc),
                       ymax=Ne_sc.iloc[lat_loc], alpha=0.8,
                       color='orange')
    # Set axis info
    axs.grid(axis='x')
    axs.set_xlim([min(swarm_check['Mag_Lat']), max(swarm_check['Mag_Lat'])])
    axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axs.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    axs.tick_params(axis='both', which='major', length=8)
    axs.tick_params(axis='both', which='minor', length=5)

    # set y limits
    axs.set_ylim([min(Ne_sc) - 0.1 * min(Ne_sc),
                  max(Ne_sc) + 0.2 * max(Ne_sc)])

    eia_lab = eia_type_slope.replace("_", " ")

    axs.set_title(eia_lab, color='#000080', fontweight='bold')

    # Change x axis tick labels to latitude format
    putils.format_latitude_labels(axs)

    # Add axis labels
    axs.set_ylabel("N$_e$")
    axs.set_xlabel("Magnetic Latitude")

    if 'south' in eia_type_slope:
        axs.legend(fontsize=fosi, framealpha=0, loc='upper right')
    else:
        axs.legend(fontsize=fosi, framealpha=0, loc='upper left')

    return axs


def pyiri_model_swarm_plot(sday, daily_dir, swarm_dir, model_name="NIMO",
                           fig_on=True, fig_save_dir='', file_save_dir='',
                           pyiri_filt='', pyiri_interpolate=2,
                           pyiri_envelope=False, pyiri_barrel=3,
                           pyiri_window=3, fosi=18):
    """Create and plot a 1 day file for PyIRI, using output from another model.

    Parameters
    ----------
    sday: dt.datetime
        Starting date with time-of-day information set to zero
    daily_dir : str
        Directory of daily files made for the other model-Swarm comparison.
    swarm_dir : str
        Swarm data directory to which data will be downloaded into an
        appropriate date/satellite directory structure
    model_name : str
        Name for the model against which PyIRI will be compared (default='NIMO')
    file_save_dir : str
        directory where file should be saved, (default='')
    fig_on : bool
        Set to true, plot will be made, if false, plot will not be made
        (default=True)
    fig_save_dir : str
        directory where figure should be saved (default='')
    pyiri_filt : str
        Desired Filter for the model data, '' is no filter (default='')
    pyiri_interpolate : int
        int that determines the number of data points in interpolation
        new length will be len(density) x interpolate, with one indicating no
        interpolation (default=2)
    pyiri_envelope : bool
        if True, barrel roll will include points inside an envelope, if false,
        no envelope will be used (default=False)
    pyiri_barrel : float
        latitudinal radius of barrel for Swarm in degrees of magnetic
        latitude (default=3)
    pyiri_window : float
        latitudinal width of moving window of degrees in magnetic latitude
        (default=3)
    fosi : int
        fontsize for plot, with the main title being 10 points larger and the
        legend being three points smaller (default=18)

    Returns
    -------
    df : mpl.Figure or NoneType
        Figure containing 2 panels for each pass between +/-MLat: Swarm and
        pyIRI or None if no data is available
    daily_df : pd.DataFrame or NoneType
        Daily file containing pyIRI information or None if no data is available

    """
    columns = ['Satellite', 'PyIRI_Time', 'PyIRI_GLon', 'PyIRI_Min_MLat',
               'PyIRI_Max_MLat', 'PyIRI_Alt', 'PyIRI_Type',
               'PyIRI_Peak_MLat1', 'PyIRI_Peak_Ne1', 'PyIRI_Peak_MLat2',
               'PyIRI_Peak_Ne2', 'PyIRI_Peak_MLat3', 'PyIRI_Peak_Ne3']

    df = pd.DataFrame(columns=columns)

    # Establish date and some pyIRI params
    year = sday.year
    month = sday.month
    day = sday.day
    f107 = 100
    ccir_or_ursi = 0

    # current day, day after, and day before
    asday = dt.datetime(year, month, day, 0, 0)
    eday = asday + dt.timedelta(days=1)
    pday = asday - dt.timedelta(days=1)

    # Open Daily File
    daily_df = io.load.load_daily_stats(asday, model_name.upper(), 'SWARM',
                                        daily_dir)

    if daily_df.empty:
        logger.info('No {:s}-Swarm comparison available on {:}'.format(
            model_name, asday))
        return None, None

    if fig_on:
        # Open Swarm Files for Plotting
        swarm_dfA = io.load.load_swarm(asday, eday, 'A', swarm_dir)
        swarm_dfB = io.load.load_swarm(asday, eday, 'B', swarm_dir)
        swarm_dfC = io.load.load_swarm(asday, eday, 'C', swarm_dir)

        # Open Previous Day File if not already open
        pre_swarm_dfA = io.load.load_swarm(pday, asday, 'A', swarm_dir)
        pre_swarm_dfB = io.load.load_swarm(pday, asday, 'B', swarm_dir)
        pre_swarm_dfC = io.load.load_swarm(pday, asday, 'C', swarm_dir)

        swarm_A_full = pd.concat([pre_swarm_dfA, swarm_dfA], ignore_index=True)
        swarm_B_full = pd.concat([pre_swarm_dfB, swarm_dfB], ignore_index=True)
        swarm_C_full = pd.concat([pre_swarm_dfC, swarm_dfC], ignore_index=True)

        # Dictionary of all Swarm satellites
        swarm_dc = {'A': swarm_A_full, 'B': swarm_B_full, 'C': swarm_C_full}

    # Convert daily file dates into datetime and get relevant model parameters
    format_date = "%Y/%m/%d_%H:%M:%S.%f"
    date_mod = pd.to_datetime(daily_df[
        '{:s}_Time'.format(model_name.capitalize())].values, format=format_date)

    # Calculate decimal hours for PyIRI input
    mod_decimal_hrs = (date_mod.hour + date_mod.minute / 60
                       + date_mod.second / 3600)
    mod_glon = daily_df['{:s}_GLon'.format(model_name.capitalize())]
    mod_alt = daily_df['{:s}_Swarm_Alt'.format(model_name.capitalize())]
    mod_max_mlats = daily_df['{:s}_Max_MLat'.format(model_name.capitalize())]
    mod_min_mlats = daily_df['{:s}_Min_MLat'.format(model_name.capitalize())]
    sat_list = daily_df['Satellite']
    swarm_date1 = pd.to_datetime(daily_df['Swarm_Time_Start'].values,
                                 format=format_date)
    swarm_date2 = pd.to_datetime(daily_df['Swarm_Time_End'].values,
                                 format=format_date)

    for i in range(len(mod_decimal_hrs)):

        # Create pyIRI dataset based on model parameters
        tim = date_mod[i]
        glon1 = mod_glon[i]
        alat = np.linspace(-90, 90, 181)
        alon = np.ones(len(alat)) * glon1
        mlat, mlon = coords.compute_magnetic_coords(alat, alon, tim)

        ahr = np.array([mod_decimal_hrs[i]])
        aalt = np.array([mod_alt[i]])

        f2, f1, e_peak, es_peak, sun, mag, edp = ml.IRI_density_1day(
            year, month, day, ahr, alon, alat, aalt, f107,
            PyIRI.coeff_dir, ccir_or_ursi)

        mlat_max = mod_max_mlats[i]
        mlat_min = mod_min_mlats[i]

        iri_df = pd.DataFrame()
        iri_df['Mag_Lat'] = mlat[(mlat <= mlat_max) & (mlat >= mlat_min)]
        iri_df['Ne'] = edp[0][0][(mlat <= mlat_max) & (mlat >= mlat_min)]
        time_ls = []

        for j in range(len(iri_df['Ne'])):
            time_ls.append(tim)
        iri_df['Time'] = time_ls

        lat = iri_df['Mag_Lat'].values
        density = iri_df['Ne'].values / 10 ** 6  # convert from m^3 to cm^3
        den_str = 'Ne'

        # Calculate EIA Type for IRI-------------------------
        iri_nlat, iri_filt, eia_type_slope, z_loc, plats, p3 = eia_complete(
            lat, density, den_str, filt=pyiri_filt,
            interpolate=pyiri_interpolate, barrel_envelope=pyiri_envelope,
            barrel_radius=pyiri_barrel, window_lat=pyiri_window)

        # Data File Inputs
        sat = sat_list[i]
        st1 = swarm_date1[i]
        st2 = swarm_date2[i]
        df.at[i, 'Satellite'] = sat
        df.at[i, 'PyIRI_Time'] = tim.strftime(format_date)
        df.at[i, 'PyIRI_GLon'] = glon1
        df.at[i, 'PyIRI_Min_MLat'] = min(mlat)
        df.at[i, 'PyIRI_Max_MLat'] = max(mlat)
        df.at[i, 'PyIRI_Alt'] = aalt[0]
        df.at[i, 'PyIRI_Type'] = eia_type_slope

        if len(plats) > 0:
            for pi, p in enumerate(plats):
                lat_loc = (abs(p - mlat).argmin())
                df_strl = 'PyIRI_Peak_MLat' + str(pi + 1)
                df_strn = 'PyIRI_Peak_Ne' + str(pi + 1)
                df.at[i, df_strl] = mlat[lat_loc]
                df.at[i, df_strn] = edp[0][0][lat_loc]

        # Ensure that something is put into peaks even if none are present
        if len(plats) == 1:
            df_strl = 'PyIRI_Peak_MLat' + str(2)
            df_strn = 'PyIRI_Peak_Ne' + str(2)
            df.at[i, df_strl] = np.nan
            df.at[i, df_strn] = np.nan
            df_strl = 'PyIRI_Peak_MLat' + str(3)
            df_strn = 'PyIRI_Peak_Ne' + str(3)
            df.at[i, df_strl] = np.nan
            df.at[i, df_strn] = np.nan
        elif len(plats) == 2:
            df_strl = 'PyIRI_Peak_MLat' + str(3)
            df_strn = 'PyIRI_Peak_Ne' + str(3)
            df.at[i, df_strl] = np.nan
            df.at[i, df_strn] = np.nan

        if fig_on:

            # Get Desired Swarm Data for Plot
            swarm_df = swarm_dc[sat]
            swarm_pass = swarm_df[((swarm_df['Time'] > st1)
                                   & (swarm_df['Time'] < st2))]

            sw_peak_mlats = np.array(
                [daily_df['Swarm_Peak_MLat1'].iloc[i],
                 daily_df['Swarm_Peak_MLat2'].iloc[i],
                 daily_df['Swarm_Peak_MLat3'].iloc[i]])

            sw_peak_nes = np.array(
                [daily_df['Swarm_Peak_Ne1'].iloc[i],
                 daily_df['Swarm_Peak_Ne2'].iloc[i],
                 daily_df['Swarm_Peak_Ne3'].iloc[i]])

            # Create Figure
            fig = plt.figure(figsize=(12, 12))
            plt.rcParams.update({'font.size': fosi})

            # PLOT SWARM ----------------------------------------
            axs = fig.add_subplot(2, 1, 1)
            axs.plot(swarm_pass['Mag_Lat'],
                     swarm_pass['Ne'],
                     label=daily_df['Swarm_EIA_Type'].iloc[i])
            axs.vlines(sw_peak_mlats,
                       ymin=min(swarm_pass['Ne']),
                       ymax=sw_peak_nes, alpha=0.5, color='black')
            if 'south' in daily_df['Swarm_EIA_Type'].iloc[i]:
                axs.legend(fontsize=fosi - 3, loc='upper right')
            else:
                axs.legend(fontsize=fosi - 3, loc='upper left')
            axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            axs.set_title("Swarm " + sat + ' ' + st1.strftime('%Y%m%d %H:%M')
                          + '-' + st2.strftime('%H:%M'))

            # Plot IRI ------------
            axi = fig.add_subplot(2, 1, 2)
            axi.plot(mlat[(mlat <= mlat_max) & (mlat >= mlat_min)],
                     edp[0][0][(mlat <= mlat_max) & (mlat >= mlat_min)],
                     label=eia_type_slope)
            axi.scatter(mlat[(mlat <= mlat_max) & (mlat >= mlat_min)],
                        edp[0][0][(mlat <= mlat_max) & (mlat >= mlat_min)],
                        label=None)

            if len(plats) > 0:
                for pi, p in enumerate(plats):
                    lat_loc = (abs(p - mlat).argmin())
                    lat_plot = mlat[lat_loc]
                    axi.vlines(lat_plot, ymin=min(edp[0][0]),
                               ymax=edp[0][0][lat_loc],
                               alpha=0.5, color='black')

            axi.set_xlabel("Latitude (\N{DEGREE SIGN})")
            axi.set_ylabel("Ne (cm$^-3$)")

            plot_date_str = tim.strftime('%Y%m%d %H:%M')
            axi.set_title("PyIRI " + plot_date_str
                          + " (" + str(int(aalt[0])) + "km)")
            if 'south' in eia_type_slope:
                axi.legend(fontsize=fosi - 3, loc='upper right')
            else:
                axi.legend(fontsize=fosi - 3, loc='upper left')

            # Plot SAVING --------------------------
            ds = st1.strftime('%Y%m%d')
            ys = st1.strftime('%Y')
            plt.suptitle(str(int(glon1)) + ' GeoLon and '
                         + str(np.round(daily_df['LT_Hour'].iloc[i], 1))
                         + 'LT', x=0.5, y=0.94, fontsize=fosi + 10)

            # Save fig to cwd if not provided
            if fig_save_dir == '':
                fig_save_dir = os.getcwd()
            save_dir = fig_save_dir + '/' + ys + '/' + ds + '/Map_Plots'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_as = os.path.join(save_dir, '_'.join([
                model_name.upper(), 'SWARM', sat, ds, st1.strftime('%H%M'),
                st2.strftime('%H%M'), 'IRI.jpg']))
            fig.savefig(save_as)
            plt.close()

    # Create Data File
    io.write.write_daily_stats(df, st1, 'PyIRI', 'SWARM', file_save_dir)

    return df, daily_df


def model_swarm_mapplot(start_day, swarm_file_dir, mod_file_dir,
                        mod_name_format, model_name='NIMO',
                        mod_load_func=io.load.load_nimo, MLat=30,
                        file_dir='', fig_on=True, fig_dir='',
                        swarm_filt='barrel_average', swarm_interpolate=1,
                        swarm_envelope=True, swarm_barrel=3, swarm_window=2,
                        mod_filt='', mod_interpolate=2, mod_envelope=False,
                        mod_barrel=3, mod_window=3, fosi=18, ne_var='dene',
                        lon_var='lon', lat_var='lat', alt_var='alt',
                        hr_var='hour', min_var='minute', tec_var='tec',
                        hmf2_var='hmf2', nmf2_var='nmf2', mod_cadence=15,
                        max_tdif=15, offset=0):
    """Plot Swarm and model data for 1 day and create file of EIA info.

    Parameters
    ----------
    start_day : datetime
        day starting at 0,0
    swarm_file_dir : str
        directory where swarm file can be found
    mod_file_dir : str
        directory where model file can be found
    mod_name_format : str
        prefix of the desired model NetCDF file including date format before
        the .nc extention, e.g., 'NIMO_AQ_%Y%j'
    mod_load_func : function
        Function for loading the model data (default=`io.load.load_nimo`)
    MLat: int
        Absolute value of the desired maximum magnetic latitude range, e.g.,
        +/- MLat (default=30)
    file_dir : str
        Directory for daily files (default='')
    fig_on : bool
        Make plots if True, or don't if False (default=True)
    fig_dir : str
        Directory for figures (default='')
    swarm_filt : str
        Desired Filter for swarm data (default='barrel_average')
    swarm_interpolate : int
        Multiple of data points to increase density by through interpolation,
        new length will be len(density) x `swarm_interpolate` (default=1)
    swarm_envelope : bool
        if True, barrel roll will include points inside an envelope, if False,
        no envelope will be used (default=True)
    swarm_barrel : float
        latitudinal radius of barrel for swarm (default=3)
    swarm_window : float
        latitudinal width of moving window (default=2)
    mod_filt : str
        Desired Filter for model data (default='')
    mod_interpolate : int
        int that determines the number of data points in interpolation
        new length will be len(density) x interpolate (default=2)
    mod_envelope : bool
        if True, barrel roll will include points inside an
        envelope, if false, no envelope will be used (default=False)
    mod_barrel : float
        latitudinal radius of barrel for swarm (default=3)
    mod_window : float
        latitudinal width of moving window (default=3)
    fosi : int
        Fontsize for plot, with super title equal to `fosi` + 10 and the
        legend text being `fosi` - 3 (default=18)
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
        (default=15)
    offset : int
        Number of days to offset Swarm data from model data to test the model
        reliability (default=0)

    Returns
    -------
    df : pd.DataFrame
        dataframe of info that went into daily file

    """
    # Initialize column names
    col_mod_name = model_name.capitalize()
    columns = ['Satellite', 'Swarm_Time_Start', 'Swarm_Time_End',
               'Swarm_MLat_Start', 'Swarm_MLat_End', 'Swarm_GLon_Start',
               'Swarm_GLon_End', 'Swarm_GLat_Start', 'Swarm_GLat_End',
               'LT_Hour', 'Swarm_EIA_Type', 'Swarm_Peak_MLat1',
               'Swarm_Peak_Ne1', 'Swarm_Peak_MLat2', 'Swarm_Peak_Ne2',
               'Swarm_Peak_MLat3', 'Swarm_Peak_Ne3', f'{col_mod_name}_Time',
               f'{col_mod_name}_GLon', f'{col_mod_name}_Min_MLat',
               f'{col_mod_name}_Max_MLat', f'{col_mod_name}_Swarm_Alt',
               f'{col_mod_name}_Swarm_Type', f'{col_mod_name}_Swarm_Peak_MLat1',
               f'{col_mod_name}_Swarm_Peak_Ne1',
               f'{col_mod_name}_Swarm_Peak_MLat2',
               f'{col_mod_name}_Swarm_Peak_Ne2',
               f'{col_mod_name}_Swarm_Peak_MLat3',
               f'{col_mod_name}_Swarm_Peak_Ne3',
               f'{col_mod_name}_Swarm_Third_Peak_MLat1',
               f'{col_mod_name}_Swarm_Third_Peak_Ne1',
               f'{col_mod_name}_hmf2_Alt', f'{col_mod_name}_hmf2_Type',
               f'{col_mod_name}_hmf2_Peak_MLat1',
               f'{col_mod_name}_hmf2_Peak_Ne1',
               f'{col_mod_name}_hmf2_Peak_MLat2',
               f'{col_mod_name}_hmf2_Peak_Ne2',
               f'{col_mod_name}_hmf2_Peak_MLat3',
               f'{col_mod_name}_hmf2_Peak_Ne3',
               f'{col_mod_name}_hmf2_Third_Peak_MLat1',
               f'{col_mod_name}_hmf2_Third_Peak_Ne1',
               f'{col_mod_name}_Swarm100_Alt',
               f'{col_mod_name}_Swarm100_Type',
               f'{col_mod_name}_Swarm100_Peak_MLat1',
               f'{col_mod_name}_Swarm100_Peak_Ne1',
               f'{col_mod_name}_Swarm100_Peak_MLat2',
               f'{col_mod_name}_Swarm100_Peak_Ne2',
               f'{col_mod_name}_Swarm100_Peak_MLat3',
               f'{col_mod_name}_Swarm100_Peak_Ne3',
               f'{col_mod_name}_Swarm100_Third_Peak_MLat1',
               f'{col_mod_name}_Swarm100_Third_Peak_Ne1']

    # Set up dataframe to save the data in
    df = pd.DataFrame(columns=columns)

    # Ensure the entire day of SWARM is loaded, since the user needs to specify
    # the time closest to the one they want to plot. Do this be removing time
    # of day elements from day-specific timestamp for start and end
    sday = start_day.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = sday + dt.timedelta(days=1)

    # Swarm Satellites
    satellites = ['A', 'B', 'C']

    # f is the index of where we are in the dataframe to add data onto the next
    # slot
    f = -1

    # Get model dictionary for whole day
    mod_start_day = start_day + dt.timedelta(days=offset)
    mod_dc = mod_load_func(
        mod_start_day, mod_file_dir, name_format=mod_name_format, ne_var=ne_var,
        lon_var=lon_var, lat_var=lat_var, alt_var=alt_var, hr_var=hr_var,
        min_var=min_var, tec_var=tec_var, hmf2_var=hmf2_var, nmf2_var=nmf2_var,
        time_cadence=mod_cadence)

    # Iterate through satellites
    for sa, sata in enumerate(satellites):
        # Load Swarm Data for day per satellite
        sw = io.load.load_swarm(sday, end_day, sata, swarm_file_dir)

        # If satellite data is not available, move onto next one
        if len(sw) == 0:
            continue

        # Set Local Time Fractional Hour for plotting purposes
        sw['LT_hr'] = (sw['LT'].dt.hour + sw['LT'].dt.minute / 60
                       + sw['LT'].dt.second / 3600)

        # Limit data by user input MLat (default 30 degrees maglat)
        sw_lat = sw[(abs(sw['Mag_Lat']) <= MLat)]

        # Get the indices of the new latitudinally limited dataset
        lat_ind = sw_lat.index.values

        # Identify the index ranges where the satellite passes over the desired
        # magnetic latitude range
        gap_all = find_all_gaps(lat_ind)

        # Append the first and last indices of lat_ind to gap array
        # to form a full list of gap indices
        start_val = [0]
        end_val = [len(lat_ind)]
        gap = start_val + gap_all + end_val

        # iterate through the desired magnetic laitude range
        for fg in range(len(gap) - 1):
            swarm_check = sw_lat[gap[fg]:gap[fg + 1]]

            # Check for funky orbits
            if abs(min(swarm_check['Longitude'])
                   - max(swarm_check['Longitude'])) > 5:
                logger.info(
                    'Odd Orbit longitude span > 5 degrees: Skipping Pass')
                continue

            # Check that latitude ranges that are +/- 5 degrees from MLat
            # If either side is too far from MLat, try including the day before
            if (min(swarm_check["Mag_Lat"]) < (-MLat + 5)) & (
                    max(swarm_check["Mag_Lat"]) > (MLat - 5)):
                f += 1
            else:
                # If it is the first gap (closes to midnight)
                # Grab day before otherwise continue
                # We do not grab night after so that there are no repeats when
                # Iterating through a whole month
                if fg == 0:
                    # look at day before if available.
                    sw_new = io.load.load_swarm(sday - dt.timedelta(days=1),
                                                sday, sata, swarm_file_dir)
                    sw_new['LT_hr'] = (sw_new['LT'].dt.hour
                                       + sw['LT'].dt.minute / 60
                                       + sw['LT'].dt.second / 3600)
                    # limit data latitudinally
                    sw_lat_new = sw_new[(abs(sw_new['Mag_Lat']) <= MLat)]
                    lat_ind_new = sw_lat_new.index.values
                    # find the places where the passes start and end
                    gap_all_new = find_all_gaps(lat_ind_new)
                    end_val_new = [len(lat_ind_new)]
                    sw_add = sw_lat_new[gap_all_new[-1]:end_val_new[0]]
                    swarm_check_old = swarm_check
                    swarm_check = pd.concat([sw_add, swarm_check_old],
                                            ignore_index=True)
                else:
                    continue

            # Start by saving the universal time of the pass, the satellite,
            swt_str1 = 'Swarm_Time_Start'
            swt_str2 = 'Swarm_Time_End'
            df.at[f, 'Satellite'] = sata  # get satellite info

            # Save time in format %Y/%m/%d_%H:%M:%S.%f to ensure that
            # np.genfromtxt sees data as 1 single string
            df.at[f, swt_str1] = swarm_check["Time"].iloc[0].strftime(
                '%Y/%m/%d_%H:%M:%S.%f')
            df.at[f, swt_str2] = swarm_check["Time"].iloc[-1].strftime(
                '%Y/%m/%d_%H:%M:%S.%f')

            # Save Magnetic Latitude range, geographic longitude range, and
            # geographic latitude range
            df.at[f, 'Swarm_MLat_Start'] = swarm_check["Mag_Lat"].iloc[0]
            df.at[f, 'Swarm_MLat_End'] = swarm_check["Mag_Lat"].iloc[-1]
            df.at[f, 'Swarm_GLon_Start'] = swarm_check["Longitude"].iloc[0]
            df.at[f, 'Swarm_GLon_End'] = swarm_check["Longitude"].iloc[-1]
            df.at[f, 'Swarm_GLat_Start'] = swarm_check["Latitude"].iloc[0]
            df.at[f, 'Swarm_GLat_End'] = swarm_check["Latitude"].iloc[-1]

            # calcualte LT hour at 0 maglat
            LT_dec_H = swarm_check['LT_hr']

            # Separate local times by magnetic hemisphere
            ml_south_all = swarm_check["Mag_Lat"][swarm_check["Mag_Lat"] < 0]
            lt_south_all = LT_dec_H[swarm_check["Mag_Lat"] < 0]
            ml_north_all = swarm_check["Mag_Lat"][swarm_check["Mag_Lat"] > 0]
            lt_north_all = LT_dec_H[swarm_check["Mag_Lat"] > 0]

            # Get closes local time to 0 degrees maglat on each hemisphere
            # if both hemispheres are present
            if (len(ml_south_all) > 0) & (len(ml_north_all) > 0):
                ml_south = ml_south_all.iloc[-1]
                ml_north = ml_north_all.iloc[0]
                lt_south = lt_south_all.iloc[-1]
                lt_north = lt_north_all.iloc[0]
                ml_all = np.array([ml_south, ml_north])
                lt_all = np.array([lt_south, lt_north])

                # Calculate a line between 2 closes LTs to 0 maglat
                # intercept will be maglat == 0
                slope, intercept, rvalue, _, _ = stats.linregress(ml_all,
                                                                  lt_all)
                df.at[f, 'LT_Hour'] = intercept
                lt_plot = np.round(intercept, 2)
            else:
                df.at[f, 'LT_Hour'] = np.nan
                lt_plot = np.round(swarm_check['LT_hr'].iloc[0], 2)

            # Housekeeping: get rid of bad values by flag.
            # \https://earth.esa.int/eogateway/documents/20142/37627/Swarm-
            # Level-1b-Product-Definition-Specification.pdf/12995649-fbcb-6ae2-
            # 5302-2269fecf5a08
            # Navigate to page 52 Table 6-4
            swarm_check.loc[(swarm_check['Ne_flag'] > 20), 'Ne'] = np.nan

            # ------------Swarm EIA STATE ------------------------------------
            slat = swarm_check['Mag_Lat'].values
            density = swarm_check['Ne'].values
            den_str = 'Ne'
            slat_new, sw_filt, eia_type_slope, z_lat, plats, p3 = eia_complete(
                slat, density, den_str, filt=swarm_filt,
                interpolate=swarm_interpolate, barrel_envelope=swarm_envelope,
                barrel_radius=swarm_barrel, window_lat=swarm_window)
            df.at[f, 'Swarm_EIA_Type'] = eia_type_slope

            # Give user a heads up for an unknown type
            if eia_type_slope == 'unknown':
                logger.info(' '.join(['Swarm type unknown for Sat', sata, 'at',
                                      swarm_check['Time'].iloc[-1].strftime(
                                          '%Y/%m/%d %H:%M')]))

            # If user specified fig_on is True, create a figure
            if fig_on:
                fig = plt.figure(figsize=(25, 27))
                plt.rcParams.update({'font.size': fosi})
                gs = gridspec.GridSpec(4, 2, width_ratios=[1, 1],
                                       height_ratios=[1, 1, 1, 1],
                                       wspace=0.1, hspace=0.3)
                axs = fig.add_subplot(gs[0, 0])
                axs.plot(swarm_check['Mag_Lat'],
                         swarm_check['Ne'], linestyle='--', label="Raw Ne")
                axs.plot(slat_new, sw_filt, label='Filtered Ne')
                axs.scatter(swarm_check['Mag_Lat'].iloc[0],
                            swarm_check['Ne'].iloc[0], color='white', s=0,
                            label=eia_type_slope)
                axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

            # Save and/or plot peak latitudes
            if len(plats) > 0:
                for pi, p in enumerate(plats):
                    lat_loc = (abs(p - swarm_check['Mag_Lat']).argmin())
                    df_strl = 'Swarm_Peak_MLat' + str(pi + 1)
                    df_strn = 'Swarm_Peak_Ne' + str(pi + 1)
                    df.at[f, df_strl] = swarm_check['Mag_Lat'].iloc[lat_loc]
                    df.at[f, df_strn] = swarm_check['Ne'].iloc[lat_loc]
                    if fig_on:
                        axs.vlines(swarm_check['Mag_Lat'].iloc[lat_loc],
                                   ymin=min(swarm_check['Ne']),
                                   ymax=swarm_check['Ne'].iloc[lat_loc],
                                   alpha=0.5, color='black')

            # Ensure that something is put into peaks even if none are present
            if len(plats) == 1:
                df_strl = 'Swarm_Peak_MLat' + str(2)
                df_strn = 'Swarm_Peak_Ne' + str(2)
                df.at[f, df_strl] = np.nan
                df.at[f, df_strn] = np.nan
                df_strl = 'Swarm_Peak_MLat' + str(3)
                df_strn = 'Swarm_Peak_Ne' + str(3)
                df.at[f, df_strl] = np.nan
                df.at[f, df_strn] = np.nan
            elif len(plats) == 2:
                df_strl = 'Swarm_Peak_MLat' + str(3)
                df_strn = 'Swarm_Peak_Ne' + str(3)
                df.at[f, df_strl] = np.nan
                df.at[f, df_strn] = np.nan

            if fig_on:
                axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                axs.xaxis.set_minor_locator(mticker.AutoMinorLocator())
                axs.tick_params(axis='both', which='major', length=8)
                axs.tick_params(axis='both', which='minor', length=5)
                axs.set_ylabel("Ne (cm$^-3$)")
                axs.set_xlabel("Magnetic Latitude (\N{DEGREE SIGN})")

                # Change location of legend if it south in eia_type_slope
                if 'south' in eia_type_slope:
                    axs.legend(fontsize=fosi - 3, loc='upper right')
                else:
                    axs.legend(fontsize=fosi - 3, loc='upper left')
                if sata == 'B':
                    axs.set_title('Swarm ' + sata + ' ' + str(511) + 'km')
                else:
                    axs.set_title('Swarm ' + sata + ' ' + str(462) + 'km')

            # Set altiudes and increments for model-Swarm conjunctions
            alt_arr = [sata, 'hmf2', sata]
            inc_arr = [0, 0, 100]

            # Set plot location using lx and r i.e. subplot(lx[i], r[i])
            lx = [0, 1, 1]
            r = [1, 0, 1]

            # Model processing
            for i in range(len(alt_arr)):

                # Initialize base_string for data saving purposes
                if i == 1:
                    base_str = '_'.join([col_mod_name, alt_arr[i]])
                else:
                    base_str = '_'.join([col_mod_name, 'Swarm'])
                if inc_arr[i] == 100:
                    base_str += str(100)

                # Choose an altitude
                alt_str = alt_arr[i]

                # RuntimeError for swarm_conjunction if no model data within
                # max_tdif of Swarm time
                try:
                    (mod_swarm_alt,
                     mod_map) = conjunctions.swarm_conjunction(
                         mod_dc, swarm_check, alt_str, inc=inc_arr[i],
                         max_tdif=max_tdif, offset=offset)

                    if not ((mod_swarm_alt['Mag_Lat'].max()
                             >= 25.0) and (mod_swarm_alt['Mag_Lat'].min()
                                           <= -25.0)):
                        logger.info("".join([
                            model_name, " data doesn't span the mag equator: ",
                            "{:.1f} to {:.1f}".format(
                                mod_swarm_alt['Mag_Lat'].max(),
                                mod_swarm_alt['Mag_Lat'].min())]))
                except ValueError:
                    logger.info(
                        "{:} falls outside the Swarm observations".format(
                            model_name))
                    continue

                # ------------- Model EIA State -----------------------
                nlat = mod_swarm_alt['Mag_Lat'].values
                density = mod_swarm_alt['Ne'].values
                den_str = 'Ne'

                (mod_lat, mod_dfilt, eia_type_slope, z_lat, plats,
                 p3) = eia_complete(
                     nlat, density, den_str, filt=mod_filt,
                     interpolate=mod_interpolate,
                     barrel_envelope=mod_envelope,
                     barrel_radius=mod_barrel, window_lat=mod_window)

                # Give User a heads up for an unknown type
                if eia_type_slope == 'unknown':
                    logger.info(''.join([
                        'Model EIA type unknown at ', alt_str, ' height: ',
                        mod_swarm_alt["Time"].iloc[0][0].strftime(
                            '%Y/%m/%d_%H:%M:%S.%f')]))

                # General model info only need to save once per alt_arr
                nts = '{:s}_Time'.format(col_mod_name)
                if i == 0:
                    df.at[f, nts] = mod_swarm_alt["Time"].iloc[0][0].strftime(
                        '%Y/%m/%d_%H:%M:%S.%f')
                    df.at[f, '{:s}_GLon'.format(col_mod_name)] = mod_swarm_alt[
                        "Longitude"].iloc[0]
                    df.at[f, '{:s}_Min_MLat'.format(col_mod_name)] = min(
                        mod_swarm_alt["Mag_Lat"])
                    df.at[f, '{:s}_Max_MLat'.format(col_mod_name)] = max(
                        mod_swarm_alt["Mag_Lat"])

                # save altitude specific information
                df.at[f, base_str + '_Alt'] = int(mod_swarm_alt["alt"].iloc[0])
                df.at[f, base_str + '_Type'] = eia_type_slope

                # Plot Model
                if fig_on:
                    axns = fig.add_subplot(gs[lx[i], r[i]])
                    plt.rcParams.update({'font.size': fosi})
                    axns.plot(mod_swarm_alt['Mag_Lat'],
                              mod_swarm_alt['Ne'],
                              linestyle='--', marker='o', label='Raw Ne')
                    axns.plot(mod_lat, mod_dfilt, color='C1',
                              label="Filtered Ne")
                    axns.scatter(mod_swarm_alt['Mag_Lat'].iloc[0],
                                 mod_swarm_alt['Ne'].iloc[0], color='white',
                                 s=0, label=eia_type_slope)

                # Save and/or plot Peak lats
                if len(plats) > 0:
                    for pi, p in enumerate(plats):
                        lat_lo = (abs(p - mod_swarm_alt['Mag_Lat']).argmin())
                        lat_plot = mod_swarm_alt['Mag_Lat'].iloc[lat_lo]
                        dfl = base_str + '_Peak_MLat' + str(pi + 1)
                        dfn = base_str + '_Peak_Ne' + str(pi + 1)
                        df.at[f, dfl] = mod_swarm_alt['Mag_Lat'].iloc[lat_lo]
                        df.at[f, dfn] = mod_swarm_alt['Ne'].iloc[lat_lo]
                        if fig_on:
                            axns.vlines(lat_plot,
                                        ymin=min(mod_swarm_alt['Ne']),
                                        ymax=mod_swarm_alt['Ne'].iloc[lat_lo],
                                        alpha=0.5, color='k')
                # save and/or plot thrid peak if present
                # (no thrid peak for ghosts)
                if len(p3) > 0:
                    for pi, p in enumerate(p3):
                        lat_loc = (abs(p - mod_swarm_alt['Mag_Lat']).argmin())
                        lat_plot = mod_swarm_alt['Mag_Lat'].iloc[lat_loc]
                        dl3 = base_str + '_Third_Peak_MLat' + str(pi + 1)
                        df_strn3 = base_str + '_Third_Peak_Ne' + str(pi + 1)
                        df.at[f, dl3] = mod_swarm_alt['Mag_Lat'].iloc[lat_loc]
                        df.at[f, df_strn3] = mod_swarm_alt['Ne'].iloc[lat_loc]
                        if fig_on:
                            axns.vlines(
                                lat_plot, ymin=min(mod_swarm_alt['Ne']),
                                ymax=mod_swarm_alt['Ne'].iloc[lat_loc],
                                linestyle='--', alpha=0.5, color='r')

                # Set labels for plots
                if fig_on:
                    axns.ticklabel_format(axis='y', style='sci',
                                          scilimits=(0, 0))
                    axns.xaxis.set_minor_locator(mticker.AutoMinorLocator())
                    axns.tick_params(axis='both', which='major', length=8)
                    axns.tick_params(axis='both', which='minor', length=5)
                    axns.set_xlabel("Magnetic Latitude (\N{DEGREE SIGN})")
                    axns.set_ylabel("Ne (cm$^-3$)")
                    if 'south' in eia_type_slope:
                        axns.legend(fontsize=fosi - 3, loc='upper right')
                    else:
                        axns.legend(fontsize=fosi - 3, loc='upper left')
                    axns.set_title('{:s} {:d} km'.format(
                        model_name, int(mod_swarm_alt['alt'].iloc[0])))
            # Terminator and Map plotting
            if fig_on:

                # Set the date and time for the terminator
                date_term = mod_swarm_alt['Time'].iloc[0][0]

                # Get terminator at the given height
                antisolarpsn, arc, ang = pydarn.terminator(date_term, 300)
                # antisolarpsn contains the latitude and longitude
                # of the antisolar point
                # arc represents the radius of the terminator arc
                # directly use the geographic coordinates from antisolarpsn.
                lat_antisolar = antisolarpsn[1]
                lon_antisolar = antisolarpsn[0]
                # Get positions along the terminator arc in geo coordinates
                lats = []
                lons = []
                # Iterate over longitudes from -180 to 180
                for b in range(-180, 180, 1):
                    lat, lon = pydarn.GeneralUtils.new_coordinate(
                        lat_antisolar, lon_antisolar, arc, b, R=pydarn.Re)
                    lats.append(lat)
                    lons.append(lon)
                lons = [(lon + 180) % 360 - 180 for lon in lons]

                # plot nmf2 map
                ax = fig.add_subplot(gs[2:, :], projection=ccrs.PlateCarree())
                ax.set_global()

                # Add coast line
                ax.add_feature(cfeature.COASTLINE)

                # Use the cvidis colormap
                heatmap = ax.pcolormesh(mod_map['glon'], mod_map['glat'],
                                        mod_map['nmf2'], cmap='cividis',
                                        transform=ccrs.PlateCarree())
                ax.plot(swarm_check['Longitude'], swarm_check['Latitude'],
                        color='white', label="Satellite Path")
                ax.text(swarm_check['Longitude'].iloc[0] + 1,
                        swarm_check['Latitude'].iloc[0], sata, color='white')
                lons = np.squeeze(lons)
                lats = np.squeeze(lats)

                # Plot terminator
                ax.scatter(lons, lats, color='orange', s=1, zorder=2.0,
                           linewidth=2.0)
                ax.plot(lons[0], lats[0], color='orange', linestyle=':',
                        zorder=2.0, linewidth=2.0, label='Terminator 300km')
                leg = ax.legend(framealpha=0, loc='upper right')
                for text in leg.get_texts():
                    text.set_color('white')

                # Set x and y labels
                ax.text(-205, -30, 'Geographic Latitude (\N{DEGREE SIGN})',
                        color='k', rotation=90)
                ax.text(-35, -105, 'Geographic Longitude (\N{DEGREE SIGN})',
                        color='k')
                ax.set_title('{:s} NmF2 at {:}'.format(
                    model_name, mod_swarm_alt['Time'].iloc[0][0]))

                # Add vertical colorbar on the side
                cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical',
                                    pad=0.02, shrink=0.8)
                cbar.set_label("NmF2 (cm$^-3$)")
                cbar.ax.tick_params(labelsize=15)

                # Show plot
                gl = ax.gridlines(draw_labels=True, linewidth=0,
                                  color='gray', alpha=0.5)
                gl.top_labels = False  # Optional: Turn off top labels
                gl.right_labels = False  # Optional: Turn off right labels

                plt.suptitle(str(int(mod_swarm_alt['Longitude'].iloc[0]))
                             + ' GeoLon and ' + str(lt_plot) + ' LT',
                             x=0.5, y=0.92, fontsize=fosi + 10)
                plt.rcParams.update({'font.size': fosi})
                ts1 = swarm_check['Time'].iloc[0].strftime('%H%M')
                ts2 = swarm_check['Time'].iloc[-1].strftime('%H%M')
                ds = swarm_check['Time'].iloc[0].strftime('%Y%m%d')
                ys = swarm_check['Time'].iloc[0].strftime('%Y')

                # Save figure - CWD IF EMPTY
                if fig_dir == '':
                    fig_dir = os.getcwd()
                save_dir = os.path.join(fig_dir, ys, ds, 'Map_Plots')
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                offstr = '' if offset == 0 else 'offset{:s}days_'.format(
                    str(offset))
                save_as = os.path.join(save_dir, '_'.join([
                    model_name.upper(), 'SWARM', sata, ds, ts1,
                    '{:s}{:s}.jpg'.format(offstr, ts2)]))
                fig.savefig(save_as)
                plt.close()

    # Write the output file
    io.write.write_daily_stats(df, start_day, model_name.upper(), 'SWARM',
                               file_dir)

    return df


def model_swarm_single_plot(stime, satellite, swarm_file_dir, mod_file_dir,
                            mod_name_format, model_name='NIMO',
                            mod_load_func=io.load.load_nimo,
                            MLat=30, swarm_filt='barrel_average',
                            swarm_interpolate=1, swarm_envelope=True,
                            swarm_barrel=3, swarm_window=2, mod_filt='',
                            mod_interpolate=2, mod_envelope=False, mod_barrel=3,
                            mod_window=3, fosi=18, plot_dir='', ne_var='dene',
                            lon_var='lon', lat_var='lat', alt_var='alt',
                            hr_var='hour', min_var='minute', tec_var='tec',
                            hmf2_var='hmf2', nmf2_var='nmf2', mod_cadence=15):
    """Plot and save a single model/Swarm EIA type plot.

    Parameters
    ----------
    stime : datetime object
        time of desired plot, nearest time within mlatitudinal window will be
        plotted
    satellite : str
        'A', 'B', or 'C' for Swarm
    swarm_file_dir : str
        directory where swarm file can be found
    mod_file_dir : str
        directory where nimo file can be found
    mod_name_format : str
        Prefix of model file including date format before the .nc extension,
        e.g., 'NIMO_AQ_%Y%j'
    model_name : str
        Model name (default='NIMO')
    mod_load_func : function
        Function for loading the model data (default=`io.load.load_nimo`)
    MLat : int
        magnetic latitude range +/-MLat (default=30)
    swarm_filt : str kwarg
        Desired Filter for swarm data (default='barrel_average')
    swarm_interpolate : int
        int that determines the number of data points in interpolation
        new length will be len(density) x interpolate (default=1)
    swarm_envelope : bool
        if True, barrel roll will include points inside an envelope, if False,
        no envelope will be used (default=True)
    swarm_barrel : float
        latitudinal radius of barrel for swarm (default=3)
    swarm_window : float
        latitudinal width of moving window (default=2)
    mod_filt : str
        Desired Filter for nimo data (default='')
    mod_interpolate : int
        int that determines the number of data points in interpolation
        new length will be len(density) x interpolate (default=2)
    mod_envelope : bool
        if True, barrel roll will include points inside an
        envelope, if False, no envelope will be used (default=False)
    mod_barrel : float
        latitudinal radius of barrel for swarm (default=3)
    mod_window : float
        latitudinal width of moving window (default=3)
    fosi : int
        fontsize for plot, with super title equal to `fosi` + 10 and the legend
        text equal to `fosi` - 3 (default=18)
    plot_dir : str
        output folder for plot, or '' to not save output. If saved, an
        additional date directory will be created: `plot_dir`/{%Y%m%d}/fig.jpg
        (default='')
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
    mod_cadence: int
        Time cadence of model data in minutes (default=15)

    Returns
    -------
    fig : matplotlib.Figure
        Figure handle

    Notes
    -----
    filt options include: 'barrel', 'average', 'median', 'barrel_average'
    'barrel_median', 'average_barrel', and 'median_barrel'

    """
    # Ensure the entire day of SWARM is loaded, since the user needs to specify
    # the time closest to the one they want to plot. Do this be removing time
    # of day elements from day-specific timestamp for start and end
    sday = stime.replace(hour=0, minute=0, second=0, microsecond=0)
    eday = sday + dt.timedelta(days=1)

    # Get full day of Swarm Data
    swarm_df = load.load_swarm(sday, eday, satellite, swarm_file_dir)

    # Housekeeping: get rid of bad values by flag.
    # \https://earth.esa.int/eogateway/documents/20142/37627/Swarm-Level-1b-
    # Product-Definition-Specification.pdf/12995649-fbcb-6ae2-5302-2269fecf5a08
    # Navigate to page 52 Table 6-4
    swarm_df['LT_hr'] = swarm_df['LT'].dt.hour + swarm_df['LT'].dt.minute / 60
    swarm_df.loc[(swarm_df['Ne_flag'] > 20), 'Ne'] = np.nan

    # Limit by user specified magnetic latitud range
    sw_lat = swarm_df[(swarm_df["Mag_Lat"] < MLat) & (swarm_df["Mag_Lat"]
                                                      > -MLat)]
    lat_ind = sw_lat.index.values

    # Identify the index ranges where the satellite passes over the desired
    # magnetic latitude range
    gap_all = find_all_gaps(lat_ind)

    # Append the first and last indices of lat_ind to gap array to form a full
    # List of gap indices
    start_val = [0]
    end_val = [len(lat_ind)]
    gaps = start_val + gap_all + end_val

    # Get closest time to Input
    tim_arg = abs(sw_lat["Time"] - stime).argmin()
    if abs(sw_lat["Time"].iloc[tim_arg] - stime) > dt.timedelta(minutes=10):
        logger.info(f'Selecting {sw_lat["Time"].iloc[tim_arg]}')

    # Choose latitudinally limited segment using gap indices
    gap_arg = abs(tim_arg - gaps).argmin()
    if gaps[gap_arg] <= tim_arg or tim_arg == 0:
        g1 = gap_arg
        g2 = gap_arg + 1
    else:
        g1 = gap_arg - 1
        g2 = gap_arg

    # Desired Swarm Data Segment
    swarm_check = sw_lat[gaps[g1]:gaps[g2]]

    # Get NIMO Dictionary
    mod_dc = mod_load_func(stime, mod_file_dir, name_format=mod_name_format,
                           ne_var=ne_var, lon_var=lon_var, lat_var=lat_var,
                           alt_var=alt_var, hr_var=hr_var, min_var=min_var,
                           tec_var=tec_var, hmf2_var=hmf2_var,
                           nmf2_var=nmf2_var, time_cadence=mod_cadence)

    # Evaluate Swarm EIA-------------------------------------------------
    slat_use = swarm_check['Mag_Lat'].values
    density = swarm_check['Ne'].values
    den_str = 'Ne'
    sw_lat, sw_filt, eia_type_slope, z_lat, plats, p3 = eia_complete(
        slat_use, density, den_str, filt=swarm_filt,
        interpolate=swarm_interpolate, barrel_envelope=swarm_envelope,
        barrel_radius=swarm_barrel, window_lat=swarm_window)

    # Create Figure
    fig = plt.figure(figsize=(14, 16))
    plt.rcParams.update({'font.size': fosi})
    gs = gridspec.GridSpec(4, 2, width_ratios=[1, 1],
                           height_ratios=[1, 1, 1, 1], wspace=0.1, hspace=0.3)

    # Plot the Swarm Data
    axs = fig.add_subplot(gs[0, 0])
    axs.plot(swarm_check['Mag_Lat'], swarm_check['Ne'], linestyle='--',
             label="Raw Ne")
    axs.plot(sw_lat, sw_filt, label='Filtered Ne')
    axs.scatter(swarm_check['Mag_Lat'].iloc[0], swarm_check['Ne'].iloc[0],
                color='white', s=0, label=eia_type_slope)
    axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Plot Swarm Peak Latitudes
    if len(plats) > 0:
        for pi, p in enumerate(plats):
            lat_loc = (abs(p - swarm_check['Mag_Lat']).argmin())
            axs.vlines(swarm_check['Mag_Lat'].iloc[lat_loc],
                       ymin=min(swarm_check['Ne']),
                       ymax=swarm_check['Ne'].iloc[lat_loc], alpha=0.5,
                       color='black')

    axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axs.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    axs.tick_params(axis='both', which='major', length=8)
    axs.tick_params(axis='both', which='minor', length=5)
    axs.set_ylabel("Ne")
    axs.set_xlabel("Magnetic Latitude")

    # Change location of legend if it south in eia_type_slope
    if 'south' in eia_type_slope:
        axs.legend(fontsize=fosi - 3, loc='upper right')
    else:
        axs.legend(fontsize=fosi - 3, loc='upper left')
    if satellite == 'B':
        axs.set_title('Swarm ' + satellite + ' ' + str(511) + 'km')
    else:
        axs.set_title('Swarm ' + satellite + ' ' + str(462) + 'km')

    # Set altiudes and increments for NIMO conjunctions
    alt_arr = [satellite, 'hmf2', satellite]
    inc_arr = [0, 0, 100]

    # Set plot location using lo and r i.e. subplot(lo[i], r[i])
    lo = [0, 1, 1]
    r = [1, 0, 1]

    # Model-------------------------
    for i in range(len(alt_arr)):

        # Choose an altitude for NIMO
        alt_str = alt_arr[i]  # Go through through Altitudes
        mod_swarm_alt, mod_map = conjunctions.swarm_conjunction(
            mod_dc, swarm_check, alt_str, inc=inc_arr[i])
        nlat_use = mod_swarm_alt['Mag_Lat'].values
        density = mod_swarm_alt['Ne'].values

        # Detect NIMO EIA Type -----------------------------------------
        den_str = 'Ne'
        mod_lat, mod_dfilt, eia_type_slope, z_lat, plats, p3 = eia_complete(
            nlat_use, density, den_str, filt=mod_filt,
            interpolate=mod_interpolate, barrel_envelope=mod_envelope,
            barrel_radius=mod_barrel, window_lat=mod_window)

        axns = fig.add_subplot(gs[lo[i], r[i]])  # plot model ne at swarm alt
        axns.plot(mod_swarm_alt['Mag_Lat'],
                  mod_swarm_alt['Ne'], linestyle='--', marker='o',
                  label='Raw Ne')
        axns.plot(mod_lat, mod_dfilt, color='C1', label="Filtered Ne")
        axns.scatter(mod_swarm_alt['Mag_Lat'].iloc[0],
                     mod_swarm_alt['Ne'].iloc[0], color='white', s=0,
                     label=eia_type_slope)

        # Plot NIMO Peak Latitudes
        if len(plats) > 0:
            for pi, p in enumerate(plats):
                lat_loc = (abs(p - mod_swarm_alt['Mag_Lat']).argmin())
                lat_plot = mod_swarm_alt['Mag_Lat'].iloc[lat_loc]
                axns.vlines(lat_plot, ymin=min(mod_swarm_alt['Ne']),
                            ymax=mod_swarm_alt['Ne'].iloc[lat_loc],
                            alpha=0.5, color='k')

        # Plot third peak if not a ghost and detected
        if len(p3) > 0:
            for pi, p in enumerate(p3):
                lat_loc = (abs(p - mod_swarm_alt['Mag_Lat']).argmin())
                lat_plot = mod_swarm_alt['Mag_Lat'].iloc[lat_loc]
                axns.vlines(lat_plot, ymin=min(mod_swarm_alt['Ne']),
                            ymax=mod_swarm_alt['Ne'].iloc[lat_loc],
                            linestyle='--', alpha=0.5, color='r')

        axns.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        axns.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        axns.tick_params(axis='both', which='major', length=8)
        axns.tick_params(axis='both', which='minor', length=5)
        axns.set_xlabel("Magnetic Latitude")

        if i == 1:
            axns.set_ylabel("Ne")

        if 'south' in eia_type_slope:
            axns.legend(fontsize=fosi - 3, loc='upper right')
        else:
            axns.legend(fontsize=fosi - 3, loc='upper left')
        axns.set_title('{:s} {:d} km'.format(
            model_name, int(mod_swarm_alt['alt'].iloc[0])))

    # ----------------- MAP PLOT --------------------
    # Set the date and time for the terminator
    date_term = mod_swarm_alt['Time'].iloc[0][0]

    # Get antisolar position and the arc (terminator) at the given height
    antisolarpsn, arc, ang = pydarn.terminator(date_term, 300)

    # antisolarpsn contains the latitude and longitude of the antisolar point
    # arc represents the radius of the terminator arc
    # Now, you can directly use the geographic coordinates from antisolarpsn.
    lat_antisolar = antisolarpsn[1]
    lon_antisolar = antisolarpsn[0]

    # Get positions along the terminator arc in geographic coordinates
    lats = []
    lons = []

    # Iterate over longitudes from -180 to 180 at a one degree resolution
    for b in range(-180, 180, 1):
        lat, lon = pydarn.GeneralUtils.new_coordinate(lat_antisolar,
                                                      lon_antisolar, arc, b,
                                                      R=pydarn.Re)
        lats.append(lat)
        lons.append(lon)
    lons = [(lon + 180) % 360 - 180 for lon in lons]

    # Plot NmF2 map
    ax = fig.add_subplot(gs[2:, :], projection=ccrs.PlateCarree())
    ax.set_global()

    # Add Coastlines
    ax.add_feature(cfeature.COASTLINE)

    # Colormap
    heatmap = ax.pcolormesh(mod_map['glon'], mod_map['glat'],
                            mod_map['nmf2'],
                            cmap=colormaps.get_cmap('cividis'),
                            transform=ccrs.PlateCarree())
    ax.plot(swarm_check['Longitude'], swarm_check['Latitude'], color='white',
            label="Satellite Path")
    ax.text(swarm_check['Longitude'].iloc[0] + 1,
            swarm_check['Latitude'].iloc[0], satellite, color='white')

    # Plot the Terminator
    lons = np.squeeze(lons)
    lats = np.squeeze(lats)
    ax.scatter(lons, lats, color='orange', s=1, zorder=2.0, linewidth=2.0)
    ax.plot(lons[0], lats[0], color='orange', linestyle=':', zorder=2.0,
            linewidth=2.0, label='Terminator 300km')
    leg = ax.legend(framealpha=0, loc='upper right')

    for text in leg.get_texts():
        text.set_color('white')

    # Set labels
    ax.text(-220, -50, 'Geographic Latitude', color='k', rotation=90)
    ax.text(-50, -110, 'Geographic Longitude', color='k')
    ax.set_title('{:s} N$_m$F$_2$ at {:}'.format(
        model_name, mod_swarm_alt['Time'].iloc[0][0]), fontsize=fosi + 5)

    # Add vertical colorbar on the side
    cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical',
                        pad=0.04, shrink=0.7)
    cbar.set_label("N$_m$F$_2$")
    cbar.ax.tick_params(labelsize=fosi)

    # Add grids and unset the grid labels
    gl = ax.gridlines(draw_labels=True, linewidth=0, color='gray', alpha=0.5)
    gl.top_labels = False  # Optional: Turn off top labels
    gl.right_labels = False  # Optional: Turn off right labels

    fig.suptitle(str(int(mod_swarm_alt['Longitude'].iloc[0]))
                 + ' GeoLon and '
                 + str(np.round(swarm_check['LT_hr'].iloc[0], 2)) + ' LT',
                 fontsize=fosi + 10)
    fig.subplots_adjust(bottom=.03, top=.92)

    # Save plot if an output directory was supplied
    if os.path.isdir(plot_dir):
        ds = swarm_check['Time'].iloc[0].strftime('%Y%m%d')
        ts1 = swarm_check['Time'].iloc[0].strftime('%H%M')
        ts2 = swarm_check['Time'].iloc[-1].strftime('%H%M')

        fig_dir = os.path.join(plot_dir, ds)
        Path(fig_dir).mkdir(parents=True, exist_ok=True)

        figname = os.path.join(fig_dir, '_'.join([
            model_name.upper(), 'SWARM', satellite, ds, ts1,
            '{:}.jpg'.format(ts2)]))
        fig.savefig(figname)

    return fig
