#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd

import PyIRI
import PyIRI.edp_update as ml


from pyValEIA.EIA_type_detection import eia_complete
from pyValEIA import io
from pyValEIA.utils import coords


def PyIRI_NIMO_SWARM_plot(sday, daily_dir, swarm_dir, fig_on=True,
                          fig_save_dir='', file_save_dir='', pyiri_filt='',
                          pyiri_interpolate=2, pyiri_envelope=False,
                          pyiri_barrel=3, pyiri_window=3, fosi=18):
    """ Creates plots and a data file for 1 day, this datafile works in
    conjunction with the NIMO_SWARM_EIA_type_[date]ascii.txt files

    Parameters
    ----------
    sday: datetime
        (day starting at 0,0)
    daily_dir : str
        directory of daily files made by
        NIMO_Swarm_Map_Plotting.NIMO_SWARM_mapplot
    swarm_dir : str
        Swarm data directory to which data will be downloaded into an
        appropriate date/satellite directory structure
    file_save_dir : str kwarg
        directory where file should be saved, default cwd
    fig_on : kwarg bool
        set to true, plot will be made, if false, plot will not be made
    fig_save_dir : str kwarg
        directory where figure should be saved, default cwd
    pyiri_filt : str kwarg
        Desired Filter for nimo data (no filter default)
    pyiri_interpolate : int kwarg
        int that determines the number of data points in interpolation
        new length will be len(density)xinterpolate
        default is 2
    pyiri_envelope : bool kwarg
        if True, barrel roll will include points inside an
        envelope, if false (default), no envelope will be used
    pyiri_barrel : double
        latitudinal radius of barrel for swarm (default: 3 degrees maglat)
    pyiri_window : double kwarg
        latitudinal width of moving window (default: 3 degrees maglat)
    fosi : int kwarg
        fontsize for plot (default is 18)
        Exceptions:
            Super Title (fosi + 10)
            legends (fosi - 3)
    Returns
    -------
    Figure containing 2 panels for each pass between +/-MLat: Swarm and pyIRI
    Data file: daily file containing pyIRI information

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
    asday = datetime(year, month, day, 0, 0)
    eday = asday + timedelta(days=1)
    pday = asday - timedelta(days=1)

    # Open Daily File
    daily_df = io.load.load_daily_stats(asday, 'NIMO', 'SWARM', daily_dir)

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

    # Convert daily file dates into datetime and get relevant NIMO parameters
    format_date = "%Y/%m/%d_%H:%M:%S.%f"
    date_nimo = pd.to_datetime(daily_df['Nimo_Time'].values,
                               format=format_date)

    # Calculate decimal hours for PyIRI input
    nim_decimal_hrs = (date_nimo.hour + date_nimo.minute / 60
                       + date_nimo.second / 3600)
    nim_glon = daily_df['Nimo_GLon']
    nim_alt = daily_df['Nimo_Swarm_Alt']
    nim_max_mlats = daily_df['Nimo_Max_MLat']
    nim_min_mlats = daily_df['Nimo_Min_MLat']
    sat_list = daily_df['Satellite']
    swarm_date1 = pd.to_datetime(daily_df['Swarm_Time_Start'].values,
                                 format=format_date)
    swarm_date2 = pd.to_datetime(daily_df['Swarm_Time_End'].values,
                                 format=format_date)

    for i in range(len(nim_decimal_hrs)):

        # Create pyIRI dataset based on NIMO parameters
        tim = date_nimo[i]
        glon1 = nim_glon[i]
        alat = np.linspace(-90, 90, 181)
        alon = np.ones(len(alat)) * glon1
        mlat, mlon = coords.compute_magnetic_coords(alat, alon, tim)

        ahr = np.array([nim_decimal_hrs[i]])
        aalt = np.array([nim_alt[i]])

        f2, f1, e_peak, es_peak, sun, mag, edp = ml.IRI_density_1day(
            year, month, day, ahr, alon, alat, aalt, f107,
            PyIRI.coeff_dir, ccir_or_ursi)

        mlat_max = nim_max_mlats[i]
        mlat_min = nim_min_mlats[i]

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
            save_as = (save_dir + '/NIMO_SWARM_' + sat + '_' + ds + '_'
                       + st1.strftime('%H%M') + '_' + st2.strftime('%H%M')
                       + '_IRI.jpg')
            fig.savefig(save_as)
            plt.close()

    # Create Data File
    io.write.write_daily_stats(df, st1, 'PyIRI', 'SWARM', file_save_dir)

    return df, daily_df
