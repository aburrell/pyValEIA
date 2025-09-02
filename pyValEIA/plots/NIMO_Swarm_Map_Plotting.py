#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
# These functions will create a plot of Swarm
# (between +/- some magnetic latitude)
# and their NIMO counterparts 5 panels Swarm, NIMO Swarm Alt,
# NIMO HMF2, NIMO Swarm alt + 100, NIMO NMF2 Map with swarm trajectory
from datetime import timedelta
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from scipy import stats
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pydarn

from pyValEIA import io
from pyValEIA.EIA_type_detection import eia_complete
from pyValEIA import nimo_conjunctions


def find_all_gaps(arr):
    """Find gap indices.

    Parameters
    ----------
    arr : array-like
        Array of indices

    Returns
    -------
    gap_indices : array-like
        Indices of gap start and end

    Notes
    -----
    For example, in an array of `arr=[2,3,5,6,7,8]`, this function will return
    `gap_inds=[1]` to indicate where the gap starts.

    """
    gap_indices = []
    # Iterate through the array and find where the gaps start
    for i in range(len(arr) - 1):
        if arr[i + 1] != arr[i] + 1:
            gap_indices.append(i + 1)  # Append the index where the gap starts
    return gap_indices


def NIMO_SWARM_mapplot(
        start_day, swarm_file_dir, nimo_file_dir, MLat=30, file_dir='',
        fig_on=True, fig_dir='', swarm_filt='barrel_average',
        swarm_interpolate=1, swarm_envelope=True, swarm_barrel=3,
        swarm_window=2, nimo_filt='', nimo_interpolate=2, nimo_envelope=False,
        nimo_barrel=3, nimo_window=3, fosi=18, nimo_name_format='NIMO_AQ_%Y%j',
        ne_var='dene', lon_var='lon', lat_var='lat', alt_var='alt',
        hr_var='hour', min_var='minute', tec_var='tec', hmf2_var='hmf2',
        nmf2_var='nmf2', nimo_cadence=15, max_tdif=15):

    """ Plot Swarm/NIMO for 1 day and create file of eia info
    Parameters
    ----------
    start_day: datetime
        day starting at 0,0
    swarm_file_dir : str
        directory where swarm file can be found
    nimo_file_dir: str
        directory where nimo file can be found
    MLat: int kwarg
        magnetic latitude 1 number
        default is 30 degrees for +/-30 maglat
    file_dir: str kwarg
        directory for daily files
        default is current working directory
    fig_on: bool kwarg
        True (default), plots will be made, if False, plot will not be made
    fig_dir: str kwarg
        directory for figures
        default is current working directory
    swarm_filt : str kwarg
        Desired Filter for swarm data (default barrel_average)
    swarm_interpolate : int kwarg
        int that determines the number of data points in interpolation
        new length will be len(density)xinterpolate
        default is 1 indicating no interpolation
    swarm_envelope : bool kwarg
        if True (default), barrel roll will include points inside an
        envelope, if false, no envelope will be used
    swarm_barrel : double
        latitudinal radius of barrel for swarm (default: 3 degrees maglat)
    swarm_window : double kwarg
        latitudinal width of moving window (default: 2 degrees maglat)
    nimo_filt : str kwarg
        Desired Filter for nimo data (no filter default)
    nimo_interpolate : int kwarg
        int that determines the number of data points in interpolation
        new length will be len(density)xinterpolate
        default is 2
    nimo_envelope : bool kwarg
        if True, barrel roll will include points inside an
        envelope, if false (default), no envelope will be used
    nimo_barrel : double
        latitudinal radius of barrel for swarm (default: 3 degrees maglat)
    nimo_window : double kwarg
        latitudinal width of moving window (default: 3 degrees maglat)
    fosi : int kwarg
        fontsize for plot (default is 18)
        Exceptions:
            Super Title (fosi + 10)
            legends (fosi - 3)
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
        maximum time distance (in minutes) between a NIMO and Swarm
        conjunction allowed (default 15)
    Returns
    -------
    Fig: Figure
        Plot containing 5 panels for each pass between +/-MLat:
        Swarm, NIMO Swarm Alt, NIMO HMF2, NIMO Swarm alt + 100,
        NIMO NMF2 Map with swarm trajectory
    Data file: file
        daily file containing NIMO and Swarm Info about the pass and eia type
        and eia peaks
    df : dataframe
        dataframe of info that went into daily file
    """
    # Initialize column names
    columns = ['Satellite', 'Swarm_Time_Start', 'Swarm_Time_End',
               'Swarm_MLat_Start', 'Swarm_MLat_End', 'Swarm_GLon_Start',
               'Swarm_GLon_End', 'Swarm_GLat_Start', 'Swarm_GLat_End',
               'LT_Hour', 'Swarm_EIA_Type', 'Swarm_Peak_MLat1',
               'Swarm_Peak_Ne1', 'Swarm_Peak_MLat2', 'Swarm_Peak_Ne2',
               'Swarm_Peak_MLat3', 'Swarm_Peak_Ne3', 'Nimo_Time', 'Nimo_GLon',
               'Nimo_Min_MLat', 'Nimo_Max_MLat', 'Nimo_Swarm_Alt',
               'Nimo_Swarm_Type', 'Nimo_Swarm_Peak_MLat1',
               'Nimo_Swarm_Peak_Ne1', 'Nimo_Swarm_Peak_MLat2',
               'Nimo_Swarm_Peak_Ne2', 'Nimo_Swarm_Peak_MLat3',
               'Nimo_Swarm_Peak_Ne3', 'Nimo_Swarm_Third_Peak_MLat1',
               'Nimo_Swarm_Third_Peak_Ne1', 'Nimo_hmf2_Alt', 'Nimo_hmf2_Type',
               'Nimo_hmf2_Peak_MLat1', 'Nimo_hmf2_Peak_Ne1',
               'Nimo_hmf2_Peak_MLat2', 'Nimo_hmf2_Peak_Ne2',
               'Nimo_hmf2_Peak_MLat3', 'Nimo_hmf2_Peak_Ne3',
               'Nimo_hmf2_Third_Peak_MLat1', 'Nimo_hmf2_Third_Peak_Ne1',
               'Nimo_Swarm100_Alt', 'Nimo_Swarm100_Type',
               'Nimo_Swarm100_Peak_MLat1', 'Nimo_Swarm100_Peak_Ne1',
               'Nimo_Swarm100_Peak_MLat2', 'Nimo_Swarm100_Peak_Ne2',
               'Nimo_Swarm100_Peak_MLat3', 'Nimo_Swarm100_Peak_Ne3',
               'Nimo_Swarm100_Third_Peak_MLat1',
               'Nimo_Swarm100_Third_Peak_Ne1']

    # Set up dataframe to save the data in
    df = pd.DataFrame(columns=columns)

    # Ensure the entire day of SWARM is loaded, since the user needs to specify
    # the time closest to the one they want to plot. Do this be removing time
    # of day elements from day-specific timestamp for start and end
    sday = start_day.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = sday + timedelta(days=1)

    # Swarm Satellites
    Satellites = ['A', 'B', 'C']

    # f is the index of where we are in the dataframe to add data onto the next
    # slot
    f = -1

    # Get nimo dictionary for whole day
    nimo_dc = io.load.load_nimo(
        start_day, nimo_file_dir, name_format=nimo_name_format, ne_var=ne_var,
        lon_var=lon_var, lat_var=lat_var, alt_var=alt_var, hr_var=hr_var,
        min_var=min_var, tec_var=tec_var, hmf2_var=hmf2_var, nmf2_var=nmf2_var,
        time_cadence=nimo_cadence)

    # Iterate through satellites
    for sa, sata in enumerate(Satellites):

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
                print('Odd Orbit longitude span > 5 degrees: Skipping Pass')
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
                    sw_new = io.load.load_swarm(sday - timedelta(days=1),
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
                print('Swarm type unknown for Sat ' + sata)
                print(swarm_check['Time'].iloc[-1].strftime('%Y/%m/%d %H:%M'))

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
                axs.xaxis.set_minor_locator(AutoMinorLocator())
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

            # Set altiudes and increments for NIMO conjunctions
            alt_arr = [sata, 'hmf2', sata]
            inc_arr = [0, 0, 100]

            # Set plot location using lx and r i.e. subplot(lx[i], r[i])
            lx = [0, 1, 1]
            r = [1, 0, 1]

            # NIMO-------------------------
            for i in range(len(alt_arr)):

                # Initialize base_string for data saving purposes
                if i == 1:
                    base_str = 'Nimo_' + alt_arr[i]
                else:
                    base_str = 'Nimo_Swarm'
                if inc_arr[i] == 100:
                    base_str += str(100)

                # Choose an altitude
                alt_str = alt_arr[i]

                # RuntimeError for nimo_conjunction if no NIMO data within
                # max_tdif of Swarm time
                try:
                    (nimo_swarm_alt,
                     nimo_map) = nimo_conjunctions.nimo_conjunction(
                         nimo_dc, swarm_check, alt_str, inc=inc_arr[i],
                         max_tdif=max_tdif)
                except RuntimeError:
                    continue

                # ------------- NIMO EIA STATE -----------------------
                nlat = nimo_swarm_alt['Mag_Lat'].values
                density = nimo_swarm_alt['Ne'].values
                den_str = 'Ne'

                (nimo_lat, nimo_dfilt, eia_type_slope, z_lat, plats,
                 p3) = eia_complete(
                     nlat, density, den_str, filt=nimo_filt,
                     interpolate=nimo_interpolate,
                     barrel_envelope=nimo_envelope,
                     barrel_radius=nimo_barrel, window_lat=nimo_window)

                # Give User a heads up for an unknown type
                if eia_type_slope == 'unknown':
                    print('NIMO type unknown at ' + alt_str + ' height')
                    print(nimo_swarm_alt["Time"].iloc[0][0].strftime(
                        '%Y/%m/%d_%H:%M:%S.%f'))

                # general Nimo Info only need to save once per alt_arr
                nts = 'Nimo_Time'
                if i == 0:
                    df.at[f, nts] = nimo_swarm_alt["Time"].iloc[0][0].strftime(
                        '%Y/%m/%d_%H:%M:%S.%f')
                    df.at[f, 'Nimo_GLon'] = nimo_swarm_alt["Longitude"].iloc[0]
                    df.at[f, 'Nimo_Min_MLat'] = min(nimo_swarm_alt["Mag_Lat"])
                    df.at[f, 'Nimo_Max_MLat'] = max(nimo_swarm_alt["Mag_Lat"])

                # save altitude specific information
                df.at[f, base_str + '_Alt'] = int(nimo_swarm_alt["alt"].iloc[0])
                df.at[f, base_str + '_Type'] = eia_type_slope

                # Plot NIMO
                if fig_on:
                    axns = fig.add_subplot(gs[lx[i], r[i]])
                    plt.rcParams.update({'font.size': fosi})
                    axns.plot(nimo_swarm_alt['Mag_Lat'],
                              nimo_swarm_alt['Ne'],
                              linestyle='--', marker='o', label='Raw Ne')
                    axns.plot(nimo_lat, nimo_dfilt, color='C1',
                              label="Filtered Ne")
                    axns.scatter(nimo_swarm_alt['Mag_Lat'].iloc[0],
                                 nimo_swarm_alt['Ne'].iloc[0], color='white',
                                 s=0, label=eia_type_slope)

                # Save and/or plot Peak lats
                if len(plats) > 0:
                    for pi, p in enumerate(plats):
                        lat_lo = (abs(p - nimo_swarm_alt['Mag_Lat']).argmin())
                        lat_plot = nimo_swarm_alt['Mag_Lat'].iloc[lat_lo]
                        dfl = base_str + '_Peak_MLat' + str(pi + 1)
                        dfn = base_str + '_Peak_Ne' + str(pi + 1)
                        df.at[f, dfl] = nimo_swarm_alt['Mag_Lat'].iloc[lat_lo]
                        df.at[f, dfn] = nimo_swarm_alt['Ne'].iloc[lat_lo]
                        if fig_on:
                            axns.vlines(lat_plot,
                                        ymin=min(nimo_swarm_alt['Ne']),
                                        ymax=nimo_swarm_alt['Ne'].iloc[lat_lo],
                                        alpha=0.5, color='k')
                # save and/or plot thrid peak if present
                # (no thrid peak for ghosts)
                if len(p3) > 0:
                    for pi, p in enumerate(p3):
                        lat_loc = (abs(p - nimo_swarm_alt['Mag_Lat']).argmin())
                        lat_plot = nimo_swarm_alt['Mag_Lat'].iloc[lat_loc]
                        dl3 = base_str + '_Third_Peak_MLat' + str(pi + 1)
                        df_strn3 = base_str + '_Third_Peak_Ne' + str(pi + 1)
                        df.at[f, dl3] = nimo_swarm_alt['Mag_Lat'].iloc[lat_loc]
                        df.at[f, df_strn3] = nimo_swarm_alt['Ne'].iloc[lat_loc]
                        if fig_on:
                            axns.vlines(
                                lat_plot, ymin=min(nimo_swarm_alt['Ne']),
                                ymax=nimo_swarm_alt['Ne'].iloc[lat_loc],
                                linestyle='--', alpha=0.5, color='r')

                # Set labels for plots
                if fig_on:
                    axns.ticklabel_format(axis='y', style='sci',
                                          scilimits=(0, 0))
                    axns.xaxis.set_minor_locator(AutoMinorLocator())
                    axns.tick_params(axis='both', which='major', length=8)
                    axns.tick_params(axis='both', which='minor', length=5)
                    axns.set_xlabel("Magnetic Latitude (\N{DEGREE SIGN})")
                    axns.set_ylabel("Ne (cm$^-3$)")
                    if 'south' in eia_type_slope:
                        axns.legend(fontsize=fosi - 3, loc='upper right')
                    else:
                        axns.legend(fontsize=fosi - 3, loc='upper left')
                    axns.set_title('NIMO {:d} km'.format(
                                   int(nimo_swarm_alt['alt'].iloc[0])))
            # Terminator and Map plotting
            if fig_on:

                # Set the date and time for the terminator
                date_term = nimo_swarm_alt['Time'].iloc[0][0]

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
                heatmap = ax.pcolormesh(nimo_map['glon'], nimo_map['glat'],
                                        nimo_map['nmf2'], cmap='cividis',
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
                ax.set_title('NIMO NmF2 at {:}'.format(
                             nimo_swarm_alt['Time'].iloc[0][0]))

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

                plt.suptitle(str(int(nimo_swarm_alt['Longitude'].iloc[0]))
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
                save_as = (save_dir + '/NIMO_SWARM_' + sata + '_' + ds + '_'
                           + ts1 + '_' + ts2 + '.jpg')
                fig.savefig(save_as)
                plt.close()

    # Write the output file
    io.write.write_daily_stats(df, start_day, 'NIMO', 'SWARM', file_dir)

    return df
