#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Functions to plot Swarm skill score statistical outcomes."""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from pyValEIA.io import load
from pyValEIA.stats import skill_score
from pyValEIA.eia.types import clean_type


def states_report_swarm(date_range, daily_dir, typ='eia',
                        NIMO_alt='swarm'):
    """ Report States for date range for Swarm comparison,
    need to make one for Madrigal comparison
    Parameters
    ----------
    date_range : pandas daterange
        Date range of desired states files
    daily_dir : str
        directory of daily files
    typ: str
        desired type to check against
        for orientation of 'state'
        'eia'(default), 'peak', 'flat', 'trough'
        for orientation of 'direction'
        'north', 'south', 'neither'
    NIMO_alt: str
        specifies which altitude to use
        'swarm'(default),'hmf2','100'
    Returns
    -------
    NiSw : DataFrame
        NIMO states, directions, and types
        also includes longitude, local times, and sat list
    Sw : DataFrame
        Swarm States, direction, and types
        also includes longitude, local times, and sat list
    Py : DataFrame
        PyIRI states, directions, and types
        also includes longitude, local times, and sat list
    """
    # Check to see what we are comparing (states or directions)
    if (typ == 'north') | (typ == 'south') | (typ == 'neither'):
        orientation = 'direction'
    else:
        orientation = 'state'
    # Set date array for given time range
    date_array = date_range.to_pydatetime()

    # Establish strings
    if NIMO_alt == 'swarm':
        NIMO_str = 'Nimo_Swarm_Type'
    elif NIMO_alt == 'hmf2':
        NIMO_str = 'Nimo_hmf2_Type'
    elif NIMO_alt == '100':
        NIMO_str = 'Nimo_Swarm100_Type'

    # initialize params for states, directions, and full types
    nim_sts = []  # NIMO
    nim_dirs = []
    nim_typ = []

    sw_sts = []  # Swarm
    sw_dirs = []
    sw_typ = []

    py_sts = []  # PyIRI
    py_dirs = []
    py_typ = []

    lts = []
    lons = []
    sats = []

    for sday in date_array:
        # load files
        nimdf = load.load_daily_stats(sday, 'NIMO', 'SWARM', daily_dir)
        pydf = load.load_daily_stats(sday, 'PyIRI', 'SWARM', daily_dir)

        # NIMO
        s, d = clean_type(nimdf[NIMO_str].values)
        nim_sts.extend(s)
        nim_dirs.extend(d)
        nim_typ.extend(nimdf[NIMO_str].values)

        # Swarm
        s, d = clean_type(nimdf['Swarm_EIA_Type'].values)
        sw_sts.extend(s)
        sw_dirs.extend(d)
        sw_typ.extend(nimdf['Swarm_EIA_Type'].values)

        # basic params
        lts.extend(nimdf['LT_Hour'].values)
        lons.extend(nimdf['Nimo_GLon'].values)
        sats.extend(nimdf['Satellite'].values)

        # PyIRI
        s, d = clean_type(pydf['PyIRI_Type'].values)
        py_sts.extend(s)
        py_dirs.extend(d)
        py_typ.extend(pydf['PyIRI_Type'].values)

    # Initialize DataFrames
    NiSw = pd.DataFrame()
    Sw = pd.DataFrame()
    PyI = pd.DataFrame()

    # NIMO
    NiSw['state'] = np.array(nim_sts)
    NiSw['direction'] = np.array(nim_dirs)
    NiSw['type'] = np.array(nim_typ)
    NiSw['GLon'] = np.array(lons)
    NiSw['LT'] = np.array(lts)
    NiSw['Sat'] = np.array(sats)

    # Swarm
    Sw['state'] = np.array(sw_sts)
    Sw['direction'] = np.array(sw_dirs)
    Sw['type'] = np.array(sw_typ)
    Sw['GLon'] = np.array(lons)
    Sw['LT'] = np.array(lts)
    Sw['Sat'] = np.array(sats)

    # PyIRI
    PyI['state'] = np.array(py_sts)
    PyI['direction'] = np.array(py_dirs)
    PyI['type'] = np.array(py_typ)
    PyI['GLon'] = np.array(lons)
    PyI['LT'] = np.array(lts)
    PyI['Sat'] = np.array(sats)

    # Compare PyIRI to Swarm and NIMO to Swarm
    NiSw['skill'] = skill_score.state_check(Sw[orientation].values,
                                            NiSw[orientation].values, state=typ)
    PyI['skill'] = skill_score.state_check(Sw[orientation].values,
                                           PyI[orientation].values, state=typ)

    return NiSw, Sw, PyI


def lss_plot_Swarm(model1, model2, eia_type, date_range, model1_name='Model1',
                   model2_name='Model2', PorC='PC',
                   DayNight=True, LT_range=[7, 19], coin=True):
    """ Plot LSS vs CSI or PC 4 panels (one for each LSS)
    NOTE: LSS can range outside of +/-1
    Parameters
    ----------
    model1 : dataframe
        first model dataframe built by states_report_swarm
    model2 : dataframe
        second model dataframe built by states_report_swarm
    eia_type : str
        desired eia type for fig title
    date_range : datetime range
        For plotting title purposes
    model1_name : str kwarg
        first model name for labelling purposes
    model2_name : str kwarg
        second model name for labelling purposes
    PorC : str kwarg
        Percent correct or Critical success index for x axes
    DayNight : bool kwarg
        True (default) if panels should have separate markers for day and night
        otherwise (false) all are plotted together
    LT_range : list kwarg
        Range of day night local time, Default is 7 LT to 19 LT for day and
        19 LT to 7 LT for Night
    coin : bool kwarg
        If True, coin LSS will be plotted for comparison (default)
        if false, coin LSS will not be plotted
    Returns
    -------
    fig : fig handle
        4 panel figure (one for each LSS)
    """

    # Set date array for given time range
    date_array = date_range.to_pydatetime()

    # model 1 and model 2 will have same sats
    sats = np.unique(model1['Sat'])

    # let's make a plot of changing PC or CSI
    fig, axs = plt.subplots(2, 2, figsize=(11, 11))
    look = ['All']
    cols1 = ['blue']
    cols2 = ['orange']
    colsc = ['black']

    # IF DayNight Separation is specified
    if DayNight:
        model1_day = model1[((model1['LT'] > LT_range[0])
                             & (model1['LT'] < LT_range[1]))]
        model1_night = model1[((model1['LT'] < LT_range[0])
                               | (model1['LT'] > LT_range[1]))]
        model2_day = model2[((model2['LT'] > LT_range[0])
                             & (model2['LT'] < LT_range[1]))]
        model2_night = model2[((model2['LT'] < LT_range[0])
                               | (model2['LT'] > LT_range[1]))]
        look = ['Day', 'Night']
        cols1 = ['skyblue', 'blue']
        cols2 = ['salmon', 'red']
        colsc = ['gray', 'black']

    look = np.array(look)
    for lo in range(len(look)):
        col1 = cols1[lo]
        col2 = cols2[lo]
        colc = colsc[lo]
        for j, s in enumerate(sats):
            if DayNight:
                if lo == 0:
                    model1_sat = model1_day[model1_day['Sat'] == s]
                    model2_sat = model2_day[model2_day['Sat'] == s]
                elif lo == 1:
                    model1_sat = model1_night[model1_night['Sat'] == s]
                    model2_sat = model2_night[model2_night['Sat'] == s]
            else:
                model1_sat = model1[model1['Sat'] == s]
                model2_sat = model2[model2['Sat'] == s]

            # Compute PC and CSI
            PC_1, CSI_1 = skill_score.calc_pc_and_csi(
                model1_sat['skill'].values, coin=False)
            PC_2, CSI_2 = skill_score.calc_pc_and_csi(
                model2_sat['skill'].values, coin=False)
            PC_coin, CSI_coin = skill_score.calc_pc_and_csi(
                model2_sat['skill'].values, coin=True)

            # Compute Skill Scores
            (lss1_mod1, lss2_mod1, lss3_mod1,
             lss4_mod1) = skill_score.liemohn_skill_score(
                model1_sat['skill'].values, coin=False)
            (lss1_mod2, lss2_mod2, lss3_mod2,
             lss4_mod2) = skill_score.liemohn_skill_score(
                model2_sat['skill'].values, coin=False)
            (lss1_coin, lss2_coin, lss3_coin,
             lss4_coin) = skill_score.liemohn_skill_score(
                model2_sat['skill'].values, coin=True)

            # MAkE lss arrays
            lss_mod1 = np.array([lss1_mod1, lss2_mod1, lss3_mod1, lss4_mod1])
            lss_mod2 = np.array([lss1_mod2, lss2_mod2, lss3_mod2, lss4_mod2])
            lss_coin = np.array([lss1_coin, lss2_coin, lss3_coin, lss4_coin])

            # Change label and variables depending on if user specified
            # CSI or PC
            if PorC == 'PC':
                lab = 'Percent Correct'
                exes = np.array([PC_coin, PC_1, PC_2])
            elif PorC == 'CSI':
                lab = 'Critical Success Index'
                exes = np.array([CSI_coin, CSI_1, CSI_2])

            # Establish axes
            for i in range(4):
                if i == 0:
                    ax = axs[0, 0]
                if i == 1:
                    ax = axs[0, 1]
                if i == 2:
                    ax = axs[1, 0]
                if i == 3:
                    ax = axs[1, 1]

                # Plot Satellite names as text
                # only plot coin toss if specified as True
                if coin:
                    ax.text(exes[0], lss_coin[i], s, fontsize=12, color=colc,
                            label=None)
                ax.text(exes[1], lss_mod1[i], s, fontsize=12, color=col1,
                        label=None)
                ax.text(exes[2], lss_mod2[i], s, fontsize=12, color=col2,
                        label=None)

                # Set labels depending on plot number
                if (i == 0) | (i == 2):
                    ax.set_ylabel('Skill Score')
                ax.set_xlabel(lab)
                ax.set_title('lss' + str(i + 1))

                # Legend:
                # If you want to plot the legend for both day and night colors
                # in the first legend remove "& (lo == 0)"
                if (i == 0) & (j == 0) & (lo == 0):
                    ax.plot(-99, -99, color=col2, label=model2_name)
                    ax.plot(-99, -99, color=col1, label=model1_name)
                    if coin:
                        ax.plot(-99, -99, color=colc, label='Coin Flip')
                    ax.legend()
                if (i == 1) & (lo == 0) & (j == 0):
                    lab1 = (str(np.round(min(model1_sat['LT']), 1))
                            + '-' + str(np.round(max(model1_sat['LT']), 1))
                            + ' LT')
                    ax.plot(-99, -99, color=col1, label=lab1)
                if (i == 1) & (lo == 1) & (j == 0):
                    lab1 = (str(np.round(min(model1_sat['LT']), 1))
                            + '-' + str(np.round(max(model1_sat['LT']), 1))
                            + ' LT')
                    ax.plot(-99, -99, color=col1, label=lab1)
                    ax.legend()

                ax.set_ylim([-1, 1])
                ax.set_xlim([0, 1])

    # Add super title
    plt.suptitle((eia_type + ' ' + date_array[0].strftime('%Y/%m/%d') + '-'
                  + date_array[-1].strftime('%Y/%m/%d')), x=0.5, y=0.92,
                 fontsize=17)
    return fig


def one_model_lss_plot_Swarm(model1, eia_type, date_range, model_name='Model',
                             PorC='PC', DayNight=True, LT_range=[7, 19],
                             coin=True):
    """ Plot LSS vs CSI or PC 4 panels (one for each LSS) for 1 model alone
    NOTE: LSS is only useful in comparison to another model, therefore,
    coin set to True is highly recommended!
    Parameters
    ----------
    model1 : dataframe
        model dataframe built by states_report_swarm
    eia_type : str
        desired eia type for fig title
    date_range : datetime range
        For plotting title purposes
    model_name : str kwarg
        first model name for labelling purposes
    PorC : str kwarg
        Percent correct or Critical success index for x axes
    DayNight : bool kwarg
        True (default) if panels should have separate markers for day and night
        otherwise (false) all are plotted together
    LT_range : list kwarg
        Range of day night local time, Default is 7 LT to 19 LT for day and
        19 LT to 7 LT for Night
    coin : bool kwarg
        If True, coin LSS will be plotted for comparison (default)
        if false, coin LSS will not be plotted
    Returns
    -------
    fig : fig handle
        4 panel figure (one for each LSS)
    """
    # Print Warning if coin is set to False
    if not coin:
        warnings.warn("Warning: Coin is False! LSS is a comparison tool!")
    # Set date array for given time range
    date_array = date_range.to_pydatetime()

    # model 1 and model 2 will have same sats
    sats = np.unique(model1['Sat'])

    # let's make a plot of changing PC or CSI
    fig, axs = plt.subplots(2, 2, figsize=(11, 11))
    look = ['All']
    cols1 = ['blue']
    colsc = ['black']

    # IF DayNight Separation is specified
    if DayNight:
        model1_day = model1[((model1['LT'] > LT_range[0])
                             & (model1['LT'] < LT_range[1]))]
        model1_night = model1[((model1['LT'] < LT_range[0])
                               | (model1['LT'] > LT_range[1]))]
        look = ['Day', 'Night']
        cols1 = ['skyblue', 'blue']
        colsc = ['gray', 'black']

    look = np.array(look)
    for lo in range(len(look)):
        col1 = cols1[lo]
        colc = colsc[lo]
        for j, s in enumerate(sats):
            if DayNight:
                if lo == 0:
                    model1_sat = model1_day[model1_day['Sat'] == s]
                elif lo == 1:
                    model1_sat = model1_night[model1_night['Sat'] == s]
            else:
                model1_sat = model1[model1['Sat'] == s]

            # Compute PC and CSI
            PC_1, CSI_1 = skill_score.calc_pc_and_csi(
                model1_sat['skill'].values, coin=False)
            PC_coin, CSI_coin = skill_score.calc_pc_and_csi(
                model1_sat['skill'].values, coin=True)

            # Compute Skill Scores
            (lss1_mod1, lss2_mod1, lss3_mod1,
             lss4_mod1) = skill_score.liemohn_skill_score(
                model1_sat['skill'].values, coin=False)
            (lss1_coin, lss2_coin, lss3_coin,
             lss4_coin) = skill_score.liemohn_skill_score(
                model1_sat['skill'].values, coin=True)

            # MAkE lss arrays
            lss_mod1 = np.array([lss1_mod1, lss2_mod1, lss3_mod1, lss4_mod1])
            lss_coin = np.array([lss1_coin, lss2_coin, lss3_coin, lss4_coin])

            # Change label and variables depending on if user specified
            # CSI or PC
            if PorC == 'PC':
                lab = 'Percent Correct'
                exes = np.array([PC_coin, PC_1])
            elif PorC == 'CSI':
                lab = 'Critical Success Index'
                exes = np.array([CSI_coin, CSI_1])

            # Establish axes
            for i in range(4):
                if i == 0:
                    ax = axs[0, 0]
                if i == 1:
                    ax = axs[0, 1]
                if i == 2:
                    ax = axs[1, 0]
                if i == 3:
                    ax = axs[1, 1]

                # Plot Satellite names as text
                # only plot coin toss if specified as True
                if coin:
                    ax.text(exes[0], lss_coin[i], s, fontsize=12, color=colc,
                            label=None)
                ax.text(exes[1], lss_mod1[i], s, fontsize=12, color=col1,
                        label=None)

                # Set labels depending on plot number
                if (i == 0) | (i == 2):
                    ax.set_ylabel('Skill Score')
                ax.set_xlabel(lab)
                ax.set_title('lss' + str(i + 1))

                # Legend:
                # If you want to plot the legend for both day and night colors
                # in the first legend remove "& (lo == 0)"
                if (i == 0) & (j == 0) & (lo == 0):
                    ax.plot(-99, -99, color=col1, label=model_name)
                    if coin:
                        ax.plot(-99, -99, color=colc, label='Coin Flip')
                    ax.legend()

                # Add Time Legend for DayNight True
                if DayNight:

                    if (i == 1) & (lo == 0) & (j == 0):
                        lab1 = (str(np.round(min(model1_sat['LT']), 1))
                                + '-' + str(np.round(max(model1_sat['LT']), 1))
                                + ' LT')
                        ax.plot(-99, -99, color=col1, label=lab1)
                    if (i == 1) & (lo == 1) & (j == 0):
                        lab1 = (str(np.round(min(model1_sat['LT']), 1))
                                + '-' + str(np.round(max(model1_sat['LT']), 1))
                                + ' LT')
                        ax.plot(-99, -99, color=col1, label=lab1)
                        ax.legend()

                ax.set_ylim([-1, 1])
                ax.set_xlim([0, 1])

    # Add super title
    plt.suptitle((eia_type + ' ' + date_array[0].strftime('%Y/%m/%d') + '-'
                  + date_array[-1].strftime('%Y/%m/%d')), x=0.5, y=0.92,
                 fontsize=17)
    return fig


def map_hist_panel(ax, model, bin_lons=37, DayNight=True, LT_range=[7, 19]):
    """ plot histogram maps on a panel
    Parameters
    ----------
    ax : plt axis
        matplotlib.plt axis
    model : dataframe
        dataframe of model data including skill and local times
        built by states_report_swarm
    bin_lons : int kwarg
        number of bins between -180 and 180 deg geo lon
        np.linspace(-180, 180, bin_lons)
    DayNight : bool kwarg
        True (default) if panels should have separate markers for day and night
        otherwise (false) all are plotted together
    LT_range : list kwarg
        Range of day night local time, Default is 7 LT to 19 LT for day and
        19 LT to 7 LT for Night
    Returns
    -------
    ax : plt axis
        original axis with data plotted
    hist_ax : plt axis
        twinx axis to ax with histogram plotted
    """
    # Initialize histogram bins
    hist_bins = np.linspace(-180, 180, bin_lons)

    # PLot Map
    ax.set_global()
    ax.add_feature(cfeature.LAND, edgecolor='gray', facecolor='none')
    ax.add_feature(cfeature.COASTLINE)
    ax.set_xticklabels([])

    # Fix aspect ratio issue
    ax.set_aspect('auto', adjustable='box')

    # Set Histogram axis
    hist_ax = ax.twinx()

    # IF DayNight Separation is specified
    if DayNight:
        model_day = model[((model['LT'] > LT_range[0])
                           & (model['LT'] < LT_range[1]))]
        model_night = model[((model['LT'] < LT_range[0])
                             | (model['LT'] > LT_range[1]))]

        look = ['Day', 'Night']
        colsh = ['salmon', 'skyblue']

        # Day
        lon_day = model_day['GLon']
        if len(model_day['LT']) > 0:
            day_str = (str(int(np.trunc(min(model_day['LT'])))) + ' to '
                       + str(int(np.round(max(model_day['LT'])))))
        else:
            day_str = ''
        hist_ax.hist(lon_day, bins=hist_bins, color=colsh[0],
                     alpha=0.5, label=look[0] + day_str + ' LT')

        # Night
        lon_night = model_night['GLon']
        if len(model_night['LT']) > 0:
            night_str = (str(int(np.trunc(min(model_night['LT']))))
                         + ' to ' + str(int(np.round(max(model_night['LT'])))))
        else:
            night_str = ''
        hist_ax.hist(lon_night, bins=hist_bins, color=colsh[1],
                     alpha=0.5, label=look[1] + night_str + ' LT')
        hist_ax.set_xticklabels([])

    else:  # Day night not specified
        lon = model['GLon']
        hist_ax.hist(lon, bins=hist_bins, color=colsh[0], alpha=0.3)

    return ax, hist_ax


def decision_table_sat(states, eia_type='eia',
                       sats=['A', 'B', 'C'], model_name='Model'):
    """ Neat decision table summing up the hits, misses,
    correct negatives, and false positives

    Parameters
    ----------
    states: dataframe
        dataframe of model data including skill and local times
        built by states_report_swarm
    eia_type : str
        eia state e.g. EIA, Peak, etc. depending on what is considered a hit
    sats : list of strings kwarg
        swarm satellites 'A', 'B', and 'C' as default
        can specify just 1 or 2
    model_name : str kwarg
        Model name for decision table label
        default 'Model'
    Returns
    -------
    df : dataframe
        dataframe in table format separated by satellite
        and event state (state, non-state)
        index using
        df.loc[(f'Swarm {satellite}', eia_type), (model_name, eia_type)]
    """
    for i, s in enumerate(sats):

        # Sum total HMFCs

        hit = sum(states['skill'][states['Sat'] == s] == 'H')
        falarm = sum(states['skill'][states['Sat'] == s] == 'F')
        miss = sum(states['skill'][states['Sat'] == s] == 'M')
        corneg = sum(states['skill'][states['Sat'] == s] == 'C')
        if i == 0:
            df = pd.DataFrame(
                [[hit, miss], [falarm, corneg]],
                index=pd.MultiIndex.from_product(
                    [['Swarm ' + s], [eia_type, 'Non-' + eia_type]]),
                columns=pd.MultiIndex.from_product(
                    [[model_name], [eia_type, 'Non-' + eia_type]]))
        else:
            df.loc[('Swarm ' + s, eia_type), :] = np.array([hit, miss])
            df.loc[('Swarm ' + s, 'Non-' + eia_type), :] = np.array([
                falarm, corneg])

    df.style
    return df


def plot_hist_quad_maps(model_states, sat, eia_type, date_range, bin_lons=37,
                        model_name='Model', fosi=16, hist_ylim=[0, 15],
                        LT_range=[7, 19]):
    """ plot histogram maps on a 4 panel figure for each score: Hit, Miss,
    False positive, and Correct Negative
    Parameters
    ----------
    model_states : dataframe
        dataframe of model data including skill and local times
        built by states_report_swarm
    sat : str
        swarm satellite 'A', 'B', or 'C'
    eia_type : str
        eia state e.g. EIA, Peak, etc. depending on what is considered a hit
    date_range : pandas daterange
        range of dates for title purposes
    bin_lons : int kwarg
        number of bins between -180 and 180 deg geo lon
        default 37
        np.linspace(-180, 180, bin_lons)
    model_name : str kwarg
        name of model for title purposes
        default 'Model'
    fosi : int kwarg
        font size for plot
        default 16
    hist_ylim : list kwarg
        y range (counts) for hist plot
        default [0,15]
    LT_range : list kwarg
        Range of day night local time, Default is 7 LT to 19 LT for day and
        19 LT to 7 LT for Night
    Returns
    -------
    fig : figure handle
        fig with 4 panels of hist maps
    """

    # Creating Figure with GridSpec
    scores = ["H", "M", "F", "C"]

    model_sat = model_states[model_states['Sat'] == sat]

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2)
    plt.rcParams.update({'font.size': fosi})

    for s, score in enumerate(scores):
        model_score = model_sat[model_sat['skill'] == score]

        # Panel 1: World Map with Longitude Histogram
        if s == 0:
            ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        elif s == 1:
            ax0 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        elif s == 2:
            ax0 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
        elif s == 3:
            ax0 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())

        ax0, hist_ax = map_hist_panel(ax0, model_score, bin_lons=bin_lons,
                                      DayNight=True, LT_range=LT_range)

        # Add skill titles
        ax0.set_aspect('auto', adjustable='box')
        if score == 'H':
            ax0.text(-160, 95, 'HIT', fontweight='bold', color='black',
                     fontsize=20)
        elif score == 'M':
            ax0.text(-160, 95, 'MISS', fontweight='bold', color='black',
                     fontsize=20)
        elif score == 'C':
            ax0.text(-160, -105, 'CORRECT NEGATIVE', fontweight='bold',
                     color='black', fontsize=20)
        elif score == 'F':
            ax0.text(-160, -105, 'FALSE POSITIVE', fontweight='bold',
                     color='black', fontsize=20)

        # Move Latitude Axis (Secondary Y-Axis) to the Left
        if (s == 1) | (s == 3):
            secax_y = ax0.twinx()
            secax_y.set_ylim(ax0.get_ylim())
            secax_y.set_ylabel("Latitude (\N{DEGREE SIGN})")
            secax_y.yaxis.set_label_position('left')
            secax_y.yaxis.tick_left()
            secax_y.spines['right'].set_visible(False)  # Hide right y-axis
            secax_y.spines['left'].set_visible(True)   # Show left y-axis
        else:
            ax0.set_yticklabels([])

        if (s == 0) | (s == 2):
            hist_ax.yaxis.set_label_position('left')
            hist_ax.yaxis.tick_left()
        hist_ax.set_xlim(-180, 180)
        hist_ax.set_ylabel('Counts')
        if s == 1:
            hist_ax.legend(bbox_to_anchor=(1, 1.1), loc='upper right',
                           borderaxespad=0, ncols=2)
        hist_ax.set_ylim(hist_ylim)

    # Secondary X-Axis for the Map
        if (s == 2) | (s == 3):
            secax_x = ax0.twiny()
            secax_x.set_xlim(ax0.get_xlim())
            secax_x.set_xlabel("Longitude (\N{DEGREE SIGN})")
        else:
            hist_ax.set_xticklabels([])
    if eia_type == 'eia':
        eia_title = 'EIA'
    else:
        eia_title = eia_type

    # Add title
    date_str = date_range[0].strftime('%B %Y')
    fig.suptitle(
        f"{date_str} {model_name} vs SWARM Satellite {sat} Type: {eia_title}",
        fontsize=25, fontweight="bold", x=0.5, y=0.98)
    # Adjust layout to prevent overlap
    return fig


def style_df_table(df_table, eia_type):
    """ Style decision table using
    Need all satellites to use this!
    Parameters
    ----------
    df_table : dataframe
        dataframe created by decision_table_sat
    eia_type : str
        string designating which eia type is being reported
    Returns
    -------
    s : styled dataframe
    """
    s = df_table.style.format('{:.0f}')

    # Adding Color
    s.set_table_styles([
        {'selector': '.true', 'props': 'background-color: #e6ffe6;'},
        {'selector': '.false', 'props': 'background-color: #ffe6e6;'},],
        overwrite=False)
    cell_color = pd.DataFrame([['true ', 'false '],
                               ['false ', 'true '],
                               ['true ', 'false '],
                               ['false ', 'true '],
                               ['true ', 'false '],
                               ['false ', 'true ']],
                              index=df_table.index,
                              columns=df_table.columns[:len(df_table)])
    for l0 in ['Swarm A', 'Swarm B', 'Swarm C']:
        s.set_table_styles(
            {(l0, 'Non-' + eia_type):
             [{'selector': '', 'props':
               'border-bottom: 2px solid black;'}],
             (l0, eia_type):
             [{'selector': '.level0', 'props':
               'border-bottom: 2px solid black'}]},
            overwrite=False, axis=1)

    s.set_td_classes(cell_color)

    return s


def HMFC_percent_panel(model_states, df_table, fig, ax, eia_type,
                       colors=['blue', 'red', 'purple']):
    """ Plot the percentages of H/(H+M), M/(H+M), F/(F+C), C/(C+F) as
    4 quadrants
    Prameters
    model_states: dataframe
        dataframe of model data including skill and local times
        built by states_report_swarm
    df_table : dataframe
        decision table build by decision_table_sat
    fig : figure
        figure for plotting on
    eia_type : str
        string designating which eia type is being reported
    colors : list of strings
        colors to be plotted for each satellite
    Returns
    ------
    fig : figure
        the resulting figure
    """
    model_name = df_table.columns[0][0]
    sats = np.unique(model_states['Sat'])

    for i, s in enumerate(sats):
        col = colors[i]

        # total values Y_tot is in state, N_tot is out of state
        Y_tot = sum(df_table.loc[('Swarm ' + s, eia_type)].values)
        N_tot = sum(df_table.loc[('Swarm ' + s, 'Non-' + eia_type)].values)

        # decimal HMCF as yes and no

        # hit (yes yes)
        yy = df_table.loc[(('Swarm ' + s, eia_type),
                           (model_name, eia_type))] / Y_tot
        # miss (yes no)
        yn = df_table.loc[(('Swarm ' + s, eia_type),
                           (model_name, 'Non-' + eia_type))] / Y_tot

        # correct negative (no no)
        nn = df_table.loc[(('Swarm ' + s, 'Non-' + eia_type),
                           (model_name, 'Non-' + eia_type))] / N_tot

        # False positive (no yes)
        ny = df_table.loc[(('Swarm ' + s, 'Non-' + eia_type),
                           (model_name, eia_type))] / N_tot

        # plot
        plt.text(yy, yy, s, fontsize=12, color=col)
        plt.text(yn, -yn, s, fontsize=12, color=col)
        plt.text(-nn, -nn, s, fontsize=12, color=col)
        plt.text(-ny, ny, s, fontsize=12, color=col)

    return fig


def HMFC_percent_figure(model1, model2, eia_type, model1_name='Model1',
                        model2_name='Model2', col1='orange', col2='purple',
                        fosi=16):
    """ Plot full figure using HMFC_percent_panel
    Parameters
    ----------
    model1 : dataframe
        first model dataframe built by states_report_swarm
    model2 : dataframe
        second model dataframe built by states_report_swarm
    eia_type : str
        desired eia type for fig title
    model1_name : str kwarg
        first model name for labelling purposes
        default Model1
    model2_name : str kwarg
        second model name for labelling purposes
        default Model2
    col1 : str
        plotting color for Model1
        defualt orange
    col2 : str
        plotting color for Model 2
        default purple
    fosi : int
        font size for plot
    Returns
    -------
    fig : figure

    Notes
    -----
    This figure has a lot going on. When you look at it, think of each
    quadrant as a separate plot defined by Hit, Miss, Correct Negative,
    and False Positive as labelled. The percentages are the percent the
    model got correct or incorrect based on event states
    For example, for Hits, ther percentage is Hit/(Hit + Miss) where Hit+Miss
    is the total in the event states, the panel below that Miss/(Hit+Miss) is
    equivalent to 100% - Hit/(Hit + Miss), so those sectors are conjugate to
    each other
    For quick viewing, there are 4 shaded regions. These represent when a
    model is doing better than a coin toss. Ideally, False positives and Misses
    would have a low % and Hits and Correct Negatives have a higher percentage
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams.update({'font.size': fosi})

    # Model1
    df_table1 = decision_table_sat(model1, model_name=model1_name)
    HMFC_percent_panel(model1, df_table1, fig, ax, eia_type,
                       colors=[col1, col1, col1])

    # Model2
    df_table2 = decision_table_sat(model2, model_name=model2_name)
    HMFC_percent_panel(model2, df_table2, fig, ax, eia_type,
                       colors=[col2, col2, col2])

    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.text(-0.9, 0.9, 'False Positive', fontsize=12, color='black')
    plt.text(-0.9, -0.9, 'Correct Negative', fontsize=12, color='black')
    plt.text(0.9, 0.9, 'Hit', fontsize=12, color='black')
    plt.text(0.8, -0.9, 'Miss', fontsize=12, color='black')

    # Add horizontal and vertical lines for x = 0 and y = 0
    plt.axvline(0, color='black', linestyle='-', linewidth=1.5)
    plt.axhline(0, color='black', linestyle='-', linewidth=1.5)

    # add 50% (coin toss) vertical and horizontal lines
    plt.axvline(0.5, color='gray', linestyle='--', linewidth=0.8)
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(-0.5, color='gray', linestyle='--', linewidth=0.8)
    plt.axhline(-0.5, color='gray', linestyle='--', linewidth=0.8)

    # Add custom ticks as percentages
    custom_ticks = [-1, -0.5, 0, 0.5, 1]
    custom_labels = ['100%', '50%', '0%', '50%', '100%']
    plt.yticks(custom_ticks, custom_labels, fontsize=fosi - 3)
    plt.xticks(custom_ticks, custom_labels, fontsize=fosi - 3)

    # Add labels along axes, red (high is bad) blue (high is good)
    plt.text(0.4, -1.15, 'M/(H+M)', color='red')
    plt.text(-0.65, -1.15, 'C/(C+F)', color='blue')

    plt.text(0.4, 1.15, 'H/(H+M)', color='blue')
    plt.text(-0.6, 1.15, 'F/(C+F)', color='red')

    plt.text(1.2, -0.6, 'M/(H+M)', rotation=90, color='red')
    plt.text(-1.25, -0.6, 'C/(C+F)', rotation=90, color='blue')

    plt.text(1.2, 0.4, 'H/(H+M)', rotation=90, color='blue')
    plt.text(-1.25, 0.4, 'F/(C+F)', rotation=90, color='red')

    # Add legend for models
    plt.text(1.3, 0.9, model1_name, color=col1)
    plt.text(1.3, 0.8, model2_name, color=col2)

    # add labels to other side of axis
    sec_ax = plt.twiny()
    sec_ticks = [0, 0.25, 0.5, 0.75, 1]
    sec_ax.set_ylim([-1, 1])
    sec_ax.set_xticks(sec_ticks, custom_labels, fontsize=fosi - 3)

    sec_ay = plt.twinx()
    sec_ay.set_ylim([-1, 1])
    sec_ay.set_yticks(custom_ticks, custom_labels, fontsize=fosi - 3)

    # Plot shaded regions indicating better than a coin toss
    plt.axhspan(0, 0.5, xmin=0.25, xmax=0.5, color='green', alpha=0.2, lw=0)
    plt.axhspan(0.5, 1, xmin=0.75, xmax=1, color='green', alpha=0.2, lw=0)
    plt.axhspan(-0.5, 0, xmin=0.5, xmax=0.75, color='green', alpha=0.2, lw=0)
    plt.axhspan(-1, -0.5, xmin=0, xmax=0.25, color='green', alpha=0.2, lw=0)

    return fig


def lss_table_sat(model1, model2, model1_name='Model1', model2_name='Model2',
                  sats=['A', 'B', 'C']):
    """ Neat table including the Liemohn Skill Scores 1-4

    Parameters
    ----------
    model1: dataframe
        dataframe of 1st model data including skill and local times
        built by states_report_swarm
    model2 : dataframe
        dataframe of 2nd model data including skill and local times
        built by states_report_swarm
    model1_name : str kwarg
        string of name of model1
    model2_name : str kwarg
        string of name for model2
    sats : list of strings kwarg
        swarm satellites 'A', 'B', and 'C' as default
        can specify just 1 or 2
    Returns
    -------
    lss_df : dataframe
        dataframe in table format separated by satellite
        and Liemohn skill score
    """
    for i, s in enumerate(sats):
        lss1_m1, lss2_m1, lss3_m1, lss4_m1 = Liemohn_Skill_Scores(
            model1['skill'][model1['Sat'] == s])
        lss1_m2, lss2_m2, lss3_m2, lss4_m2 = Liemohn_Skill_Scores(
            model2['skill'][model2['Sat'] == s])
        if i == 0:
            lss_df = pd.DataFrame(
                [[lss1_m1, lss1_m2], [lss2_m1, lss2_m2],
                 [lss3_m1, lss3_m2], [lss4_m1, lss4_m2]],
                index=pd.MultiIndex.from_product(
                    [['Swarm ' + s], ['lss1', 'lss2', 'lss3', 'lss4']]),
                columns=[model1_name, model2_name])
        else:
            lss_df.loc[('Swarm ' + s,
                        'LSS1'), :] = np.array([lss1_m1, lss1_m2])
            lss_df.loc[('Swarm ' + s,
                        'LSS2'), :] = np.array([lss2_m1, lss2_m2])
            lss_df.loc[('Swarm ' + s,
                        'LSS3'), :] = np.array([lss3_m1, lss3_m2])
            lss_df.loc[('Swarm ' + s,
                        'LSS4'), :] = np.array([lss4_m1, lss4_m2])

    lss_df.style
    return lss_df


def style_lss_table(lss_df, sat_list=['A', 'B', 'C']):
    """ Style decision table using

    Parameters
    ----------
    lss_df : dataframe
        dataframe created by lss_table_sat
    sat_list: list of strings kwarg
        satellite list for lss_df
    Returns
    -------
    LSS table with dividers
    """
    s = lss_df.style.format()

    for sat in sat_list:
        l0 = 'Swarm ' + sat
        s.set_table_styles(
            {(l0, 'LSS4'):
             [{'selector': '', 'props':
               'border-bottom: 2px solid black;'}],
             (l0, 'LSS1'):
             [{'selector': '.level0', 'props':
               'border-bottom: 2px solid black'}]},
            overwrite=False, axis=1)
    return s
