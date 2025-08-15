#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
# Madrigal Statistics
# Created 08/07/2025
# By alanahco@umich.edu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from open_daily_files import open_daily


def clean_type(arr):
    """ Simplifies EIA states into 4 base categories and 3 directions
    Parameters
    ----------
    arr : array-like
        array of eia states
    Returns
    -------
    base_types : array-like of strings
        states into 4 categories
        eia
        peak
        trough
        flat
    base_dirs : array-like of string
        state directions in 3 categories
        north
        south
        neither

    Notes
    -----
    flat
    ----
    flat_north
    flat_south
    flat

    trough
    -----
    trough

    peak
    -----
    peak
    peak_north
    peak_south

    eia
    ---
    eia_symmetric
    eia_north
    eia_south
    eia_saddle_peak
    eia_saddle_peak_north
    eia_saddle_peak_south
    eia_ghost_symmetric
    eia_ghost_north
    eia_ghost_south
    eia_ghost_peak_north
    eia_ghost_peak_south
    """
    base_types = []
    base_dirs = []

    for typ in arr:

        # Base Types
        if 'eia' in typ:
            base_types.append('eia')
        elif (typ == 'peak_north') | (typ == 'peak_south') | (typ == 'peak'):
            base_types.append('peak')
        elif typ == 'trough':
            base_types.append('trough')
        elif 'flat' in typ:
            base_types.append('flat')

        # Base Directions
        if 'north' in typ:
            base_dirs.append('north')
        elif 'south' in typ:
            base_dirs.append('south')
        else:
            base_dirs.append('neither')

    base_types = np.array(base_types)
    base_dirs = np.array(base_dirs)

    return base_types, base_dirs


def state_check(obs_type, mod_type, state='eia'):
    """calculates if something is H,M,C, or F using swarm as a
    reference and nimo as the check

    Parameters
    ----------
    obs_type : array-like
        observation base types
    mod_type : array-like
        model base types
    state : kwarg str
        base state we are comparing
        'eia', 'peak', 'flat', 'trough', 'north', 'south', 'neither'

    Returns
    -------
    event_states : array-like of strings
        categories of model to observations
        H-hit
        M-miss
        C-correct negative
        F-false alarm

    """
    event_states = []
    for i, otype in enumerate(obs_type):
        mtype = mod_type[i]
        if otype == state:

            # either a hit or a miss
            if mtype == state:
                event_states.append('H')
            else:
                event_states.append('M')
        elif otype != state:

            # either a hit or a miss
            if mtype != state:
                event_states.append('C')
            else:
                event_states.append('F')

    event_states = np.array(event_states)

    return event_states


def states_report_mad(date_range, daily_dir, typ='eia', mad_lon=-90):
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
    mad_lon : int
        starting longitude for Madrigal Daily Files

    Returns
    -------
    Ni : DataFrame
        NIMO states, directions, and types
        also includes longitude and local times
    Mad : DataFrame
        Madrigal States, direction, and types
        also includes longitude and local times
    """
    # Check to see what we are comparing (states or directions)
    if (typ == 'north') | (typ == 'south') | (typ == 'neither'):
        orientation = 'direction'
    else:
        orientation = 'state'
    # Set date array for given time range
    date_array = date_range.to_pydatetime()

    # initialize params for states, directions, and full types
    nim_sts = []  # NIMO
    nim_dirs = []
    nim_typ = []

    mad_sts = []  # Madrigal
    mad_dirs = []
    mad_typ = []

    lts = []
    lons = []

    for sday in date_array:

        # load files from date_array list
        nimdf = open_daily(sday, 'NIMO_MADRIGAL', daily_dir, mad_lon=mad_lon)

        # NIMO
        s, d = clean_type(nimdf['Nimo_Type'].values)
        nim_sts.extend(s)
        nim_dirs.extend(d)
        nim_typ.extend(nimdf['Nimo_Type'].values)

        # Madrigal
        s, d = clean_type(nimdf['Mad_EIA_Type'].values)
        mad_sts.extend(s)
        mad_dirs.extend(d)
        mad_typ.extend(nimdf['Mad_EIA_Type'].values)

        # basic params
        lts.extend(nimdf['LT_Hour'].values)
        lons.extend(nimdf['Nimo_GLon'].values)

    # Initialize DataFrames
    NiMad = pd.DataFrame()
    Mad = pd.DataFrame()

    # NIMO
    NiMad['state'] = np.array(nim_sts)
    NiMad['direction'] = np.array(nim_dirs)
    NiMad['type'] = np.array(nim_typ)
    NiMad['GLon'] = np.array(lons)
    NiMad['LT'] = np.array(lts)

    # Madrigal
    Mad['state'] = np.array(mad_sts)
    Mad['direction'] = np.array(mad_dirs)
    Mad['type'] = np.array(mad_typ)
    Mad['GLon'] = np.array(lons)
    Mad['LT'] = np.array(lts)

    # Compare PyIRI to Swarm and NIMO to Swarm
    NiMad['skill'] = state_check(Mad[orientation].values,
                                 NiMad[orientation].values, state=typ)

    return NiMad, Mad


def Liemohn_Skill_Scores(event_states, coin=False):
    """ Calcualted Skill scores from Liemohn 2024/2025

    Parameters
    ----------
    event_states : array-like
        array of event states 'H', 'M', 'F', and 'C'
    coin : bool
        if True, returns will be LSS for a coin
        if False, returns will be LSS of event_states
    Returns
    -------
    LSS1 : double
        Liemohn Skill Score 1
    LSS2 : double
        Liemohn Skill Score 2
    LSS3 : double
        Liemohn Skill Score 3
    LSS4 : double
        Liemohn Skill Score 4

    Notes: Paper Liemohn 2025 under review
    """
    H = sum(event_states == 'H')
    F = sum(event_states == 'F')
    M = sum(event_states == 'M')
    C = sum(event_states == 'C')

    if coin:  # LSS of a coin toss
        coin_HM = H+M
        coin_FC = F+C

        # HMFC of a coin is half of H+M for H and M and C+F for C and F
        H = coin_HM/2
        M = coin_HM/2
        F = coin_FC/2
        C = coin_FC/2

    # Liemohn Skill Score 1
    LSS1 = ((2 * H * C + M * C + H * F - H * M - M ** 2 - F ** 2 - F * C)
            / (2 * (H + M) * (F + C)))

    # Liemohn Skill Score 2 (LSS2t/LSS2b)
    LSS2t = (H * ((H + M) ** 2 + 2 * (H + M) * (F + C))
             - (H + M) ** 2 * (H + M + F))
    LSS2b = ((H + M + F) * ((H + M) ** 2 + 2 * (H + M) * (F + C)) -
             (H + M) ** 2 * (H + M + F))
    LSS2 = LSS2t / LSS2b

    # Liemohn Skill Score 3
    LSS3 = ((H + C) - (M + F)) / (H + M + F + C)

    # Liemohn Skill Score 4
    LSS4 = ((H * (2 * (H + M) + F + C) - (H + M) * (H + M + F))
            / ((H + M + F) * (2 * (H + M) + F + C) - (H + M) * (H + M + F)))

    return LSS1, LSS2, LSS3, LSS4


def PC_CSI(state, coin=False):
    """ Calculates percent correct and critical success index

    Parameters
    ----------
    event_states : array-like
        array of event states 'H', 'M', 'F', and 'C'
    coin : bool
        if True, returns will be LSS for a coin
        if False, returns will be LSS of event_states
    Returns
    -------
    PC : double
        percent correct as a decimal between 0 and 1
    CSI : double
        critical success index as a decimal between 0 and 1

    """
    H = sum(state == 'H')
    F = sum(state == 'F')
    M = sum(state == 'M')
    C = sum(state == 'C')

    if coin:  # Use a coin toss instead
        coin_HM = H + M
        coin_FC = F + C

        # HMFC of a coin
        H = coin_HM / 2
        M = coin_HM / 2
        F = coin_FC / 2
        C = coin_FC / 2

    PC = (H + C) / (H + M + F + C)
    CSI = H / (H + M + F)

    return PC, CSI


def Mad_LSS_plot(model1, eia_type, date_range, model_name='Model',
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

    # let's make a plot of changing PC or CSI
    fig, axs = plt.subplots(2, 2, figsize=(11, 11))
    look = ['All']
    cols1 = ['blue']
    colsc = ['black']

    # Start with whole model without separating into
    # Day and Night
    model_use = model1

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
        if DayNight:
            if lo == 0:
                model_use = model1_day
            else:
                model_use = model1_night

        col1 = cols1[lo]
        colc = colsc[lo]

        # Compute PC and CSI
        PC_1, CSI_1 = PC_CSI(model_use['skill'].values, coin=False)
        PC_coin, CSI_coin = PC_CSI(model_use['skill'].values, coin=True)

        # Compute Skill Scores
        LSS1_mod1, LSS2_mod1, LSS3_mod1, LSS4_mod1 = Liemohn_Skill_Scores(
            model_use['skill'].values, coin=False)
        LSS1_coin, LSS2_coin, LSS3_coin, LSS4_coin = Liemohn_Skill_Scores(
            model_use['skill'].values, coin=True)

        # MAkE LSS arrays
        LSS_mod1 = np.array([LSS1_mod1, LSS2_mod1, LSS3_mod1, LSS4_mod1])
        LSS_coin = np.array([LSS1_coin, LSS2_coin, LSS3_coin, LSS4_coin])

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

            # only plot coin toss if specified as True
            if coin:
                ax.text(exes[0], LSS_coin[i], 'O', fontsize=12, color=colc,
                        label=None)
            ax.text(exes[1], LSS_mod1[i], 'N', fontsize=12, color=col1,
                    label=None)

            # Set labels depending on plot number
            if (i == 0) | (i == 2):
                ax.set_ylabel('Skill Score')
            ax.set_xlabel(lab)
            ax.set_title('LSS' + str(i + 1))

            # Legend:
            # If you want to plot the legend for both day and night colors
            # in the first legend remove "& (lo == 0)"
            if (i == 0) & (lo == 0):
                ax.plot(-99, -99, color=col1, label=model_name)
                if coin:
                    ax.plot(-99, -99, color=colc, label='Coin Flip')
                ax.legend()
            if DayNight:

                # Getting LT hours for legend
                sm_arm = abs(model_use['LT'] - LT_range[0]).argmin()
                bg_arm = abs(model_use['LT'] - LT_range[1]).argmin()
                sm_lt = model_use['LT'].iloc[sm_arm]
                bg_lt = model_use['LT'].iloc[bg_arm]

                # Add Legend
                if (i == 1) & (lo == 0):
                    lab1 = (str(np.round(sm_lt, 1))
                            + '-' + str(np.round(bg_lt, 1))
                            + ' LT')
                    ax.plot(-99, -99, color=col1, label=lab1)
                if (i == 1) & (lo == 1):
                    lab1 = (str(np.round(bg_lt, 1))
                            + '-' + str(np.round(sm_lt, 1))
                            + ' LT')
                    ax.plot(-99, -99, color=col1, label=lab1)
                    ax.legend()

            ax.set_ylim([-1, 1])
            ax.set_xlim([0, 1])

    # Add super title
    plt.suptitle((eia_type + ' ' + date_array[0].strftime('%Y/%m/%d') + '-' +
                  date_array[-1].strftime('%Y/%m/%d')), x=0.5, y=0.92,
                 fontsize=17)
    return fig
