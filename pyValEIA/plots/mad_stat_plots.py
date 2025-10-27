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

from pyValEIA.eia.types import clean_type
from pyValEIA.io import load
from pyValEIA.stats import skill_score


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
        nimdf = load.load_daily_stats(sday, 'NIMO', 'MADRIGAL', daily_dir,
                                      mad_lon=mad_lon)

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
    NiMad['skill'] = skill_score.state_check(Mad[orientation].values,
                                             NiMad[orientation].values,
                                             state=typ)

    return NiMad, Mad


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
        PC_1, CSI_1 = skill_score.calc_pc_and_csi(
            model_use['skill'].values, coin=False)
        PC_coin, CSI_coin = skill_score.calc_pc_and_csi(
            model_use['skill'].values, coin=True)

        # Compute Skill Scores
        (LSS1_mod1, LSS2_mod1, LSS3_mod1,
         LSS4_mod1) = skill_score.liemohn_skill_score(
             model_use['skill'].values, coin=False)
        (LSS1_coin, LSS2_coin, LSS3_coin,
         LSS4_coin) = skill_score.liemohn_skill_score(
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
    plt.suptitle((eia_type + ' ' + date_array[0].strftime('%Y/%m/%d') + '-'
                  + date_array[-1].strftime('%Y/%m/%d')), x=0.5, y=0.92,
                 fontsize=17)
    return fig
