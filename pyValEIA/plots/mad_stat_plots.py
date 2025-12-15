#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Functions for visualizing Madrigal TEC EIA statistics."""

import numpy as np
import matplotlib.pyplot as plt
import warnings

from pyValEIA.stats import skill_score


def Mad_LSS_plot(model1, eia_type, date_range, model_name='Model',
                 PorC='PC', DayNight=True, LT_range='day', coin=True):
    """Plot LSS vs CSI or PC 4 panels (one for each LSS) for 1 model alone.

    Parameters
    ----------
    model1 : pd.DataFrame
        model DataFrame built by states_report_swarm
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
    LT_range : list-like or str
        Range of day night local time, or 'day' for 7 LT to 19 LT and 'night'
        for 19 LT to 7 LT (default='day')
    coin : bool kwarg
        If True, coin LSS will be plotted for comparison (default)
        if false, coin LSS will not be plotted
    Returns
    -------
    fig : fig handle
        4 panel figure (one for each LSS)

    Raises
    ------
    ValueError
        Unexpected input values

    Notes
    -----
    LSS is only useful in comparison to another model, therefore,
    coin set to True is highly recommended!

    """
    if LT_range == 'day':
        LT_range = [7, 19]
    elif LT_range == 'night':
        LT_range = [19, 7]
    elif len(LT_range) != 2:
        raise ValueError('unknown LT range: {:}'.format(LT_range))

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
    fig.suptitle((eia_type + ' ' + date_array[0].strftime('%Y/%m/%d') + '-'
                  + date_array[-1].strftime('%Y/%m/%d')), x=0.5, y=0.92,
                 fontsize=17)
    return fig
