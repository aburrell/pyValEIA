#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Functions to plot Swarm skill score statistical outcomes."""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import pyValEIA.plots.utils as putils
from pyValEIA.stats import skill_score
from pyValEIA.stats import tables


def lss_plot_Swarm(model1, model2, eia_type, date_range, model1_name='Model1',
                   model2_name='Model2', PorC='PC', DayNight=True,
                   LT_range=None, coin=True, lssylim=None, lssxlim=None):
    """Plot LSS vs CSI or PC 4 panels (one for each LSS).

    Parameters
    ----------
    model1 : DataFrame
        first model DataFrame built by multiday_states_report
    model2 : DataFrame
        second model DataFrame built by multiday_states_report
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
    LT_range : list-like or NoneType
        Range of day night local time, or None for default of [7, 19]
        (default=None)
    coin : bool kwarg
        If True, coin LSS will be plotted for comparison (default)
        if false, coin LSS will not be plotted
     lssylim : list-like or NoneType
        y axis limit, defaults to [-1,1] if None is provided (default=None)
    lssxlim : list-like or NoneType
        y axis limit, defaults to [0,1] if None is provided (default=None)

    Returns
    -------
    fig : fig handle
        4 panel figure (one for each LSS)

    See Also
    --------
    io.load.multiday_states_report

    Notes
    -----
    LSS can range outside of +/-1

    """
    # Update to default for kwargs
    if LT_range is None:
        LT_range = [7, 19]

    if lssxlim is None:
        lssxlim = [0, 1]

    if lssylim is None:
        lssylim = [-1, 1]

    # Set date array for given time range
    date_array = date_range.to_pydatetime()

    # model 1 and model 2 will have same sats
    sats = np.unique(model1['Sat'])

    # let's make a plot of changing PC or CSI
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.25], hspace=0.1)

    # Initialize Axes
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    look = ['All']
    cols2 = ['purple']
    cols1 = ['darkorange']
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
        cols2 = ['plum', 'purple']
        cols1 = ['#FFD984', 'darkorange']
        colsc = ['gray', 'black']

    look = np.array(look)
    for lo in range(len(look)):
        col1 = cols1[lo]
        col2 = cols2[lo]
        colc = colsc[lo]
        for j, s in enumerate(sats):
            if DayNight:
                if look[lo] == 'Day':
                    model1_sat = model1_day[model1_day['Sat'] == s]
                    model2_sat = model2_day[model2_day['Sat'] == s]
                elif look[lo] == 'Night':
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
            for i, ax in enumerate([ax1, ax2, ax3, ax4]):
                # Plot Satellite names as text
                # only plot coin toss if specified as True
                if coin:
                    ax.text(exes[0], lss_coin[i], s, fontsize=12, color=colc,
                            label=None)

                # Model 1
                ax.scatter(exes[1], lss_mod1[i], marker=f'${s}$', color=col1,
                           s=80)
                ax.scatter(exes[1], lss_mod1[i], marker='o', edgecolors=col1,
                           facecolors='none', s=160, linewidth=2, zorder=0)
                # Model 2
                ax.scatter(exes[2], lss_mod2[i], marker=f'${s}$', color=col2,
                           s=100)

                # Set labels depending on plot number
                if (i == 0) | (i == 2):
                    ax.set_ylabel('Skill Score')
                if (i == 0) | (i == 1):
                    ax.xaxis.set_ticklabels([])

                if (i == 2) | (i == 3):
                    ax.set_xlabel(lab)

                if (i == 1) | (i == 3):
                    ax.yaxis.set_ticklabels([])

                ax.set_title('LSS' + str(i + 1))
                ax.grid(True)
                ax.set_ylim(lssylim)
                ax.set_xlim(lssxlim)

    leg_ax = fig.add_subplot(gs[0, 2])

    # Get day and night labels
    day_lab, night_lab = putils.daynight_label(model1, LT_range=LT_range)

    # set up legend params
    leg_labs = [model1_name, model2_name, 'Swarm A', 'Swarm B', 'Swarm C',
                day_lab, night_lab]
    leg_cols = ['orange', 'purple', 'k', 'k', 'k', 'lightgray', 'k']
    leg_modes = ['scatter', 'line', 'scatter', 'scatter', 'scatter',
                 'line', 'line']
    leg_styles = ['$O$', '-', '$A$', '$B$', '$C$', '-', '-']

    putils.make_legend(leg_ax, leg_labs, leg_cols, leg_styles, leg_modes,
                       frameon=False, loc='center')

    plt.suptitle(('Liemohn Skill Score ' + date_array[0].strftime('%b %Y')),
                 x=0.45, y=0.92)
    return fig


def one_model_lss_plot_Swarm(model1, eia_type, date_range, model_name='Model',
                             PorC='PC', DayNight=True, LT_range=None,
                             coin=True):
    """Plot LSS vs CSI or PC 4 panels (one for each LSS) for 1 model alone.

    Parameters
    ----------
    model1 : DataFrame
        model DataFrame built by multiday_states_report
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
    LT_range : list-like or NoneType
        Range of day night local time, or if None is supplied, [7, 19] is used
        to specify 07:00-19:00 LT for daytime and 19:00-07:00 LT for nighttime
        (default=None)
    coin : bool kwarg
        If True, coin LSS will be plotted for comparison (default)
        if false, coin LSS will not be plotted

    Returns
    -------
    fig : fig handle
        4 panel figure (one for each LSS)

    See Also
    --------
    io.load.multiday_states_report

    Notes
    -----
    LSS is only useful in comparison to another model, therefore,
    coin set to True is highly recommended!

    """
    # Update to default for kwargs
    if LT_range is None:
        LT_range = [7, 19]

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


def map_hist_panel(ax, model, bin_lons=37, DayNight=True, LT_range=None):
    """Plot histogram maps on a panel.

    Parameters
    ----------
    ax : plt axis
        matplotlib.plt axis
    model : DataFrame
        DataFrame of model data including skill and local times
        built by states_report_swarm
    bin_lons : int
        Number of bins between -180 and 180 deg geo lon (default=37)
    DayNight : bool
        True if panels should have separate markers for day and night or False
        for all to be plotted together (default=True)
    LT_range : list-like or NoneType
        Range of day night local time, or if None is supplied, [7, 19] is used
        to specify 07:00-19:00 LT for daytime and 19:00-07:00 LT for nighttime
        (default=None)

    Returns
    -------
    ax : plt axis
        original axis with data plotted
    hist_ax : plt axis
        twinx axis to ax with histogram plotted

    """
    # Update to default for kwargs
    if LT_range is None:
        LT_range = [7, 19]

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


def plot_hist_quad_maps(model_states, sat, eia_type, date_range, bin_lons=37,
                        model_name='Model', fosi=16, hist_ylim=None,
                        LT_range=None):
    """Plot histograms for each Hit, Miss, False Pos, and Cor Neg.

    Parameters
    ----------
    model_states : pd.DataFrame
        DataFrame of model data including skill and local times built by
        multiday_states_report
    sat : str
        swarm satellite 'A', 'B', or 'C'
    eia_type : str
        eia state e.g. EIA, Peak, etc. depending on what is considered a hit
    date_range : pd.DateRange
        range of dates for title purposes
    bin_lons : int
        Number of bins between -180 and 180 deg geo lon (default=37)
    model_name : str
        name of model for title purposes (default='Model')
    fosi : int
        font size for plot (default=16)
    hist_ylim : list-like or NoneType
        y range (counts) for hist plot, if None uses [0, 15] (default=None)
    LT_range : list-like or NoneType
        Range of day night local time, or if None is supplied, [7, 19] is used
        to specify 07:00-19:00 LT for daytime and 19:00-07:00 LT for nighttime
        (default=None)

    Returns
    -------
    fig : figure handle
        fig with 4 panels of hist maps

    See Also
    --------
    io.load.multiday_states_report

    """
    # Update to default for kwargs
    if LT_range is None:
        LT_range = [7, 19]

    if hist_ylim is None:
        hist_ylim = [0, 15]

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


def HMFC_percent_panel(model_states, df_table, fig, ax, eia_type, colors=None):
    """Plot percentages of H/(H+M), M/(H+M), F/(F+C), C/(C+F) as 4 quadrants.

    Parameters
    ----------
    model_states : pd.DataFrame
        DataFrame of model data including skill and local times built by
        multiday_states_report
    df_table : pd.DataFrame
        Decision table build by decision_table_sat
    fig : figure
        Figure for plotting on
    eia_type : str
        String designating which EIA type is being reported
    colors : list of strings or NoneType
        colors to be plotted for each satellite (default=None)

    Returns
    ------
    fig : figure
        the resulting figure

    See Also
    --------
    io.load.multiday_states_report
    stats.tables.decision_table_sat

    """
    if colors is None:
        colors = ['blue', 'red', 'purple']

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
        plt.scatter(yy, yy, marker=f'${s}$', color=col, s=90)
        plt.scatter(yn, -1 * yn, marker=f'${s}$', color=col, s=90)
        plt.scatter(-1 * nn, -1 * nn, marker=f'${s}$', color=col, s=90)
        plt.scatter(-1 * ny, ny, marker=f'${s}$', color=col, s=90)

    return fig


def HMFC_percent_figure(model1, model2, eia_type, model1_name='Model1',
                        model2_name='Model2', col1='orange', col2='purple',
                        fosi=16):
    """Plot full figure using HMFC_percent_panel.

    Parameters
    ----------
    model1 : pd.DataFrame
        First model DataFrame built by states_report_swarm
    model2 : pd.DataFrame
        Second model DataFrame built by states_report_swarm
    eia_type : str
        Desired eia type for fig title
    model1_name : str
        First model name for labelling purposes (default='Model1')
    model2_name : str
        Second model name for labelling purposes (default='Model2')
    col1 : str
        Plotting color for Model1 (default='orange')
    col2 : str
        plotting color for Model 2 (default='purple')
    fosi : int
        font size for plot

    Returns
    -------
    fig : figure
        Figure handle

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
    df_table1 = tables.decision_table_sat(model1, model_name=model1_name)
    HMFC_percent_panel(model1, df_table1, fig, ax, eia_type,
                       colors=[col1, col1, col1])

    # Model2
    df_table2 = tables.decision_table_sat(model2, model_name=model2_name)
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


def plot_2hist_quad_maps(model_states, model2_states, sat, eia_type, date_range,
                         bin_lons=37, model_name='Model', model2_name='Model2',
                         fosi=16, hist_ylim=None, LT_range=None):
    """Plot histogram maps for each score: Hit, Miss, False Pos, and Cor Neg.

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
    hist_ylim : list-like or NoneType
        y range (counts) for hist plot, or None for default of [0, 15]
        (default=None)
     LT_range : list-like or NoneType
        Range of day night local time, or None for default of [7, 19]
        (default=None)

    Returns
    -------
    fig : figure handle
        fig with 4 panels of hist maps

    """
    # Update defaults
    if LT_range is None:
        LT_range = [7, 19]

    if hist_ylim is None:
        hist_ylim = [0, 15]

    # Creating Figure with GridSpec
    scores = ["H", "M", "F", "C"]

    # Get a specific Satellite
    model_sat = model_states[model_states['Sat'] == sat]
    model2_sat = model2_states[model2_states['Sat'] == sat]

    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 0.25], wspace=0.3)
    plt.rcParams.update({'font.size': fosi})

    for s, score in enumerate(scores):

        # Get models by score (HMFC)
        model_score = model_sat[model_sat['skill'] == score]
        model2_score = model2_sat[model2_sat['skill'] == score]

        # Panel 1: World Map with Longitude Histogram
        if s == 0:
            ax0 = fig.add_subplot(gs[0, 0:2], projection=ccrs.PlateCarree())
        elif s == 1:
            ax0 = fig.add_subplot(gs[0, 2:], projection=ccrs.PlateCarree())
        elif s == 2:
            ax0 = fig.add_subplot(gs[1, 0:2], projection=ccrs.PlateCarree())
        elif s == 3:
            ax0 = fig.add_subplot(gs[1, 2:], projection=ccrs.PlateCarree())

        ax0, hist_ax = map_2hist_panel(ax0, model_score, model2_score,
                                       bin_lons=bin_lons,
                                       DayNight=True, LT_range=LT_range)

        # Add skill titles
        ax0.set_aspect('auto', adjustable='box')
        if score == 'H':
            ax0.text(-160, 95, 'HIT', fontweight='bold', color='black')
        elif score == 'M':
            ax0.text(-160, 95, 'MISS', fontweight='bold', color='black')
        elif score == 'C':
            ax0.text(-160, -105, 'CORRECT NEGATIVE', fontweight='bold',
                     color='black')
        elif score == 'F':
            ax0.text(-160, -105, 'FALSE POSITIVE', fontweight='bold',
                     color='black')

        # Move Latitude Axis (Secondary Y-Axis) to the right
        if (s == 1) | (s == 3):
            secax_y = ax0.twinx()
            secax_y.set_ylim(ax0.get_ylim())
            secax_y.set_ylabel("Latitude", color='gray',
                               rotation=270)
            secax_y.tick_params(axis='y', colors='gray')
            secax_y.yaxis.set_major_locator(mticker.MultipleLocator(base=30))
            secax_y.yaxis.set_label_position('right')
            secax_y.yaxis.tick_right()
            secax_y.spines['left'].set_visible(False)  # Hide right y-axis
            secax_y.spines['right'].set_visible(True)   # Show left y-axis
            putils.format_latitude_labels(secax_y, xy='y')
        else:
            ax0.set_yticklabels([])

        if (s == 0) | (s == 2):
            hist_ax.yaxis.set_label_position('left')
            hist_ax.yaxis.tick_left()
            hist_ax.set_ylabel('Counts', color='k')
            hist_ax.tick_params(axis='y', colors='k')  # Y-axis tick labels
            hist_ax.set_ylim(hist_ylim)
            hist_ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            hist_ax.yaxis.set_major_locator(mticker.MultipleLocator(base=2))
        else:  # remove y axis
            hist_ax.spines['right'].set_visible(False)
            hist_ax.set_yticks([])
            hist_ax.set_ylim(hist_ylim)

        hist_ax.set_xlim(-180, 180)

    # Secondary X-Axis for the Map
        if (s == 2) | (s == 3):
            secax_x = ax0.twiny()
            secax_x.set_xlim(ax0.get_xlim())
            secax_x.set_xlabel("Longitude")
            secax_x.xaxis.set_major_locator(mticker.MultipleLocator(base=60))
            putils.format_longitude_labels(secax_x, xy='x')
            secax_x.grid(True)
        else:
            secax_x = ax0.twiny()
            secax_x.set_xlim(ax0.get_xlim())
            secax_x.xaxis.set_major_locator(mticker.MultipleLocator(base=60))
            secax_x.set_xticklabels([])
            secax_x.grid(True)
            secax_x.tick_params(axis='both', which='major', length=0, width=0)
            hist_ax.set_xticklabels([])

    # Add legend
    # Get day and night labels
    day_lab, night_lab = putils.daynight_label(model_states, LT_range=LT_range)

    # legend axis
    leg_ax = fig.add_subplot(gs[2, 0])

    leg_cols = ['#FFD984']
    leg_labs = [f'{model_name} {day_lab}']
    leg_modes = ['shading']
    leg_styles = ['-']
    putils.make_legend(leg_ax, leg_labs, leg_cols, leg_styles, leg_modes,
                       frameon=False)

    # legend axis
    leg_ax = fig.add_subplot(gs[2, 1])

    leg_cols = ['darkorange']
    leg_labs = [f'{model_name} {night_lab}']
    leg_modes = ['shading']
    leg_styles = ['-']
    putils.make_legend(leg_ax, leg_labs, leg_cols, leg_styles, leg_modes,
                       frameon=False)

    # legend axis
    leg_ax = fig.add_subplot(gs[2, 2])
    leg_cols = ['#B65FCF']
    leg_labs = [f'{model2_name} {day_lab}']
    leg_modes = ['line']
    leg_styles = ['-']

    putils.make_legend(leg_ax, leg_labs, leg_cols, leg_styles, leg_modes,
                       frameon=False)

    # legend axis
    leg_ax = fig.add_subplot(gs[2, 3])
    leg_cols = ['purple']
    leg_labs = [f'{model2_name} {night_lab}']
    leg_modes = ['line']
    leg_styles = ['--']

    putils.make_legend(leg_ax, leg_labs, leg_cols, leg_styles, leg_modes,
                       frameon=False)

    # Add title
    date_str = date_range[0].strftime('%b %Y')
    fig.suptitle(f"{date_str} Swarm {sat}", x=0.5, y=0.93)  # fontweight="bold"

    return fig


def map_2hist_panel(ax, model, model2, bin_lons=37, DayNight=True,
                    LT_range=None):
    """Plot histogram maps on a panel for 2 models.

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
    LT_range : list-like or NoneType
        Range of day night local time, or None for default of [7, 19]
        (default=None)

    Returns
    -------
    ax : plt axis
        original axis with data plotted
    hist_ax : plt axis
        twinx axis to ax with histogram plotted

    """
    # Update defaults
    if LT_range is None:
        LT_range = [7, 19]

    # Initialize histogram bins
    hist_bins = np.linspace(-180, 180, bin_lons)

    # PLot Map
    ax.set_global()
    ax.add_feature(cfeature.LAND, edgecolor='gray', facecolor='none')
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray', facecolor='none')
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

        model2_day = model2[((model2['LT'] > LT_range[0])
                             & (model2['LT'] < LT_range[1]))]
        model2_night = model2[((model2['LT'] < LT_range[0])
                               | (model2['LT'] > LT_range[1]))]

        look = ['Day', 'Night']
        colsh = ['salmon', 'lightblue']
        colsh = ['#FFD984', 'darkorange']

        # Day
        lon_day = model_day['GLon']
        if len(model_day['LT']) > 0:
            day_str = (str(int(np.trunc(min(model_day['LT'])))) + ' to '
                       + str(int(np.round(max(model_day['LT'])))))
        else:
            day_str = ''
        hist_ax.hist(lon_day, bins=hist_bins, color=colsh[0],
                     alpha=0.8, label=look[0] + ' ' + day_str + ' LT')

        # Plot Model 2 hist as a line
        hist_ax.hist(model2_day['GLon'], bins=hist_bins, color='#B65FCF',
                     histtype='step', linewidth=2)

        # Night
        lon_night = model_night['GLon']
        if len(model_night['LT']) > 0:
            night_str = (str(int(np.trunc(min(model_night['LT']))))
                         + ' to ' + str(int(np.round(max(model_night['LT'])))))
        else:
            night_str = ''
        hist_ax.hist(lon_night, bins=hist_bins, color=colsh[1],
                     alpha=0.5, label=look[1] + ' ' + night_str + ' LT')

        # Plot Model 2 hist as a line
        hist_ax.hist(model2_night['GLon'], bins=hist_bins, color='purple',
                     histtype='step', linestyle='--', linewidth=2)

    else:  # Day night not specified
        lon = model['GLon']
        hist_ax.hist(lon, bins=hist_bins, color=colsh[0], alpha=0.3)

    return ax, hist_ax
