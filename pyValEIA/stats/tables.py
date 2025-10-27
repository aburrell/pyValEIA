#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Functions to create tables of statistics."""

import numpy as np
import pandas as pd

from pyValEIA.stats import skill_score


def decision_table_sat(states, sats=None, sat_key='Sat', eia_type='eia',
                       model_name='Model', const_name='Swarm'):
    """Decision table summing hit/miss/corr-neg/false-pos states by satellite.

    Parameters
    ----------
    states : pd.DataFrame
        DataFrame of model data including skill and local times built by
        multiday_states_report
    sats : list-like or NoneType
        List of satellites specified by `states[sat_key]` to include in table,
        if None all will be included (default=None)
    sat_key : str
        Key in `states` to access the satellite separator (default='Sat')
    eia_type : str
        EIA state, e.g. 'eia', 'peak', etc., that declairs what is considered a
        hit.
    model_name : str
        Model name for decision table label (default='Model')
    const_name : str
        Satellite constellation name (default='Swarm')

    Returns
    -------
    table_frame : pd.DataFrame
        DataFrame in table format separated by satellite and event state
        (state, non-state). Index using: table_frame.loc[(
            f'{const_name} {satellite}', eia_type), (model_name, eia_type)]

    """
    # If not provided, get the unique satellite IDs
    if sats is None:
        sats = np.unique(states['Sat'].values)

    # Cycle over each unique satellite
    for i, s in enumerate(sats):
        # Sum total HMFCs
        hit = sum(states['skill'][states['Sat'] == s] == 'H')
        falarm = sum(states['skill'][states['Sat'] == s] == 'F')
        miss = sum(states['skill'][states['Sat'] == s] == 'M')
        corneg = sum(states['skill'][states['Sat'] == s] == 'C')

        # Initialize or update the output table
        sat_name = ' '.join([const_name, s])
        if i == 0:
            table_frame = pd.DataFrame(
                [[hit, miss], [falarm, corneg]],
                index=pd.MultiIndex.from_product(
                    [[sat_name], [eia_type, 'Non-' + eia_type]]),
                columns=pd.MultiIndex.from_product(
                    [[model_name], [eia_type, 'Non-' + eia_type]]))
        else:
            table_frame.loc[(sat_name, eia_type), :] = np.array([hit, miss])
            table_frame.loc[(sat_name, 'Non-' + eia_type), :] = np.array([
                falarm, corneg])

    table_frame.style
    return table_frame


def style_df_table(df_table, eia_type, sat_names=None):
    """Style decision table.

    Parameters
    ----------
    df_table : pd.DataFrame
        DataFrame created by decision_table_sat
    eia_type : str
        string designating which eia type is being reported
    sat_names : list-like or NoneType
        List of satellite names in `df_table` or None to use Swarm defaults.

    Returns
    -------
    styled_frame : pd.DataFrame
        Styled DataFrame

    Raises
    ------
    ValueError
        For unknown constellation name

    """
    if sat_names is None:
        sat_names = ['Swarm A', 'Swarm B', 'Swarm C']

    # Initialize the output
    styled_frame = df_table.style.format('{:.0f}')

    # Adding Color
    styled_frame.set_table_styles([
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

    # Add borders
    for l0 in sat_names:
        styled_frame.set_table_styles(
            {(l0, 'Non-' + eia_type):
             [{'selector': '', 'props':
               'border-bottom: 2px solid black;'}],
             (l0, eia_type):
             [{'selector': '.level0', 'props':
               'border-bottom: 2px solid black'}]},
            overwrite=False, axis=1)

    # Assign the cell color
    styled_frame.set_td_classes(cell_color)

    return styled_frame


def lss_table_sat(model1, model2, model1_name='Model1', model2_name='Model2',
                  sats=None, sat_key='Sat', const_name='Swarm'):
    """Create table including the Liemohn Skill Scores 1-4.

    Parameters
    ----------
    model1 : pd.DataFrame
        DataFrame of 1st model data including skill and local times built by
        multiday_states_report
    model2 : pd.DataFrame
        DataFrame of 2nd model data including skill and local times built by
        multiday_states_report
    model1_name : str
        String of name of model1 (default='Model1')
    model2_name : str
        String of name for model2 (default='Model2')
    sats : list of strings kwarg
        swarm satellites 'A', 'B', and 'C' as default
        can specify just 1 or 2
    sats : list-like or NoneType
        List of satellites specified by `states[sat_key]` to include in table,
        if None all will be included (default=None)
    sat_key : str
        Key in `model1` and `model2` used to access the satellite separator
        (default='Sat')
    const_name : str
        Satellite constellation name (default='Swarm')

    Returns
    -------
    lss_df : pd.DataFrame
        DataFrame in table format separated by satellite
        and Liemohn skill score

    See Also
    --------
    io.load.multiday_states_report

    """
    # If not provided, get the unique satellite IDs from the first DataFrame
    if sats is None:
        sats = np.unique(model1['Sat'].values)

    # Cycle through all the desired satellites
    for i, s in enumerate(sats):
        # Calculate the Liemohn skill scores
        lss1_m1, lss2_m1, lss3_m1, lss4_m1 = skill_score.Liemohn_Skill_Scores(
            model1['skill'][model1['Sat'] == s])
        lss1_m2, lss2_m2, lss3_m2, lss4_m2 = skill_score.Liemohn_Skill_Scores(
            model2['skill'][model2['Sat'] == s])

        # Initalize or update the output table
        sat_name = ' '.join([const_name, s])
        if i == 0:
            lss_df = pd.DataFrame(
                [[lss1_m1, lss1_m2], [lss2_m1, lss2_m2],
                 [lss3_m1, lss3_m2], [lss4_m1, lss4_m2]],
                index=pd.MultiIndex.from_product(
                    [[sat_name], ['lss1', 'lss2', 'lss3', 'lss4']]),
                columns=[model1_name, model2_name])
        else:
            lss_df.loc[(sat_name, 'LSS1'), :] = np.array([lss1_m1, lss1_m2])
            lss_df.loc[(sat_name, 'LSS2'), :] = np.array([lss2_m1, lss2_m2])
            lss_df.loc[(sat_name, 'LSS3'), :] = np.array([lss3_m1, lss3_m2])
            lss_df.loc[(sat_name, 'LSS4'), :] = np.array([lss4_m1, lss4_m2])

    # Set the style and output
    lss_df.style
    return lss_df


def style_lss_table(lss_df, sat_names=None):
    """Style the LSS decision table.

    Parameters
    ----------
    lss_df : pd.DataFrame
        DataFrame created by lss_table_sat
    sat_names : list-like or NoneType
        List of satellite names in `lss_df` or None to use Swarm defaults.

    Returns
    -------
    styled_table : pd.DataFrame
        LSS table with dividers

    """
    if sat_names is None:
        sat_names = ['Swarm A', 'Swarm B', 'Swarm C']

    styled_table = lss_df.style.format()

    for l0 in sat_names:
        styled_table.set_table_styles(
            {(l0, 'LSS4'): [{'selector': '', 'props':
                             'border-bottom: 2px solid black;'}],
             (l0, 'LSS1'): [{'selector': '.level0', 'props':
                             'border-bottom: 2px solid black'}]},
            overwrite=False, axis=1)

    return styled_table
