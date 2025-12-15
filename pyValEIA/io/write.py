#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Functions to write standard output files."""

import os
from pathlib import Path


def build_daily_stats_filename(stime, model, obs, file_dir, **kwargs):
    """Build the filename and directory for daily EIA stat files.

    Parameters
    ----------
    stime : datetime
        day of desired file
    model : str
        Case-sensitive name of model requested (e.g., 'NIMO', 'PyIRI').
    obs : str
        Name of data set requested (e.g., 'SWARM', 'MADRIGAL')
    file_dir : str
        File directory
    kwargs : dict
        Optional kwargs by data type.  Includes 'mad_lon', which expects
        longitudes of either -90 deg E or 60 deg E for Madrigal data.

    Returns
    -------
    date_dir : str
        Directory path in which file should exist (e.g., 'file_dir/yyyy')
    fname : str
        Filename without directory

    """
    # Build the dataset name from the model and observation type
    dataset = "_".join([model, obs.upper()])

    if obs.upper() == 'MADRIGAL':
        if 'mad_lon' in kwargs.keys():
            end_str = "_{:.0f}glon_ascii.txt".format(kwargs['mad_lon'])
        else:
            end_str = "_{:.0f}glon_ascii.txt".format(-90.0)
    else:
        end_str = "ascii.txt"

    # Build the directory path
    date_dir = os.path.join(file_dir, stime.strftime('%Y'))

    # Build the filename
    fname = "{:s}{:s}".format("_".join([
        dataset, "EIA", "type", stime.strftime('%Y%m%d')]), end_str)

    return date_dir, fname


def write_daily_stats(stat_data, stime, model, obs, file_dir, **kwargs):
    """Write the daily statistics file for model-data comparisons.

    Parameters
    ----------
    stat_data : pd.DataFrame
        DataFrame that includes all EIA statistics for the obs type
    stime : datetime
        day of desired file
    model : str
        Case-sensitive name of model requested (e.g., 'NIMO', 'PyIRI').
    obs : str
        Name of data set requested (e.g., 'SWARM', 'MADRIGAL')
    file_dir : str
        File directory, if it does not exist it will use the current directory
    kwargs : dict
        Optional kwargs by data type.  Includes 'mad_lon', which expects
        longitudes of either -90 deg E or 60 deg E for Madrigal data.

    """
    # Test the output directory
    if not os.path.isdir(file_dir):
        file_dir = os.getcwd()

    # Build the output directory path and filename
    date_dir, fname = build_daily_stats_filename(stime, model, obs, file_dir,
                                                 **kwargs)

    # Ensure the directory exists
    Path(date_dir).mkdir(parents=True, exist_ok=True)
    save_file = os.path.join(date_dir, fname)

    # Create the custom header row with a hashtag
    header_line = '#{:s}\n'.format('\t'.join(stat_data.columns))

    # Write the header to the file
    with open(save_file, 'w') as fout:
        fout.write(header_line)

    # Append the DataFrame data without the header and index
    stat_data.to_csv(save_file, sep='\t', index=False,
                     na_rep='NaN', header=False, mode='a', encoding='ascii')
    return
