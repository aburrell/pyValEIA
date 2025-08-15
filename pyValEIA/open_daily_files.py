#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
# Created by Alanah Cardenas-O'Toole
# Summer 2025
# Latest update: 08/07/2025
# Email alanahco@umich.edu

import numpy as np
import pandas as pd


def open_daily(sday, dataset, fdir, mad_lon=-90):
    """Open the daily files created in
    NIMO_Swarm_Map_Plotting.py and described in README_daily_files.txt

    Parameters
    ----------
    sday : datetime
        day of desired file
    dataset : str
        type of file requested
        i.e. 'NIMO_MADRIGAL', 'NIMO_SWARM', and 'PyIRI'
    fdir : string
        file directory
    mad_lon : int
        longitude of madrigal file either -90 or 60
    Returns
    -------
    df : pandas dataframe
        Dataframe that includes all information from type file
    """
    # Year and Day strings
    y = sday.strftime('%Y')
    dy = sday.strftime('%Y%m%d')

    if dataset == 'NIMO_MADRIGAL':  # MADRIGAL/NIMO FILES
        lon_str = str(mad_lon)
        fname = f'{fdir}/{y}/{dataset}_EIA_type_{dy}_{lon_str}glon_ascii.txt'
    else:  # SWARM/NIMO and PyIRI files
        fname = f'{fdir}/{y}/{dataset}_EIA_type_{dy}ascii.txt'

    # Open File
    dat = np.genfromtxt(fname, delimiter=None, dtype=None, skip_header=0,
                        names=True, encoding='utf-8',
                        missing_values='NaN', filling_values=np.nan)

    # If an entire column is np.nan, then genfromtxt cannot interpret the
    # dtype, so it assigns it to True. The following code replaces True with
    # The original np.nan values
    for name in dat.dtype.names:
        col = dat[name]
        if (col.dtype == bool):
            nan_col = np.full(col.shape, np.nan, dtype='float64')
            dat = dat.astype([(n, 'float64',) if n == name else
                              (n, dat.dtype[n]) for n in dat.dtype.names])
            dat[name] = nan_col

    # Save as dataframe
    df = pd.DataFrame(dat, columns=dat.dtype.names)

    return df
