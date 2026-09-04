#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Functions for cleaning data products."""

import numpy as np

from pyValEIA.utils import filters


def mad_tec_clean(mad_tec_meas, mad_std_meas, mad_mlat, mlat_val=20.0,
                  max_nan=20, min_tec=5.0, mlat_res=1.0):
    """Clean Madrigal TEC data.

    Parameters
    ----------
    mad_tec_meas : array-like
        averaged TEC over longitude and time
    mad_std_meas : array-like
        Standard deviation of `mad_tec_meas`
    mad_mlat : array-like
        magnetic latitude of `mad_tec_meas`
    mlat_val : int or float
        magnetic co-latitude cut-off (default=20.0)
    max_nan : float or int
        Maximum acceptable percent nan values in a pass (default=20)
    min_tec : float or int
        Minimum TEC to consider in TECU (default=5.0)
    mlat_res : float or int
        Latitude resolution in degrees (default=1.0)

    Returns
    -------
    mad_tec_lat : array-like
        Array of clean TEC values over longitude and time
    mad_std_lat : array-like
        Array of clean TEC standard deviations over longitude and time
    nan_perc : float
        Percentage of TEC data that is NaN
    mlat_try : float
        Magnetic latitude cut-off used to create results

    """
    # Minimum is 20 degree cutoff on either side
    # filter by by magnetic latitude (start with given mlat_val)
    lat_mask = abs(mad_mlat) < mlat_val
    mad_tec_lat = mad_tec_meas[lat_mask]
    mad_std_lat = mad_std_meas[lat_mask]

    # If all data is below the minimum TEC, remove it
    if np.all(mad_tec_lat[np.isfinite(mad_tec_lat)] < min_tec):
        mad_tec_lat[:] = np.nan
        mad_std_lat[:] = np.nan

    nan_perc = (np.isnan(mad_tec_lat).mean() * 100)

    if nan_perc != 100:
        # Remove oultier tec values
        out_tec = filters.detect_outliers(mad_tec_lat)
        mad_tec_lat[out_tec] = np.nan
        mad_std_lat[out_tec] = np.nan

    # Calculate nan percent
    nan_perc = np.isnan(mad_tec_lat).mean() * 100
    mlat_try = mlat_val

    # If nan_perc is greater than max_nan and less than 80%, try to reduce the
    # magnetic latitude range until there is enough good data
    while (nan_perc > max_nan) & (mlat_try >= max_nan * mlat_res) & (
            nan_perc < 80):
        # Adjsut the latitude range
        mlat_try = mlat_try - mlat_res
        lat_mask = abs(mad_mlat) < mlat_try
        mad_tec_lat = mad_tec_meas[lat_mask]
        mad_std_lat = mad_std_meas[lat_mask]

        # Remove oultier tec values
        out_tec = filters.detect_outliers(mad_tec_lat)
        mad_tec_lat[out_tec] = np.nan
        mad_std_lat[out_tec] = np.nan

        # Calculate the NaN percentage
        nan_perc = np.isnan(mad_tec_lat).mean() * 100

    # If all data is below the TEC floor, then remove completely
    if np.all(mad_tec_lat[np.isfinite(mad_tec_lat)] < min_tec):
        mad_tec_lat[:] = np.nan
        mad_std_lat[:] = np.nan

    # Calculate nan percent one final time
    nan_perc = np.isnan(mad_tec_lat).mean() * 100

    return mad_tec_lat, mad_std_lat, nan_perc, mlat_try
