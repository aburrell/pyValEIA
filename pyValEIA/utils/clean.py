#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Functions for cleaning data products."""

import numpy as np

from pyValEIA.utils import filters


def mad_tec_clean(mad_tec_meas, mad_std_meas, mad_mlat, mlat_val, max_nan=20):
    """Clean Madrigal TEC data.

    Parameters
    ----------
    mad_tec_meas : array-like
        averaged TEC over longitude and time
    mad_std_meas : array-like
        Standard deviation of `mad_tec_meas`
    mad_mlat : array-like
        magnetic laittude of `mad_tec_meas`
    mlat_val : int
        magnetic latitude cutoff
    max_nan : float or int
        Maximum acceptable percent nan values in a pass (default=20)

    """
    # minimum is 20 degree cutoff on either side
    # filter by by magnetic latitude (start with given mlat_val)
    mad_tec_lat = mad_tec_meas[abs(mad_mlat) < mlat_val]
    mad_std_lat = mad_std_meas[abs(mad_mlat) < mlat_val]

    if np.all(mad_tec_lat[np.isfinite(mad_tec_lat)] < 5):
        mad_tec_lat[:] = np.nan
        mad_std_lat[:] = np.nan

    nan_perc = (np.isnan(mad_tec_lat).mean() * 100)

    if nan_perc != 100:
        # Remove oultier tec values
        out_tec = filters.detect_outliers(mad_tec_lat)
        mad_tec_lat[out_tec] = np.nan
        mad_std_lat[out_tec] = np.nan

    # calculate nan percent
    nan_perc = np.isnan(mad_tec_lat).mean() * 100
    mlat_try = mlat_val

    # if nan_perc is greater than max_nan,
    # we want to try to get it below 20 until we hit max_nan degrees mag lat
    if (nan_perc > max_nan) & (nan_perc < 80):
        while (nan_perc > max_nan) & (mlat_try >= max_nan) & (nan_perc < 80):
            mlat_try = mlat_try - 1
            mad_tec_lat = mad_tec_meas[abs(mad_mlat) < mlat_try]
            mad_std_lat = mad_std_meas[abs(mad_mlat) < mlat_try]

            # remove oultier tec values
            out_tec = filters.detect_outliers(mad_tec_lat)
            mad_tec_lat[out_tec] = np.nan
            mad_std_lat[out_tec] = np.nan

            # calculate nan percent
            nan_perc = np.isnan(mad_tec_lat).mean() * 100

    # if all data is below 5, then remove completely
    if np.all(mad_tec_lat[np.isfinite(mad_tec_lat)] < 5):
        mad_tec_lat[:] = np.nan
        mad_std_lat[:] = np.nan

    # calculate nan percent one final time
    nan_perc = np.isnan(mad_tec_lat).mean() * 100

    return mad_tec_lat, mad_std_lat, nan_perc, mlat_try
