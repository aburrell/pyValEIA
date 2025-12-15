#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Functions that support mathematical calculations."""

import numpy as np


def get_exponent(number):
    """Calculate the exponent of number using base 10.

    Parameters
    ----------
    number : double
        Number for which the base-10 exponent will be calculated

    Returns
    -------
    exp_val : float
        Exponent of `number` as a whole value

    """
    if number == 0:
        # This is the same result, but without the RuntimeWarning
        exp_val = -np.inf
    else:
        exp_val = np.floor(np.log10(abs(number)))

    return exp_val


def base_round(xvals, base=5):
    """Round values to the nearest base.

    Parameters
    ----------
    xvals : array-like
        Values to be rounded
    base : int
        Base to be rounded to (default=5)

    Returns
    -------
    round_vals : array-like
        Values rounded to nearest base

    """
    round_vals = np.floor(base * np.round(np.asarray(xvals).astype(np.float64)
                                          / base))

    return round_vals


def unique_threshold(xvals, thresh=0.01):
    """Round values to the desired threshold and return a unique array.

    Parameters
    ----------
    xvals : array-like
        Values to be rounded
    thresh : float
        Threshold for uniqueness (default=0.01)

    Returns
    -------
    uvals : array-like
        Unique values at the desired threshold level

    """
    # Number of decimal places
    ndec = int(abs(np.log10(thresh)))

    # Threshold here is our base in rounding
    uvals = np.unique(np.round(xvals, ndec))

    return uvals


def set_dif_thresh(lat_span, percent=0.05):
    """Set a difference threshold.

    Parameters
    ----------
    lat_span: double
        span of latitude array e.g. max(latitude) - min(latitude)
    percent : kwarg double
        Percent as a decimal for difference  threshold from 0-1 (default=0.05)

    Returns
    -------
    float
        Percentage times the span

    Notes
    -----
    Set the threshold for what is different, input scale (if lat_span) = 50,
    then our max tec/ne is 50 so set thresh to 5 for 10%
    can also use this for maximum difference between peak and trough,
    so can use smaller threshold

    """

    return percent * lat_span
