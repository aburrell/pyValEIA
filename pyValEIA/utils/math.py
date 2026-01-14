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
    number : float or array-like
        Number(s) for which the base-10 exponent will be calculated

    Returns
    -------
    exp_val : float or array
        Exponent of `number` as a whole value

    """
    array_num = np.asarray(number)
    zero_mask = array_num == 0

    if np.any(zero_mask):
        # This is the same result, but without the RuntimeWarning
        if array_num.shape == ():
            exp_val = -np.inf
        else:
            exp_val = np.full(shape=array_num.shape, fill_value=-np.inf)

            # Fill the non-zero values in an array
            if not np.all(zero_mask):
                exp_val[~zero_mask] = np.floor(np.log10(abs(array_num[
                    ~zero_mask])))
    else:
        # Do not need to consider zero values raising a RuntimeWarning
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
        Threshold for uniqueness with a maximum value of 1.0 (default=0.01)

    Returns
    -------
    uvals : array-like
        Unique values at the desired threshold level

    Raises
    ------
    ValueError
        If a threshold greater than 1 is requested

    """
    if thresh > 1.0:
        raise ValueError('Maximum threshold for uniqueness is 1.0')

    # Number of decimal places
    ndec = int(abs(np.log10(thresh)))

    # Threshold here is our base in rounding
    uvals = np.unique(np.round(xvals, ndec))

    return uvals


def set_dif_thresh(span, percent=0.05):
    """Set a difference threshold.

    Parameters
    ----------
    span: float or array-like
        Span of an array; e.g., max - min
    percent : float
        Percent as a decimal for difference  threshold from 0-1 (default=0.05)

    Returns
    -------
    float
        Percentage multiplied by the span

    Notes
    -----
    Set the threshold for what is different, input scale (if span) = 50,
    then our max tec/ne is 50 so set thresh to 5 for 10%
    can also use this for maximum difference between peak and trough,
    so can use smaller threshold

    """

    return percent * span
