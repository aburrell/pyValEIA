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
    round_vals = np.floor(base * np.round(xvals.astype(np.float64) / base))

    return round_vals
