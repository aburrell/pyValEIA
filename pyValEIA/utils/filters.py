#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Functions to filter data."""

import numpy as np


def simple_barrel_roll(xvar, yvar, barrel_radius, envelope=True,
                       envelope_lower=0.6, envelope_upper=0.2):
    """Roll barrel over data to detrended over large decreases in value.

    Parameters
    ----------
    xvar: array-like
        Independent data variable along which `yvar` will be smoothed.
    yvar: array-like
        Dependent data variable of the same size as `xvar`; needs to be scaled
        such that its magnitude is similar to `yvar`
    barrel_radius : double
        Radius of the 'barrel' rolling over the data in the same units as `xvar`
    envelope : bool
        If True, the barrel roll results will be used as constraints for the
        original `yvar` values (default=True)
    envelope_lower : float
        Fraction starting from zero to multiply the minimum by, used to
        calculate the lower limit of the barrel roll envelope; a larger value
        creates a larger lower envelope (default=0.6)
    envelope_upper : float
        Fraction starting from zero to multiply the maximum by, used to
        calculate the upper limit of the barrel roll envelope; a larger value
        creates a larger upper envelope (default=0.2)

    Returns
    -------
    yvar_det : array-like
        Detrended values for the dependent variable, `yvar`

    """
    # Initialize the x and y values for forward rolling
    strt_con_y = yvar[0]
    strt_con_x = xvar[0]

    # Initialize the list of contact points for forward rolling
    f_con_xs = []
    f_con_ys = []

    # Cycle through the data looking for contact points
    j = 0
    while j < len(yvar) - 1:
        # Forward Rolling only
        f_con_xs.append(strt_con_x)
        f_con_ys.append(strt_con_y)

        # Get the regions of interest (within barrel view). For forward rolling,
        # this is greater than the start time and less than the barrel diameter
        # (in the same units as `xvar`)
        x_roi = xvar[(xvar > strt_con_x) & (xvar <= strt_con_x
                                            + 2 * barrel_radius)]
        y_roi = yvar[(xvar > strt_con_x) & (xvar <= strt_con_x
                                            + 2 * barrel_radius)]

        # Calcualte angular distance delta for each delta = beta - theta for
        # each x-value region of interest
        deltas = []
        for i, xr_val in enumerate(x_roi):
            # Calculate the difference between each data value and the start
            del_x = xr_val - strt_con_x
            del_y = y_roi[i] - strt_con_y

            # Calculate the rolling angles
            theta = np.arctan(del_y / del_x)
            if 2 * barrel_radius >= np.sqrt((del_x) ** 2 + (del_y) ** 2):
                beta = np.arcsin(np.sqrt((del_x) ** 2 + (del_y) ** 2)
                                 / (2 * barrel_radius))
            else:
                beta = np.pi / 2.0
            delta = beta - theta
            deltas.append(delta * 180 / np.pi)

        # Move the barrel forward along the x-axis
        if len(x_roi) != 0:
            # Identify the new forward contacts
            strt_con_y = y_roi[deltas.index(min(deltas))]  # minimum delta
            strt_con_x = x_roi[deltas.index(min(deltas))]
            j = np.where(strt_con_x == xvar)[0]  # update j for while loop
        else:
            # Append last value if there is no region of interest
            strt_con_y = yvar[len(yvar) - 1]
            strt_con_x = xvar[len(yvar) - 1]
            j = len(yvar)  # update j for while loop

    # The contact points identify gaps or depletions in the data. Filter these
    # out using linear interpolation between the contact points
    int_y = np.interp(np.setdiff1d(xvar, f_con_xs), f_con_xs, f_con_ys)

    # Combine the contact points and the interpolated data
    x_combined = np.concatenate((f_con_xs, np.setdiff1d(xvar, f_con_xs)))
    y_combined = np.concatenate((f_con_ys, int_y))

    # Sort combined data by x values
    sorted_indices = np.argsort(x_combined)
    x_combined = x_combined[sorted_indices]
    y_combined = y_combined[sorted_indices]

    # If desired, replace data from inside the barrel envelope with original
    # y-values
    if envelope:
        # Set the upper and lower limits of the envelope
        brc_upper = y_combined + envelope_upper * max(y_combined)
        brc_lower = y_combined - envelope_lower * min(y_combined)

        # Initialize the output array
        yvar_det = np.full(shape=y_combined.shape, fill_value=np.nan)

        # Retrieve all the original values from between the upper and lower
        # barrel roll limits
        yvar_det[(yvar < brc_upper) & (yvar > brc_lower)] = yvar[
            ((yvar < brc_upper) & (yvar > brc_lower))]

        # Check to see if there are NaN values left in the output
        nan_mask = np.isnan(yvar_det)
        if sum(nan_mask):
            yvar_det[nan_mask] = np.interp(xvar[nan_mask], xvar[~nan_mask],
                                           yvar_det[~nan_mask])
    else:
        # If no envelope is desired, the output is the combined results
        yvar_det = np.array(y_combined)

    return yvar_det


def rolling_nanmeasure(arr, window, measure='mean'):
    """Calculate the rolling mean or median of an array with or without nans.

    Parameters
    ----------
    arr: array-like
        array of values to roll over
    window : int
        window size
    measure : str
        Method to apply to data; 'mean', 'median', 'average' (default='mean')

    Returns
    -------
    out : array-like
        rolling measured array of same length as original

    """
    # Initialize array of same length as input
    out = np.full_like(arr, np.nan, dtype=float)
    half_w = window // 2

    # Iterate through array
    for i in range(len(arr)):
        left = max(0, i - half_w)
        right = min(len(arr), i + half_w + 1)
        window_vals = arr[left:right]

        # Choose between mean/average and median
        if measure.lower() in ['mean', 'average']:
            if np.all(np.isnan(window_vals)):
                out[i] = np.nan
            else:
                out[i] = np.nanmean(window_vals)
        elif measure.lower() == 'median':
            if np.all(np.isnan(window_vals)):
                out[i] = np.nan
            else:
                out[i] = np.nanmedian(window_vals)
        else:
            raise ValueError('unknown method for smoothing: {:}'.format(
                measure))

    return out


def find_nan_ranges(arr):
    """Identify continuous ranges of NaN values in an array.

    Parameters
    ----------
    arr : array-like
        array with nans

    Returns
    -------
    nan_list : list-like
        List of (start_idx, end_idx) for each contiguous NaN section

    """
    # Get continuous ranges of nan values
    isnan = np.isnan(arr)
    edges = np.diff(isnan.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1

    if isnan[0]:
        starts = np.insert(starts, 0, 0)
    if isnan[-1]:
        ends = np.append(ends, len(arr))
    nan_list = list(zip(starts, ends))

    return nan_list


def find_all_gaps(inds):
    """Identify gaps in a list of indices.

    Parameters
    ----------
    inds : list-like
        Array of indices

    Returns
    -------
    gap_indices : list-like
        Indices of gap start and end

    Notes
    -----
    For example, in an array of `arr=[2,3,5,6,7,8]`, this function will return
    `gap_inds=[1]` to indicate where the gap starts.

    """
    gap_indices = []

    # Iterate through the array and find where the gaps start
    for i in range(len(inds) - 1):
        if inds[i + 1] != inds[i] + 1:
            gap_indices.append(i + 1)  # Append the index where the gap starts

    return gap_indices


def detect_outliers(arr):
    """Detect outliers in an array.

    Parameters
    ----------
    arr : array-like
        Array of numbers

    Returns
    -------
    outlier_indices : array-like
        array of indices where `arr` has outliers

    Notes
    -----
    Uses InterQuartile Range (IQR)
    IQR = q3 - q1
    outlier > q3 + 1.5 * IQR
    outlier < q1 - 1.5 * IQR

    """
    # Ensure input is array-like
    arr = np.asarray(arr)
    if arr.shape == ():
        arr = np.array([arr])

    # Get the quartiles and IQR
    q1 = np.percentile(arr[np.isfinite(arr)], 25)
    q3 = np.percentile(arr[np.isfinite(arr)], 75)
    iqr = q3 - q1

    # Get the upper and lower limits
    upper_lim = q3 + 1.5 * iqr
    lower_lim = q1 - 1.5 * iqr

    # Identify the desired indices
    outlier_indices = np.where((arr > upper_lim) | (arr < lower_lim))[0]

    return outlier_indices
