#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Functions to detect the EIA."""

import numpy as np
from scipy import stats

from pyValEIA.eia import types
from pyValEIA.utils import filters
from pyValEIA.utils import math


def eia_complete(lat, density, den_type, filt='', interpolate=1,
                 barrel_envelope=False, envelope_lower=0.6, envelope_upper=0.2,
                 barrel_radius=3, window_lat=3):
    """Detect and classify EIAs in plasma density data.

    Parameters
    ----------
    lat : array-like
        Magnetic latitude in degrees
    density : array-like
        Plasma density data, e.g., TEC, electron density, or ion density
    den_type : str
        String specifying 'tec' if density is TEC or 'ne' for ion or electron
        density
    filt : str
        Filter method(s) for density. An empty string means no filtering, and
        and underscore combines two methods in the order they are specified.
        Valid methods include 'barrel', 'median', 'mean', and 'average'
        (default='')
    interpolate : int
        Interpolate data to a higher resolution; the integer determines the
        number of data points in the interpolated output (e.g.,
        len(`density`) * `interpolate), so a value of one or less means there
        will not be any interpolation (default=1)
    barrel_envelope : bool kwarg
        if True, barrel roll will include points inside an
        envelope, if false (default) no envelope will be used
    envelope_lower : double kwarg
        lower limit of envelope
        default 0.6 (6%) of min value from contact points
    envelope_upper : double kwarg
        upper limit of envelope
        default 0.2 (2%) of max value from contact points
    barrel_radius : double kwarg
        latitudinal radius of barrel
    window_lat : double kwarg
        latitudinal width of moving window (default: 3 degrees maglat)

    Returns
    -------
    lat_use : array-like
        latitudes either original lat returned or
        interpolated lat depending on interpolate
    den_filt2 : array-like
        filtered density
    eia_type_slope : str
        EIA type see eia_slope_state for types
    z_lat : array-like
        zero latitudes found for checking purposes
    plats : arrray-like
        latitudes of peaks found from eia_slope_state
    p3lats : array-like
        Additional peak latitudes, if additional peak(s) are between the EIA
        double peaks and the type is not ghost; these are likely artifacts

    Raises
    ------
    ValueError
        If inputs do not allow for EIA detection.

    """
    # Test the input
    if den_type.lower() not in ['ne', 'tec']:
        raise ValueError("unknown plasma density type, use 'ne' or 'tec'")

    filt = filt.lower()
    if len(filt.split("_")) > 2:
        raise ValueError('data may be only filtered using two or less methods')

    # Calculate the latitude range
    lat_span = int(max(lat) - min(lat))

    if lat_span <= 3:
        raise ValueError('Insufficient data to calculate EIA')

    # sort from south hemisphere to north hemisphere
    sort_in = np.argsort(lat)
    sort_lat = lat[sort_in]
    sort_density = density[sort_in]

    # If desired, interpolate data to a higher resolution
    if interpolate > 1:
        x_new = np.linspace(min(sort_lat), max(sort_lat),
                            interpolate * len(sort_lat))
        sort_density = np.interp(x_new, sort_lat, sort_density)

    # Perform the first round of smoothing
    if filt[:6] == 'barrel':
        # Perform the barrel roll first. Start by scaling the density data
        # so that it is on the same scale as the latitude
        den_scale = lat_span / max(sort_density)
        den_barrel = sort_density * den_scale
        den_filt_barrel = filters.simple_barrel_roll(
            sort_lat, den_barrel, barrel_radius, envelope=barrel_envelope,
            envelope_lower=envelope_lower, envelope_upper=envelope_upper)

        # Scale the filtered output back up to the normal range
        den_filt1 = den_filt_barrel / den_scale

    elif filt == '':
        # No filtering is needed
        den_filt1 = sort_density
    else:
        # First perform either median or average filtering. Determine the
        # window size
        window = int(np.round(abs(window_lat / np.median(np.diff(sort_lat)))))

        measure = filt.split("_")[0]
        den_filt1 = filters.rolling_nanmeasure(sort_density, window, measure)

    # Perform the second round of filtering
    if '_barrel' in filt:
        # Filter using a barrel roll. First scale down the plasma density so
        # that is is scaled the same as latitude
        dens_scale = lat_span / max(den_filt1)
        den_barrel = den_filt1 * dens_scale
        den_filt_barrel = filters.simple_barrel_roll(
            sort_lat, den_barrel, barrel_radius, envelope=barrel_envelope,
            envelope_lower=envelope_lower, envelope_upper=envelope_upper)

        # Scale the barrel filtered data back to the plasma density range
        den_filt2 = den_filt_barrel / dens_scale

    elif "_" not in filt:
        # No additional filtering is requested
        den_filt2 = den_filt1

    else:
        # A moving filter window using a median or average is requested.
        # Determine the window size
        window = int(np.round(abs(window_lat / np.median(np.diff(sort_lat)))))
        measure = filt.split('_', "")[-1]
        den_filt2 = filters.rolling_nanmeasure(den_filt1, window, measure)

    # Calculate gradient
    grad_den = np.gradient(np.array(den_filt2), sort_lat)

    # Get latitudes of zero gradient points and process them
    zero_lat = evaluate_eia_gradient(sort_lat, grad_den)
    z_lat = process_zlats(zero_lat, sort_lat, den_filt2, lat_base=3)

    # Ion/electron density permits ghost check, but not for TEC
    ghosts = True if den_type.lower() == 'ne' else False

    # Evaluate EIA gradient using zero lats, filtered density and lat
    eia_type_slope, plats, p3lats = types.eia_slope_state(
        z_lat, sort_lat, den_filt2, ghost=ghosts)

    return sort_lat, den_filt2, eia_type_slope, z_lat, plats, p3lats


def process_zlats(z_lat, lat, den, lat_base=3):
    """Detect valid latitude locations.

    Parameters
    ----------
    z_lat : array-like
        Latitudes where the density gradient is zero
    lat : array-like
        All latitudes for the density measurments
    den : array-like
        Plasma denisty measurements
    lat_base : int
        Number of degrees latitude to round to when filtering (default=3)

    Returns
    ------
    z_lat : array-like
        Quality checked array of zero-density gradient latitudes

    """
    # Get nearest indices to z_lat
    ilocz = np.array([abs(z - lat).argmin() for z in z_lat])

    # ensure that there are indices
    if len(ilocz) != 0:
        # get density at z_lat
        denz = den[ilocz]

        # round z_lat by lat_base
        z_round = math.base_round(z_lat, base=lat_base)
        z_lat5 = []

        # choose z_lats associated with maximum density in lat_base window
        for u in range(len(set(z_round))):
            uu = list(set(z_round))[u]
            z_u = z_lat[z_round == uu]
            z_lat5.append(z_u[denz[z_round == uu].argmax()])

        z_lat = np.sort(z_lat5)

        # combine points between +/- 2.5 degrees using maximum density
        if np.any((z_lat <= 2.5) & (z_lat >= -2.5)):
            z_eqs = z_lat[(z_lat <= 2.5) & (z_lat >= -2.5)]

            # recalculate iloc for new z_lat array
            ilocz = []
            for z in z_eqs:
                ilocz.append(abs(z - lat).argmin())
            ilocz = np.array(ilocz)
            z_lat[((z_lat <= 2.5)
                   & (z_lat >= -2.5))] = z_eqs[den[ilocz].argmax()]

        # make sure z_lat is a unique array
        z_lat = math.unique_threshold(z_lat, 0.01)

    #  Apply quality control to the sign changes by removing adjacent indices
    iadjacent = np.where((z_lat[1:] - z_lat[:-1]) <= 0.5)[0]

    # Get TEC of z_lat
    z_den = []
    if len(z_lat) > 0:
        for z in z_lat:
            z_den.append(den[abs(z - lat).argmin()])
    z_den = np.array(z_den)
    ipops = []
    if len(iadjacent) > 0:
        for ia in iadjacent:
            icheck = np.flip(np.unique([[ia, ia + 1]]).flatten())

            # pop lower tec value only
            ipops.append(icheck[z_den[icheck].argmin()])

    if len(ipops) > 0:
        z_lat = list(z_lat)
        for p in ipops:
            if p < len(z_lat):
                z_lat.pop(p)

    z_lat = np.array(z_lat)

    # make sure z_lat is a unique array once more
    z_lat = math.unique_threshold(z_lat, 0.01)

    return z_lat


def evaluate_eia_gradient(lat, grad_dat, edge_lat=5):
    """Evaluate the density gradient for intersections revealing the EIA state.

    Parameters
    ----------
    lat : array-like
        Apex latitude in degrees
    grad_dat : array-like
        Plasma density gradient data in density units
    edge_lat : double
        Latitude from edge to exclude (default=5)

    Returns
    -------
    zero_lat : array-like
        Locations of EIA peaks and troughs in degrees latitude

    Raises
    ------
    ValueError
        If `lat` and `grad_dat` have different shapes

    """
    # Get the signs of the gradient values
    grad_sign = np.sign(grad_dat)
    lat = np.asarray(lat)

    # Test input
    if len(grad_dat) != len(lat):
        raise ValueError('len(lat) != len(grad_dat)')

    # Get the locations of sign changes
    ichange = np.where(grad_sign[1:] != grad_sign[:-1])[0]

    # Use a linear fit to estimate the latitude of the sign change
    zero_lat = list()
    for cind in ichange:
        find = cind + 1
        slope = (grad_dat[find] - grad_dat[cind]) / (lat[find] - lat[cind])
        intercept = grad_dat[cind] - slope * lat[cind]
        zero_lat.append(-intercept / slope)

    zero_lat = np.array(zero_lat)

    # Remove potential spurious peaks near the data edges
    zero_lat = zero_lat[((zero_lat < lat.max() - edge_lat)
                         & (zero_lat > lat.min() + edge_lat))]
    return zero_lat


def flat_rules(p1, tec, lat, zero_slope=0.5):
    """Determine if a peak is actually flat along with direciton.

    Parameters
    ----------
    p1 : array-like of length 1
        index of maxima
    lat : array-like
        latitude
    tec : array-like
        tec or ne
    zero_slope : float
        Threshold for the zero-slope value (default=0.5)

    Returns
    -------
    flat : int
        1 is flat_north, -1 is flat south, 0 is not flat
        2 if trough

    """
    # tec and lat of peak
    tec_max = tec[p1]
    lat_max = lat[p1]

    # initialize flat as 0
    flat = 0

    #  tec on north and south sides of peak
    south_tec_p1 = tec[np.where(lat < lat[p1])]
    north_tec_p1 = tec[np.where(lat > lat[p1])]

    # Calculate % of tec on each side of peak greater than the tec at peak
    south_perc1 = (len(south_tec_p1[south_tec_p1 > tec_max])
                   / len(south_tec_p1) * 100)
    north_perc1 = (len(north_tec_p1[north_tec_p1 > tec_max])
                   / len(north_tec_p1) * 100)

    # calculate peak span
    n, s = peak_span(p1, tec, lat)

    #  tec on each side of equator
    south_tec_all = tec[lat < 0]
    north_tec_all = tec[lat > 0]

    # minimum tec on each side of equator
    ntec_min = min(north_tec_all)
    stec_min = min(south_tec_all)

    # Calculate % of tec on south (north) < minimum tec on north (south) side
    south_perc2 = (len(south_tec_all[south_tec_all < ntec_min])
                   / len(south_tec_all) * 100)
    north_perc2 = (len(north_tec_all[north_tec_all < stec_min])
                   / len(north_tec_all) * 100)

    # if any of the following conidions are met,
    # flat is defined as 1 or -1 depending on north or south
    if (n == -99) | (s == -99):  # checks if peak is undefined (no span)
        if south_perc1 > north_perc1:
            flat = -1
        else:
            flat = 1
    # checks if 50% of either side is greater than the peak
    elif (south_perc1 > 50) ^ (north_perc1 > 50):
        if south_perc1 > north_perc1:
            flat = -1
        elif north_perc1 > south_perc1:
            flat = 1
        if (south_perc1 > 40) & (north_perc1 > 40):
            flat = 2  # trough not flat

    # checks if 80% of north or south side is under south or north side min
    elif (south_perc2 > 80) | (north_perc2 > 80):
        if south_perc2 < north_perc2:
            flat = -1
        elif north_perc2 < south_perc2:
            flat = 1

    # check if peak is within 5 degrees of edges
    elif (lat_max < min(lat) + 5):
        flat = -1
    elif (lat_max > max(lat) - 5):
        flat = 1

    # if flat is defined, do a second check
    if (n != -99) & (s != -99):  # peaks need to be defined on both sides

        if flat != 0:

            # fit a line to tec
            slope, intercept, rvalue, _, _ = stats.linregress(lat, tec)
            tec_filt = slope * lat + intercept

            # detrend tec
            tec_detrend = tec - tec_filt

            # Get slope between south point, peak, and north point
            loc_check = np.array([s, lat[p1], n])
            zlope, ztec, zlat = getzlopes(loc_check, lat, tec_detrend)
            # Filter slopes using zero slope definition
            if abs(zlope[0]) < zero_slope:
                zlope[0] = 0
            if abs(zlope[1]) < zero_slope:
                zlope[1] = 0

            # if the slope would be a primary (+-)
            # or secondary peak (0- or +0), then it is not flat
            if (np.sign(zlope[0]) == 1) & (np.sign(zlope[1]) == -1):
                flat = 0
            elif (np.sign(zlope[0]) == 0) & (np.sign(zlope[1]) == -1):
                flat = 0
            elif (np.sign(zlope[0]) == 1) & (np.sign(zlope[1]) == 0):
                flat = 0

    # flat = 1 if flat 0 if not, NS = -1 if south, 1 if north, 0 if flat
    # flat = 2 if trough
    return flat


def third_peak(z_lat, tec, lat, ghosts=False):
    """Identify a third peak, if present.

    Parameters
    ----------
    z_lat : int
        single index of first maxima
    lat : array-like
        latitude
    tec : array-like
        tec or ne
    ghosts : bool
        if False, don't look for ghosts, if True look for ghosts (default=False)

    Returns
    -------
    p_third : list-like
        List of latitudes if 3 peaks are found

    """
    # calculate slopes between z-locs
    zlope, ztec, zlat = getzlopes(z_lat, lat, tec)
    zlope = np.array(zlope)

    ilocz = []  # get locs of zero points
    for z in zlat:
        ilocz.append(abs(z - lat).argmin())

    # calcualte maximas
    zmaxima, zmaxi, zminima, zmini = find_maxima(zlope, ztec, ilocz)

    # if ghost and we only find 2 max, check for 1 handed ghost
    if (ghosts) & (len(zmaxima) == 2):
        max_lats = lat[zmaxi]
        max_i = abs(max_lats).argmax()  # find farthest value from equator
        min_i = abs(max_lats).argmin()  # find closest value to equator

        # NORTH, look in southern hemisphere for a thrid peak
        if max_lats[max_i] > 0:

            # get lats between equator point and -15
            z_new = z_lat[(z_lat < max_lats[min_i]) & (z_lat > -15)]
            if len(z_new) > 0:
                z_add = min(z_new)  # get farthest point from equator

                # add a min value between z_add and max_lats[min_i]
                tec_check = tec[(lat > z_add) & (lat < max_lats[min_i])]
                lat_check = lat[(lat > z_add) & (lat < max_lats[min_i])]

                if len(tec_check) > 0:  # check if there are any values between
                    z_min = lat_check[tec_check.argmin()]
                    z_lat_new = [-15, z_add, z_min]
                    zlope, ztec, zlat = getzlopes(z_lat_new, lat, tec)

                    # if it is a peak, then add it in
                    if (zlope[0] > 0) & (zlope[1] < 0):
                        zi = abs(lat - z_add).argmin()
                        zmaxi = np.insert(zmaxi, 0, zi)

        elif max_lats[max_i] < 0:  # SOUTH, look in northern hemisphere

            # get lats between equator point and 15
            z_new = z_lat[(z_lat > max_lats[min_i]) & (z_lat < 15)]
            if len(z_new) > 0:
                z_add = max(z_new)  # get farthest point from equator

                # add a min value between z_add and max_lats[min_i]
                tec_check = tec[(lat < z_add) & (lat > max_lats[min_i])]
                lat_check = lat[(lat < z_add) & (lat > max_lats[min_i])]

                if len(tec_check) > 0:
                    z_min = lat_check[tec_check.argmin()]
                    z_lat_new = [z_min, z_add, 15]
                    zlope, ztec, zlat = getzlopes(z_lat_new, lat, tec)

                    # if it is a peak, then add it in
                    if (zlope[0] > 0) & (zlope[1] < 0):
                        zi = abs(lat - z_add).argmin()
                        zmaxi = np.insert(zmaxi, 0, zi)
    if ghosts:
        if len(zmaxi) > 1:  # for ghosts, report maxima if there is more than 1
            p_third = lat[zmaxi]
        else:
            p_third = []
    else:
        if len(zmaxi) == 3:  # for reg third peak check, only report if 3 peaks
            p_third = lat[zmaxi]
        else:
            p_third = []
    return p_third


def peak_span(pm, tec, lat, trough_tec=-99, trough_lat=-99, div=0.5):
    """Calculate the latitudinal span of the peak.

    Parameters
    ----------
    pm : int
        peak index
    tec: array-like
        tec or ne
    lat: array-like
        latitude
    trough_tec : int or float
        TEC at trough, minimum TEC for double or triple peaks, or unspecified
        if set to -99 (default=-99)
    trough_lat : int or float
        Latitude of trough if `trough_tec` is also supplied (default=-99)
    div : float
        Decimal between 0 and 1 indicating desired peak width location; e.g.,
        0.5 indicates the half-width (default=0.5)

    Returns
    -------
    north_point : float
        northern latitude of peak width
    south_point : float
        southern latitude of peak width

    """
    # make sure that pm is an integer
    check_int = isinstance(pm, np.int64)
    if not check_int:
        if not isinstance(pm, int):
            pm_new = np.array(pm)
            pm = pm_new[0]

    # Peak lat and peak tec
    p_tec = tec[pm]

    # Lats and tec north and south of peak
    south_tec = tec[np.where(lat < lat[pm])]
    north_tec = tec[np.where(lat > lat[pm])]

    div = 1 / div  # divide height by 1/div

    # Establish lists for span indices
    ngap = []
    sgap = []

    for i in range(1, 31):  # Segment the tec
        if trough_tec == -99:  # from peak down
            t_base = p_tec * (32 - i) / 32

        else:  # from peak to trough tec
            t_base = (p_tec - trough_tec) * (32 - i) / 32 + trough_tec

        if np.any(north_tec < t_base):  # check for north tec below than t_base
            nex = 0
            j = 1
            while nex == 0:
                if pm + j < len(tec):  # start at center point
                    tec_new = tec[pm + j]
                    if tec_new > t_base:
                        j += 1
                    else:
                        ngap.append(pm + j - 1)
                        nex = 1
                else:
                    ngap.append(-99)
                    nex = 1
        else:
            ngap.append(-99)
        if np.any(south_tec < t_base):  # check for south tec below than t_base
            nex = 0
            j = 1
            while nex == 0:
                if pm - j >= 0:
                    tec_new = tec[pm - j]
                    if tec_new > t_base:
                        j += 1
                    else:
                        sgap.append(pm - j + 1)
                        nex = 1
                else:
                    sgap.append(-99)
                    nex = 1
        else:
            sgap.append(-99)

    # Save original array
    ngap_og = np.array(ngap)
    sgap_og = np.array(sgap)

    # Mask array by removing indices less than 0
    ngap = np.array(ngap)
    sgap = np.array(sgap)
    mask = (ngap < 0) | (sgap < 0)
    ngap = ngap[~mask]
    sgap = sgap[~mask]

    north_mask = (ngap_og < 0)
    south_mask = (sgap_og < 0)
    ngap_og = ngap_og[~north_mask]
    sgap_og = sgap_og[~south_mask]

    # check if lens of ngap/sgap are greater than 0 after masking
    if (len(ngap) > 0):
        north_ind = ngap[int(len(ngap) / div)]
        south_ind = sgap[int(len(sgap) / div)]

        north_point = lat[north_ind]
        south_point = lat[south_ind]

    else:
        # if original of both have some defined values
        if (len(ngap_og) > 0) & (len(sgap_og) > 0):
            north_point = lat[ngap_og[int(len(ngap_og) / div)]]
            south_point = lat[sgap_og[int(len(sgap_og) / div)]]

        # if only north edge is defined
        elif (len(ngap_og) > 0) & (len(sgap_og) == 0):
            north_point = lat[ngap_og[int(len(ngap_og) / div)]]
            south_point = -99

        # if south edge is defined
        elif (len(ngap_og) == 0) & (len(sgap_og) > 0):
            north_point = -99
            south_point = lat[sgap_og[int(len(sgap_og) / div)]]

        else:  # if neither edge can be defined, report -99
            north_point = -99
            south_point = -99

    return north_point, south_point


def toomanymax(z_lat, lat, tec, max_lat=None):
    """Reduce the number of peaks.

    Parameters
    ----------
    z_lat : array-like
        array of latitudes at zero gradient points
    lat : array-like
        array of latitudes in degrees
    tec : array-like
        Totel electron content or plasma density
    max_lat : array-like or NoneType
        if a peak is already found, it can be input to guarantee it is in the
        array new array (default=None)

    Returns
    -------
    z_lat : array-like
        a new array of latitudes zero points
    z_lat_new : list-like
        A list that will contain a maximum of 5 values: south edge,
        closest south, equator max, closest north, and north edge.

    """
    # Process the inputs
    if max_lat is None:
        max_lat = [-99]

    # Initalize the indices of z_lat
    ilocz = [abs(z - lat).argmin() for z in z_lat]

    # Corresponding TEC
    tecz = tec[ilocz[1:-1]]
    latz = lat[ilocz[1:-1]]

    # TEC from z_lats north (> 3 and 20 mlat), south (<-3 and -20 mlat),
    # and equator (between 3 and -3 mlat)
    tecz_south = tecz[(latz < -3) & (latz > -20)]  # 5 vs 3
    latz_south = latz[(latz < -3) & (latz > -20)]
    tecz_north = tecz[(latz > 3) & (latz < 20)]
    latz_north = latz[(latz > 3) & (latz < 20)]
    tecz_eq = tecz[(latz >= -3) & (latz <= 3)]
    latz_eq = latz[(latz >= -3) & (latz <= 3)]

    # Set up new list
    z_lat_new = []
    z_lat_new.append(z_lat[0])

    # If there are south tec, get largest value
    if len(tecz_south) > 0:
        # If max_lat is not provided or it is in the north
        if (max_lat[0] == -99) | (np.sign(max_lat[0]) == 1):
            # This is the value closest to the equator
            z_lat_new.append(latz_south[-1])
        else:
            # If a max_lat is provided and in south
            z_lat_new.append(max_lat[0])

    if len(tecz_eq) > 0:  # look for max tec value in equatorial region
        tez = tecz_eq.argmax()
        z_lat_new.append(latz_eq[tez])

    # Look for max tec value in north
    if len(tecz_north) > 0:

        # if max_lat is not provided or it is in the south
        if (max_lat[0] == -99) | (np.sign(max_lat[0]) == -1):
            # This is the value closest to the equator
            z_lat_new.append(latz_north[0])
        else:
            # If max_lat is in provided and in north, assign a new max
            z_lat_new.append(max_lat[0])

    z_lat_new.append(z_lat[-1])

    # Make sure array is unique
    z_lat_new = math.unique_threshold(z_lat_new, 0.01)

    return np.array(z_lat_new)


def getzlopes(z_lat_ends, lat, tec):
    """Calculate slopes between zero points.

    Parameters
    ----------
    z_lat_ends : array-like
        gradient zero latitudes including end points
    lat : array-like
        latitude
    tec : array-like
        tec

    Returns
    -------
    zlope : list-like
        slope between zero points length is lengeth of z_lat_ends-1
    ztec : list-like
        closest tec of z_lat_ends
    zlat : list-like
        closest latitude of z_lat_ends

    Notes
    -----
    This function returns the slopes, latitudes, and TEC or density nearest to
    the z points.

    """
    # Initialize the output
    zlope = []
    ztec = []
    zlat = []

    # Iterate through zero lats
    for zl in range(len(z_lat_ends) - 1):
        ilat1 = abs(z_lat_ends[zl] - lat).argmin()  # find index of nearest lat
        lat1 = lat[ilat1]  # get the latitude and tec of the first zero lat
        tec1 = tec[ilat1]
        ilat2 = abs

        # Find index of nearest next latitude
        ilat2 = abs(z_lat_ends[zl + 1] - lat).argmin()
        lat2 = lat[ilat2]
        tec2 = tec[ilat2]

        # Either set the slope to zero or ensure that the latitudes differ
        if abs(lat2 - lat1) > 1.0e-4:
            # The latitudes are not the same, calculate the slope
            slope = (tec2 - tec1) / (lat2 - lat1)
        else:
            # The latitudes are the same, prevent an infinite slope calculation
            # by setting the value to zero
            slope = 0

        zlope.append(slope)
        ztec.append(tec1)
        zlat.append(lat1)

        #  for the last value append the tec and lat of last point
        if zl == len(z_lat_ends) - 2:
            ztec.append(tec2)
            zlat.append(lat2)

    return zlope, ztec, zlat


def find_maxima(zlope, ztec, ilocz):
    """Find the local maxima based on the slopes.

    Parameters
    ----------
    zlope : array-like
        slopes outputted from getzlopes
    ztec : array-like
        tec of zero locations
    ilocz: array-like
        indices of zero locations

    Returns
    -------
    zmaxima : list-like
        Maximum TEC
    zmaxi: list-like
        Indices of maximum TEC
    zminima : list-like
        Minimum TEC
    zmini : list-like
        Indices of minimum TEC

    """
    # Initalize the output
    zmaxima = []
    zmaxi = []
    zminima = []
    zmini = []

    # Cycle through the zero-slopes
    for s in range(len(zlope) - 1):
        # Positive to negative slope = local maximum
        if (zlope[s] > 0) and (zlope[s + 1] < 0):
            # Exclude ends from being counted as max or min
            # len(ztec) is greater than len(zlope) by 1
            zmaxima.append(ztec[s + 1])
            zmaxi.append(ilocz[s + 1])

        # Negative slope to positive slope = local minimum
        elif (zlope[s] < 0) and (zlope[s + 1] > 0):
            zminima.append(ztec[s + 1])
            zmini.append(ilocz[s + 1])

    return zmaxima, zmaxi, zminima, zmini


def find_second_maxima(zlope, zdens, ilocz):
    """Find the secondary maxima.

    Parameters
    ----------
    zlope : array-like
        Slopes outputted from `getzlopes`
    zdens : array-like
        tec of zero locations
    ilocz: array-like
        indices of zero locations

    Returns
    -------
    sec_max : array-like
        secondary maxima tec
    sec_maxi : array-like
        indices of secondary maxima

    """
    # Initalize output
    sec_max = []
    sec_maxi = []

    # Cycle through the zero-slopes, comparing adjacent values
    for s in range(len(zlope) - 1):
        # Positive to 0 slope = secondary maximum
        if (zlope[s] > 0) and (zlope[s + 1] == 0):
            # Exclude ends from being counted as max or min len(zdens)
            # is greater than len(zlope) by 1
            sec_max.append(zdens[s + 1])
            sec_maxi.append(ilocz[s + 1])

        # Zero to negative slope = secondary maximum
        elif (zlope[s] == 0) and (zlope[s + 1] < 0):
            sec_max.append(zdens[s + 1])
            sec_maxi.append(ilocz[s + 1])

    return sec_max, sec_maxi


def zero_max(lat, dens, zlats, maxes=None):
    """Identify potential peaks using the maxima.

    Parameters
    ----------
    lat : array-like
        Magnetic latitudes in degrees
    dens : array-like
        Plasma density as TEC, electron density, or ion density
    zlats : array-like
        Latitudes of potential peaks
    maxes : list-like or NoneType
        Indices of identified peaks or None (default=None)

    Returns
    -------
    p1 : int or NoneType
        Index of the first peak, or None if not found
    p2 : int or NoneType
        Index of the second peak, or None if not found

    """
    # Get indices of the zero points
    ilocz = np.array([abs(zlat - lat).argmin() for zlat in zlats])

    # Get the location and density at the potential peaks
    lat_all = lat[ilocz]
    dens_all = dens[ilocz]

    # Separate by location: North, South, and Equatorial
    densz = dens_all[1:-1]
    latz = lat_all[1:-1]
    densz_south = densz[latz < 0]
    latz_south = latz[latz < 0]
    densz_north = densz[latz > 0]
    latz_north = latz[latz > 0]

    # Initalize the first and second peak indicators
    p1 = None
    p2 = None

    # Search through each of the potential peak locations
    if (len(densz_south) > 0):
        # If second peak found, double_peak_rules
        ts = densz_south.argmax()
        ps = abs(lat_all - latz_south[ts]).argmin()

        # south check
        dens_b4 = dens_all[ps - 1]
        dens_af = dens_all[ps + 1]
        dens_at = dens_all[ps]

        if (dens_at > dens_b4) & (dens_at > dens_af):
            # Ff it is a peak, set the first index
            p1 = abs(lat - latz_south[ts]).argmin()

        else:
            # Check for a centeral peak
            lat_eq = lat_all[(lat_all > -1) & (lat_all < 1)]
            dens_eq = dens_all[(lat_all > -1) & (lat_all < 1)]
            if len(dens_eq) != 0:
                te = dens_eq.argmax()
                pe = abs(lat_all - lat_eq[te]).argmin()

                # Check for a southern peak
                dens_b4e = dens_all[pe - 1]
                dens_afe = dens_all[pe + 1]
                dens_ate = dens_all[pe]
                if (dens_ate > dens_b4e) & (dens_ate > dens_afe):
                    # If it's a peak, set the first index
                    p1 = abs(lat - lat_eq[te]).argmin()

        # If p1 is still None after a south and center check,
        # check for secondary maximum
        if p1 is None:
            if (dens_at > dens_b4) | (dens_at > dens_af):
                p1 = abs(lat - latz_south[ts]).argmin()

    if len(densz_north) > 0:
        tn = densz_north.argmax()
        pn = abs(lat_all - latz_north[tn]).argmin()

        # Check for a northern peak
        dens_b4 = dens_all[pn - 1]
        dens_af = dens_all[pn + 1]
        dens_at = dens_all[pn]
        if (dens_at > dens_b4) & (dens_at > dens_af):
            # If it is a peak, set the second index
            p2 = abs(lat - latz_north[tn]).argmin()
        else:
            # Check for center peak instead
            lat_eq = lat_all[(lat_all > -1) & (lat_all < 1)]
            dens_eq = dens_all[(lat_all > -1) & (lat_all < 1)]
            if len(dens_eq) != 0:
                te = dens_eq.argmax()
                pe = abs(lat_all - lat_eq[te]).argmin()

                # Check for a southern peak
                dens_b4e = dens_all[pe - 1]
                dens_afe = dens_all[pe + 1]
                dens_ate = dens_all[pe]
                if (dens_ate > dens_b4e) & (dens_ate > dens_afe):
                    # If it's a peak, set the second index
                    p2 = abs(lat - lat_eq[te]).argmin()

        # If p2 is still None after a south and center check,
        # check for secondary maximum
        if p2 is None:
            if (dens_at > dens_b4) | (dens_at > dens_af):
                # If it is a peak, set the second index
                p2 = abs(lat - latz_north[tn]).argmin()

    if len(maxes) > 0:
        # If one peak is given, replace either p1 (souther) or p2 (northern)
        if lat[maxes[0]] > 0:
            p2 = maxes[0][0]
        elif lat[maxes[0]] < 0:
            p1 = maxes[0][0]
    else:
        # No current maxes, use a sinlge max
        if (p1 < 0) & (p2 < 0):
            t_last = densz.argmax()
            p1 = abs(lat - latz[t_last]).argmin()

    return p1, p2
