#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
# EIA slope_detection Clean

# EIA state detection using slopes between zero points
# state definitions taken from Angeline code
import numpy as np
from scipy import stats


def eia_complete(lat, density, den_str, filt='', interpolate=1,
                 barrel_envelope=False, envelope_lower=0.6, envelope_upper=0.2,
                 barrel_radius=3, window_lat=3):
    """ Conduct full analysis of eia
    Parameters
    ----------
    lat : array-like
        magnetic latitude for eia detection
    density : array-like
        ne or tec
    den_str : string
        string specifying 'tec' or 'ne'
    filt : str kwarg kwarg
        filter for denisty default empty string/ no filt
    interpolate : int kwarg
        int that determines the number of data points in interpolation
        new length will be len(density)xinterpolate
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
    p3 : array-like
        latitude of third peak if one is found and
        not ghost from eia_slope_state

    Notes
    -----
    filt options (order matters):
        'barrel'
        'average'
        'median'
        'barrel_average'
        'barrel_median'
        'average_barrel'
        'median_barrel'
    """

    lat_span = int(max(lat) - min(lat))  # calculate lat_span

    # sort from south hemisphere to north hemisphere
    sort_in = np.argsort(lat)
    lat = lat[sort_in]
    density = density[sort_in]

    if interpolate > 1:  # Interpolate if necessary
        lat_len = len(lat)
        x_new = np.linspace(min(lat), max(lat), interpolate * lat_len)
        y_new = np.interp(x_new, lat, density)
        lat_use = x_new
        den_use = np.array(y_new)
    else:
        den_use = density
        lat_use = lat

    # first round of smoothing
    # if barrel rolling first
    if ('barrel_' in filt) | (filt == 'barrel'):

        # scale down for barrel so x and y are on same scale
        den_barrel = den_use / max(den_use) * lat_span
        den_filt_barrel = simple_barrel_roll(lat_use, den_barrel,
                                             barrel_radius,
                                             envelope=barrel_envelope,
                                             envelope_lower=envelope_lower,
                                             envelope_upper=envelope_upper)

        #  scale back up to normal
        den_filt1 = den_filt_barrel * max(den_use) / lat_span

    # if moving measure first
    elif (('average_' in filt) | (filt == 'average')
          | ('median_' in filt) | (filt == 'median')):

        # roughly window_lat degree smoothing window by converting to indices
        window = int(np.round(abs(window_lat / np.median(np.diff(lat_use)))))

        substring_to_remove = '_barrel'
        measure = filt.replace(substring_to_remove, "")
        filt_meas = rolling_nanmeasure(den_use, window, measure)
        den_filt1 = np.array(filt_meas)

    # If No filter
    else:
        den_filt1 = den_use

    # second round of smoothing
    # if barrel rolling second
    if ('_barrel' in filt):

        # scale down for barrel so x and y are on same scale
        den_barrel = den_filt1 / max(den_filt1) * lat_span
        den_filt_barrel = simple_barrel_roll(lat_use, den_barrel,
                                             barrel_radius,
                                             envelope=barrel_envelope,
                                             envelope_lower=envelope_lower,
                                             envelope_upper=envelope_upper)

        #  scale back up
        den_filt2 = den_filt_barrel * max(den_use) / lat_span

    # if moving measure second
    elif ('_average' in filt) | (filt == '_median'):

        # roughly window_lat degree smoothing window by converting to indices
        window = int(np.round(abs(window_lat / np.median(np.diff(lat_use)))))
        substring_to_remove = 'barrel_'
        measure = filt.replace(substring_to_remove, "")
        filt_meas = rolling_nanmeasure(den_filt1, window, measure)
        den_filt2 = np.array(filt_meas)

    # If no filter
    else:
        den_filt2 = den_filt1

    # Calculate gradient
    grad_den = np.gradient(np.array(den_filt2), lat_use)

    # Get latitudes of zero gradient points and process them
    zero_lat = evaluate_eia_gradient(lat_use, grad_den)
    z_lat = process_zlats(zero_lat, lat_use, den_filt2, lat_base=3)

    # electron density ghost check, but not for TEC
    if (den_str == 'ne') | (den_str == 'Ne'):
        ghost_check = True
    else:  # tec no ghost check
        ghost_check = False

    # Evaluate EIA gradient using zero lats, filtered density and lat
    eia_type_slope, plats, p3 = eia_slope_state(z_lat, lat_use,
                                                den_filt2, ghost=ghost_check)

    return (lat_use, den_filt2, eia_type_slope, z_lat, plats, p3)


def process_zlats(z_lat, lat, den, lat_base=3):
    """ filter z_lat by latitudes

    Parameters
    ----------
    z_lat : array-like
        latitudes where gradient = 0
    lat : array-like
        latitudes
    den :
        denisty (tec or ne)
    lat_base : int kwarg
        round to the nearest lat_base for filtering default = 3

    Returns
    ------
    z_lat : array-like
        new filtered z_lat array
    """
    # Get nearest indices to z_lat
    ilocz = []
    for z in z_lat:
        ilocz.append(abs(z - lat).argmin())
    ilocz = np.array(ilocz)

    # ensure that there are indices
    if len(ilocz) != 0:

        # get density at z_lat
        denz = den[ilocz]

        # round z_lat by lat_base
        z_round = myround(z_lat, base=lat_base)
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
        z_lat = np.unique(z_lat)

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
    z_lat = np.unique(z_lat)

    return z_lat


def set_zero_slope():
    """set the threshold for what is considered a zero slope
    """
    return 0.5


def set_dif_thresh(lat_span, percent=0.05):
    """set the threshold for what is different, input scale (if lat_span) = 50,
    then our max tec/ne is 50 so set thresh to 5 for 10%
    can also use this for maximum difference between peak and trough,
    so can use smaller threshold
    percent: decimal out of 1
    Parameters
    ----------
    lat_span: double
        span of latitude array e.g. max(latitude) - min(latitude)
    percent : kwarg double
        percent as a decimal for difference  threshold
    """
    return percent * lat_span


def get_exponent(number):
    """ calculate exponent of number
    Parameters
    ----------
    number : double

    Returns
    -------
    exponent of number
    """
    if number == 0:
        return float('-inf')  # Or handle as appropriate for your use case
    return np.floor(np.log10(abs(number)))


def myround(x, base=5):
    """ Round array to the nearest base
    Parameters
    ----------
    x : array-like
        array of values to be rounded
    base : int kwarg
        base to be rounded to, 5 default

    Returns
    -------
    rounded array to nearest base
    """
    rounded_array = []

    for xx in x:
        rounded_array.append(int(base * np.round(float(xx) / base)))

    return np.array(rounded_array)


def rolling_nanmeasure(arr, window, measure='mean'):
    """ Calculate the rolling mean or median of an array with or without nans
    Parameters
    ----------
    arr: array-like
        array of values to roll over
    window : integer
        window size
    measure : str kwarg
        string 'mean', 'median', 'average'

    Returns
    -------
    out: array-like
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
        if (measure == 'mean') | (measure == 'average'):
            if np.all(np.isnan(window_vals)):
                out[i] = np.nan
            else:
                out[i] = np.nanmean(window_vals)
        elif measure == 'median':
            if np.all(np.isnan(window_vals)):
                out[i] = np.nan
            else:
                out[i] = np.nanmedian(window_vals)
    return out


def find_nan_ranges(arr):
    """
    Identify continuous ranges of NaN values in an array
    Parameters
    ----------
    arr : array-like
        array with nans
    Returns
    -------
    nan_list : array-like
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


def simple_barrel_roll(lat, ne, barrel_radius, envelope=True,
                       envelope_lower=0.6, envelope_upper=0.2):
    """
    Roll barrel over an array to get detrended tec/ne (1 direciton, 1 radius)

    Parameters
    ----------
    lat: array-like
        latitude array
    ne: array-like
        density array (needs to be scaled so that similar magnitude to lat)
    barrel_radius : double
        latitudinal radius in degrees
    envelope : bool
        True (default) envelope used for return, False envelope not used

    Returns
    -------
    a : array-like
        detrended array
    """
    # starting x and y values for forward rolling
    strt_con_y = ne[0]
    strt_con_x = lat[0]

    # empty array of contact points for forward rolling
    f_con_xs = []
    f_con_ys = []

    # set j to 0 to keep track of index of array of next contact point
    j = 0

    # while index is less than last point in the array
    # keep lookin for contact points
    while (j < len(ne) - 1):
        r_sc = barrel_radius
        if j < len(ne) - 1:

            # Forward Rolling only
            f_con_xs .append(strt_con_x)
            f_con_ys.append(strt_con_y)

            # get the regions of interest (within barrel view) FORWARD ROLLING
            # > than start time and less than + diameter (same units as x)
            x_roi = lat[(lat > strt_con_x) & (lat <= strt_con_x + 2 * r_sc)]
            y_roi = ne[(lat > strt_con_x) & (lat <= strt_con_x + 2 * r_sc)]

            # Calcualte angular distance delta for each delta = beta - theta
            deltas = []

            # iterate through x region of interest
            for i in range(len(x_roi)):
                del_x = x_roi[i] - strt_con_x
                del_y = y_roi[i] - strt_con_y
                theta = np.atan(del_y / del_x)
                if (2 * r_sc) >= (((del_x) ** 2 + (del_y) ** 2) ** 0.5):
                    beta = np.asin((((del_x) ** 2 + (del_y) ** 2) ** 0.5)
                                   / (2 * r_sc))
                else:
                    beta = np.pi / 2
                delta = beta - theta
                deltas.append(delta * 180 / np.pi)

            if len(x_roi) != 0:

                # FORWARD CONTACTS
                strt_con_y = y_roi[deltas.index(min(deltas))]  # minimum delta
                strt_con_x = x_roi[deltas.index(min(deltas))]
                j = np.where(strt_con_x == lat)[0]  # update j for while loop
            else:
                # Append last value if there is no region of interest
                strt_con_y = ne[len(ne) - 1]
                strt_con_x = lat[len(ne) - 1]
                j = len(ne)

    # linear interpolation between contact points to return full array
    int_y = np.interp(np.setdiff1d(lat, f_con_xs), f_con_xs, f_con_ys)

    x_combined = np.concatenate((f_con_xs, np.setdiff1d(lat, f_con_xs)))
    y_combined = np.concatenate((f_con_ys, int_y))

    # Sort combined data by x values
    sorted_indices = np.argsort(x_combined)
    x_combined = x_combined[sorted_indices]
    y_combined = y_combined[sorted_indices]

    ne_cont = np.array(y_combined)

    if envelope:  # use an envelope
        BRC_upper = ne_cont + envelope_upper * max(ne_cont)
        BRC_lower = ne_cont - envelope_lower * min(ne_cont)
        a = np.empty((len(y_combined)))
        a[:] = np.nan

        # this is a combination of nan and values inside envelope
        a[(ne < BRC_upper) & (ne > BRC_lower)] = ne[((ne < BRC_upper)
                                                     & (ne > BRC_lower))]
        # Check to see if there are NaN values left from a
        int_y2 = np.interp(lat[np.isnan(a)],
                           lat[~np.isnan(a)], a[~np.isnan(a)])

        # replace nan values with interpolated values
        a[np.isnan(a)] = int_y2
    else:
        a = np.array(y_combined)
    return a


def evaluate_eia_gradient(lat, grad_dat, edge_lat=5):
    """Evaluate the TEC gradient for intersections revealing the EIA state

    Parameters
    ----------
    lat : array-like
        Apex latitude in degrees
    grad_dat : array-like
        TEC gradient data in TECU
    edge_lat : double
        latitude from edge to exclude, default is 5 deg

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
    lat = np.array(lat)

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
    return (zero_lat)


def single_peak_rules(p1, tec, lat):
    """ Determine if a peak is a peak, flat, or trough
    Parameters:
    -------------
    p1: array-like of length 1
        index of maxima
    lat: array-like
        latitude
    tec: array-like
        tec or ne

    Returns:
    ---------
    eia_state: (str)
        saddle, peak (north, south, (saddle) peak, (saddle) trough)
    plats: array-like
        latitude of peak
    """
    # calculate the latitudinal span of the peak
    n, s = peak_span(p1, tec, lat)

    # fit a line to the tec
    slope, intercept, rvalue, _, _ = stats.linregress(lat, tec)

    # detrend the tec for zlope
    tec_filt = slope * lat + intercept
    tec_detrend = tec - tec_filt

    loc_check = np.array([-15, 0, 15])
    zlope, ztec, zlat = getzlopes(loc_check, lat, tec_detrend)

    tr_check = 0  # Check if the slope decreases then increases
    if (np.sign(zlope[0]) == -1) & (np.sign(zlope[1]) == 1):
        tr_check = 1

    # check if there is span of the peak
    if ((n == -99) | (s == -99)) & (tr_check == 1):
        eia_state = 'trough'
        plats = []
    else:
        flat = flat_rules(p1, tec, lat)  # check if flat
        if flat == 0:  # Use location of peak to find orientation
            eia_state = "peak"
            peak_lat = lat[p1]
            if peak_lat > 3:
                eia_state += '_north'
            elif peak_lat < -3:
                eia_state += '_south'
            plats = [lat[p1]]
        elif flat == 2:  # trough
            plats = []
            eia_state = 'trough'
        else:
            plats = []
            eia_state = 'flat'
            if np.sign(flat) == 1:
                eia_state += '_north'
            else:
                eia_state += '_south'
    return eia_state, plats


def flat_rules(p1, tec, lat):
    """ Determines if a peak is actually flat along with direciton
    Parameters:
    -------------
    p1: array-like of length 1
        index of maxima
    lat: array-like
        latitude
    tec: array-like
        tec or ne

    Returns:
    ---------
    flat: int
        1 is flat_north, -1 is flat south, 0 is not flat
        2 if trough
    """
    # tec and lat of peak
    tec_max = tec[p1]
    lat_max = lat[p1]

    # set zero slope
    zero_slope = set_zero_slope()

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


def double_peak_rules(p1a, p2b, tec, lat):
    """ Determine if something is a saddle, eia, or single peak

    Parameters:
    -------------
    p1a: int
        single index of first maxima
    p2b: int
        single index of second maxima
    lat : array-like
        latitude
    tec : array-like
        tec or ne

    Returns:
    ---------
    eia_state : str
        string of eia state including: 'eia_north',
        'eia_south', 'eia_symmetric', 'eia_saddle_peak',
        'eia_saddle_peak_north', 'eia_saddle_peak_south',
        and orientations output by single_peak_rules
    """

    # set zero slope and symmetrical tec values
    zero_slope = set_zero_slope()
    lat_span = (max(lat) - min(lat))
    sym_tec = set_dif_thresh(lat_span)

    # peak lat and tec
    p_i = [p1a, p2b]
    p_l = lat[p_i]
    p_t = tec[p_i]

    # check that p1a does not equal p2b
    if p1a != p2b:
        if tec[p1a] != tec[p2b]:  # check if the tec is the same at each peak
            max_lat = p_l[max(p_t) == p_t]  # latitude at higher peak
            min_lat = p_l[min(p_t) == p_t]  # latitude at lower peak
            max_tec = p_t[max(p_t) == p_t]  # tec at higher peak
            min_tec = p_t[min(p_t) == p_t]  # tec at lower peak
            pmax = np.where(lat == max_lat)[0][0]
            pmin = np.where(lat == min_lat)[0][0]
        else:
            max_lat = lat[p1a]
            max_tec = tec[p1a]
            min_lat = lat[p2b]
            min_tec = tec[p2b]
            pmax = p1a
            pmin = p2b
    else:
        max_lat = p_l[0]
        min_lat = max_lat
        max_tec = p_t[0]
        min_tec = max_tec
        pmax = p1a
        pmin = pmax

    # Check if both peaks are different enough in lat
    # and not on same side of equator are p1 and p2
    if (abs(max_lat - min_lat) > 1) & (np.sign(max_lat) != np.sign(min_lat)):

        # trough defined as lowest point between peaks (non-inclusive)
        t_lats = lat[min(p_i) + 1:max(p_i)]
        tr_tec = tec[min(p_i) + 1:max(p_i)]

        # Limit trough lats to +/- 3 degrees Maglat
        t_lats_lim = t_lats[(t_lats < 3) & (t_lats > -3)]
        tp = (tr_tec[(t_lats < 3) & (t_lats > -3)]).argmin()
        trough_min = min(tr_tec[(t_lats < 3) & (t_lats > -3)])
        trough_lat = t_lats_lim[tp]  # latitude of trough minimum

        # calculate the north and south points of both peaks
        north_point_max, south_point_max = peak_span(pmax, tec,
                                                     lat,
                                                     trough_tec=trough_min,
                                                     trough_lat=trough_lat)
        north_point_min, south_point_min = peak_span(pmin, tec,
                                                     lat,
                                                     trough_tec=trough_min,
                                                     trough_lat=trough_lat)

        # Peak span tests
        # north and south points should be on same side of equator
        max_test = (np.sign(north_point_max) == np.sign(south_point_max))
        min_test = (np.sign(north_point_min) == np.sign(south_point_min))
        point_check = np.array([south_point_min, south_point_max,
                                north_point_min, north_point_max])

        # if the north or south point are very close to 0, then make true
        # (same side of equator)
        if (not max_test) & (min_test):
            if (abs(north_point_max) < 0.5) ^ (abs(south_point_max) < 0.5):
                max_test = True

        if (max_test) & (not min_test):

            # check if it is still within 0.5 degrees of equator
            if (abs(north_point_min) < 0.5) ^ (abs(south_point_min) < 0.5):
                min_test = True

        # if either peak is between 0.5 and -0.5,
        # then max test and min test are False
        if (abs(max_lat) < 0.5) | (abs(min_lat) < 0.5):
            max_test = False
            min_test = False

        # if the difference btween the north point and
        # south point is < 1 degree, opposite test is False
        if abs(north_point_min - south_point_min) < 1:
            max_test = False

        if abs(north_point_max - south_point_max) < 1:
            min_test = False

        # if the peaks are all undefined, both tests are false
        if np.all(point_check == -99):
            max_test = False
            min_test = False

        # if 1 peak has undefined span, opposite test false
        if (south_point_min == -99) | (north_point_min == -99):
            max_test = False
        elif (south_point_max == -99) | (north_point_max == -99):
            min_test = False

        # if both max test and min test are True, then we have an eia type
        if (max_test) & (min_test):
            eia_state = "eia"  # state is eia, eia_saddle, or saddle

            # Calculate slopes between min peak and trough
            slope_min = (min_tec[0] - trough_min) / (min_lat[0] - trough_lat)
            plats = [max_lat[0], min_lat[0]]

            # if slope_min is > zero_slope
            if abs(slope_min) > zero_slope:

                # get difference between peak max_tec and peak min_tec
                del_tec = max_tec[0] - min_tec[0]

                # symmetric if < sym_tec
                if abs(del_tec) <= sym_tec:
                    eia_state += '_symmetric'
                elif np.sign(max_lat) > 0:  # if not symmetric
                    eia_state += '_north'
                elif np.sign(max_lat) < 0:
                    eia_state += '_south'
            else:  # if not, eia_saddle
                eia_state += '_saddle'
                eia_state += saddle_N_S(p1a, p2b, tec, lat)

        # if not, send to single peak rules for peak that failed test
        elif (max_test) & (not min_test):  # smaller peak spans over 0
            eia_state, plats = single_peak_rules(pmin, tec, lat)
        else:  # both are False or max peak is false
            eia_state, plats = single_peak_rules(pmax, tec, lat)

    elif abs(max_lat - min_lat) <= 1:  # peaks are too close together,
        eia_state, plats = single_peak_rules(pmax, tec, lat)

    # same side of magnetic equator, choose peak closest to equator
    elif np.sign(max_lat) == np.sign(min_lat):
        max_eq_dist = abs(0 - max_lat)
        min_eq_dist = abs(0 - min_lat)
        if max_eq_dist < min_eq_dist:
            eia_state, plats = single_peak_rules(pmax, tec, lat)
        else:
            eia_state, plats = single_peak_rules(pmin, tec, lat)

    return eia_state, plats


def third_peak(z_lat, tec, lat, ghost_check=False):
    """ Look for a third peak

    Parameters:
    -------------
    z_lat : int
        single index of first maxima
    lat : array-like
        latitude
    tec : array-like
        tec or ne
    ghost_check : bool kwarg default False
        if False, don't look for ghosts, if True look for ghosts

    Returns:
    ---------
    p_third : array-like
        array of latitudes if 3 peaks are found
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
    if (ghost_check) & (len(zmaxima) == 2):
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
    if ghost_check:
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


def ghost_check(z_lat_og, lat, tec):
    """ Check for a ghost

    Parameters:
    -------------
    z_lat_og : int
        single index of first maxima
    lat : array-like
        latitude
    tec : array-like
        tec or ne

    Returns:
    ---------
    eia_state : str
        string of eia state-- options: '' (no ghost),
        'eia_ghost_' + north or south or symmetric,
        'ghost_peak_' + north or south
    spooky : bool
        if spooky = True, then there is a ghost
        if spooky = False, then no ghosts
    plats : array-like
        array of latitudes if ghost is found
    """
    spooky = False
    eia_state = ''
    plats = []

    # Establish symmetric threshold between trough and peak for ghosts
    lat_span = (max(lat) - min(lat))
    sym_ghost = set_dif_thresh(lat_span) / 2
    z_lat_og = np.array(z_lat_og)

    # Conduct ghost check
    # Limit latitudes between +/-15
    ghost_lat = z_lat_og[abs(z_lat_og) < 15]

    # Add +/- 15 in latitude check
    ghost_lat_ends = np.insert(ghost_lat, 0, -15)
    g_lats = np.insert(ghost_lat_ends, len(ghost_lat_ends), 15)

    # use third_check with ghost controls
    p3_check = third_peak(g_lats, tec, lat, ghost_check=True)

    # coninue if 3 peaks are returned also need to check if 1 is going
    # over equator or vs ghost vs saddle
    if len(p3_check) == 3:
        ghost_lat_check = np.all(abs(p3_check) < 15)
        if (ghost_lat_check):  # double check that everything is between +/-15

            # All not on one side of equator
            if ((not np.all(p3_check > 0)) & (not np.all(p3_check < 0))):

                # Peaks at equator, north, and south
                pn = max(p3_check)
                ps = min(p3_check)
                pe = p3_check[(p3_check != pn) & (p3_check != ps)]

                #  Locs of north, south, equator, north trough, south trough
                ttn = abs(pn - lat).argmin()
                tts = abs(ps - lat).argmin()
                tte = abs(pe - lat).argmin()
                trn = tec[tte + 1:ttn].argmin()
                trs = tec[tts + 1:tte].argmin()
                pm = abs(lat - pe).argmin()

                # Check tec difference between each peak and trough
                north_tec_check = abs(tec[ttn] - tec[tte + 1:ttn][trn])
                south_tec_check = abs(tec[tts] - tec[tts + 1:tte][trs])

                # calculate the span of the center based on the trough lats
                eqn, ex = peak_span(pm, tec, lat,
                                    trough_tec=tec[tte + 1:ttn][trn],
                                    trough_lat=lat[tte + 1:ttn][trn])
                ex, eqs = peak_span(pm, tec, lat,
                                    trough_tec=tec[tts + 1:tte][trs],
                                    trough_lat=lat[tts + 1:tte][trs])

                equator_check = False
                if (eqn > -1) & (eqs < 1):  # Center span +/- 1 deg of equator
                    equator_check = True

                if (equator_check):

                    # If the TEC is different from troughs, proper EIA ghost
                    if ((north_tec_check > sym_ghost) & (south_tec_check
                                                         > sym_ghost)):
                        eia_state += 'eia_ghost'
                        plats = p3_check
                        eia_state += ghost_NS_rules(plats, lat, tec)
                        spooky = True
                    elif ((north_tec_check < sym_ghost)
                          & (south_tec_check > sym_ghost)):
                        p3_check = np.array([lat[tts], lat[tte]])
                    elif ((north_tec_check > sym_ghost)
                          & (south_tec_check < sym_ghost)):
                        p3_check = np.array([lat[tte], lat[ttn]])

    if len(p3_check) == 2:  # if there are 2 peaks, may be a one armed ghost

        # double check that all are within +/- 15 degrees
        ghost_lat_check = np.all(abs(p3_check) < 15)
        if (ghost_lat_check):
            pn = max(p3_check)
            ps = min(p3_check)
            if pn != ps:  # make sure that the peaks are not at same lat

                # find trough between the peaks
                ttn = abs(pn - lat).argmin()
                tts = abs(ps - lat).argmin()
                tr = tec[tts + 1:ttn].argmin()
                north_tec_check = abs(tec[ttn] - tec[tts + 1:ttn][tr])
                south_tec_check = abs(tec[tts] - tec[tts + 1:ttn][tr])

                # caclualte span of each peak
                nn, ns = peak_span(ttn, tec, lat,
                                   trough_tec=tec[tts + 1:ttn][tr],
                                   trough_lat=tec[tts + 1:ttn][tr])
                sn, ss = peak_span(tts, tec, lat,
                                   trough_tec=tec[tts + 1:ttn][tr],
                                   trough_lat=tec[tts + 1:ttn][tr])

                n_eq = False
                s_eq = False
                if (nn > 0) & (ns < 0):
                    n_eq = True
                if (sn > 0) & (ss < 0):
                    s_eq = True

                if (s_eq) ^ (n_eq):  # Only one peak has to cross the equator

                    if not s_eq:  # Ghost north/south from arm hemisphere
                        if (south_tec_check > sym_ghost):
                            eia_state += 'eia_ghost_peak_south'
                            plats = p3_check
                            spooky = True
                    elif not n_eq:
                        if (north_tec_check > sym_ghost):
                            eia_state += 'eia_ghost_peak_north'
                            plats = p3_check
                            spooky = True
    return eia_state, spooky, plats


def ghost_NS_rules(plats, lat, tec):
    """ Determines if a ghost is symmetric, north, or south
    Parameters:
    -------------
    plats : array-like
        latitudes of peaks
    lat : array-like
        latitude
    tec : array-like
        tec or ne

    Returns:
    ---------
    eia_NS : str
        string of '_symmetric', '_south', or '_north'
    """
    # Establish symmetric threshold
    lat_span = (max(lat) - min(lat))
    sym_tec = set_dif_thresh(lat_span)

    # Array of TEC from peak latitude location
    tec_max = []
    for g in plats:
        tec_max.append(tec[abs(g - lat).argmin()])
    tec_max = np.array(tec_max)

    # Establish location of peaks
    southest_tec = tec_max[plats.argmin()]
    northest_tec = tec_max[plats.argmax()]

    eia_NS = ''

    # compare north and south
    n_s = northest_tec - southest_tec

    # Symmetric threshold
    if (abs(n_s) <= sym_tec):
        n_s = 0

    # Get direction based on n_s
    if (n_s == 0):
        eia_NS = '_symmetric'
    elif (n_s < 0):
        eia_NS = '_south'
    elif (n_s > 0):
        eia_NS = '_north'
    return eia_NS


def saddle_N_S(p1, p2, tec, lat):
    """ this evaluates whether or not a saddle should be labelled
    North or South or neither/Peak or Trough

    Parameters:
    -------------
    p1 : int
        index of first peak
    p2 : int
        index of second peak
    lat : array-like
        latitude
    tec : array-like
        tec or ne

    Returns:
    eia_NS : str
        direction of saddle '_peak', '_peak_north', '_peak_south'

    """
    eia_NS = '_peak'

    # Establish symmetric threshold
    lat_span = (max(lat) - min(lat))
    sym_tec = set_dif_thresh(lat_span)

    lat1 = lat[p1]
    lat2 = lat[p2]
    tec1 = tec[p1]
    tec2 = tec[p2]

    # compare tec at peak 1 to tec at peak 2
    dif21 = tec2 - tec1

    #  Adjust for symmetric criteria
    if (abs(dif21) <= sym_tec):
        dif21 = 0

    # North or south based on which peak is higher
    lat_high = 0
    if (dif21 < 0):
        lat_high = lat1
    elif (dif21 > 0):
        lat_high = lat2

    if lat_high > 0:
        eia_NS += '_north'
    elif lat_high < 0:
        eia_NS += '_south'

    return eia_NS


def peak_span(pm, tec, lat, trough_tec=-99, trough_lat=-99, div=0.5):
    """ Calculate latitudinal span of the peak
    Inputs:
    pm : int
        peak index
    tec: array-like
        tec or ne
    lat: array-like
        latitude
    trough_tec : kwarg double
        tec at trough, minimum tec for double or triple peaks,
        set to -99 if not specified
    trough_lat : kwarg trough_lat
        lat of trough if trough_tec reported, also report trough lat
    div : kwarg double
        decimal between 0 and 1 indicating desired peak width location
        default: 0.5 indicating half-width

    output:
    north_point : double
        northern latitude of peak width
    south_point : double
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


def toomanymax(z_lat, lat, tec, max_lat=[-99]):
    """ Reduce the number of peaks
    Parameters
    ------
        z_lat : array of latitudes at zero gradient points
        lat : array of latitudes
        tec : array of tec
        max_lat : array of length 1
            if a peak is already found,
            it can be input to guarantee it is in the array new array
            default is -99, undefined
    Output
    ------
        z_lat : array-like
        a new array of latitudes zero points
        z_lat_new should contain a maximum of 5 values
        [south edge, closest south, equator max, closest north, and north edge]
    """
    # indices of z_lat
    ilocz = []
    for z in z_lat:
        ilocz.append(abs(z - lat).argmin())

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

    # set up new array
    z_lat_new = []
    z_lat_new.append(z_lat[0])

    if len(tecz_south) > 0:  # if there are south tec, get largest value

        # if max_lat is not provided or it is in the north
        if (max_lat[0] == -99) | (np.sign(max_lat[0]) == 1):
            z_lat_new.append(latz_south[-1])  # closest to equator

        else:  # if a max_lat is provided and in south
            z_lat_new.append(max_lat[0])

    if len(tecz_eq) > 0:  # look for max tec value in equatorial region
        tez = tecz_eq.argmax()
        z_lat_new.append(latz_eq[tez])

    if len(tecz_north) > 0:  # look for max tec value in north

        # if max_lat is not provided or it is in the south
        if (max_lat[0] == -99) | (np.sign(max_lat[0]) == -1):
            z_lat_new.append(latz_north[0])  # closest to equator

        else:  # if max_lat is in provided and in north
            z_lat_new.append(max_lat[0])

    z_lat_new.append(z_lat[-1])

    # Make sure array is unique
    z_lat_new = np.unique(z_lat_new)

    return np.array(z_lat_new)


def getzlopes(z_lat_ends, lat, tec):
    """ Calculate slopes between zero points
    it returns the slopes, the nearest latitude and tec
    to the z points
    Parameters
    -------------
    z_lat_ends : array-like
        gradient zero latitudes including end points
    lat : array-like
        latitude
    tec : array-like
        tec

    Returns
    -------
    zlope : array-like
        slope between zero points length is lengeth of z_lat_ends-1
    ztec : array-like
        closest tec of z_lat_ends
    zlat : array-like
        closest latitude of z_lat_ends
    """

    zlope = []
    ztec = []
    zlat = []
    for zl in range(len(z_lat_ends) - 1):  # iterate through zero lats
        ilat1 = abs(z_lat_ends[zl] - lat).argmin()  # find index of nearest lat
        lat1 = lat[ilat1]  # get the latitude and tec of the first zero lat
        tec1 = tec[ilat1]
        ilat2 = abs

        # find index of nearest next latitude
        ilat2 = abs(z_lat_ends[zl + 1] - lat).argmin()
        lat2 = lat[ilat2]
        tec2 = tec[ilat2]
        if lat2 - lat1 != 0:  # make sure that the latitudes are not the same
            slope = (tec2 - tec1) / (lat2 - lat1)  # rise/run
        else:  # if they are the same, slope = 0, no difference
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
    """ Find the local maxima based on the slopes
    Parameters:
    -------------
    zlope : array-like
        slopes outputted from getzlopes
    ztec : array-like
        tec of zero locations
    ilocz: array-like
        indices of zero locations

    Returns:
    ---------
    zmaxima : array-like
        maximum tec
    zmaxi: array-like
        indices of maximum tec
    zminima : array-like
        minimum tec
    zmini : array-like
        indices of minimum tec
    """
    # set up arrays
    zmaxima = []
    zmaxi = []
    zminima = []
    zmini = []

    # go through slopes
    for s in range(len(zlope) - 1):

        # positive to negative slope = local maximum
        if (zlope[s] > 0) and (zlope[s + 1] < 0):

            # exclude ends from being counted as max or min
            # len(ztec) is greater than len(zlope) by 1
            zmaxima.append(ztec[s + 1])
            zmaxi.append(ilocz[s + 1])

        # negative slope to positive slope = local minimum
        elif (zlope[s] < 0) and (zlope[s + 1] > 0):
            zminima.append(ztec[s + 1])
            zmini.append(ilocz[s + 1])
    return zmaxima, zmaxi, zminima, zmini


def find_second_maxima(zlope, ztec, ilocz):
    """ Find secondary maxima

    Parameters:
    -------------
    zlope : array-like
        slopes outputted from getzlopes
    ztec : array-like
        tec of zero locations
    ilocz: array-like
        indices of zero locations

    Returns:
    ---------
    sec_max : array-like
        secondary maxima tec
    sec_maxi : array-like
        indices of secondary maxima
    """
    # set up arrays
    sec_max = []
    sec_maxi = []
    for s in range(len(zlope) - 1):

        # positive to 0 slope = secondary maximum
        if (zlope[s] > 0) and (zlope[s + 1] == 0):

            # exclude ends from being counted as max or min len(ztec)
            # is greater than len(zlope) by 1
            sec_max.append(ztec[s + 1])
            sec_maxi.append(ilocz[s + 1])

        # 0 to negative slope = secondary maximum
        elif (zlope[s] == 0) and (zlope[s + 1] < 0):
            sec_max.append(ztec[s + 1])
            sec_maxi.append(ilocz[s + 1])
    return sec_max, sec_maxi


def zero_max(lat, tec, zlats, maxes=[]):
    """ check for maxes if none are found
    lat : array-like
        latitudes
    tec : array-like
        tec or ne
    ilocz :
        inidces of potential peaks
    maxes : array-like kwarg
        given if we are looking only for a 1 peak instead of 2
    """
    # get locs of zero points
    ilocz = []
    for z in zlats:
        ilocz.append(abs(z - lat).argmin())

    lat_all = lat[ilocz]
    tec_all = tec[ilocz]
    tecz = tec_all[1:-1]
    latz = lat_all[1:-1]
    tecz_south = tecz[latz < 0]
    latz_south = latz[latz < 0]
    tecz_north = tecz[latz > 0]
    latz_north = latz[latz > 0]
    p1 = -99
    p2 = -99

    if (len(tecz_south) > 0):

        # if second peak found, double_peak_rules
        ts = tecz_south.argmax()
        ps = abs(lat_all - latz_south[ts]).argmin()

        # south check
        tec_b4 = tec_all[ps - 1]
        tec_af = tec_all[ps + 1]
        tec_at = tec_all[ps]

        if (tec_at > tec_b4) & (tec_at > tec_af):  # if it is a peak
            p1 = abs(lat - latz_south[ts]).argmin()

        else:  # check for center peak
            lat_eq = lat_all[(lat_all > -1) & (lat_all < 1)]
            tec_eq = tec_all[(lat_all > -1) & (lat_all < 1)]
            if len(tec_eq) != 0:
                te = tec_eq.argmax()
                pe = abs(lat_all - lat_eq[te]).argmin()

                # south check
                tec_b4e = tec_all[pe - 1]
                tec_afe = tec_all[pe + 1]
                tec_ate = tec_all[pe]
                if (tec_ate > tec_b4e) & (tec_ate > tec_afe):  # if it's a peak
                    p1 = abs(lat - lat_eq[te]).argmin()

        # if p1 is still -99 after a south and center check,
        # check for secondary maximum
        if p1 == -99:
            if (tec_at > tec_b4) | (tec_at > tec_af):
                p1 = abs(lat - latz_south[ts]).argmin()

    if (len(tecz_north) > 0):
        tn = tecz_north.argmax()
        pn = abs(lat_all - latz_north[tn]).argmin()

        # north check
        tec_b4 = tec_all[pn - 1]
        tec_af = tec_all[pn + 1]
        tec_at = tec_all[pn]
        if (tec_at > tec_b4) & (tec_at > tec_af):  # if it is a peak
            p2 = abs(lat - latz_north[tn]).argmin()
        else:  # check for center peak instead
            lat_eq = lat_all[(lat_all > -1) & (lat_all < 1)]
            tec_eq = tec_all[(lat_all > -1) & (lat_all < 1)]
            if len(tec_eq) != 0:
                te = tec_eq.argmax()
                pe = abs(lat_all - lat_eq[te]).argmin()

                # south check
                tec_b4e = tec_all[pe - 1]
                tec_afe = tec_all[pe + 1]
                tec_ate = tec_all[pe]
                if (tec_ate > tec_b4e) & (tec_ate > tec_afe):  # if it's a peak
                    p2 = abs(lat - lat_eq[te]).argmin()

        # if p1 is still -99 after a south and center check,
        # check for secondary maximum
        if p2 == -99:
            if (tec_at > tec_b4) | (tec_at > tec_af):  # if it is a peak
                p2 = abs(lat - latz_north[tn]).argmin()

    if len(maxes) > 0:  # if one peak is given, replace either p1 or p2 with it
        if lat[maxes[0]] > 0:
            p2 = maxes[0][0]
        elif lat[maxes[0]] < 0:
            p1 = maxes[0][0]
    else:  # no current maxes, use a sinlge max
        if (p1 < 0) & (p2 < 0):
            t_last = tecz.argmax()
            p1 = abs(lat - latz[t_last]).argmin()
    return p1, p2


def eia_slope_state(z_lat, lat, filt_tec, ghost=True):
    """Set the EIA state for a set of peaks and troughs in TEC data.

    Parameters
    ----------
    z_lat : array-like
        Latitude locations of the peaks and troughs in degrees
    lat : array-like
        Latitude locations of the TEC measurements in degrees
    filt_tec : array-like
        Ne data at a set of latitude locations in cm^-3/10**5 or
        TEC data at a set of latitude locations in TECU
    ghost : bool (kwarg default True)
        indicates whether or not ghost type should be included in analysis
        True (default) include ghost type, False exclude ghost type

    Returns
    -------
    eia_state : str
        String specifying the EIA state, one of 21 possible options
        eia_symmetric
    plats: array-like
        latitudes of peaks
    p3: array-like
        peak latitudes, if additional peak is found between
        eia type double peaks (not reported if ghost)

    Notes
    -----
    EIA States              | Description
    ------------------------|-----------------------
    flat_north              | A peakless TEC gradient that is higher in north
    flat_south              | A peakless TEC gradient that is higher in south
    flat                    | A roughly flat TEC across the equator
    peak                    | A single peak within 5 degrees of the equator
    peak_north              | A single peak in the northern hemisphere
    peak_south              | A single peak in the southern hemisphere
    eia_symmetric           | A hemispherically symmetric EIA
    eia_north               | An EIA with a higher peak in the north
    eia_south               | An EIA with a higher peak in the south
    eia_saddle_peak         | An EIA where there is a saddle and a peak
    eia_saddle_peak_north   | An EIA where there is a saddle and a peak, and
                            | the peak is in the North
    eia_saddle_peak_south   | An EIA where there is a saddle and a peak, and
                            | the peak is in the South
    trough                  | concave like TEC, dip in center
    eia_ghost_symmetric     | Triple peak between +/-15 degrees maglat where
                            | 1 peak crosses over 0 maglat
                            | North and South "arms" are symmetric
    eia_ghost_north         | Triple peak between +/-15 degrees maglat where
                            | 1 peak crosses over 0 maglat
                            | North "arm" higher than South "arm" are symmetric
    eia_ghost_south         | Triple peak between +/-15 degrees maglat where
                            | 1 peak crosses over 0 maglat
                            | South "arm" higher than North "arm" are symmetric
    eia_ghost_peak_north    | One armed ghost where 1 peak crosses 0 maglat
                            | Second peak in north
    eia_ghost_peak_south    | One armed ghost where 1 peak crosses 0 maglat
                            | Second peak in south

    """

    # Ensure we are working in - to + increasing latitude framework
    sort_in = np.argsort(lat)
    lat = lat[sort_in]
    tec = filt_tec[sort_in]
    lat_span = (max(lat) - min(lat))
    # Scale the density
    tec = tec / max(tec) * lat_span
    plats = []
    p3 = []

    # abs(slope) less than zero_slope is 0
    zero_slope = set_zero_slope()

    # save original z_lats from updates
    z_lat_og = z_lat

    # set initial state as unknown
    eia_state = 'unknown'

    if len(z_lat) == 0:

        # Fit a line to the TEC data to determine if flat north or south
        slope, intercept, rvalue, _, _ = stats.linregress(lat, tec)
        eia_state = "flat"

        if slope > zero_slope / 5:
            eia_state += "_north"
        elif slope < -zero_slope / 5:
            eia_state += "_south"

    # Single peak, single_peak_rules
    elif len(z_lat) == 1:

        p1 = abs(z_lat[0] - lat).argmin()
        eia_state, plats = single_peak_rules(p1, tec, lat)

    # Double peak or single peak
    elif len(z_lat) == 2:
        # get the zlopes
        zero_lat_ends = np.insert(z_lat, 0, np.nanmean(lat[0:5]))

        z_lat_ends = np.insert(zero_lat_ends, len(zero_lat_ends),
                               np.nanmean(lat[(len(lat) - 5):len(lat)]))
        zlope, ztec, zlat = getzlopes(z_lat_ends, lat, tec)

        # set 0 slopes
        zlope = np.array(zlope)
        zlope[abs(zlope) <= zero_slope] = 0

        if (zlope[0] > 0) & (zlope[2] > 0):  # choose z_lat[0] as peak
            p1 = abs(z_lat[0] - lat).argmin()
            eia_state, plats = single_peak_rules(p1, tec, lat)

        elif (zlope[0] < 0) & (zlope[2] < 0):  # choose z_lat[1] as peak
            p2 = abs(z_lat[1] - lat).argmin()
            eia_state, plats = single_peak_rules(p2, tec, lat)
        else:  # otherwise try double peak rules
            p1 = abs(z_lat[0] - lat).argmin()
            p2 = abs(z_lat[1] - lat).argmin()
            eia_state, plats = double_peak_rules(p1, p2, tec, lat)

    elif len(z_lat) > 2:

        # Add zero points close to end of TEC as padding for zlope calculation
        zero_lat_ends = np.insert(z_lat, 0, np.nanmean(lat[0:5]))
        z_lat_ends = np.insert(zero_lat_ends, len(zero_lat_ends),
                               np.nanmean(lat[(len(lat) - 5):len(lat)]))

        # get locs of zero points
        ilocz = []
        for z in z_lat_ends:
            ilocz.append(abs(z - lat).argmin())

        # see if there are triple peaks
        p3 = third_peak(z_lat_ends, tec, lat)

        # get zlopes and maxima from slopes
        zlope, ztec, zlat = getzlopes(z_lat_ends, lat, tec)
        zlope = np.array(zlope)
        zlope[abs(zlope) <= zero_slope] = 0
        zmaxima, zmaxi, zminima, zmini = find_maxima(zlope, ztec, ilocz)

        # use the length of the maxima to determine EIA type

        if len(zmaxima) > 2:
            # recalculate z_lat, zlopes, and zmaxima
            z_lat = toomanymax(z_lat_ends, lat, tec)
            zlope, ztec, zlat = getzlopes(z_lat, lat, tec)
            zlope = np.array(zlope)
            zlope[abs(zlope) <= zero_slope] = 0
            ilocz = []  # get locs of zero points
            for z in z_lat:
                ilocz.append(abs(z - lat).argmin())
            zmaxima, zmaxi, zminima, zmini = find_maxima(zlope, ztec, ilocz)

        if len(zmaxima) == 2:

            # 2 peaks, double peak rules
            p1a = zmaxi[0]
            p2b = zmaxi[1]
            peak_is = [p1a, p2b]  # Peak locations
            p_i = np.sort(peak_is)
            p1 = p_i[0]
            p2 = p_i[1]
            eia_state, plats = double_peak_rules(p1, p2, tec, lat)

        elif len(zmaxima) == 1:

            # look for secondary peak
            sec_max, sec_maxi = find_second_maxima(zlope, ztec, ilocz)
            if len(sec_maxi) == 1:
                p1 = zmaxi[0]
                p2 = sec_maxi[0]
                eia_state, plats = double_peak_rules(p1, p2, tec, lat)

            elif len(sec_maxi) == 0:  # use original zlat
                p1, p2 = zero_max(lat, tec, z_lat_ends, maxes=[zmaxi])

                if (p2 > 0) & (p1 > 0):

                    # 2 peaks found, double peak rules
                    eia_state, plats = double_peak_rules(p1, p2, tec, lat)
                else:

                    # only single peak
                    p1 = zmaxi[0]
                    eia_state, plats = single_peak_rules(p1, tec, lat)

            elif len(sec_maxi) >= 2:

                #  primary peak + 2 or more secondary peaks
                # recalculate z_lat, zlope, and zmaxima
                z_lat = toomanymax(z_lat_ends, lat, tec, max_lat=lat[zmaxi])
                zlope, ztec, zlat = getzlopes(z_lat, lat, tec)
                zlope = np.array(zlope)

                ilocz = []  # get locs of zero points
                for z in z_lat:
                    ilocz.append(abs(z - lat).argmin())
                zmaxima, zmaxi, zminima, zmini = find_maxima(zlope,
                                                             ztec, ilocz)
                if len(zmaxima) == 2:
                    # double peak rules
                    p1 = zmaxi[0]
                    p2 = zmaxi[1]
                    eia_state, plats = double_peak_rules(p1, p2, tec, lat)
                elif len(zmaxima) == 1:

                    # single peak rules with original peak
                    p1 = zmaxi[0]
                    eia_state, plats = single_peak_rules(p1, tec, lat)

        # If NO maximas
        if len(zmaxima) == 0:

            # look for secondary peaks
            sec_max, sec_maxi = find_second_maxima(zlope, ztec, ilocz)
            if len(sec_max) > 2:

                # if too many secondary peaks, use secondary peaks and ends
                # to recalculate zlope and secondary maxima
                z_too_many = np.insert(lat[sec_maxi], 0, z_lat_ends[0])
                z_too_many = np.insert(z_too_many,
                                       len(z_too_many), z_lat_ends[-1])  # Pad

                # instead of z_lat_ends ensures we are keeping peaks
                z_lat = toomanymax(z_too_many, lat, tec)
                zlope, ztec, zlat = getzlopes(z_lat, lat, tec)
                zlope = np.array(zlope)

                ilocz = []  # get locs of zero points
                for z in z_lat:
                    ilocz.append(abs(z - lat).argmin())

                # Calculate maxima without zero slopes
                sec_max, sec_maxi, zminima, zmini = find_maxima(zlope,
                                                                ztec, ilocz)

            if len(sec_max) == 2:

                # 2 max, double peak rules
                p1 = sec_maxi[0]
                p2 = sec_maxi[1]
                eia_state, plats = double_peak_rules(p1, p2, tec, lat)
            elif len(sec_max) == 1:

                # single peak, check for secondary peak
                p1, p2 = zero_max(lat, tec, z_lat_ends, maxes=[sec_maxi])

                if (p2 > 0) & (p1 > 0):

                    # 2 peaks found, double peak rules
                    eia_state, plats = double_peak_rules(p1, p2, tec, lat)
                else:

                    # only single peak
                    p1 = sec_maxi[0]
                    eia_state, plats = single_peak_rules(p1, tec, lat)

            elif len(sec_max) == 0:

                # no primary or secondary peaks found
                # check for peaks using maxima tec
                # at z_lat_ends on each side of equator
                p1, p2 = zero_max(lat, tec, z_lat_ends)

                if (p2 > 0) & (p1 > 0):

                    # 2 peaks found, double peak rules
                    eia_state, plats = double_peak_rules(p1, p2, tec, lat)
                elif (p1 > 0) & (p2 < 0):
                    eia_state, plats = single_peak_rules(p1, tec, lat)
                elif (p1 < 0) & (p2 > 0):
                    eia_state, plats = single_peak_rules(p2, tec, lat)

    if ghost:  # if ghost is specified
        eia_update, spooky, plat_ghost = ghost_check(z_lat_og, lat, tec)

        if spooky:  # spooky True indicates ghost presence
            eia_state = eia_update
            plats = plat_ghost

    # initialize new p3 as empty before checking for p3 to return that is not
    # ghostly
    p3_new = []

    # report triple non-ghost peaks for model purposes
    if len(plats) != 2:
        p3 = []
    elif len(plats) == 2:
        if len(p3) > 0:
            for ii in range(3):

                # if p3 is in between the two then report, otherwise do not
                if (p3[ii] < max(plats)) & (p3[ii] > min(plats)):
                    p3_new.append(p3[ii])
    p3 = np.array(p3_new)

    return eia_state, plats, p3
