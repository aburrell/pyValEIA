#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Functions to specify the EIA morphology."""

import numpy as np

from pyValEIA.eia import detection


def clean_type(state_array):
    """Simplifies EIA states into 4 base categories and 3 directions.

    Parameters
    ----------
    state_array : array-like
        Array of strings specifying the EIA state

    Returns
    -------
    base_types : array-like
        Simplifies EIA type strings into one of: flat, trough, peak, or EIA.
    base_dirs : array-like
        Simplifies EIA type strings into state directions: north, south, or
        neither.

    Raises
    ------
    ValueError
        If a `base_type` value cannot be established for any input value.

    See Also
    --------


    """
    base_types = []
    base_dirs = []

    for in_state in state_array:
        # Determine base type
        if 'eia' in in_state:
            base_types.append('eia')
        elif 'peak' in in_state:
            base_types.append('peak')
        elif in_state == 'trough':
            base_types.append('trough')
        elif 'flat' in in_state:
            base_types.append('flat')
        else:
            raise ValueError(
                'unknown EIA type encountered: {:}'.format(in_state))

        # Determine the base direction
        if 'north' in in_state:
            base_dirs.append('north')
        elif 'south' in in_state:
            base_dirs.append('south')
        else:
            base_dirs.append('neither')

    base_types = np.asarray(base_types)
    base_dirs = np.asarray(base_dirs)

    return base_types, base_dirs


def eia_slope_state(z_lat, lat, dens, ghost=True, zero_slope=0.5):
    """Set the EIA state for a set of peaks and troughs in plasma density data.

    Parameters
    ----------
    z_lat : array-like
        Latitude locations of the peaks and troughs in degrees
    lat : array-like
        Latitude locations of the plasma density measurements in degrees
    dens : array-like
        Electron density, ion density, or TEC data at a set of latitudes
    ghost : bool (kwarg default True)
        indicates whether or not ghost type should be included in analysis
        True (default) include ghost type, False exclude ghost type
    zero_slope : float
        Threshold for the zero-slope values (default=0.5)

    Returns
    -------
    eia_state : str
        String specifying the EIA state, one of 21 possible options
        eia_symmetric
    plats : array-like
        Latitudes of peaks
    p3lats : array-like
        Additional peak latitudes, if additional peak(s) are between the EIA
        double peaks and the type is not ghost; these are likely artifacts

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
    sort_lat = lat[sort_in]
    sort_dens = dens[sort_in]
    lat_span = (max(sort_lat) - min(sort_lat))

    # Scale the density
    sort_dens /= max(sort_dens) * lat_span

    # Save original zero-latitude locations
    z_lat_og = np.asarray(z_lat)

    # Initialize the output
    eia_state = 'unknown'
    plats = []
    p3lats = []

    # Determine the EIA morphology by examining different key characteristics
    if len(z_lat) == 0:
        # There is no latitude with a zero slope, so no peaks. Fit a line to
        # the plasma density data to determine if flat north or south
        slope, intercept, rvalue, _, _ = stats.linregress(sort_lat, sort_dens)
        eia_state = "flat"

        # Set the direction
        if slope > zero_slope / 5:
            eia_state += "_north"
        elif slope < -zero_slope / 5:
            eia_state += "_south"

    elif len(z_lat) == 1:
        # Single peak, single_peak_rules
        p1 = abs(z_lat[0] - sort_lat).argmin()
        eia_state, plats = single_peak_rules(p1, sort_dens, sort_lat)

    elif len(z_lat) == 2:
        # This could be a double peak or single peak. Start by getting the
        # slopes in the regions of the peaks (zlopes)
        zero_lat_ends = np.insert(z_lat, 0, np.nanmean(sort_lat[0:5]))
        z_lat_ends = np.insert(
            zero_lat_ends, len(zero_lat_ends),
            np.nanmean(sort_lat[(len(sort_lat) - 5):len(sort_lat)]))
        zlope, zsort_dens, zlat = getzlopes(z_lat_ends, sort_lat, sort_dens)

        # Set the zero-slopes
        zlope = np.array(zlope)
        zlope[abs(zlope) <= zero_slope] = 0

        if (zlope[0] > 0) & (zlope[2] > 0):
            # Choose z_lat[0] as peak
            p1 = abs(z_lat[0] - sort_lat).argmin()
            eia_state, plats = single_peak_rules(p1, sort_dens, sort_lat)

        elif (zlope[0] < 0) & (zlope[2] < 0):
            # Choose z_lat[1] as peak
            p2 = abs(z_lat[1] - sort_lat).argmin()
            eia_state, plats = single_peak_rules(p2, sort_dens, sort_lat)
        else:
            # Otherwise try double peak rules
            p1 = abs(z_lat[0] - sort_lat).argmin()
            p2 = abs(z_lat[1] - sort_lat).argmin()
            eia_state, plats = double_peak_rules(p1, p2, sort_dens, sort_lat)

    elif len(z_lat) > 2:
        # Add zero points close to end of the plasma density as padding for
        # zlope calculation
        zero_lat_ends = np.insert(z_lat, 0, np.nanmean(sort_lat[0:5]))
        z_lat_ends = np.insert(zero_lat_ends, len(zero_lat_ends),
                               np.nanmean(sort_lat[(len(sort_lat) - 5):
                                                   len(sort_lat)]))

        # get locs of zero points
        ilocz = []
        for z in z_lat_ends:
            ilocz.append(abs(z - sort_lat).argmin())

        # see if there are triple peaks
        p3lats = third_peak(z_lat_ends, sort_dens, sort_lat)

        # get zlopes and maxima from slopes
        zlope, zsort_dens, zlat = getzlopes(z_lat_ends, sort_lat, sort_dens)
        zlope = np.array(zlope)
        zlope[abs(zlope) <= zero_slope] = 0
        zmaxima, zmaxi, zminima, zmini = detection.find_maxima(
            zlope, zsort_dens, ilocz)

        # use the length of the maxima to determine EIA type
        if len(zmaxima) > 2:
            # recalculate z_lat, zlopes, and zmaxima
            z_lat = toomanymax(z_lat_ends, sort_lat, sort_dens)
            zlope, zsort_dens, zlat = getzlopes(z_lat, sort_lat, sort_dens)
            zlope = np.array(zlope)
            zlope[abs(zlope) <= zero_slope] = 0
            ilocz = []  # get locs of zero points
            for z in z_lat:
                ilocz.append(abs(z - sort_lat).argmin())
            zmaxima, zmaxi, zminima, zmini = detection.find_maxima(
                zlope, zsort_dens, ilocz)

        if len(zmaxima) == 2:

            # 2 peaks, double peak rules
            p1a = zmaxi[0]
            p2b = zmaxi[1]
            peak_is = [p1a, p2b]  # Peak locations
            p_i = np.sort(peak_is)
            p1 = p_i[0]
            p2 = p_i[1]
            eia_state, plats = double_peak_rules(p1, p2, sort_dens, sort_lat)

        elif len(zmaxima) == 1:

            # look for secondary peak
            sec_max, sec_maxi = detection.find_second_maxima(
                zlope, zsort_dens, ilocz)
            if len(sec_maxi) == 1:
                p1 = zmaxi[0]
                p2 = sec_maxi[0]
                eia_state, plats = double_peak_rules(p1, p2, sort_dens, sort_lat)

            elif len(sec_maxi) == 0:  # use original zlat
                p1, p2 = detection.zero_max(sort_lat, sort_dens, z_lat_ends,
                                            maxes=[zmaxi])

                if (p2 > 0) & (p1 > 0):

                    # 2 peaks found, double peak rules
                    eia_state, plats = double_peak_rules(p1, p2, sort_dens, sort_lat)
                else:

                    # only single peak
                    p1 = zmaxi[0]
                    eia_state, plats = single_peak_rules(p1, sort_dens, sort_lat)

            elif len(sec_maxi) >= 2:

                #  primary peak + 2 or more secondary peaks
                # recalculate z_lat, zlope, and zmaxima
                z_lat = toomanymax(z_lat_ends, sort_lat, sort_dens, max_lat=sort_lat[zmaxi])
                zlope, zsort_dens, zlat = getzlopes(z_lat, sort_lat, sort_dens)
                zlope = np.array(zlope)

                ilocz = []  # get locs of zero points
                for z in z_lat:
                    ilocz.append(abs(z - sort_lat).argmin())
                zmaxima, zmaxi, zminima, zmini = detection.find_maxima(
                    zlope, zsort_dens, ilocz)
                if len(zmaxima) == 2:
                    # double peak rules
                    p1 = zmaxi[0]
                    p2 = zmaxi[1]
                    eia_state, plats = double_peak_rules(p1, p2, sort_dens, sort_lat)
                elif len(zmaxima) == 1:

                    # single peak rules with original peak
                    p1 = zmaxi[0]
                    eia_state, plats = single_peak_rules(p1, sort_dens, sort_lat)

        # If NO maximas
        if len(zmaxima) == 0:

            # look for secondary peaks
            sec_max, sec_maxi = detection.find_second_maxima(
                zlope, zsort_dens, ilocz)
            if len(sec_max) > 2:

                # if too many secondary peaks, use secondary peaks and ends
                # to recalculate zlope and secondary maxima
                z_too_many = np.insert(sort_lat[sec_maxi], 0, z_lat_ends[0])
                z_too_many = np.insert(z_too_many,
                                       len(z_too_many), z_lat_ends[-1])  # Pad

                # instead of z_lat_ends ensures we are keeping peaks
                z_lat = toomanymax(z_too_many, sort_lat, sort_dens)
                zlope, zsort_dens, zlat = getzlopes(z_lat, sort_lat, sort_dens)
                zlope = np.array(zlope)

                ilocz = []  # get locs of zero points
                for z in z_lat:
                    ilocz.append(abs(z - sort_lat).argmin())

                # Calculate maxima without zero slopes
                sec_max, sec_maxi, zminima, zmini = detection.find_maxima(
                    zlope, zsort_dens, ilocz)

            if len(sec_max) == 2:

                # 2 max, double peak rules
                p1 = sec_maxi[0]
                p2 = sec_maxi[1]
                eia_state, plats = double_peak_rules(p1, p2, sort_dens,
                                                     sort_lat)
            elif len(sec_max) == 1:

                # single peak, check for secondary peak
                p1, p2 = detection.zero_max(sort_lat, sort_dens, z_lat_ends,
                                            maxes=[sec_maxi])

                if None not in [p1, p2]:
                    # 2 peaks found, double peak rules
                    eia_state, plats = double_peak_rules(p1, p2, sort_dens,
                                                         sort_lat)
                else:
                    # Only single peak
                    p1 = sec_maxi[0]
                    eia_state, plats = single_peak_rules(p1, sort_dens,
                                                         sort_lat)

            elif len(sec_max) == 0:

                # no primary or secondary peaks found
                # check for peaks using maxima tec
                # at z_lat_ends on each side of equator
                p1, p2 = detection.zero_max(sort_lat, sort_dens, z_lat_ends)

                if None not in [p1, p2]:
                    # 2 peaks found, double peak rules
                    eia_state, plats = double_peak_rules(p1, p2, sort_dens,
                                                         sort_lat)
                elif p2 is None and p1 is not None:
                    eia_state, plats = single_peak_rules(p1, sort_dens,
                                                         sort_lat)
                elif p1 is None and p2 is not None:
                    eia_state, plats = single_peak_rules(p2, sort_dens,
                                                         sort_lat)

    if ghost:
        # Check for ghosts
        eia_update, spooky, plat_ghost = ghost_check(z_lat_og, sort_lat,
                                                     sort_dens)

        if spooky:
            # spooky True indicates ghost presence
            eia_state = eia_update
            plats = plat_ghost

    # initialize new p3lats as empty before checking for p3lats to return that
    # is not ghostly
    p3lats_new = []

    # Report triple non-ghost peaks for model purposes
    if len(plats) != 2:
        p3lats = []
    elif len(plats) == 2:
        if len(p3lats) > 0:
            for ii in range(3):
                # If p3lats is in between the two then report, otherwise do not
                if (p3lats[ii] < max(plats)) & (p3lats[ii] > min(plats)):
                    p3lats_new.append(p3lats[ii])
    p3lats = np.array(p3lats_new)

    return eia_state, plats, p3lats


def ghost_check(z_lat_og, lat, tec):
    """Check for a ghost formation.

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

    Notes
    -----
    A ghost is formed when the EIA is developing or deteriorating

    """
    spooky = False
    eia_state = ''
    plats = []

    # Establish symmetric threshold between trough and peak for ghosts
    lat_span = (max(lat) - min(lat))
    sym_ghost = math.set_dif_thresh(lat_span) / 2
    z_lat_og = np.array(z_lat_og)

    # Conduct ghost check
    # Limit latitudes between +/-15
    ghost_lat = z_lat_og[abs(z_lat_og) < 15]

    # Add +/- 15 in latitude check
    ghost_lat_ends = np.insert(ghost_lat, 0, -15)
    g_lats = np.insert(ghost_lat_ends, len(ghost_lat_ends), 15)

    # use third_check with ghost controls
    p3_check = third_peak(g_lats, tec, lat, ghosts=True)

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

                # Center span +/- 1 deg of equator
                if (eqn > -1) & (eqs < 1):
                    # If the TEC is different from troughs, proper EIA ghost
                    if ((north_tec_check > sym_ghost) & (south_tec_check
                                                         > sym_ghost)):
                        eia_state += 'eia_ghost'
                        plats = p3_check
                        eia_state += ghost_ns_rules(plats, lat, tec)
                        spooky = True
                    elif ((north_tec_check < sym_ghost)
                          & (south_tec_check > sym_ghost)):
                        p3_check = np.array([lat[tts], lat[tte]])
                    elif ((north_tec_check > sym_ghost)
                          & (south_tec_check < sym_ghost)):
                        p3_check = np.array([lat[tte], lat[ttn]])

    if len(p3_check) == 2:
        # If there are 2 peaks, may be a one armed ghost. Double check that all
        # are within +/- 15 degrees
        ghost_lat_check = np.all(abs(p3_check) < 15)
        if (ghost_lat_check):
            pn = max(p3_check)
            ps = min(p3_check)
            if pn != ps:
                # Make sure that the peaks are not at same lat. Find the trough
                # between the peaks
                ttn = abs(pn - lat).argmin()
                tts = abs(ps - lat).argmin()
                tr = tec[tts + 1:ttn].argmin()
                north_tec_check = abs(tec[ttn] - tec[tts + 1:ttn][tr])
                south_tec_check = abs(tec[tts] - tec[tts + 1:ttn][tr])

                # Caclualte span of each peak
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

                if (s_eq) ^ (n_eq):
                    # Only one peak has to cross the equator
                    if not s_eq:
                        # Ghost north/south from arm hemisphere
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


def ghost_ns_rules(plats, lat, tec):
    """Determine if a ghost is symmetric, dominate north, or dominate south.

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
    eia_ns : str
        string of '_symmetric', '_south', or '_north'

    """
    # Establish symmetric threshold
    lat_span = (max(lat) - min(lat))
    sym_tec = math.set_dif_thresh(lat_span)

    # Array of TEC from peak latitude location
    tec_max = []
    for g in plats:
        tec_max.append(tec[abs(g - lat).argmin()])
    tec_max = np.array(tec_max)

    # Establish location of peaks
    southest_tec = tec_max[plats.argmin()]
    northest_tec = tec_max[plats.argmax()]

    # compare north and south
    n_s = northest_tec - southest_tec

    # Symmetric threshold
    if (abs(n_s) <= sym_tec):
        n_s = 0

    # Get direction based on n_s
    if n_s < 0:
        eia_ns = '_south'
    elif n_s > 0:
        eia_ns = '_north'
    else:
        eia_ns = '_symmetric'

    return eia_ns


def saddle_ns(p1, p2, tec, lat):
    """Evaluate whether or not a saddle should be labelled with a direction.

    Parameters
    ----------
    p1 : int
        index of first peak
    p2 : int
        index of second peak
    lat : array-like
        latitude
    tec : array-like
        tec or ne

    Returns
    -------
    eia_ns : str
        direction of saddle '_peak', '_peak_north', '_peak_south'

    """
    eia_ns = '_peak'

    # Establish symmetric threshold
    lat_span = (max(lat) - min(lat))
    sym_tec = math.set_dif_thresh(lat_span)

    lat1 = lat[p1]
    lat2 = lat[p2]
    tec1 = tec[p1]
    tec2 = tec[p2]

    # compare tec at peak 1 to tec at peak 2
    dif21 = tec2 - tec1

    #  Adjust for symmetric criteria
    if abs(dif21) <= sym_tec:
        dif21 = 0

    # North or south based on which peak is higher
    lat_high = 0
    if (dif21 < 0):
        lat_high = lat1
    elif (dif21 > 0):
        lat_high = lat2

    if lat_high > 0:
        eia_ns += '_north'
    elif lat_high < 0:
        eia_ns += '_south'

    return eia_ns
