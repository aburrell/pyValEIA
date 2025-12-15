#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Utilities for different coordinate systems."""

import numpy as np
import pandas as pd

from apexpy import Apex


def longitude_to_local_time(longitude, utc_time):
    """Convert geographic/geodetic longiutde to solar local time.

    Parameters
    ----------
    longitude : array-like
        longitudes
    utc_time : array-like
        Array of timestamps in UT

    Returns
    -------
    local_times : array-like
        Solar local times array as datetime objects

    """
    # Calcualte the longitude offset in seconds
    offset_sec = (3600.0 * np.asarray(longitude)) / 15.0

    # Use pandas to perform conversion as an array
    offset = pd.to_timedelta(offset_sec, unit='s')
    local_times = pd.to_datetime(utc_time) + offset

    return local_times


def compute_magnetic_coords(lat, lon, epoch_time, mag_type='qd'):
    """Calculate magnetic coordinates from geodetic coordinates.

    Parameters
    ----------
    lat : array-like
        Latitudes in degrees North
    lon : array-like
        Longitudes in degrees East
    epoch_time : dt.datetime
        Universal time for IGRF coefficients
    mag_type : str
        Magnetic coordinate type (default='qd')

    Returns
    -------
    mlat : array-like
        Magnetic latitude in degrees
    mlon : array-like
        Magnetic longitude in degrees

    See Also
    --------
    apexpy.Apex.convert

    Notes
    -----
    `mag_type` inputs are defined in `apexpy`.

    """
    # Initalize the Apex object
    apex = Apex(date=epoch_time)

    # Calculate the quasi-dipole coordinates
    mlat, mlon = apex.convert(lat, lon, 'geo', mag_type)

    return mlat, mlon


def earth_radius(lat, Re=6378137, Rp=6356752):
    """Calculate Earth's radius in meters at given latitudes.

    Parameters
    ----------
    lat : array-like
        latitude array
    Re : float
        Radius of Earth's equator in meters (default=6378137)
    Rp : float
        Radius of Earth's poles in meters (default=6356752)

    Returns
    -------
    Rearth : array-like
        Earth's radius in meters at given latitudes

    """

    Rearth = []

    # iterate through latitudes
    for i, l in enumerate(lat):
        # convert latitude to raidans
        lat_rad = l * (np.pi / 180)

        # caluclat earth's raidus at a specific altitude
        eq_top = (((Re ** 2 * np.cos(lat_rad)) ** 2)
                  + ((Rp ** 2 * np.sin(lat_rad)) ** 2))
        eq_bot = (((Re * np.cos(lat_rad)) ** 2)
                  + ((Rp * np.sin(lat_rad)) ** 2))

        # take square root and append
        Rearth.append((eq_top / eq_bot) ** 0.5)

    # convert to array
    Rearth = np.array(Rearth)

    return Rearth
