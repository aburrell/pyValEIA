# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Utility functions for formatting or creating plots."""

import matplotlib.ticker as mticker


def format_latitude_labels(ax, xy='x'):
    """Format the latitude axis labels with degree symbols and N/S suffixes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Matplotlib axes object to format
    xy : str
        Specifies whether the x or y axis is being formatted (default='x')

    """
    if xy.lower() == 'x':
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(latitude_formatter))
    elif xy.lower() == 'y':
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(latitude_formatter))
    elif xy.lower() == 'z':
        ax.zaxis.set_major_formatter(mticker.FuncFormatter(latitude_formatter))
    else:
        raise ValueError('unknown axis requested: {:}'.format(xy))

    return


def latitude_formatter(latitude, pos):
    """Format latitude ticks to include degrees and hemisphere, removing signs.

    Parameters
    ----------
    latitude : float
        Latitude tick value in degrees from -90 to 90.
    pos : float
        Position, not used but required for use as FuncFormatter

    Returns
    -------
    lat_str : str
        Formatted latitude string

    Notes
    -----
    Designed for use within mpl.ticker.FuncFormatter

    """
    if latitude > 0:
        lat_str = r"{:.0f}$^\circ$N".format(latitude)
    elif latitude < 0:
        lat_str = r"{:.0f}$^\circ$S".format(abs(latitude))
    else:
        lat_str = r"0$^\circ$"

    return lat_str


def longitude_formatter(longitude, pos):
    """Format longitude ticks to include degrees and hemisphere, removing signs.

    Parameters
    ----------
    longitude : float
        Longitude tick value in degrees from -180 to 360.
    pos : float
        Position, not used but required for use as FuncFormatter

    Returns
    -------
    lon_str : str
        Formatted latitude string

    Notes
    -----
    Designed for use within mpl.ticker.FuncFormatter

    """
    if longitude > 0.0 and longitude <= 180.0:
        lon_str = r"{:.0f}$^\circ$E".format(longitude)
    elif longitude < 0.0 or longitude > 180.0:
        lon_str = r"{:.0f}$^\circ$W".format(abs(longitude))
    else:
        lon_str = r"0$^\circ$"

    return lon_str
