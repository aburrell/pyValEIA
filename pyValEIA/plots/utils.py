# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Utility functions for formatting or creating plots."""

from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from matplotlib.patches import Patch


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


def format_longitude_labels(ax, xy='x'):
    """Format the longitude axis labels with degree symbols and E/W suffixes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Matplotlib axes object
    xy : str kwarg
        'x', 'y', or 'z' depending on which axis you want to have the degree
        symbol with E/W formatting (default='x')

    Raises
    ------
    ValueError
        If unknown axis supplied.

    """
    if xy.lower() == 'x':
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(longitude_formatter))
    elif xy.lower() == 'y':
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(longitude_formatter))
    elif xy.lower() == 'z':
        ax.zaxis.set_major_formatter(mticker.FuncFormatter(latitude_formatter))
    else:
        raise ValueError('unknown axis requested: {:}'.format(xy))

    return


def make_legend(leg_ax, leg_labs, leg_cols, leg_styles, modes, **kwargs):
    """Create a custom legend on a given axis.

    Parameters
    ----------
    leg_ax : matplotlib axis
        Axis to place the legend on.
    leg_labs : list of str
        Labels for the legend entries.
    leg_cols : list of str
        Colors for the legend entries.
    leg_styles : list of str
        Marker styles (if scatter) or line styles (if line).
    modes : list of str
        Type of legend entry for each label.
        Options: "line", "scatter", "shading", "line+shading".
    kwargs : dict
        Additional keyword arguments passed to ax.legend().

    """
    handles = []

    for lab, col, style, mode in zip(leg_labs, leg_cols, leg_styles, modes):
        if mode == "line":
            h = Line2D([], [], color=col, linestyle=style, marker=None,
                       label=lab)
        elif mode == "scatter":
            h = Line2D([], [], marker=style, linestyle="None",
                       markerfacecolor=col, markeredgecolor=col,
                       markersize=8, label=lab)
        elif mode == "shading":
            h = Patch(facecolor=col, edgecolor="none", alpha=0.5, label=lab)
        elif mode == "line+shading":
            line = Line2D([], [], color=col, linestyle=style)
            patch = Patch(facecolor=col, edgecolor="none", alpha=0.3)
            h = (line, patch)  # composite handle
        else:
            raise ValueError(f"Unknown mode: {mode}")
        handles.append(h)

    leg_ax.legend(handles=handles, labels=leg_labs, **kwargs)
    leg_ax.axis("off")  # hide axes completely)

    return


def daynight_label(model, LT_range=None):
    """Generate Label for local time day and night.

    Parameters
    ----------
    model : pd.DataFrame
        model dataframe build by states_report_swarm
    LT_range : list-like or NoneType
        Range of day night local time, or None for default of [7, 19]
        (default=None)

    Returns
    -------
    lab_day : string
        legend label for daytime
    lab_night : string
        legend label for nighttime

    """
    if LT_range is None:
        LT_range = [7, 19]

    # Separate day and night
    model_day = model[((model['LT'] > LT_range[0])
                       & (model['LT'] < LT_range[1]))]
    model_night = model[((model['LT'] < LT_range[0])
                         | (model['LT'] > LT_range[1]))]

    # Day label form minimum LT to maximum LT
    day_lab = str(int(min(model_day['LT']))) + '-' + str(
        int(max(model_day['LT']))) + ' LT'

    # night label depends on what the max and min are
    max_night = max(model_night['LT'])
    min_night = min(model_night['LT'])

    if (min_night < 12) & (max_night > 12):
        # minimum is between 0 and 12 do max to min, i.e. 23 to 3
        night_lab = str(int(max_night)) + '-' + str(int(min_night)) + ' LT'
    else:
        # minimum is greater than 12 do min to max, i.e. 20 to 23 or 1 to 5
        night_lab = str(int(min_night)) + '-' + str(int(max_night)) + ' LT'
    return day_lab, night_lab
