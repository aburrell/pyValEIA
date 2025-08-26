#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
import numpy as np
from datetime import timedelta
import matplotlib.ticker as mticker

from pyValEIA.EIA_type_detection import eia_complete
from pyValEIA.io import load
from pyValEIA.NIMO_Swarm_Map_Plotting import find_all_gaps


def format_latitude_labels(ax, xy='x'):
    """
    Formats the latitude axis labels with degree symbols and N/S suffixes.

    Args:
        ax (matplotlib.axes.Axes): The Matplotlib axes object.
    """

    def latitude_formatter(latitude, pos):
        if latitude > 0:
            return f"{latitude:.0f}°N"
        elif latitude < 0:
            return f"{abs(latitude):.0f}°S"
        else:
            return "0°"
    if xy == 'x':
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(latitude_formatter))
    elif xy == 'y':
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(latitude_formatter))


def swarm_panel(axs, stime, satellite, swarm_file_dir, MLat=30,
                swarm_filt='barrel_average', swarm_interpolate=1,
                swarm_envelope=True, swarm_barrel=3, swarm_window=2, fosi=14,
                scale=False, scale_by='num', scale_num=10**5):
    """ Plot a single swarm panel no NIMO
    Parameters
    ----------
    axs : matplotlib axis
        axis for the data to be plotted onto
    stime : datetime object
        time of desired plot, nearest time within mlatitudinal window will be
        plotted
    satellite: str
        'A', 'B', or 'C' for Swarm
    swarm_file_dir : str
        directory where swarm file can be found
    MLat : int kwarg
        magnetic latitude range +/-MLat (default 30)
    swarm_filt : str kwarg
        Desired Filter for swarm data (default barrel_average)
    swarm_interpolate : int kwarg
        int that determines the number of data points in interpolation
        new length will be len(density)xinterpolate
        default is 1 indicating no interpolation
    swarm_envelope : bool kwarg
        if True (default), barrel roll will include points inside an
        envelope, if false, no envelope will be used
    swarm_barrel : double
        latitudinal radius of barrel for swarm (default: 3 degrees maglat)
    swarm_window : double kwarg
        latitudinal width of moving window (default: 2 degrees maglat)
    fosi : int kwarg
        fontsize for the legend
    scale : bool kwarg
        Default to False, will scale the data if True
    scale_by : str kwarg
        2 options: 'num' (default) and 'max'
        Can scale the data by a number (scale_num) or the maximum of the data
    scale_num : double kwarg
        if scale is True and scale_by is 'num', the density data will be
        divided by scale_num

    Notes
    -----
    filt options include: 'barrel', 'average', 'median', 'barrel_average'
    'barrel_median', 'average_barrel', and 'median_barrel'
    """

    # Convert to Day if not already
    sday = stime.replace(hour=0, minute=0, second=0, microsecond=0)
    eday = sday + timedelta(days=1)

    # Get full day of Swarm Data
    swarm_df = load.load_swarm(sday, eday, satellite, swarm_file_dir)

    # Housekeeping
    swarm_df['LT_hr'] = swarm_df['LT'].dt.hour + swarm_df['LT'].dt.minute / 60
    swarm_df.loc[(swarm_df['Ne_flag'] > 20), 'Ne'] = np.nan

    sw_lat = swarm_df[(swarm_df["Mag_Lat"] < MLat) & (swarm_df["Mag_Lat"]
                                                      > -MLat)]
    lat_ind = sw_lat.index.values
    gap_all = find_all_gaps(lat_ind)
    start_val = [0]
    end_val = [len(lat_ind)]  # add the beginning and end to gap indices
    gaps = start_val + gap_all + end_val

    # Get closest time to Input
    tim_arg = abs(sw_lat["Time"] - stime).argmin()
    if abs(sw_lat["Time"].iloc[tim_arg] - stime) > timedelta(minutes=10):
        print(f'Selecting {sw_lat["Time"].iloc[tim_arg]}')

    # Choose latitudinally limited segment using gap indices
    gap_arg = abs(tim_arg - gaps).argmin()
    if gaps[gap_arg] <= tim_arg:
        g1 = gap_arg
        g2 = gap_arg + 1
    else:
        g1 = gap_arg - 1
        g2 = gap_arg

    # Desired Swarm Data Segment
    swarm_check = sw_lat[gaps[g1]:gaps[g2]]

    # Evaluate Swarm EIA-------------------------------------------------
    lat_use = swarm_check['Mag_Lat'].values
    density = swarm_check['Ne'].values
    den_str = 'Ne'
    sw_lat, sw_filt, eia_type_slope, z_lat, plats, p3 = eia_complete(
        lat_use, density, den_str, filt=swarm_filt,
        interpolate=swarm_interpolate, barrel_envelope=swarm_envelope,
        barrel_radius=swarm_barrel, window_lat=swarm_window)

    # Plot Findings

    # Ne scaling...
    if scale:
        if scale_by == 'max':
            Ne_sc = swarm_check['Ne'] / max(swarm_check['Ne'])
        elif scale_by == 'num':
            Ne_sc = swarm_check['Ne'] / scale_num
    else:
        Ne_sc = swarm_check['Ne']

    # Add legend labels with Satelltie and times
    d1 = swarm_check['Time'].iloc[0].strftime('%Y %b %d')
    t1 = swarm_check['Time'].iloc[0].strftime('%H:%M')
    t2 = swarm_check['Time'].iloc[-1].strftime('%H:%M')
    axs.plot(swarm_check['Mag_Lat'], Ne_sc, linestyle='-')
    axs.scatter(swarm_check['Mag_Lat'].iloc[0], Ne_sc.iloc[0],
                color='white', s=0, label=f'Swarm {satellite}')
    axs.scatter(swarm_check['Mag_Lat'].iloc[0], Ne_sc.iloc[0],
                color='white', s=0, label=f'{d1}')
    axs.scatter(swarm_check['Mag_Lat'].iloc[0], Ne_sc.iloc[0],
                color='white', s=0, label=f'{t1}-{t2}UT')

    # Add peak lat lines
    if len(plats) > 0:
        for pi, p in enumerate(plats):
            lat_loc = (abs(p - swarm_check['Mag_Lat']).argmin())
            axs.vlines(swarm_check['Mag_Lat'].iloc[lat_loc],
                       ymin=min(Ne_sc),
                       ymax=Ne_sc.iloc[lat_loc], alpha=0.8,
                       color='orange')
    # Set axis info
    axs.grid(axis='x')
    axs.set_xlim([min(swarm_check['Mag_Lat']), max(swarm_check['Mag_Lat'])])
    axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axs.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    axs.tick_params(axis='both', which='major', length=8)
    axs.tick_params(axis='both', which='minor', length=5)

    # set y limits
    axs.set_ylim([min(Ne_sc) - 0.1 * min(Ne_sc),
                  max(Ne_sc) + 0.2 * max(Ne_sc)])

    eia_lab = eia_type_slope.replace("_", " ")

    axs.set_title(eia_lab, color='#000080', fontweight='bold')

    # Change x axis tick labels to latitude format
    format_latitude_labels(axs)

    # Add axis labels
    axs.set_ylabel("N$_e$")
    axs.set_xlabel("Magnetic Latitude")

    if 'south' in eia_type_slope:
        axs.legend(fontsize=fosi, framealpha=0, loc='upper right')
    else:
        axs.legend(fontsize=fosi, framealpha=0, loc='upper left')

    return axs
