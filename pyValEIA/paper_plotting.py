#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
# Paper Plots

# Single NIMO Swarm Plot for paper purposes
from datetime import timedelta, datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pydarn

from pyValEIA.EIA_type_detection import eia_complete
from pyValEIA.io import load
from pyValEIA import nimo_conjunction
from pyValEIA.NIMO_Swarm_Map_Plotting import find_all_gaps


def decHr_astime(LT):
    """ Take a decimal hour and convert to timestring
    Parameters
    ----------
    LT: double
        single decimal hour time

    Returns String time as H:M
    """
    LT_hr = int(np.trunc(LT))
    LT_min = int(np.trunc((LT % 1) * 60))

    # Arbitrary date
    dat = datetime(2000, 1, 1, LT_hr, LT_min)

    return dat.strftime('%H:%M')


def format_latitude_labels(ax, xy='x'):
    """
    Formats the latitude axis labels with degree symbols and N/S suffixes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Matplotlib axes object
    xy : str kwarg
        'x' (defualt) or 'y' depending on which axis you want to have
        degree symbol N/S formatting
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


def nimo_swarm_single_plot(
        stime, satellite, swarm_file_dir, nimo_file_dir, MLat=30,
        swarm_filt='barrel_average', swarm_interpolate=1, swarm_envelope=True,
        swarm_barrel=3, swarm_window=2, nimo_filt='', nimo_interpolate=2,
        nimo_envelope=False, nimo_barrel=3, nimo_window=3, fosi=18,
        plot_dir='', nimo_name_format='NIMO_AQ_%Y%j', ne_var='dene',
        lon_var='lon', lat_var='lat', alt_var='alt', hr_var='hour',
        min_var='minute', tec_var='tec', hmf2_var='hmf2', nmf2_var='nmf2',
        nimo_cadence=15):
    """ Plot and save a single NIMO/Swarm EIA Type Plot for paper
    Parameters
    ----------
    stime : datetime object
        time of desired plot, nearest time within mlatitudinal window will be
        plotted
    satellite: str
        'A', 'B', or 'C' for Swarm
    swarm_file_dir : str
        directory where swarm file can be found
    nimo_file_dir: str
        directory where nimo file can be found
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
    nimo_filt : str kwarg
        Desired Filter for nimo data (no filter default)
    nimo_interpolate : int kwarg
        int that determines the number of data points in interpolation
        new length will be len(density)xinterpolate
        default is 2
    nimo_envelope : bool kwarg
        if True, barrel roll will include points inside an
        envelope, if false (default), no envelope will be used
    nimo_barrel : double
        latitudinal radius of barrel for swarm (default: 3 degrees maglat)
    nimo_window : double kwarg
        latitudinal width of moving window (default: 3 degrees maglat)
    fosi : int kwarg
        fontsize for plot (default is 18)
        Exceptions:
            Super Title (fosi + 10)
            legends (fosi - 3)
    plot_dir : str kwarg
        output folder for plot, or '' to not save output. If saved, an
        additional date directory will be created: plot_dir/{%Y%m%d}/fig.jpg
        (default='')
    nimo_name_format : str kwarg
        prefix of NIMO file including date format before .nc
        Default: 'NIMO_AQ_%Y%j'
    *_var : str kwarg
        variable names to be opened in the NIMO file
        * ne, lon, lat, alt, hr, min, tec, hmf2, nmf2
        Defaults
        --------
        electron density - 'dene'
        geo longitude - 'lon'
        geo latitude - 'lat'
        altitude - 'alt'
        hour - 'hour'
        minute - 'minute'
        TEC - 'tec'
        hmf2 - 'hmf2'
        nmf2 - 'nmf2'
    nimo_cadence: int kwarg
        time cadence of NIMO data in minutes
        Default is 15 minutes

    Notes
    -----
    filt options include: 'barrel', 'average', 'median', 'barrel_average'
    'barrel_median', 'average_barrel', and 'median_barrel'

    """

    # Ensure the entire day of SWARM is loaded, since the user needs to specify
    # the time closest to the one they want to plot. Do this be removing time
    # of day elements from day-specific timestamp for start and end
    sday = stime.replace(hour=0, minute=0, second=0, microsecond=0)
    eday = sday + timedelta(days=1)

    # Get full day of Swarm Data
    swarm_df = load.load_swarm(sday, eday, satellite, swarm_file_dir)

    # Housekeeping: get rid of bad values by flag.
    # \https://earth.esa.int/eogateway/documents/20142/37627/Swarm-Level-1b-
    # Product-Definition-Specification.pdf/12995649-fbcb-6ae2-5302-2269fecf5a08
    # Navigate to page 52 Table 6-4
    swarm_df['LT_hr'] = swarm_df['LT'].dt.hour + swarm_df['LT'].dt.minute / 60
    swarm_df.loc[(swarm_df['Ne_flag'] > 20), 'Ne'] = np.nan

    # Limit by user specified magnetic latitud range
    sw_lat = swarm_df[(swarm_df["Mag_Lat"] < MLat) & (swarm_df["Mag_Lat"]
                                                      > -MLat)]
    lat_ind = sw_lat.index.values

    # Identify the index ranges where the satellite passes over the desired
    # magnetic latitude range
    gap_all = find_all_gaps(lat_ind)

    # Append the first and last indices of lat_ind to gap array to form a full
    # List of gap indices
    start_val = [0]
    end_val = [len(lat_ind)]
    gaps = start_val + gap_all + end_val

    # Get closest time to Input
    tim_arg = abs(sw_lat["Time"] - stime).argmin()
    if abs(sw_lat["Time"].iloc[tim_arg] - stime) > timedelta(minutes=10):
        print(f'Selecting {sw_lat["Time"].iloc[tim_arg]}')

    # Choose latitudinally limited segment using gap indices
    gap_arg = abs(tim_arg - gaps).argmin()
    if gaps[gap_arg] <= tim_arg or tim_arg == 0:
        g1 = gap_arg
        g2 = gap_arg + 1
    else:
        g1 = gap_arg - 1
        g2 = gap_arg

    # Desired Swarm Data Segment
    swarm_check = sw_lat[gaps[g1]:gaps[g2]]

    # Get NIMO Dictionary
    nimo_dc = load.load_nimo(
        stime, nimo_file_dir, name_format=nimo_name_format, ne_var=ne_var,
        lon_var=lon_var, lat_var=lat_var, alt_var=alt_var, hr_var=hr_var,
        min_var=min_var, tec_var=tec_var, hmf2_var=hmf2_var, nmf2_var=nmf2_var,
        time_cadence=nimo_cadence)

    # Evaluate Swarm EIA-------------------------------------------------
    slat_use = swarm_check['Mag_Lat'].values
    density = swarm_check['Ne'].values
    den_str = 'Ne'
    sw_lat, sw_filt, eia_type_slope, z_lat, plats, p3 = eia_complete(
        slat_use, density, den_str, filt=swarm_filt,
        interpolate=swarm_interpolate, barrel_envelope=swarm_envelope,
        barrel_radius=swarm_barrel, window_lat=swarm_window)

    # Create Figure
    fig = plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': fosi})
    gs = mpl.gridspec.GridSpec(4, 3, width_ratios=[1, 1, 1],
                               height_ratios=[1, 1, 1, 1], wspace=0.1,
                               hspace=0.8)

    # Make first panel the legend for top panel
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(0, 0, linestyle='--',
             color='blue', label='Swarm ' + satellite + ' $N_{e}$')
    ax0.plot(0, 0, color='orange', label='Swarm Barrel Average')

    ax0.plot(0, 0, linestyle='-.', color='k', label='NIMO $N_{e}$')
    ax0.set_ylim([-99, -89])
    ax0.set_xlim([-100, -99])
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.legend(loc='center')
    ax0.axis('off')

    # Plot the Swarm Data --------------------------------------------------
    axs = fig.add_subplot(gs[0:2, 1:])
    axs.plot(swarm_check['Mag_Lat'], swarm_check['Ne'], linestyle='--',
             color='blue', label=None)
    axs.plot(sw_lat, sw_filt, color='orange', label=eia_type_slope)
    axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Plot Swarm Peak Latitudes
    if len(plats) > 0:
        for pi, p in enumerate(plats):
            lat_loc = (abs(p - swarm_check['Mag_Lat']).argmin())
            axs.vlines(swarm_check['Mag_Lat'].iloc[lat_loc],
                       ymin=min(swarm_check['Ne']),
                       ymax=swarm_check['Ne'].iloc[lat_loc], alpha=0.7,
                       color='orange')

    axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axs.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    axs.tick_params(axis='both', which='major', length=8)
    axs.tick_params(axis='both', which='minor', length=5)
    axs.set_ylabel("$N_{e}$ cm$^{-3}$")
    axs.set_xlabel(r'Magnetic Latitude')

    # Add NIMO Panel
    # Go through through Altitudes
    nimo_swarm_alt, nimo_map = nimo_conjunction.nimo_conjunction(
        nimo_dc, swarm_check, satellite, inc=0)
    nlat_use = nimo_swarm_alt['Mag_Lat'].values
    density = nimo_swarm_alt['Ne'].values

    den_str = 'Ne'
    nimo_lat, nimo_dfilt, eia_type_slope, z_lat, plats, p3 = eia_complete(
        nlat_use, density, den_str, filt=nimo_filt,
        interpolate=nimo_interpolate, barrel_envelope=nimo_envelope,
        barrel_radius=nimo_barrel, window_lat=nimo_window)

    axs.plot(nimo_swarm_alt['Mag_Lat'],
             nimo_swarm_alt['Ne'], linestyle='-.', color='black',
             label=eia_type_slope)

    # Plot NIMO Peak Latitudes
    if len(plats) > 0:
        for pi, p in enumerate(plats):
            lat_loc = (abs(p - nimo_swarm_alt['Mag_Lat']).argmin())
            lat_plot = nimo_swarm_alt['Mag_Lat'].iloc[lat_loc]
            axs.vlines(lat_plot, ymin=min(nimo_swarm_alt['Ne']),
                       ymax=nimo_swarm_alt['Ne'].iloc[lat_loc],
                       alpha=0.7, linestyle='-.', color='k')

    # Add Grid and limits to x axis
    axs.grid(axis='x')
    axs.set_xlim([min(swarm_check['Mag_Lat']), max(swarm_check['Mag_Lat'])])
    format_latitude_labels(axs)
    # Change location of legend if it south in eia_type_slope
    if 'south' in eia_type_slope:
        axs.legend(fontsize=fosi - 3, loc='upper right')
    else:
        axs.legend(fontsize=fosi - 3, loc='upper left')

    ts1 = swarm_check['Time'].iloc[0].strftime('%H:%M')
    ts2 = swarm_check['Time'].iloc[-1].strftime('%H:%M')

    axs.set_title(f'Swarm between {ts1}UT and {ts2}UT')

    # ----------------- MAP PLOT --------------------
    # Set the date and time for the terminator
    date_term = nimo_swarm_alt['Time'].iloc[0][0]

    # Get antisolar position and the arc (terminator) at the given height
    antisolarpsn, arc, ang = pydarn.terminator(date_term, 300)

    # antisolarpsn contains the latitude and longitude of the antisolar point
    # arc represents the radius of the terminator arc
    # Now, you can directly use the geographic coordinates from antisolarpsn.
    lat_antisolar = antisolarpsn[1]
    lon_antisolar = antisolarpsn[0]

    # Get positions along the terminator arc in geographic coordinates
    lats = []
    lons = []

    for b in range(-180, 180, 1):  # Iterate over longitudes from -180 to 180
        lat, lon = pydarn.GeneralUtils.new_coordinate(lat_antisolar,
                                                      lon_antisolar, arc, b,
                                                      R=pydarn.Re)
        lats.append(lat)
        lons.append(lon)
    lons = [(lon + 180) % 360 - 180 for lon in lons]

    # plot nmf2 map ------------------------------
    ax = fig.add_subplot(gs[2:, 1:], projection=ccrs.PlateCarree())

    ax.set_global()

    # Add Coastlines
    ax.add_feature(cfeature.COASTLINE)

    # Colormap
    heatmap = ax.pcolormesh(nimo_map['glon'], nimo_map['glat'],
                            nimo_map['nmf2'],
                            cmap=mpl.colormaps.get_cmap('cividis'),
                            transform=ccrs.PlateCarree())
    ax.plot(swarm_check['Longitude'], swarm_check['Latitude'], color='white',
            label="Satellite Path")
    ax.text(swarm_check['Longitude'].iloc[0] + 1,
            swarm_check['Latitude'].iloc[0], satellite, color='white')

    # Plot the Terminator
    lons = np.squeeze(lons)
    lats = np.squeeze(lats)
    ax.scatter(lons, lats, color='orange', s=1, zorder=2.0, linewidth=2.0)
    ax.plot(lons[0], lats[0], color='orange', linestyle=':', zorder=2.0,
            linewidth=2.0, label='Terminator 300km')
    leg = ax.legend(framealpha=0, loc='upper right')

    # Set text color
    for text in leg.get_texts():
        text.set_color('white')

    # set_aspect helps sizing
    ax.set_aspect('auto', adjustable='box')

    # Set labels
    ax.text(210, -50, 'Geographic Latitude', color='k', rotation=270)
    ax.text(-50, -110, 'Geographic Longitude', color='k')

    ax.set_title('NIMO N$_m$F$_2$ at {:s} UT'.format(
        nimo_swarm_alt['Time'].iloc[0][0].strftime('%H:%M')), fontsize=fosi + 5)

    cax = fig.add_subplot(gs[2:, 0], projection=ccrs.PlateCarree())

    # Add vertical colorbar on the side
    cbar = plt.colorbar(heatmap, ax=cax, orientation='vertical',
                        location='right', pad=0.04, shrink=0.8)
    cbar.ax.set_title("N$_m$F$_2$")
    cbar.set_label("cm$^{-3}$")
    cbar.ax.tick_params(labelsize=fosi)
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.tick_left()
    cax.spines['top'].set_visible(False)
    cax.spines['right'].set_visible(False)
    cax.spines['bottom'].set_visible(False)
    cax.spines['left'].set_visible(False)
    cax.axis('off')

    # Add grids and unset the grid labels
    gl = ax.gridlines(draw_labels=True, linewidth=0, color='gray', alpha=0.5)
    gl.top_labels = False   # Turn off top labels
    gl.right_labels = True  # Turn on right labels
    gl.left_labels = False  # Turn off left labels
    ds = swarm_check['Time'].iloc[0].strftime('%d %b %Y')

    # Plot Title

    lt_plot = decHr_astime(swarm_check['LT_hr'].iloc[0])
    fig.suptitle(ds + ' ' + ' at ' + lt_plot + 'LT',
                 fontsize=fosi + 10)
    fig.subplots_adjust(bottom=.03, top=.9, hspace=0.3)

    # Save plot if an output directory was supplied
    if os.path.isdir(plot_dir):
        ds = swarm_check['Time'].iloc[0].strftime('%Y%m%d')
        ts1 = swarm_check['Time'].iloc[0].strftime('%H%M')
        ts2 = swarm_check['Time'].iloc[-1].strftime('%H%M')

        fig_dir = os.path.join(plot_dir, ds)
        Path(fig_dir).mkdir(parents=True, exist_ok=True)

        figname = os.path.join(fig_dir, '_'.join(['NIMO_SWARM_Paper_',
                                                  satellite, ds,
                                                  ts1, '{:}.jpg'.format(ts2)]))
        fig.savefig(figname)

    return fig
