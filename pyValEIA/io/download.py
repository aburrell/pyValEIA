#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Download functions for supported data."""

import datetime as dt
import glob
import requests
import os
import zipfile

from pyValEIA import logger

swarm_url = "https://swarm-diss.eo.esa.int/?do=download&file=swarm%2FLevel"


def download_and_unzip_swarm(ddate, satellite, out_dir, base_url=swarm_url,
                             level='1b', baseline='Latest_baselines',
                             instrument='EFI', dataset='LP',
                             f_end='0602', stime_str='000000',
                             etime_str='235959', num_days=0, remove=False):
    """Download daily Swarm files and unzip them into instrument-date dirs.

    Parameters
    ----------
    ddate: datetime object
        Date of the desired Swarm file
    satellite : str
        Satellite string 'A', 'B', or 'C'
    out_dir : str
        String specifying base directory for file output
    base_url : str
        Base URL where data can be found before Level specification
        (default=`swarm_url`)
    level : str
        Data level, only tested on '1b' (default='1b')
    baseline : str
        Desired baseline, have not tested 'Entire_mission_data'
        (default='Latest_baselines')
    instrument : str
        Desired insturment acronym, e.g. 'EFI' is the Electric Field
        Instrument (default='EFI')
    dataset : str
        Desired dataset acronym from instrument, e.g. 'LP' is Langmuir Probe
        (default='LP')
    f_end : str
        For different data products there are different numbers at the end
        The most common for EFIxLP is '0602' where '0602' represents
        the file version. Other data products also have a record type string.
        (default='0602')
    stime_str : str
        Starting time using the string format "HHMMSS". Most files start with
        "000000", but if the file is not the whole day it will be different.
        Check website if download fails (default="000000")
    etime_str : str
        Ending time using the string format "HHMMSS". Most files end with
        "235959", but if the file is not the whole day it will be different.
        Check website if download fails (default="235959")
    num_days : int
        Number of days after the starting date to be downloaded after the
        initial day (default=0)
    remove : bool
        If True, remove zip archive after unpacking (default=False)

    Notes
    -----
    Different file options found at: https://swarm-diss.eo.esa.int/#
    File format information found at:
    https://swarmhandbook.earth.esa.int/article/product

    Raises
    ------
    ValueError
        If an unknown level is supplied

    """
    # Adjsut the name based on if it is level 1b or level 2daily
    full_url = ''.join([base_url, level, "%2F", baseline, "%2F", instrument,
                        'x_' if level == '1b' else '%2F', dataset])

    # Create the output folder
    yr = ddate.year
    mnth = ddate.month
    dy = ddate.day
    out_folder = os.path.join(out_dir, instrument, '_'.join(['Sat', satellite]),
                              ddate.strftime('%Y'))

    # Make the path if it does not exist
    if not os.path.exists(out_folder):
        logger.info(f'Making path {out_folder}')
        os.makedirs(out_folder)

    # Start at first day and go for num_days
    start_date = dt.datetime(yr, mnth, dy)
    end_date = start_date + dt.timedelta(days=num_days)

    # Start with start date and go until end date is reached
    while start_date <= end_date:
        date_str = start_date.strftime("%Y%m%d")
        f_bse = "SW_OPER_"
        d_str = ''.join([date_str, "T", stime_str, "_", date_str, "T",
                         etime_str, "_", f_end])

        if level == '1b':
            filename = ''.join([f_bse, instrument, satellite, "_", dataset,
                                "_1B_", d_str, ".CDF.ZIP"])
        elif level == '2daily':
            filename = ''.join([f_bse, instrument, satellite, dataset, "_2F_",
                                d_str, ".ZIP"])
        else:
            raise ValueError('unknown level: {:}'.format(level))

        # Set full file URL
        file_url = ''.join([full_url, "%2FSat_", satellite, "%2F", filename])

        # Set the full file path for the zip archive
        zip_path = os.path.join(out_folder, filename)

        # Set the output folder for unzipped data
        extract_folder = os.path.join(out_folder, date_str)

        # Find file if it already exists
        if level == '1b':
            efile = ''.join([f_bse, instrument, satellite, "_", dataset,
                             "_1B_", d_str, "*.cdf"])
        elif level == '2daily':
            efile = ''.join([f_bse, instrument, satellite, dataset, "_2F_",
                             d_str, "*.cdf"])

        extracted_files = os.path.join(extract_folder, efile)
        found_file = extracted_files
        if len(glob.glob(extracted_files)) > 0:
            found_file = glob.glob(extracted_files)[0]

        if os.path.exists(found_file):
            logger.info(f"File already exists: {found_file}.Skipping download.")
        else:
            # Download file from the file URL
            response = requests.get(file_url)
            if response.status_code == 200:
                with open(zip_path, 'wb') as fout:
                    fout.write(response.content)
                logger.info("Downloading: {:s}".format(filename))

                # Unzip file into date folder
                extract_folder = os.path.join(out_folder, date_str)
                os.makedirs(extract_folder, exist_ok=True)

                try:
                    # Extract the zip archive
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_folder)

                    logger.info("Extracted to: {:s}".format(extract_folder))

                    # Remove zip archive, if desired
                    if remove:
                        os.remove(zip_path)
                except zipfile.BadZipFile:
                    logger.warning(
                        f"Failed filename {filename} does not exist")

        # Cycle to the next day
        start_date += dt.timedelta(days=1)

    return
