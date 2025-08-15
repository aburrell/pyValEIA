#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------

import requests
import os
import zipfile
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import glob


def download_and_unzip(
        ymd, satellite, out_dir,
        s_url="https://swarm-diss.eo.esa.int/?do=download&file=swarm%2FLevel",
        level='1b', baseline='Latest_baselines',
        instrument1='EFI', instrument2='LP',
        f_end='0602', T1='000000', T2='235959', num_days=0):
    """ Download Swarm daily File
        Parameters
        ----------
        ymd: datetime object
            year month day of desired swarm file
        satellite : str
            satellite string 'A', 'B', or 'C'
        out_dir : str
            directory string for file output
        bse_url : str
            Base URL where data can be found before Level specification
        level : str kwarg
            This works for Level1b, not tested n Level2daily
        baseline : str kwarg
            desired baseline 'Latest_baselines' (defaut) or
            'Entire_mission_data' (not tested)
        instrument1 : str kwarg
            desired insturment default is 'EFI'
            for Electric Field Instrument
        instrument2 : str kwarg
            desired dataset from instrument1 default is 'LP'
            for Langmuir Probe
        f_end : str kwarg
            For different data products there are different numbers at the end
            The most common for EFIxLP is (Default) '0602_MDR_EFI_LP' where
             0602 represents the file version
             MDR_EFI_LP represents the Record Type
        T1 : str kwarg
            starting Time string format "HHMMSS"
            Most files contain "000000" to start, but if the file is not the
            whole day it will be something else
            Check website if download fails
        T2 : str kwarg
            ending Time string format "HHMMSS"
            Most files contain "235959" to end, but if the file is not the
            whole day it will be something else
            Check website if download fails
        num_days : int kwarg
            number of days from starting date to be downloaded after initial
            file (default is 0)
        Returns
        -------
            No returns, just file downloaded to out_dir
        Notes
        -----
            Default is an EFI file
            Options found at https://swarm-diss.eo.esa.int/#
            File format found at
                https://swarmhandbook.earth.esa.int/article/product
    """
    # Adjsut the name based on if it is level 1b or level 2daily
    if level == '1b':
        full_url = (s_url + level + "%2F" + baseline + "%2F" + instrument1
                    + 'x_' + instrument2)
    elif level == '2daily':
        full_url = (s_url + level + "%2F" + baseline + "%2F" + instrument1
                    + "%2F" + instrument2)
    # Out Folder
    yer = ymd.year
    mnth = ymd.month
    dy = ymd.day
    out_folder = f'{out_dir}/{instrument1}/Sat_{satellite}/{yer}/'

    # Make the path if it does not exist
    if not os.path.exists(out_folder):
        print(f'Making path {out_folder}')
        os.makedirs(out_folder)

    # Start at first day and go for num_days
    start_date = datetime(yer, mnth, dy)
    end_date = start_date + relativedelta(days=num_days)

    # Start with start date and go until end date is reached
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        f_bse = "SW_OPER_"
        d_str = date_str + "T" + T1 + "_" + date_str + "T" + T2 + "_" + f_end

        if level == '1b':
            filename = (f_bse + instrument1 + satellite + "_" + instrument2
                        + "_1B_" + d_str + ".CDF.ZIP")
        elif level == '2daily':
            filename = (f_bse + instrument1 + satellite + instrument2 + "_2F_"
                        + ".ZIP")

        # Set Full File URL
        file_url = full_url + "%2FSat_" + satellite + "%2F" + filename
        zip_path = os.path.join(out_folder, filename)
        current_date = current_date + timedelta(days=1)
        extract_folder = os.path.join(out_folder, date_str)

        # Find file if it already exists
        if level == '1b':
            efile = (f_bse + instrument1 + satellite + "_" + instrument2
                     + "_1B_" + d_str + "*.cdf")
        elif level == '2daily':
            efile = (f_bse + instrument1 + satellite + instrument2 + "_2F_"
                     + "*.cdf")

        extracted_files = os.path.join(extract_folder, efile)
        found_file = extracted_files

        if len(glob.glob(extracted_files)) > 0:
            found_file = glob.glob(extracted_files)[0]
        if os.path.exists(found_file):
            print(f"File already exists: {found_file}.Skipping download.")
        else:
            # Download file
            response = requests.get(file_url)
            if response.status_code == 200:
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                print("Downloading: "+filename)

                # Unzip file into date folder
                extract_folder = os.path.join(out_folder, date_str)
                os.makedirs(extract_folder, exist_ok=True)

                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_folder)
                    print("Extracted to:" + extract_folder)
                    os.remove(zip_path)
                except zipfile.BadZipFile:
                    print(f"Failed filename {filename} does not exist")
