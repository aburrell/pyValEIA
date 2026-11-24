#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `io.write` and their counterparts in `io.load`."""

import datetime as dt
from glob import glob
import numpy as np
import os
import pandas as pd
import tempfile
import unittest

import pyValEIA
from pyValEIA.io import load
from pyValEIA.io import write


class TestWriteUtils(unittest.TestCase):
    """Tests for the write utilities that do not create files."""

    def setUp(self):
        """Set up the test runs."""
        self.start_time = dt.datetime(2026, 3, 14)
        self.file_dir = os.path.join("file", "directory", "location")
        self.model = "Model"
        self.out = None
        return

    def tearDown(self):
        """Clean up the test environment."""

        del self.start_time, self.file_dir, self.model, self.out
        return

    def test_madrigal_build_daily_stats_filename(self):
        """Test the filename building of the daily stats files for Madrigal."""
        # Cycle through the possible states
        for lon_type in [None, 'mag']:
            for mad_lon in [None, -10, 10.1]:
                # Set the optional inputs
                if lon_type is None and mad_lon is None:
                    in_kwargs = {}
                elif lon_type is None:
                    in_kwargs = {'mad_lon': mad_lon}
                elif mad_lon is None:
                    in_kwargs = {'lon_type': lon_type}
                else:
                    in_kwargs = {'lon_type': lon_type, 'mad_lon': mad_lon}

                with self.subTest(in_kwargs=in_kwargs):
                    # Build the file name
                    self.out = write.build_daily_stats_filename(
                        self.start_time, self.model, "madrigal", self.file_dir,
                        **in_kwargs)

                    # Test the date directory
                    self.assertEqual(os.path.split(self.out[0])[0],
                                     self.file_dir)
                    self.assertEqual(int(os.path.split(self.out[0])[-1]),
                                     self.start_time.year)

                    # Test the file name
                    comp_str = "".join(['-90' if mad_lon is None else
                                        '{:.0f}'.format(mad_lon),
                                        'g' if lon_type is None
                                        else lon_type[0], 'lon'])
                    self.assertRegex(self.out[1], comp_str)
                    self.assertRegex(self.out[1], "_".join([
                        self.model, "MADRIGAL"]))
                    self.assertRegex(self.out[1], 'lon_ascii.txt')
        return

    def test_sat_build_daily_stats_filename(self):
        """Test the filename building of the daily stats files for sats."""
        # Build the file name
        self.out = write.build_daily_stats_filename(
            self.start_time, self.model, "cnofs", self.file_dir)

        # Test the date directory
        self.assertEqual(os.path.split(self.out[0])[0], self.file_dir)
        self.assertEqual(int(os.path.split(self.out[0])[-1]),
                         self.start_time.year)

        # Test the file name
        self.assertRegex(self.out[1], "_".join([self.model, "CNOFS"]))
        self.assertRegex(self.out[1], 'ascii.txt')
        self.assertNotRegex(self.out[1], 'lon')
        return


class TestWriteLoadData(unittest.TestCase):
    """Tests for the writing and loading stat files."""

    @classmethod
    def setUpClass(self):
        """Set up the class by creating the test data directory."""
        self.tempdir = tempfile.TemporaryDirectory(
            ignore_cleanup_errors=True)
        return

    @classmethod
    def tearDownClass(self):
        """Clean up the class by removing the test data directory."""
        self.tempdir.cleanup()

        del self.tempdir
        return

    def setUp(self):
        """Set up the test runs."""
        self.start_time = dt.datetime(2026, 3, 14)
        self.end_time = dt.datetime(2026, 3, 15)
        self.file_dir = self.tempdir.name
        self.out = None
        self.stats = None
        self.outfiles = []
        return

    def tearDown(self):
        """Clean up the test environment."""

        for out_file in self.outfiles:
            os.remove(out_file)

        del self.start_time, self.file_dir, self.out, self.stats, self.outfiles
        del self.end_time
        return

    def create_stats(self):
        """Create a statistics DataFrame for writing."""
        stimes = [
            (self.start_time + dt.timedelta(hours=i)).strftime(
                pyValEIA.utils.coords.get_date_format_str()) for i in range(24)]
        slen = len(stimes)
        stat_dict = {'Satellite': np.full(shape=slen, fill_value='A'),
                     'LT_Hour': np.linspace(0, 24, slen),
                     'Mod_Time': stimes,
                     'Mod_GLon': np.linspace(0, 360, slen),
                     'Mod_Min_MLat': np.full(shape=slen, fill_value=-90),
                     'Mod_Max_MLat': np.full(shape=slen, fill_value=90),
                     'Mod_Alt': np.full(shape=slen, fill_value=450),
                     'Mod_Swarm_Type': np.full(shape=slen,
                                               fill_value='eia_south'),
                     'Mod_Peak_MLat1': np.full(shape=slen, fill_value=10),
                     'Mod_Peak_Ne1': np.full(shape=slen, fill_value=1e8),
                     'Mod_Peak_MLat2': np.full(shape=slen, fill_value=-10),
                     'Mod_Peak_Ne2': np.full(shape=slen, fill_value=5e7),
                     'Mod_Peak_MLat3': np.full(shape=slen, fill_value=np.nan),
                     'Mod_Peak_Ne3': np.full(shape=slen, fill_value=np.nan)}
        self.stats = pd.DataFrame(stat_dict)
        return

    def eval_file_creation(self):
        """Evaluate the presence of a file in a new yearly directory."""
        year_dir = os.path.join(self.file_dir, self.start_time.strftime("%Y"))
        self.assertTrue(os.path.isdir(year_dir))

        # Get the files
        self.outfiles = glob(os.path.join(year_dir, "*ascii.txt"))

        # Check the number of files
        self.assertEqual(len(self.outfiles),
                         np.floor((self.end_time
                                   - self.start_time).total_seconds() / 86400))
        return

    def eval_stats(self):
        """Evaluate the written and loaded DataFrames."""
        self.assertListEqual(list(self.out.columns), list(self.stats.columns))

        for col in self.stats.columns:
            if col[-1] == '3':
                self.assertTrue(np.isnan(self.out[col]).all())
            else:
                self.assertListEqual(list(self.out[col].values),
                                     list(self.stats[col].values))
        return

    def test_write_daily_stats(self):
        """Test the file writing/loading for daily statistics files."""
        # Create data to write
        self.create_stats()

        # Write the data
        write.write_daily_stats(self.stats, self.start_time, 'Mod', 'Swarm',
                                self.file_dir)

        # Evalute the creation of a file
        self.eval_file_creation()

        # Read the data
        self.out = load.load_daily_stats(self.start_time, 'Mod', 'Swarm',
                                         self.file_dir)

        # Evaluate the output
        self.eval_stats()
        return

    def test_write_multiday_stats(self):
        """Test the file writing/loading for many daily statistics files."""
        orig_start = self.start_time
        for i in range(3):
            # Create data to write
            if i > 0:
                self.start_time += dt.timedelta(days=1)
                self.end_time += dt.timedelta(days=1)
            self.create_stats()

            # Write the data
            write.write_daily_stats(self.stats, self.start_time, 'Mod', 'Swarm',
                                    self.file_dir)

        # Evalute the creation of the files
        self.start_time = orig_start
        self.eval_file_creation()

        # Read the data
        self.out = load.multiday_states_report(
            pd.date_range(self.start_time, self.end_time, freq='D',
                          inclusive='left'), self.file_dir, 'Mod', 'Swarm',
            'swarm')

        # Evaluate the output dictionaries
        self.assertEqual(len(self.out), 2)
        self.assertListEqual(list(self.out[0].keys()),
                             list(self.out[1].keys()))

        return
