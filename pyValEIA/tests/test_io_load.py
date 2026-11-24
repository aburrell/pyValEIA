#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `io.load`."""

import datetime as dt
import pandas as pd
import tempfile
import unittest

from pyValEIA.io import load


class TestLoadErrors(unittest.TestCase):
    """Tests for the loading error catches."""

    def setUp(self):
        """Set up the test runs."""
        self.start_time = dt.datetime(2020, 3, 14)
        self.end_time = dt.datetime(2020, 3, 15)
        self.file_dir = "."
        return

    def tearDown(self):
        """Clean up the test environment."""

        del self.start_time, self.end_time, self.file_dir
        return

    def test_load_swarm_dataset_error(self):
        """Test ValueError raised for an unknown Swarm dataset."""

        with self.assertRaisesRegex(ValueError, "unknown Swarm dataset"):
            load.load_swarm(self.start_time, self.end_time, 'A', self.file_dir,
                            instrument='NotAnInstrument', dataset='NoData')
        return

    def test_load_madrigal_nodata_error(self):
        """Test ValueError raised for missing Madrigal TEC data files."""

        with self.assertRaisesRegex(ValueError, "No Madrigal File Found"):
            load.load_madrigal(self.start_time, self.file_dir)
        return

    def test_load_nimo_nodata_error(self):
        """Test ValueError raised for missing NIMO data files."""

        with self.assertRaisesRegex(ValueError, "No NIMO file found"):
            load.load_nimo(self.start_time, self.file_dir)
        return

    def test_load_daily_stats_nodata_error(self):
        """Test ValueError raised for missing daily stats files."""

        with self.assertRaisesRegex(ValueError, "Could not find file"):
            load.load_daily_stats(self.start_time, "Model", "Obs",
                                  self.file_dir)
        return

    def test_multiday_states_report_alt_error(self):
        """Raised ValueError for bad altitude constraint in multiday report."""

        with self.assertRaisesRegex(ValueError, "Unknown altitude type"):
            load.multiday_states_report(
                pd.date_range(self.start_time, self.end_time), self.file_dir,
                "Model", "SWARM", "NotAConstraint")
        return

    def test_multiday_states_report_pyiri_tec_error(self):
        """Raised ValueError for using PyIRI and TEC in multiday report."""

        with self.assertRaisesRegex(ValueError, "Have not yet implemented TEC"):
            load.multiday_states_report(
                pd.date_range(self.start_time, self.end_time), self.file_dir,
                "PyIRI", "madrigal", "NotAConstraint")
        return

    def test_multiday_states_report_obs_error(self):
        """Raised ValueError for an unknown observation in multiday report."""

        with self.assertRaisesRegex(ValueError, "Unknown observation source"):
            load.multiday_states_report(
                pd.date_range(self.start_time, self.end_time), self.file_dir,
                "PyIRI", "NotAnObs", "NotAConstraint")
        return


class TestLoadData(unittest.TestCase):
    """Tests for the loading data files."""

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
        return

    def tearDown(self):
        """Clean up the test environment."""

        del self.start_time, self.end_time, self.file_dir, self.out
        return

    def test_load_swarm(self):
        """Test the successful download and loading of Swarm data."""

        self.out = load.load_swarm(self.start_time, self.end_time, 'A',
                                   self.file_dir)

        # Test the output
        self.assertTrue('Ne' in self.out.columns)
        self.assertTrue('Ne_error' in self.out.columns)
        self.assertTrue('Ne_flag' in self.out.columns)
        return
