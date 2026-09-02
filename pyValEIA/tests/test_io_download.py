#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `utils.io.download`."""

import datetime as dt
from io import StringIO
from glob import glob
import logging
import os
import tempfile
import unittest

import pyValEIA
from pyValEIA.io import download


class TestSwarmDownload(unittest.TestCase):
    """Tests for the Swarm download functions."""

    def setUp(self):
        """Set up the test runs."""
        # Set test variables
        self.ddate = dt.datetime(2013, 12, 2)
        self.sat = "A"
        self.stime_str = "101113"
        self.etime_str = "140109"
        self.f_end = "0701"

        # Set up the test download directory
        self.tempdir = tempfile.TemporaryDirectory()
        self.file_dir = None

        # Set up the log capture
        self.lout = u''
        self.log_capture = StringIO()
        pyValEIA.logger.addHandler(logging.StreamHandler(self.log_capture))
        return

    def tearDown(self):
        """Clean up the test environment."""
        # Remove the temporary directory
        self.tempdir.cleanup()

        # Clear the attributes
        del self.ddate, self.stime_str, self.etime_str, self.f_end
        del self.tempdir, self.sat, self.lout, self.log_capture, self.file_dir
        return

    def eval_dir_structure(self, level=4):
        """Evaluate the created directory structure.

        Parmeters
        ---------
        level : int
            Number of sub directories to test (default=4)

        """
        # Ensure the temporary directory exists
        self.assertTrue(os.path.isdir(self.tempdir.name),
                        msg="Unable to find temporary directory")

        dir_list = [dval[0] for dval in os.walk(self.tempdir.name)]

        # See if the data directory exists
        if level > 0:
            self.file_dir = os.path.join(self.tempdir.name, "EFI")
            self.assertTrue(
                os.path.isdir(self.file_dir),
                msg="Unable to find the data directory in {:}".format(dir_list))

        # See if the satellite directory exists
        if level > 1:
            self.file_dir = os.path.join(self.file_dir,
                                         "_".join(['Sat', self.sat]))
            self.assertTrue(
                os.path.isdir(self.file_dir),
                msg="Unable to find satellite directory in {:}".format(
                    dir_list))

        if level > 2:
            # Find the year directory
            self.file_dir = os.path.join(self.file_dir,
                                         self.ddate.strftime('%Y'))
            self.assertTrue(
                os.path.isdir(self.file_dir),
                msg="Unable to find year directory in {:}".format(dir_list))

        if level > 3:
            # Find the date directory
            self.file_dir = os.path.join(self.file_dir,
                                         self.ddate.strftime('%Y%m%d'))
            self.assertTrue(os.path.isdir(self.file_dir),
                            msg="Unable to find date directory in {:}".format(
                                dir_list))
        return

    def eval_files(self):
        """Evaluate the downloaded files.

        Returns
        -------
        is_zip : bool
            True if there is a top-level zip file, False if there is not

        """
        is_zip = len(glob(os.path.join(os.path.split(self.file_dir)[0],
                                       "*.ZIP"))) > 0

        if self.file_dir is not None:
            self.assertEqual(len(glob(os.path.join(self.file_dir, "*"))), 3)

        return is_zip

    def test_bad_level(self):
        """Test raises ValueError with bad level input."""

        with self.assertRaisesRegex(ValueError, "unknown level"):
            download.download_and_unzip_swarm(self.ddate, self.sat,
                                              self.tempdir.name,
                                              level='NotALevel')
        return

    def test_bad_url_construction(self):
        """Test download fails without correct time and level limits."""
        # Raise the expected logging warning
        pyValEIA.logger.setLevel(logging.WARNING)
        download.download_and_unzip_swarm(self.ddate, self.sat,
                                          self.tempdir.name)

        # Ensure the first two subdirectories were created
        self.eval_dir_structure(level=3)

        # Test logging warning message and data output
        self.lout = self.log_capture.getvalue()
        self.assertRegex(self.lout, "Failed to access file URL")
        return

    def test_good_download_with_info(self):
        """Test a good download and all its logging messages."""
        # Raise the expected logging messages and get the data file
        pyValEIA.logger.setLevel(logging.INFO)
        download.download_and_unzip_swarm(
            self.ddate, self.sat, self.tempdir.name, stime_str=self.stime_str,
            etime_str=self.etime_str, f_end=self.f_end)

        # Test logging messages and data output
        self.lout = self.log_capture.getvalue()
        self.assertRegex(self.lout, "Making path")
        self.assertRegex(self.lout, "Downloading")
        self.assertRegex(self.lout, "Extracted to")

        # Ensure the subdirectories were created
        self.eval_dir_structure(level=4)

        # Ensure the file exists and the zip file is still there
        self.assertTrue(self.eval_files(), msg="Zip file missing: {:}".format(
            self.lout))

        return

    def test_good_download_with_remove(self):
        """Test a good download that removes the zip file."""
        # Raise the expected logging messages and get the data file
        download.download_and_unzip_swarm(
            self.ddate, self.sat, self.tempdir.name, stime_str=self.stime_str,
            etime_str=self.etime_str, f_end=self.f_end, remove=True)

        # Ensure the subdirectories were created
        self.eval_dir_structure(level=4)

        # Ensure the file exists and the zip file is not there
        self.assertFalse(self.eval_files(), msg="Zip file present: {:}".format(
            self.lout))

        return

    def test_download_repeat(self):
        """Test a repeated download."""
        self.test_good_download_with_info()

        # Repeat the same download
        download.download_and_unzip_swarm(
            self.ddate, self.sat, self.tempdir.name, stime_str=self.stime_str,
            etime_str=self.etime_str, f_end=self.f_end)

        # Test logging messages and data output
        self.lout = self.log_capture.getvalue()
        self.assertRegex(self.lout, "Skipping download")

        # Ensure the subdirectories were created
        self.eval_dir_structure(level=4)

        # Ensure the file exists and the zip file is still there
        self.assertTrue(self.eval_files(), msg="Zip file missing: {:}".format(
            self.lout))

        return
