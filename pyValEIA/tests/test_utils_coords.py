#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `utils.coords`."""

import datetime as dt
import numpy as np
import unittest

from pyValEIA.utils import coords


class TestTimeFuncs(unittest.TestCase):
    """Tests for time-handling functions."""

    def setUp(self):
        """Set up the test runs."""
        self.dtime = dt.datetime(1999, 2, 11)
        self.lon = 10.0
        self.out = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.dtime, self.lon, self.out
        return

    def evaluate_offset(self):
        """Evaluate the offset between UT and local time."""
        # Get the time difference in seconds regardless of the timezone
        sec = (dt.datetime.strptime(
            self.out[0].strftime("%Y-%m-%d %H:%M:%S:%u"),
            "%Y-%m-%d %H:%M:%S:%u") - self.dtime).total_seconds()
        self.assertEqual(self.lon, sec / 240.0)
        return

    def test_longitude_to_local_time_list(self):
        """Test success for datetime casting with list inputs."""
        # Cycle through potential time formats
        for in_time in [dt.date(self.dtime.year, self.dtime.month,
                                self.dtime.day), self.dtime,
                        np.datetime64(self.dtime.strftime('%Y-%m-%d')),
                        dt.datetime(self.dtime.year, self.dtime.month,
                                    self.dtime.day, tzinfo=dt.timezone.utc)]:
            with self.subTest(in_time=in_time):
                # Convert the time
                self.out = coords.longitude_to_local_time([self.lon], [in_time])
                self.assertTupleEqual(self.out.shape, (1,))
                self.evaluate_offset()
        return

    def test_longitude_to_local_time_value(self):
        """Test success for datetime casting with single value inputs."""
        # Cycle through potential time formats
        for in_time in [dt.date(self.dtime.year, self.dtime.month,
                                self.dtime.day), self.dtime,
                        np.datetime64(self.dtime.strftime('%Y-%m-%d')),
                        dt.datetime(self.dtime.year, self.dtime.month,
                                    self.dtime.day, tzinfo=dt.timezone.utc)]:
            with self.subTest(in_time=in_time):
                # Convert the time
                self.out = [coords.longitude_to_local_time(self.lon, in_time)]
                self.evaluate_offset()
        return

    def test_longitude_to_local_time_array(self):
        """Test success for datetime casting with array inputs."""
        # Cycle through potential time formats
        for in_time in [dt.date(self.dtime.year, self.dtime.month,
                                self.dtime.day), self.dtime,
                        np.datetime64(self.dtime.strftime('%Y-%m-%d')),
                        dt.datetime(self.dtime.year, self.dtime.month,
                                    self.dtime.day, tzinfo=dt.timezone.utc)]:
            with self.subTest(in_time=in_time):
                # Convert the time
                self.out = coords.longitude_to_local_time(
                    np.array([self.lon]), np.array([in_time]))
                self.assertTupleEqual(self.out.shape, (1,))
                self.evaluate_offset()
        return

    def test_longitude_to_local_time_mult_time(self):
        """Test success for datetime casting with multiple times."""
        # Cycle through different unequal length combinations
        for lon_len, time_len in [[1, 5], [3, 1], [3, 5]]:
            lon_in = np.full(shape=(lon_len,), fill_value=self.lon)
            ut_in = np.full(shape=(time_len,), fill_value=self.dtime)
            with self.subTest(lon_len=lon_len, time_len=time_len):
                with self.assertRaisesRegex(
                        ValueError, 'cannot add indices of unequal length'):
                    coords.longitude_to_local_time(lon_in, ut_in)
        return
