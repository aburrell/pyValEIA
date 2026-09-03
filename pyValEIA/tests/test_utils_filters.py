#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `utils.coords`."""

import numpy as np
import unittest

from pyValEIA.utils import filters


class TestFiltersFuncs(unittest.TestCase):
    """Tests for fitlers functions."""

    def setUp(self):
        """Set up the test runs."""
        self.in_arr = np.array([0, 6, 6, 7, 8, 50, np.nan])
        self.out = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.in_arr, self.out
        return

    def test_detect_outliers(self):
        """Test success of detect outliers."""
        # Find outliers
        self.out = filters.detect_outliers(self.in_arr)

        # Test output, outliers must be numbers
        self.assertListEqual(list(self.in_arr[self.out]), [0, 50])
        return

    def test_detect_outliers_nan(self):
        """Test detect outliers routine with only NaN."""
        # Find outliers
        self.in_arr = np.full(shape=self.in_arr.shape, fill_value=np.nan)
        self.out = filters.detect_outliers(self.in_arr)

        # Test no outliers, as no real values
        self.assertEqual(len(self.out), 0)
        return

    def test_find_all_gaps(self):
        """Test success of find_all_gaps function."""
        # Include only real values as input
        good_mask = ~np.isnan(self.in_arr)
        self.out = filters.find_all_gaps(self.in_arr[good_mask])

        # Test each gap
        for i in self.out:
            self.assertNotEqual(self.in_arr[good_mask][i],
                                self.in_arr[good_mask][i - 1] + 1,
                                msg='Failed for index {:d}/{:d}'.format(
                                    i, sum(good_mask)))
        return

    def test_find_nan_ranges(self):
        """Test find_nan_ranges success."""
        # Add another NaN range
        self.in_arr[2:4] = np.nan
        self.out = filters.find_nan_ranges(self.in_arr)

        # Evaluate output
        self.assertEqual(len(self.out), 2)

        for i in self.out:
            self.assertTrue(np.isnan(self.in_arr[slice(*i)]).all(),
                            msg="Not all NaN: {:}".format(
                                self.in_arr[slice(*i)]))
        return

    def test_rolling_nanmeasure_bad_measure_method(self):
        """Test raises ValueError with a bad `measure` kwarg."""
        with self.assertRaisesRegex(ValueError, 'unknown method for'):
            filters.rolling_nanmeasure(self.in_arr, 3, measure='No Method')
        return

    def test_rolling_nanmeasure_success(self):
        """Test success of `rolling_nanmeasure`."""
        for meas in ['mean', 'median', 'average']:
            with self.subTest(measure=meas):
                # Get the rolling central values
                self.out = filters.rolling_nanmeasure(self.in_arr, 3,
                                                      measure=meas)

                # Test the output
                self.assertTrue(np.isfinite(self.out).all())
                self.assertLessEqual(self.out.max(), np.nanmax(self.in_arr))
                self.assertGreaterEqual(self.out.min(), np.nanmin(self.in_arr))
        return

    def test_simple_barrel_roll(self):
        """Test success of `simple_barrel_roll`."""
        # Set a second array of the same shape and range, but different values
        xvar = np.linspace(np.nanmin(self.in_arr), np.nanmax(self.in_arr),
                           self.in_arr.shape[0])

        for env, low, up in ([True, 0.6, 0.2], [True, 0.2, 0.6], [False, 0, 0]):
            with self.subTest(envelope=env, envelope_lower=low,
                              envelope_upper=up):
                self.out = filters.simple_barrel_roll(
                    xvar, self.in_arr, 30, envelope=env, envelope_lower=low,
                    envelope_upper=up)

                # Test the output
                self.assertTrue(np.isfinite(self.out).all())
                self.assertLessEqual(self.out.max(), np.nanmax(self.in_arr))
                self.assertGreaterEqual(self.out.min(), np.nanmin(self.in_arr))
        return
