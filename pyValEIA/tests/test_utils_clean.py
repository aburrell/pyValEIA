#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `utils.clean`."""

import numpy as np
import unittest

from pyValEIA.utils import clean


class TestCleanFuncs(unittest.TestCase):
    """Tests for data cleaning functions."""

    def setUp(self):
        """Set up the test runs."""
        self.tec = np.linspace(0, 40, 10)
        self.std = abs(np.random.default_rng(None).gamma(
            np.full(shape=self.tec.shape, fill_value=2.0)))
        self.mlat = np.linspace(-30, 30, self.tec.shape[0])
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.tec, self.std, self.mlat
        return

    def test_mad_tec_clean_finite(self):
        """Test default Madrigal TEC cleaning with all finite data."""
        # Run cleaning with defaults
        out_tec, out_std, nan_perc, lat_cut = clean.mad_tec_clean(
            self.tec, self.std, self.mlat)

        # Test the output
        self.assertTupleEqual(out_tec.shape, (6,))
        self.assertTrue(np.isfinite(out_tec).all())
        self.assertEqual(nan_perc, 0.0)
        self.assertEqual(lat_cut, 20.0)
        return

    def test_mad_tec_clean_low_data(self):
        """Test Madrigal TEC cleaning with all data below the minimum TEC."""
        # Run cleaning with defaults
        out_tec, out_std, nan_perc, lat_cut = clean.mad_tec_clean(
            self.tec, self.std, self.mlat, min_tec=self.tec.max() + 1.0)

        # Test the output
        self.assertTupleEqual(out_tec.shape, (6,))
        self.assertTrue(np.isnan(out_tec).all())
        self.assertEqual(nan_perc, 100.0)
        self.assertEqual(lat_cut, 20.0)
        return

    def test_mad_tec_clean_nonfinite(self):
        """Test default Madrigal TEC cleaning with some fill values."""
        # Run cleaning with defaults
        self.tec[5] = np.inf
        out_tec, out_std, nan_perc, lat_cut = clean.mad_tec_clean(
            self.tec, self.std, self.mlat)

        # Test the output
        self.assertTupleEqual(out_tec.shape, (6,))
        self.assertFalse(np.isfinite(out_tec).all())
        self.assertEqual(np.isnan(out_tec).sum(), 1)
        self.assertAlmostEqual(nan_perc, 100.0 / 6.0)
        self.assertEqual(lat_cut, 20.0)
        return

    def test_mad_tec_clean_with_fill(self):
        """Test default Madrigal TEC cleaning with some fill values."""
        # Run cleaning with defaults
        self.tec[[2, 4, 5, 7]] = np.nan
        out_tec, out_std, nan_perc, lat_cut = clean.mad_tec_clean(
            self.tec, self.std, self.mlat)

        # Test the output
        self.assertTupleEqual(out_tec.shape, (6,))
        self.assertFalse(np.isfinite(out_tec).all())
        self.assertEqual(np.isnan(out_tec).sum(), 4)
        self.assertAlmostEqual(nan_perc, 400.0 / 6.0)
        self.assertEqual(lat_cut, 19.0)
        return

    def test_mad_tec_clean_with_wide_lat_Res(self):
        """Test Madrigal TEC cleaning with a different MLat resolution."""
        # Run cleaning with defaults
        self.tec[[2, 4, 5, 7]] = np.nan
        out_tec, out_std, nan_perc, lat_cut = clean.mad_tec_clean(
            self.tec, self.std, self.mlat, mlat_res=self.mlat[1] - self.mlat[0],
            max_nan=1)

        # Test the output
        self.assertTupleEqual(out_tec.shape, (2,))
        self.assertTrue(np.isnan(out_tec).all())
        self.assertEqual(nan_perc, 100.0)
        self.assertAlmostEqual(lat_cut, 6.666666666)
        return
