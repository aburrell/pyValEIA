#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `utils.coords`."""

import numpy as np
import unittest

from pyValEIA.utils import math


class TestValueFuncs(unittest.TestCase):
    """Tests for math functions that manipulate values."""

    def setUp(self):
        """Set up the test runs."""
        self.in_val = 101.6
        self.out = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.in_val, self.out
        return

    def test_get_exponent_non_zero(self):
        """Test success for non-zero float input."""
        self.out = 10 ** math.get_exponent(self.in_val)
        self.assertGreaterEqual((self.in_val - self.out) / self.out, 0.0)
        self.assertLess((self.in_val - self.out) / self.out, 1.0)
        return

    def test_get_exponent_zero(self):
        """Test success for a float input of zero."""
        self.out = math.get_exponent(0.0)
        self.assertTrue(np.isinf(self.out))
        self.assertEqual(np.sign(self.out), -1)
        return

    def test_get_exponent_non_zero_array(self):
        """Test success for non-zero float input."""
        self.out = 10 ** math.get_exponent(np.full(shape=(2, 2),
                                                   fill_value=self.in_val))
        self.assertTupleEqual(self.out.shape, (2, 2))
        self.assertTrue(np.all((self.in_val - self.out) / self.out >= 0.0))
        self.assertTrue(np.all((self.in_val - self.out) / self.out < 1.0))
        return

    def test_get_exponent_zero_array(self):
        """Test success for an array input of zeros."""
        self.out = math.get_exponent(np.zeros(shape=(4,)))
        self.assertTrue(np.isinf(self.out).all())
        self.assertTrue((np.sign(self.out) == -1).all())
        self.assertTupleEqual(self.out.shape, (4, ))
        return

    def test_get_exponent_mixed_list(self):
        """Test success for mixed zero and non-zero list input."""
        self.out = 10 ** math.get_exponent([self.in_val, 0.0])

        # Evaluate the output shape
        self.assertTupleEqual(self.out.shape, (2,))

        # Evaluate the second, zero value
        self.assertEqual(self.out[-1], 0.0)

        # Evaluate the first, non-zero value
        self.assertGreaterEqual((self.in_val - self.out[0]) / self.out[0], 0.0)
        self.assertLess((self.in_val - self.out[0]) / self.out[0], 1.0)
        return

    def test_base_one_round(self):
        """Test success of base rounding for the standard base."""
        self.out = math.base_round(self.in_val, base=1)
        self.assertEqual(np.round(self.in_val), self.out)
        return

    def test_base_round_default(self):
        """Test success of base rounding for the default base."""
        self.out = math.base_round(self.in_val)
        self.assertEqual(self.out, 100.0)
        return

    def test_base_round_array(self):
        """Test success of base rounding for the array input."""
        self.out = math.base_round(np.full(shape=(3, 2),
                                           fill_value=self.in_val))
        self.assertTupleEqual(self.out.shape, (3, 2))
        self.assertTrue((self.out == 100.0).all())
        return

    def test_base_round_list(self):
        """Test success of base rounding for list input."""
        self.out = math.base_round([self.in_val, self.in_val - 1.0])
        self.assertTupleEqual(self.out.shape, (2, ))
        self.assertTrue((self.out == 100.0).all())
        return


class TestThresholdFuncs(unittest.TestCase):
    """Tests for math functions that assist with thresholds."""

    def setUp(self):
        """Set up the test runs."""
        self.in_vals = np.arange(0.0, 5.0, 0.1)
        self.out = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.in_vals, self.out
        return

    def test_unique_threshold(self):
        """Test the success of downselecting unique points from an array."""
        self.out = math.unique_threshold(self.in_vals, thresh=1.0)

        # Evaluate the output
        self.assertLess(len(self.out), len(self.in_vals))
        self.assertTrue((self.out.astype(int) == self.out).all())
        return

    def test_unique_threshold_list(self):
        """Test the success of downselecting unique points from a list."""
        self.out = math.unique_threshold(list(self.in_vals), thresh=1.0)

        # Evaluate the output
        self.assertLess(len(self.out), len(self.in_vals))
        self.assertTrue((self.out.astype(int) == self.out).all())
        return

    def test_high_unique_threshold(self):
        """Test failure with a requested threshold that is too big."""
        with self.assertRaisesRegex(ValueError, 'Maximum threshold'):
            math.unique_threshold(self.in_vals, thresh=10.0)
        return

    def test_dif_thresh_float(self):
        """Test difference threshold calculation with float inputs."""
        self.out = math.set_dif_thresh(self.in_vals[0])
        self.assertEqual(self.out, self.in_vals[0] * 0.05)
        return

    def test_dif_thresh_array(self):
        """Test difference threshold calculation with array inputs."""
        self.out = math.set_dif_thresh(self.in_vals, percent=1.0)
        self.assertTupleEqual(self.out.shape, self.in_vals.shape)
        self.assertTrue((self.out == self.in_vals).all())
        return
