#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `utils.stats.skill_score`."""

import numpy as np
import unittest

from pyValEIA.stats import skill_score


class TestSkillScoreFuncs(unittest.TestCase):
    """Tests for skill score functions."""

    def setUp(self):
        """Set up the test runs."""
        self.truth_vals = np.arange(0, 10) 
        self.test_vals = np.ones(shape=self.truth_vals.shape)
        self.out = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.truth_vals, self.test_vals, self.out
        return

    def test_state_check_all_pos(self):
        """Test success for checking event state."""
        self.out = skill_score.state_check(self.truth_vals, self.test_vals,
                                           event_val=1)

        # Test the output values and shape
        self.assertTupleEqual(self.out.shape, np.asarray(self.test_vals).shape)
        self.assertEqual(sum(self.out == 'H'), 1)
        self.assertEqual(sum(self.out == 'F'), self.out.shape[0] - 1)
        return

    def test_state_check_all_neg(self):
        """Test success for checking event state."""
        self.out = skill_score.state_check(self.truth_vals, self.test_vals,
                                           event_val=0)

        # Test the output values and shape
        self.assertTupleEqual(self.out.shape, np.asarray(self.test_vals).shape)
        self.assertEqual(sum(self.out == 'M'), 1, msg="Output was: {:}".format(
            self.out))
        self.assertEqual(sum(self.out == 'C'), self.out.shape[0] - 1,
                         msg="Output was: {:}".format(self.out))
        return

    def test_state_check_bad_input(self):
        """Test failure for test and truth inputs of different sizes."""
        with self.assertRaisesRegex(ValueError, "Number of test values"):
            skill_score.state_check(self.truth_vals, self.test_vals[:-2])
        return

    def test_state_check_list_inputs(self):
        """Test success and failures with list inputs."""
        self.truth_vals = [val for val in self.truth_vals]
        self.test_vals = [val for val in self.test_vals]

        self.test_state_check_all_pos()
        self.test_state_check_all_neg()
        self.test_state_check_bad_input()
        return
