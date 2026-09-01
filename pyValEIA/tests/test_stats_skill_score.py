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

    def evaluate_coin_toss(self, event_state):
        """Evaluate the output based on a two-element, identical event state.

        Parameters
        ----------
        event_state : str
            Value within the event states

        """
        out_states = {'H': (1, 1, 0, 0), 'M': (1, 1, 1, 1), 'C': (0, 0, 1, 1),
                      'F': (0, 0, 0, 0)}

        self.assertTupleEqual(self.out, out_states[event_state])
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

    def test_coin_toss_state_array(self):
        """Test performance of the coin toss output with array inputs."""
        for event_state in ['H', 'M', 'C', 'F']:
            event_states = np.full(shape=(2,), fill_value=event_state)
            with self.subTest(event_states=event_states):
                self.out = skill_score.coin_toss_state(event_states)

                # Evaluate the output
                self.evaluate_coin_toss(event_state)
        return

    def test_coin_toss_state_list(self):
        """Test performance of the coin toss output with list inputs."""
        for event_state in ['H', 'M', 'C', 'F']:
            event_states = [event_state, event_state]
            with self.subTest(event_states=event_states):
                self.out = skill_score.coin_toss_state(event_states)

                # Evaluate the output
                self.evaluate_coin_toss(event_state)
        return

    def test_leimohn_skill_score(self):
        """Calculate the Leimohn skill score."""
        # Get event states
        self.test_state_check_all_pos()

        # Calculate the Leimohn skill scores
        self.out = skill_score.liemohn_skill_score(self.out)

        # Test the output
        self.assertEqual(len(self.out), 4)
        self.assertEqual(self.out[0], -4.0)
        self.assertEqual(self.out[1], 0.05)
        self.assertEqual(self.out[2], -0.8)
        self.assertEqual(self.out[3], 0.01)
        return

    def test_leimohn_skill_score_coin(self):
        """Calculate the Leimohn skill score for the coin toss."""
        # Get event states
        self.test_state_check_all_neg()

        # Calculate the Leimohn skill scores
        self.out = skill_score.liemohn_skill_score(self.out, coin=True)

        # Test the output
        self.assertEqual(len(self.out), 4)
        self.assertEqual(self.out[0], -2.3)
        self.assertEqual(self.out[1], -0.05)
        self.assertAlmostEqual(self.out[2], -0.09090909091)
        self.assertAlmostEqual(self.out[3], -0.09090909091)
        return

    def test_leimohn_skill_score_coin_infinite(self):
        """Calculate the coint-toss Leimohn skill score with no results."""

        # Calculate the Leimohn skill scores
        self.out = skill_score.liemohn_skill_score(['F', 'F'], coin=True)

        # Test the output
        self.assertEqual(len(self.out), 4)
        self.assertTrue(np.isinf(self.out).all())
        return

    def test_calc_pc_and_csi(self):
        """Calculate the percent correct and critical success index."""
        # Get event states
        self.test_state_check_all_pos()

        # Calculate the scores
        self.out = skill_score.calc_pc_and_csi(self.out)

        # Test the output
        self.assertEqual(len(self.out), 2)
        self.assertEqual(self.out[0], 0.1)
        self.assertEqual(self.out[1], 0.1)
        return

    def test_calc_pc_and_csi_coin(self):
        """Calculate the coin-toss percent correct and critical success."""
        # Get event states
        self.test_state_check_all_pos()

        # Calculate the scores
        self.out = skill_score.calc_pc_and_csi(self.out, coin=True)

        # Test the output
        self.assertEqual(len(self.out), 2)
        self.assertEqual(self.out[0], 0.0)
        self.assertEqual(self.out[1], 0.0)
        return

    def test_calc_pc_and_csi_infinite(self):
        """Calculate the percent correct and critical success with no events."""
        # Calculate the Leimohn skill scores
        self.out = skill_score.calc_pc_and_csi(['F', 'F'], coin=True)

        # Test the output
        self.assertEqual(len(self.out), 2)
        self.assertTrue(np.isinf(self.out).all())
        return
