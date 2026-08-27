#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `utils.stats.tables`."""

import numpy as np
import pandas as pd
import unittest

from pyValEIA.stats import tables


class TestTablesFuncs(unittest.TestCase):
    """Tests for table functions."""

    def setUp(self):
        """Set up the test runs."""
        self.df = pd.DataFrame({'state': ['peak', 'flat', 'eia'],
                                'direction': ['south', 'neither', 'north'],
                                'type': ['peak_south', 'flat', 'eia_north'],
                                'GLon': [148.0, -44.0, 124.0],
                                'LT': [10.1, 22.3, 10.19],
                                'Sat': ['A', 'A', 'B'],
                                'skill': ['C', 'M', 'H']})
        self.tf = None
        self.out = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.df, self.tf, self.out
        return

    def test_style_df_table_bad_levels(self):
        """Test raises ValueError with too many levels."""
        # Create a table with too many levels
        self.tf = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [['A', 'B', 'C'], ['EIA', 'Non-EIA'], ['C', 'M', 'H', 'F']]),
            columns=pd.MultiIndex.from_product([['Model'], ['EIA', 'Non-EIA']]))

        # Ensure the correct error is raised
        with self.assertRaisesRegex(ValueError, "MultiIndex with two levels"):
            tables.style_df_table(self.tf)
        return

    def test_style_df_table_bad_states(self):
        """Test raises ValueError with too many state options."""
        # Create a table with too many levels
        self.tf = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [['A', 'B', 'C'], ['EIA', 'North-EIA', 'South-EIA']]),
            columns=pd.MultiIndex.from_product(
                [['Model'], ['EIA', 'North-EIA', 'South-EIA']]))

        # Ensure the correct error is raised
        with self.assertRaisesRegex(ValueError, "state with two options"):
            tables.style_df_table(self.tf)
        return

    def test_style_df_table_success(self):
        """Test successful dataframe table formatting."""
        # Create a table from the data frame
        self.tf = tables.decision_table_sat(self.df, set_style=False)

        # Style the table
        self.out = tables.style_df_table(self.tf)

        # Test the output data
        self.assertTrue((self.out.index == self.tf.index).all())
        self.assertTrue((self.out.columns == self.tf.columns).all())
        self.assertTrue((self.out.data.values == self.tf.values).all())

        # Test the formatting has key elements
        self.assertGreater(self.out.to_latex().find('true#e6ffe6'), 0)
        self.assertGreater(self.out.to_latex().find('false#ffe6e6'), 0)

        return

    def test_style_lss_table_bad_levels(self):
        """Test LSS styling raises ValueError with too many levels."""
        # Create a table with too many levels
        self.tf = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [['A', 'B', 'C'], ['EIA', 'Non-EIA'], ['C', 'M', 'H', 'F']]),
            columns=pd.MultiIndex.from_product([['Model'], ['EIA', 'Non-EIA']]))

        # Ensure the correct error is raised
        with self.assertRaisesRegex(ValueError, "MultiIndex with two levels"):
            tables.style_df_table(self.tf)
        return

    def test_style_lss_table_success(self):
        """Test successful LSS table formatting."""
        # Create a table from the data frame
        self.tf = tables.lss_table_sat(self.df, self.df, set_style=False)

        # Style the table
        self.out = tables.style_lss_table(self.tf)

        # Test the output data
        self.assertTrue((self.out.index == self.tf.index).all())
        self.assertTrue((self.out.columns == self.tf.columns).all())
        self.assertTrue((self.out.data.values == self.tf.values).all())

        # Test the formatting has key elements
        self.assertGreater(self.out.to_latex().find(
            'level0.row02px solid black'), 0)

        return

    def test_decision_table_sat_success(self):
        """Test successful creation of a decision table from states."""

        for sats in [None, ['A'], ['NotSwarm']]:
            with self.subTest(sats=sats):
                # Create the table
                self.tf = tables.decision_table_sat(self.df, sats=sats)

                if sats is None:
                    self.assertEqual(len(self.tf.index.levels[0]),
                                     len(np.unique(self.df['Sat'])))
                else:
                    self.assertEqual(len(self.tf.index.levels[0]), 1)

                    if sats[0] not in self.df['Sat'].values:
                        self.assertEqual(self.tf.values.sum(), 0)
                    else:
                        self.assertEqual(self.tf.values.sum(), 2)
        return

    def test_decision_table_model_name(self):
        """Test successful model name labeling in decision table."""
        # Create the table
        self.tf = tables.decision_table_sat(self.df, model_name='Best')

        # Test the first level of the columns for the model name
        self.assertListEqual(list(self.tf.columns.levels[0]), ['Best'])
        return

    def test_decision_table_const_name(self):
        """Test successful constellation name labeling in decision table."""
        # Create the table
        self.tf = tables.decision_table_sat(self.df, const_name='Best')

        # Test the first level index for the constellation name
        self.assertEqual(self.tf.index.levels[0][0].find('Best'), 0)
        return

    def test_decision_table_type_name(self):
        """Test successful EIA state name labeling in decision table."""
        # Create the table
        self.tf = tables.decision_table_sat(self.df, eia_type='Best')

        # Test the indices and columns for the state labelling
        for item in ['Best', 'Non-Best']:
            self.assertTrue(item in self.tf.index.levels[1])
            self.assertTrue(item in self.tf.index.levels[1])
        return

    def test_lss_table_sat_success(self):
        """Test successful building of an LSS table."""

        for sats in [None, ['A'], ['NotSwarm']]:
            with self.subTest(sats=sats):
                # Create the table
                self.tf = tables.lss_table_sat(self.df, self.df, sats=sats)

                if sats is None:
                    self.assertEqual(len(self.tf.index.levels[0]),
                                     len(np.unique(self.df['Sat'])))
                    self.assertEqual(self.tf.values[0].sum(), 0)
                    self.assertEqual(self.tf.values[-1].sum(), 2)
                else:
                    self.assertEqual(len(self.tf.index.levels[0]), 1)

                    if sats[0] not in self.df['Sat'].values:
                        self.assertTrue(np.isinf(self.tf.values).all())
                    else:
                        self.assertEqual(self.tf.values[0].sum(), 0)
                        self.assertEqual(self.tf.values[-1].sum(), -1)
        return

    def test_lss_table_model_name(self):
        """Test successful model name labeling in decision table."""
        # Create the table
        self.tf = tables.lss_table_sat(self.df, self.df, model1_name='Best',
                                       model2_name='Worst')

        # Test the first level of the columns for the model name
        self.assertListEqual(list(self.tf.columns), ['Best', 'Worst'])
        return

    def test_lss_table_const_name(self):
        """Test successful constellation name labeling in decision table."""
        # Create the table
        self.tf = tables.lss_table_sat(self.df, self.df, const_name='Best')

        # Test the first level index for the constellation name
        self.assertEqual(self.tf.index.levels[0][0].find('Best'), 0)
        return
