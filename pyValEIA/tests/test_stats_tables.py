#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `utils.stats.tables`."""

import unittest

from pyValEIA.stats import tables


class TestTablesFuncs(unittest.TestCase):
    """Tests for table functions."""

    def setUp(self):
        """Set up the test runs."""
        self.eia_type = 'eia'
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.eia_type
        return

    def test_style_df_table_bad_names(self):
        """Test raises ValueError with bad satellite names."""
        # TODO: initialize with a good data frame
        # with self.assertRaisesRegex(ValueError):
        #    tables.style_df_table(pd.DataFrame([]), self.eia_type, ['Viking'])
        return
