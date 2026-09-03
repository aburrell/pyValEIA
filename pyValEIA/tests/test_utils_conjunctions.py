#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests for functions in `utils.conjunctions`."""

import datetime as dt
import numpy as np
import pandas as pd
import unittest

from pyValEIA.utils import conjunctions


class TestSwarmConjFuncs(unittest.TestCase):
    """Tests for functions that support model-data conjunctions."""

    def setUp(self):
        """Set up the test runs."""
        self.swarm_alt = {'A': 462.0, 'B': 511.0, 'C': 462.0}
        self.mod_dict = {
            'time': np.array([dt.datetime(2013, 12, 1)
                              + dt.timedelta(minutes=15 * i)
                              for i in range(20)]),
            'glon': np.arange(-180, 180.1, 15),
            'glat': np.arange(-90, 90.1, 1),
            'alt': np.arange(100, 1500, 50)}
        self.mod_dict['dene'] = np.ones(shape=(self.mod_dict['time'].shape[0],
                                               self.mod_dict['alt'].shape[0],
                                               self.mod_dict['glat'].shape[0],
                                               self.mod_dict['glon'].shape[0]))
        self.mod_dict['hmf2'] = np.full(shape=(self.mod_dict['time'].shape[0],
                                               self.mod_dict['glat'].shape[0],
                                               self.mod_dict['glon'].shape[0]),
                                        fill_value=350.0)
        self.mod_dict['nmf2'] = np.ones(shape=self.mod_dict['hmf2'].shape)
        self.swarm_dat = pd.DataFrame({
            'Time': np.array([dt.datetime(2013, 12, 1)
                              + dt.timedelta(minutes=1 * i)
                              for i in range(20)]),
            'Longitude': np.linspace(40.0, 42.0, 20),
            'Mag_Lat': np.linspace(-20.0, 20.0, 20),
            'Altitude': np.full(shape=(20,), fill_value=self.swarm_alt['A'])})
        self.out = None
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.swarm_alt, self.mod_dict, self.swarm_dat, self.out
        return

    def test_set_swarm_alt(self):
        """Test successful `set_swarm_alt` calls."""
        for sat_id in ['a', 'A', 'b', 'B', 'c', 'C']:
            with self.subTest(sat_id=sat_id):
                # Get the altitude for this satellite
                self.out = conjunctions.set_swarm_alt(sat_id)

                # Evaluate the output
                self.assertEqual(self.out, self.swarm_alt[sat_id.upper()])
        return

    def test_set_swarm_alt_bad_sat_id(self):
        """Test `set_swarm_alt` raises ValueError with a bad sat_id."""
        with self.assertRaisesRegex(ValueError, "unknown Swarm satellite"):
            conjunctions.set_swarm_alt('NotASatID')
        return

    def test_swarm_conjunction_geo(self):
        """Test a successful run of `swarm_conjunction` with geo inputs."""
        # Run the conjunction with the defaults
        self.out = conjunctions.swarm_conjunction(self.mod_dict,
                                                  self.swarm_dat)

        # Test the outputs
        self.assertEqual(len(self.out), 2)

        # Test the first output element
        self.assertTrue(np.all([self.swarm_dat['Time'][0] == otime[0]
                                for otime in self.out[0]['Time']]))
        self.assertLessEqual(self.out[0]['Mag_Lat'].max(),
                             self.swarm_dat['Mag_Lat'].max())
        self.assertGreaterEqual(self.out[0]['Mag_Lat'].min(),
                                self.swarm_dat['Mag_Lat'].min())
        self.assertEqual(self.out[0]['alt'][0],
                         self.mod_dict['hmf2'][0, 0, 0])

        # Test the second output element
        self.assertTrue(np.all(self.out[1]['nmf2'] == self.mod_dict['nmf2'][0]))
        self.assertTrue(np.all(self.out[1]['glon'] == self.mod_dict['glon']))
        self.assertTrue(np.all(self.out[1]['glat'] == self.mod_dict['glat']))
        return

    def test_swarm_conjunction_mag(self):
        """Test a successful run of `swarm_conjunction` with mag inputs."""
        # Run the conjunction with the defaults
        self.out = conjunctions.swarm_conjunction(self.mod_dict,
                                                  self.swarm_dat,
                                                  mod_loc_type='mag')

        # Test the outputs
        self.assertEqual(len(self.out), 2)

        # Test the first output element
        self.assertTrue(np.all([self.swarm_dat['Time'][0] == otime[0]
                                for otime in self.out[0]['Time']]))
        self.assertEqual(self.out[0]['Mag_Lat'].max(),
                         self.swarm_dat['Mag_Lat'].max())
        self.assertEqual(self.out[0]['Mag_Lat'].min(),
                         self.swarm_dat['Mag_Lat'].min())
        self.assertEqual(self.out[0]['alt'][0],
                         self.mod_dict['hmf2'][0, 0, 0])

        # Test the second output element
        self.assertTrue(np.all(self.out[1]['nmf2'] == self.mod_dict['nmf2'][0]))
        self.assertTrue(np.all(self.out[1]['glon'] == 45))
        self.assertTrue(np.all(self.out[1]['glat'] == self.mod_dict['glat']))
        return

    def test_swarm_conjunction_swarm_alt(self):
        """Test a successful run of `swarm_conjunction` with Swarm alt kwarg."""
        # Run the conjunction with the defaults
        self.out = conjunctions.swarm_conjunction(self.mod_dict,
                                                  self.swarm_dat, alt_str='A')

        # Test the outputs
        self.assertEqual(len(self.out), 2)

        # Test the first output element
        self.assertTrue(np.all([self.swarm_dat['Time'][0] == otime[0]
                                for otime in self.out[0]['Time']]))
        self.assertLessEqual(self.out[0]['Mag_Lat'].max(),
                             self.swarm_dat['Mag_Lat'].max())
        self.assertGreaterEqual(self.out[0]['Mag_Lat'].min(),
                                self.swarm_dat['Mag_Lat'].min())
        self.assertLessEqual(abs(self.out[0]['alt'][0] - self.swarm_alt['A']),
                             50)

        # Test the second output element
        self.assertTrue(np.all(self.out[1]['nmf2'] == self.mod_dict['nmf2'][0]))
        self.assertTrue(np.all(self.out[1]['glon'] == self.mod_dict['glon']))
        self.assertTrue(np.all(self.out[1]['glat'] == self.mod_dict['glat']))
        return

    def test_swarm_conjunction_alt_inc(self):
        """Test a successful run of `swarm_conjunction` with alt increase."""
        # Run the conjunction with the defaults
        self.out = conjunctions.swarm_conjunction(self.mod_dict,
                                                  self.swarm_dat, inc=100.0)

        # Test the outputs
        self.assertEqual(len(self.out), 2)

        # Test the first output element
        self.assertTrue(np.all([self.swarm_dat['Time'][0] == otime[0]
                                for otime in self.out[0]['Time']]))
        self.assertLessEqual(self.out[0]['Mag_Lat'].max(),
                             self.swarm_dat['Mag_Lat'].max())
        self.assertGreaterEqual(self.out[0]['Mag_Lat'].min(),
                                self.swarm_dat['Mag_Lat'].min())
        self.assertEqual(self.out[0]['alt'][0],
                         self.mod_dict['hmf2'][0, 0, 0] + 100)

        # Test the second output element
        self.assertTrue(np.all(self.out[1]['nmf2'] == self.mod_dict['nmf2'][0]))
        self.assertTrue(np.all(self.out[1]['glon'] == self.mod_dict['glon']))
        self.assertTrue(np.all(self.out[1]['glat'] == self.mod_dict['glat']))
        return

    def test_swarm_conjunction_tdif(self):
        """Test a successful run of `swarm_conjunction` with large tdif."""
        # Alter the Swarm times
        self.swarm_dat['Time'] = self.swarm_dat['Time'] + dt.timedelta(hours=6)
        
        # Run the conjunction with the defaults
        self.out = conjunctions.swarm_conjunction(self.mod_dict,
                                                  self.swarm_dat, max_tdif=360)

        # Test the outputs
        self.assertEqual(len(self.out), 2)

        # Test the first output element
        self.assertTrue(np.all([self.mod_dict['time'][-1] == otime[0]
                                for otime in self.out[0]['Time']]))
        self.assertLessEqual(self.out[0]['Mag_Lat'].max(),
                             self.swarm_dat['Mag_Lat'].max())
        self.assertGreaterEqual(self.out[0]['Mag_Lat'].min(),
                                self.swarm_dat['Mag_Lat'].min())
        self.assertEqual(self.out[0]['alt'][0], self.mod_dict['hmf2'][0, 0, 0])

        # Test the second output element
        self.assertTrue(np.all(self.out[1]['nmf2'] == self.mod_dict['nmf2'][0]))
        self.assertTrue(np.all(self.out[1]['glon'] == self.mod_dict['glon']))
        self.assertTrue(np.all(self.out[1]['glat'] == self.mod_dict['glat']))
        return

    def test_swarm_conjunction_offset(self):
        """Test a successful run of `swarm_conjunction` with a time offset."""
        # Alter the Swarm times
        self.swarm_dat['Time'] = self.swarm_dat['Time'] + dt.timedelta(days=1)

        # Run the conjunction with the defaults
        self.out = conjunctions.swarm_conjunction(self.mod_dict, self.swarm_dat,
                                                  offset=-1)

        # Test the outputs
        self.assertEqual(len(self.out), 2)

        # Test the first output element
        self.assertTrue(np.all([self.mod_dict['time'][0] == otime[0]
                                for otime in self.out[0]['Time']]))
        self.assertLessEqual(self.out[0]['Mag_Lat'].max(),
                             self.swarm_dat['Mag_Lat'].max())
        self.assertGreaterEqual(self.out[0]['Mag_Lat'].min(),
                                self.swarm_dat['Mag_Lat'].min())
        self.assertEqual(self.out[0]['alt'][0], self.mod_dict['hmf2'][0, 0, 0])

        # Test the second output element
        self.assertTrue(np.all(self.out[1]['nmf2'] == self.mod_dict['nmf2'][0]))
        self.assertTrue(np.all(self.out[1]['glon'] == self.mod_dict['glon']))
        self.assertTrue(np.all(self.out[1]['glat'] == self.mod_dict['glat']))
        return

    def test_swarm_conjunction_bad_alt(self):
        """Test `swarm_conjunction` raises ValueError with bad altitude."""
        # Alter the Swarm altitudes
        self.swarm_dat['Altitude'] += 1000.0
        
        # Run the conjunction and evaluate error
        with self.assertRaisesRegex(ValueError, 'not reasonable for Swarm'):
            conjunctions.swarm_conjunction(self.mod_dict, self.swarm_dat)
        return

    def test_swarm_conjunction_bad_time(self):
        """Test a successful run of `swarm_conjunction` with bad times."""
        # Alter the Swarm times
        self.swarm_dat['Time'] = self.swarm_dat['Time'] + dt.timedelta(days=1)

        # Run the conjunction and evaluate error
        with self.assertRaisesRegex(ValueError, '> 15 min'):
            conjunctions.swarm_conjunction(self.mod_dict, self.swarm_dat)
        return

    def test_swarm_conjunction_bad_coord(self):
        """Test a successful run of `swarm_conjunction` with bad coord type."""
        # Run the conjunction and evaluate error
        with self.assertRaisesRegex(ValueError, 'unknown coordinate type'):
            conjunctions.swarm_conjunction(self.mod_dict, self.swarm_dat,
                                           mod_loc_type='NotACoord')
        return
