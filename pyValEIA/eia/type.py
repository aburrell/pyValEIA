#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Functions to specify EIA types."""

import numpy as np


def clean_type(state_array):
    """Simplifies EIA states into 4 base categories and 3 directions.

    Parameters
    ----------
    state_array : array-like
        Array of strings specifying the EIA state

    Returns
    -------
    base_types : array-like
        Simplifies EIA type strings into one of: flat, trough, peak, or EIA.
    base_dirs : array-like
        Simplifies EIA type strings into state directions: north, south, or
        neither.

    Raises
    ------
    ValueError
        If a `base_type` value cannot be established for any input value.

    See Also
    --------


    """
    base_types = []
    base_dirs = []

    for in_state in state_array:
        # Determine base type
        if 'eia' in in_state:
            base_types.append('eia')
        elif 'peak' in in_state:
            base_types.append('peak')
        elif in_state == 'trough':
            base_types.append('trough')
        elif 'flat' in in_state:
            base_types.append('flat')
        else:
            raise ValueError(
                'unknown EIA type encountered: {:}'.format(in_state))

        # Determine the base direction
        if 'north' in in_state:
            base_dirs.append('north')
        elif 'south' in in_state:
            base_dirs.append('south')
        else:
            base_dirs.append('neither')

    base_types = np.asarray(base_types)
    base_dirs = np.asarray(base_dirs)

    return base_types, base_dirs
