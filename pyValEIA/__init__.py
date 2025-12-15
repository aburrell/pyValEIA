#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# ----------------------------------------------------------------------------
"""Package to validate the EIA in model data against observations."""

import logging

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

# Define a logger object to allow easier log handling
logging.raiseExceptions = False
logger = logging.getLogger('pyValEIA_logger')

# Import the package modules and top-level classes
from pyValEIA import eia  # noqa F401
from pyValEIA import io  # noqa F401
from pyValEIA import plots  # noqa F401
from pyValEIA import stats  # noqa F401
from pyValEIA import utils  # noqa F401

# Define the global variables
__version__ = metadata.version('pyValEIA')
