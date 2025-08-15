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
from pyValEIA import download_swarm  # noqa F401
from pyValEIA import EIA_type_detection  # noqa F401
from pyValEIA import Load_NIMO2  # noqa F401
from pyValEIA import Load_Swarm2  # noqa F401
from pyValEIA import Mad_Stats  # noqa F401
from pyValEIA import Madrigal_NIMO2  # noqa F401
from pyValEIA import NIMO_Swarm_Map_Plotting  # noqa F401
from pyValEIA import NIMO_SWARM_single  # noqa F401
from pyValEIA import offset_codes  # noqa F401
from pyValEIA import open_daily_files  # noqa F401
from pyValEIA import paper_plotting  # noqa F401
from pyValEIA import swarm_panel_ax  # noqa F401
from pyValEIA import Swarm_Stats  # noqa F401
from pyValEIA import SwarmPyIRI  # noqa F401


# Define the global variables
__version__ = metadata.version('pyValEIA')
