Package for validating the Equatorial Ionization Anomaly (EIA) within
ionospheric models against in situ plasma density data and Vertical
Total Electron Content (VTEC).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16884149.svg)](https://doi.org/10.5281/zenodo.16884149) [PYPI HERE] [![Test Status](https://github.com/aburrell/pyValEIA/actions/workflows/main.yml/badge.svg)](https://github.com/aburrell/pyValEIA/actions/workflows/main.yml) [![Documentation Status](https://readthedocs.org/projects/pyValEIA/badge/?version=latest)](http://pyValEIA.readthedocs.io/en/latest/?badge=latest) [![Coverage Status](https://coveralls.io/repos/github/aburrell/pyValEIA/badge.svg?branch=main)](https://coveralls.io/github/aburrell/pyValEIA?branch=main)

Example
-------

To compare Swarm and pyIRI:

```
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# Package modules
from SwarmPyIRI import PyIRI_NIMO_SWARM_plot

# Create new PyIRI files and compare to Swarm
# Set the directories for figures, EIA info files, and Swarm data
fig_dir='~/Plots/pyIRI_SWARM_offsets'
daily_dir='~/Type_Files/pyIRI_SWARM_offsets'
swarm_fdir = '~/swarm_data'

# Set the comparison day and time
stime1 = datetime(2020, 4, 15, 0, 0)

# Create new PyIRI files and compare to Swarm for a range of days (2)
for i in range(2):
    stime = stime1 + timedelta(days=i)
    print(stime)
    pdf_out = PyIRI_NIMO_SWARM_plot(stime, daily_dir, swarm_fdir, fig_on=True
        fig_save_dir=fig_dir, file_save_dir=daily_dir)

```

Notes
-----

This package is under active development and will be published in an upcoming
manuscript. When using the alpha version, we encourage you to contact one of
the authors for guidance or to provide suggestions for code development.