Package for validating the Equatorial Ionization Anomaly (EIA) within
ionospheric models against in situ plasma density data and Vertical
Total Electron Content (VTEC).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16884149.svg)](https://doi.org/10.5281/zenodo.16884149) [![PyPI version](https://badge.fury.io/py/pyvaleia.svg)](https://badge.fury.io/py/pyvaleia) [![Test Status](https://github.com/aburrell/pyValEIA/actions/workflows/main.yml/badge.svg)](https://github.com/aburrell/pyValEIA/actions/workflows/main.yml) [![Documentation Status](https://readthedocs.org/projects/pyValEIA/badge/?version=latest)](http://pyValEIA.readthedocs.io/en/latest/?badge=latest) [![Coverage Status](https://coveralls.io/repos/github/aburrell/pyValEIA/badge.svg?branch=main)](https://coveralls.io/github/aburrell/pyValEIA?branch=main)

Example
-------

To compare Swarm and pyIRI:

```
import datetime as dt
import pyValEIA

# Create new PyIRI files and compare to Swarm
# Set the directories for figures, EIA info files, and Swarm data
fig_dir='~/Plots/Model_SWARM_offsets'
daily_dir='~/Type_Files/Model_SWARM_offsets'
swarm_fdir = '~/swarm_data'
model_fdir = '~/model_data'

# Set the comparison day and time
stime1 = dt.datetime(2025, 4, 1, 0, 0)

# Download the Swarm A data
pyValEIA.io.download.download_and_unzip_swarm(stime1, 'A', swarm_fdir,
                                              f_end="0701")

# Create plots and daily comparison files between Swarm A and your model
daily_df = pyValEIA.plots.swarm_diagnostic_plots.model_swarm_mapplot(
    stime, swarm_fdir, model_fdir, 'Model_%Y%j.nc',
    mod_load_func=function_to_load_model_data, fig_dir=fig_dir,
    file_dir=daily_dir)

```

Notes
-----

This package is under active development and will be published in an upcoming
manuscript. When using the alpha version, we encourage you to contact one of
the authors for guidance or to provide suggestions for code development.