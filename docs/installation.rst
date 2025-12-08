.. _install:

Installation
============

The following instructions will allow you to install pyValEIA


.. _install-prereq:

Prerequisites
-------------

pyValEIA uses common Python modules, as well as modules developed by and for the
Space Physics community.  This module officially supports Python 3.10+.

 ============== =================
 Common modules Community modules
 ============== =================
  cartopy       aacgmv2
  cdflib        apexpy
  matplotlib    pydarn
  netCDF4       PyIRI
  numpy         pysatMadrigal
  pandas
  requests
  scipy
 ============== =================


.. _install-opt:


Installation Options
--------------------


.. _install-opt-pip:

PyPi
^^^^
All public pyValEIA releases will be made available through the PyPi package
manager *at a future time*.
::


   pip install pyValEIA



.. _install-opt-git:

Git Repository
^^^^^^^^^^^^^^
You can keep up to date with the latest changes at the git repository.

1. Clone the git repository
::


   git clone https://github.com/PACKAGE_REPOSITORY


2. Install pyValEIA:
   Change directories into the repository folder and run the pyproject.toml
   file. There are a few ways you can do this:

   A. Install on the system (will install locally without root privileges)::


        python -m build
	pip install .

   B. Install with the intent to develop locally::


        python -m build
	pip install -e .
