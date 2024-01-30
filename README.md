# LAPD plasma analysis repository
### Description
This repository holds Python code for analyzing plasma physics experiment data. It reads and processes Langmuir probe and interferometry data from HDF5 files from UCLA's LAPD experiment facility.
### Managed by
This GitHub page is managed by Leo Murphy as part of a research project under Prof. Saskia Mordijck for William & Mary plasma physics.
### Use
Download all Python files to the same directory. Download desired HDF5 files and note their file paths. Change necessary parameters inside main.py before running. Make sure numpy, astropy, plasmapy, xarray, bottleneck, PyQt5, h5py, and bapsflib-2.0.0b3.dev11+g936e493 (alternatively: use command line pip install git+https://github.com/BaPSF/bapsflib/releases/tag/v2.0.0b) are installed. This code was written in Python 3.11.
