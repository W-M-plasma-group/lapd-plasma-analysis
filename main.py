# Test file for getting accustomed to Python translation.

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os

from pprint import pprint

from plasmapy.diagnostics.langmuir import Characteristic, swept_probe_analysis
import plasmapy.plasma.sources.openpmd_hdf5 as pmd

path = os.path.join(os.curdir, "HDF5", "8-3500A.hdf5")
contents = pmd.HDF5Reader(path)

# bias, current = np.load(path)

# Create the Characteristic object, taking into account the correct units
# characteristic = Characteristic(u.Quantity(bias,  u.V), u.Quantity(current, u.A))


"""
# Calculate the cylindrical probe surface area
probe_length = 1.145 * u.mm
probe_diameter = 1.57 * u.mm
probe_area = (probe_length * np.pi * probe_diameter +
              np.pi * 0.25 * probe_diameter**2)
"""
probe_area = 1/(1000)^2

"""
pprint(swept_probe_analysis(characteristic,
                           probe_area, 'He-4+',
                           visualize=True,
                           plot_EEDF=True))
"""

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
