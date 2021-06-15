import matplotlib.pyplot as plt

from getIVsweep import *
from astropy import visualization
from pprint import pprint

print("Imported helper files")
# plateau = get_isweep_vsweep('HDF5/8-3500A.hdf5')
# sample_sec = (100 / 16 * 10 ** 6) ** (-1) * u.s  # Note that this is a small number. 10^6 is in the denominator
filename = 'HDF5/8-3500A.hdf5'

# Eliminate the hard-coded plateau range and the create_range_characteristic function; transfer unit conversion/etc.
""" plateau = create_ranged_characteristic(filename, 25005, 25620)
print("Finished creating I-V plasma Characteristic object")
smooth_plateau = smooth_characteristic(plateau, 7)
"""
"""
# debug
with visualization.quantity_support():
    plt.plot(plateau.bias, plateau.current)
    plt.plot(smooth_plateau.bias, smooth_plateau.current)
    # plt.plot(smooth_plateau.bias)
    # plateau.plot()
    # smooth_plateau.plot()
    plt.show()
"""

# Calculate the cylindrical probe surface area
# probe_area = 1/(1000)**2 (From MATLAB code)
probe_area = (1.*u.mm)**2

# pprint(swept_probe_analysis(characteristic, probe_area, 'He-4+', visualize=True, plot_EEDF=True))
# Make ion type customizable by user
# pprint(swept_probe_analysis(smooth_plateau, probe_area, 'He-4+', bimaxwellian=True, visualize=True, plot_EEDF=True))
# plt.show()

bias, current = get_isweep_vsweep(filename)
frames = isolate_plateaus(bias, current)
# print(split_plateaus(bias, current, frames)[0])
sample_indices = (30, 0, 7)  # x position, y position, plateau number within frame
split_bias, split_current, plateau_range = split_plateaus(bias, current, frames)
middle_bias = split_bias[sample_indices][plateau_range[sample_indices][0]:plateau_range[sample_indices][1]]
middle_current = split_current[sample_indices][plateau_range[sample_indices][0]:plateau_range[sample_indices][1]]
middle_plateau = Characteristic(middle_bias * 100 * u.V, middle_current * (-1. / 11) * u.A)
middle_plateau_smooth = smooth_characteristic(middle_plateau, 9)
# middle_plateau_smooth.plot()
# plt.show()
pprint(swept_probe_analysis(middle_plateau_smooth, probe_area, 'He-4+', bimaxwellian=False, visualize=True, plot_EEDF=True))
plt.show()

# Note: The (non-bimaxwellian) plasmapy electron temperature seems almost to be the *reciprocal* of the correct value?
# Attempt to make a (basic) contour or surface plot of electron temperature across positions to investigate further

