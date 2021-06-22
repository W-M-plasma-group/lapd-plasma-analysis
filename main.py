import matplotlib.pyplot as plt

from getIVsweep import *
from astropy import visualization
from pprint import pprint

print("Imported helper files")

# Global parameters
# sample_sec = (100 / 16 * 10 ** 6) ** (-1) * u.s  # Note that this is a small number. 10^6 is in the denominator
probe_area = (1. * u.mm) ** 2  # From MATLAB code
ion_type = 'He-4+'
filename = 'HDF5/8-3500A.hdf5'

bias, current = get_isweep_vsweep(filename)
frames = isolate_plateaus(bias, current)
sample_indices = (30, 0, 7)  # x position, y position, plateau number within frame

"""
smooth_plateau = smooth_characteristic(create_ranged_characteristic(bias[sample_indices[:2]], 
                                                                    current[sample_indices[:2]],  32000, 35000), 7)
print("Finished creating I-V plasma Characteristic object")
# debug
with visualization.quantity_support():
    # plt.plot(plateau.current)
    smooth_plateau.plot()
    plt.show()
"""

# pprint(swept_probe_analysis(smooth_plateau, probe_area, 'He-4+', bimaxwellian=True, visualize=True, plot_EEDF=True))
# plt.show()

split_bias, split_current, plateau_ranges = split_plateaus(bias, current, frames)
characteristics = get_characteristic_array(split_bias, split_current, plateau_ranges)
middle_plateau_smooth = characteristics[sample_indices]

middle_plateau_smooth.plot()
plt.show()
pprint(swept_probe_analysis(middle_plateau_smooth, probe_area, ion_type, visualize=True, plot_EEDF=True))
plt.show()

# Note: The (non-bimaxwellian) plasmapy electron temperature seems almost to be the *reciprocal* of the correct value?
# Attempt to make a (basic) contour or surface plot of electron temperature across positions to investigate further

