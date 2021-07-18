import matplotlib.pyplot as plt
from getIVsweep import *
from pprint import pprint
from diagnostics import *
from plots import *

print("Imported helper files")

# Global parameters
sample_sec = (100 / 16 * 10 ** 6) ** (-1) * u.s  # Note that this is a small number. 10^6 is in the denominator
probe_area = (1. * u.mm) ** 2  # From MATLAB code
ion_type = 'He-4+'
filename = 'HDF5/8-3500A.hdf5'
smoothing_margin = 10
# End of global parameters

bias, current = get_isweep_vsweep(filename)  # get isweep and vsweep arrays
adjusted_bias, smooth_current = smooth_current_array(bias, current, margin=smoothing_margin)  # 0.02 s
plateau_ranges = isolate_plateaus(adjusted_bias, smooth_current, margin=smoothing_margin)

sample_indices = (30, 0, 7)  # x position, y position, plateau number within frame

# debug
"""
# pprint(swept_probe_analysis(smooth_plateau, probe_area, 'He-4+', bimaxwellian=True, visualize=True, plot_EEDF=True))
plt.plot(adjusted_bias[sample_indices[:2]], 'b-',
         plateau_ranges[sample_indices[0], sample_indices[1], :, 0], adjusted_bias[sample_indices[0], sample_indices[1], plateau_ranges[sample_indices[0], sample_indices[1], :, 0]], 'go',
         plateau_ranges[sample_indices[0], sample_indices[1], :, 1], adjusted_bias[sample_indices[0], sample_indices[1], plateau_ranges[sample_indices[0], sample_indices[1], :, 1]], 'yo')
plt.show()
# """


time_array = get_time_array(plateau_ranges, sample_sec)
# characteristics = get_characteristic_array(bias, current, plateau_ranges, 10)
characteristics = get_characteristic_array(adjusted_bias, smooth_current, plateau_ranges)
sample_plateau_smooth = characteristics[sample_indices]

# Analysis of single sample Isweep-Vsweep curve
# sample_plateau_smooth.plot()
pprint(swept_probe_analysis(sample_plateau_smooth, probe_area, ion_type, visualize=True, plot_EEDF=True))
plt.show()
print("Done analyzing sample characteristic")
# for characteristic in characteristics:
#     pprint(swept_probe_analysis(characteristic, probe_area, ion_type, visualize=False, plot_EEDF=False))
    # plt.show()

diagnostics_xarray = plasma_diagnostics(characteristics, probe_area, ion_type)
plt.show()

plot_ne_te(diagnostics_xarray)

# Note: The (non-bimaxwellian) plasmapy electron temperature seems almost to be the *reciprocal* of the correct value?
# Attempt to make a (basic) contour or surface plot of electron temperature across positions to investigate further
