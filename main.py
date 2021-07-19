import matplotlib.pyplot as plt
from pprint import pprint

from getIVsweep import *
from characterization import *
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
# adjusted_bias, smooth_current = smooth_current_array(bias, current, margin=smoothing_margin)  # 0.02 s

sample_indices = (30, 0, 7)  # x position, y position, plateau number within frame

# debug
"""
# pprint(swept_probe_analysis(smooth_plateau, probe_area, 'He-4+', bimaxwellian=True, visualize=True, plot_EEDF=True))
plt.plot(adjusted_bias[sample_indices[:2]], 'b-',
         plateau_ranges[sample_indices[0], sample_indices[1], :, 0], adjusted_bias[sample_indices[0], sample_indices[1], plateau_ranges[sample_indices[0], sample_indices[1], :, 0]], 'go',
         plateau_ranges[sample_indices[0], sample_indices[1], :, 1], adjusted_bias[sample_indices[0], sample_indices[1], plateau_ranges[sample_indices[0], sample_indices[1], :, 1]], 'yo')
plt.show()
# """



# characteristics = get_characteristic_array(bias, current, plateau_ranges, 10)
# Put bias and current arrays in real units!
characteristics = characterize_sweep_array(bias, current, margin=smoothing_margin, sample_sec=sample_sec)
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
# plt.show()

radial_plot(diagnostics_xarray, diagnostic='T_e', plot='contour')

# Note: The non-bimaxwellian plasmapy electron temperature seems to be the *reciprocal* of the correct value.

