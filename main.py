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
sample_indices = (30, 0, 7)  # x position, y position, plateau number within frame

# Put bias and current arrays in real units!
characteristics = characterize_sweep_array(bias, current, margin=smoothing_margin, sample_sec=sample_sec)

# Analysis of single sample Isweep-Vsweep curve
sample_plateau = characteristics[sample_indices]
pprint(swept_probe_analysis(sample_plateau, probe_area, ion_type, visualize=True, plot_EEDF=True))
plt.show()
print("Done analyzing sample characteristic")

diagnostics_xarray = plasma_diagnostics(characteristics, probe_area, ion_type)
radial_plot(diagnostics_xarray, diagnostic='T_e', plot='contour')

# Note: The non-bimaxwellian plasmapy electron temperature seems to be the *reciprocal* of the correct value.

