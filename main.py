import matplotlib.pyplot as plt
from pprint import pprint

from getIVsweep import *
from characterization import *
from diagnostics import *
from plots import *
from netCDFreader import *

print("Imported helper files")

# Global parameters
sample_sec = (100 / 16 * 10 ** 6) ** (-1) * u.s  # Note that this is a small number. 10^6 is in the denominator
probe_area = (1. * u.mm) ** 2  # From MATLAB code
core_region = 26. * u.cm
ion_type = 'He-4+'
hdf5_filename = 'HDF5/8-3500A.hdf5'
save_filename = 'diagnostics.nc'
smoothing_margin = 10
# End of global parameters

# ** Set the below variable to True to open an existing diagnostic dataset from a NetCDF file
#    or False to create a new diagnostic dataset from the given HDF5 file. **
use_existing = False
# ** Set the below variable to True when creating a new diagnostic dataset to save the dataset to a NetCDF file. **
save_diagnostics = True

if use_existing:
    pass  # open existing NetCDF file
else:
    bias, current = get_isweep_vsweep(hdf5_filename)  # get isweep and vsweep arrays
    # Put bias and current arrays in real units!
    characteristics = characterize_sweep_array(bias, current, margin=smoothing_margin, sample_sec=sample_sec)
    # Analysis of single sample Isweep-Vsweep curve
    """
    sample_indices = (30, 0, 7)  # x position, y position, plateau number within frame
    sample_plateau = characteristics[sample_indices]
    pprint(swept_probe_analysis(sample_plateau, probe_area, ion_type, visualize=True, plot_EEDF=True, bimaxwellian=True))
    plt.show()
    print("Done analyzing sample characteristic")
    """
    diagnostics_xarray = plasma_diagnostics(characteristics, probe_area, ion_type, bimaxwellian=False)
    if save_diagnostics:
        pass  # save as NetCDF file

radial_plot(diagnostics_xarray, diagnostic='T_e', plot='contour')

# Note: The non-bimaxwellian plasmapy electron temperature seems to be the *reciprocal* of the correct value.
