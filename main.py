import matplotlib.pyplot as plt
from pprint import pprint

from getIVsweep import *
from characterization import *
from diagnostics import *
from plots import *
from netCDFaccess import *
from interferometry import *

print("Imported helper files")

# Global parameters
sample_sec = (100 / 16 * 10 ** 6) ** (-1) * u.s               # Note that this is small. 10^6 is in the denominator
probe_area = (1. * u.mm) ** 2                                 # From MATLAB code
core_region = 26. * u.cm                                      # From MATLAB code
ion_type = 'He-4+'
smoothing_margin = 10
steady_state_start_plateau, steady_state_end_plateau = 5, 11  # From MATLAB code
# End of global parameters

# File path names
hdf5_filename = 'HDF5/8-3500A.hdf5'
# save_filename = 'netCDF/diagnostic_dataset.nc'
save_filename = 'diagnostic_dataset.nc'
open_filename = save_filename            # write to and read from the same location
interferometry_filename = hdf5_filename  # interferometry data stored in same HDF5 file
# End of file path names

# File options
""" Set the below variable to True to open an existing diagnostic dataset from a NetCDF file
    or False to create a new diagnostic dataset from the given HDF5 file. """
use_existing = False
""" Set the below variable to True when creating a new diagnostic dataset to save the dataset to a NetCDF file. """
save_diagnostics = True
# End of file options


create_diagnostics = not use_existing
# Check if user path valid; if invalid, allow user to choose path to use or to create a new dataset
if use_existing:
    verified_path = verify_path(open_filename)
    if not verified_path:
        create_diagnostics = True

bias, current = get_isweep_vsweep(hdf5_filename)  # Needed for interferometry data; only need if interferometry wanted?
if not create_diagnostics:
    diagnostics_dataset = open_netcdf(open_filename)
else:
    # Put bias and current arrays in real units!
    characteristics = characterize_sweep_array(bias, current, margin=smoothing_margin, sample_sec=sample_sec)
    diagnostics_dataset = plasma_diagnostics(characteristics, probe_area, ion_type, bimaxwellian=False)
    if save_diagnostics:
        write_netcdf(diagnostics_dataset, save_filename)

radial_plot(diagnostics_dataset, diagnostic='T_e', plot='contour')

# Analysis of single sample Isweep-Vsweep curve
"""
sample_indices = (30, 0, 7)  # x position, y position, plateau number within frame
sample_plateau = characteristics[sample_indices]
pprint(swept_probe_analysis(sample_plateau, probe_area, ion_type, 
                            visualize=True, plot_EEDF=True, bimaxwellian=True))
plt.show()
print("Done analyzing sample characteristic")
"""

electron_density, density_scaling = interferometry_calibration(
    diagnostics_dataset['n_e'], diagnostics_dataset['T_e'], interferometry_filename,
    bias, current, steady_state_start_plateau, steady_state_end_plateau, core_region=core_region)

# Note: The non-bimaxwellian plasmapy electron temperature seems to be the *reciprocal* of the correct value.
