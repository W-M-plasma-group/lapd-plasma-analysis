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

bias, current, x, y = get_isweep_vsweep(hdf5_filename)

diagnostics_dataset = read_netcdf(open_filename) if use_existing else False  # the desired dataset, or False to use HDF5
if not diagnostics_dataset:  # diagnostic dataset not loaded; create new from HDF5 file
    characteristics = characterize_sweep_array(bias, current, x, y, margin=smoothing_margin, sample_sec=sample_sec)
    diagnostics_dataset = plasma_diagnostics(characteristics, probe_area, ion_type, bimaxwellian=False)
    if save_diagnostics:
        write_netcdf(diagnostics_dataset, save_filename)

radial_plot(diagnostics_dataset, diagnostic='n_e', plot='contour')
plt.show()

# Analysis of single sample Isweep-Vsweep curve
# """
sample_indices = (30, 0, 7)  # x position, y position, plateau number within frame
sample_plateau = characteristics[sample_indices].item()
pprint(swept_probe_analysis(sample_plateau, probe_area, ion_type, 
                            visualize=True, plot_EEDF=True, bimaxwellian=False))
plt.show()
print("Done analyzing sample characteristic")
# """

density_scaling, has_xy, electron_density = interferometry_calibration(
    diagnostics_dataset['n_e'], interferometry_filename,
    steady_state_start_plateau, steady_state_end_plateau, core_region=core_region)

# debug
# print(density_scaling, has_xy, electron_density, sep="\n")
electron_density.squeeze().plot.contourf(robust=True)
plt.show()

# Note: The non-bimaxwellian plasmapy electron temperature seems to be the *reciprocal* of the correct value.
