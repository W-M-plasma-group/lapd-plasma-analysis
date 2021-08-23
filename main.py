import matplotlib.pyplot as plt

from getIVsweep import *
from characterization import *
from diagnostics import *
from plots import *
from netCDFaccess import *
from interferometry import *
from neutrals import *
from setup import *

print("Imported helper files")

# User global parameters
sample_sec = (100 / 16 * 1e6) ** (-1) * u.s               # Note that this is small. 10^6 is in the denominator
probe_area = (1. * u.mm) ** 2                                 # From MATLAB code
core_region = 26. * u.cm                                      # From MATLAB code
ion_type = 'He-4+'
bimaxwellian = False
smoothing_margin = 10
steady_state_start_plateau, steady_state_end_plateau = 5, 11  # From MATLAB code
# End of global parameters

# User file path names
hdf5_path = "/Users/leo/Plasma code/HDF5/8-3500A.hdf5"
interferometry_filename = hdf5_path  # interferometry data stored in same HDF5 file
save_diagnostic_filename = "diagnostic_dataset.nc"    # Note: file saves to subfolder 'netcdf' to be created if needed
open_diagnostic_filename = save_diagnostic_filename   # write to and read from the same location
netcdf_subfolder_name = "netcdf"                      # subfolder to save and read netcdf files; "" for current folder
# End of file path names

# User file options
""" Set the below variable to True to open an existing diagnostic dataset from a NetCDF file
    or False to create a new diagnostic dataset from the given HDF5 file. """
use_existing = True
""" Set the below variable to True when creating a new diagnostic dataset to save the dataset to a NetCDF file. """
save_diagnostics = True
# End of file options


# Establish paths to create files in specified netcdf subfolder
netcdf_subfolder_path = ensure_netcdf_directory(netcdf_subfolder_name)
save_diagnostic_path = netcdf_path(save_diagnostic_filename, netcdf_subfolder_path, bimaxwellian)
open_diagnostic_path = netcdf_path(open_diagnostic_filename, netcdf_subfolder_path, bimaxwellian)

experimental_parameters = setup_lapd(hdf5_path)
print("Experimental parameters:", {key: str(value) for key, value in experimental_parameters.items()})
bias, current, x, y = get_isweep_vsweep(hdf5_path)

diagnostics_dataset = read_netcdf(open_diagnostic_path) if use_existing else None  # desired dataset or None to use HDF5
if not diagnostics_dataset:  # diagnostic dataset not loaded; create new from HDF5 file
    characteristics = characterize_sweep_array(bias, current, x, y, margin=smoothing_margin, sample_sec=sample_sec)
    diagnostics_dataset = plasma_diagnostics(characteristics, probe_area, ion_type, bimaxwellian=bimaxwellian)
    if save_diagnostics:
        write_netcdf(diagnostics_dataset, save_diagnostic_path)

radial_diagnostic_plot(diagnostics_dataset, diagnostic='T_e', plot='contour')
plt.show()

# Analysis of single sample Isweep-Vsweep curve
"""
sample_indices = (30, 0, 7)  # x position, y position, plateau number within frame
sample_plateau = characteristics[sample_indices].item()
print(swept_probe_analysis(sample_plateau, probe_area, ion_type, 
                            visualize=True, plot_EEDF=True, bimaxwellian=False))
plt.show()
print("Done analyzing sample characteristic")
# """

electron_density, density_scaling = interferometry_calibration(
    diagnostics_dataset['n_e'], interferometry_filename,
    steady_state_start_plateau, steady_state_end_plateau, core_region=core_region)

# debug
# print(density_scaling, has_xy, electron_density, sep="\n")
electron_density.squeeze().plot.contourf(robust=True)
plt.show()

neutral_ratio(diagnostics_dataset['n_e'], experimental_parameters, steady_state_start_plateau, steady_state_end_plateau)
plt.show()

# TODO Finish adding code from the MATLAB main method to Python code!

# Note: The non-bimaxwellian plasmapy electron temperature seems to be the *reciprocal* of the correct value.
