"""
The lapd-plasma-analysis repository was written by Leo Murphy based on code
written in MATLAB by Conor Perks (MIT) and using the PlasmaPy online code library.
Comments are added inline. A separate documentation page is not yet complete.
"""
from getIVsweep import *
from characterization import *
from diagnostics import *
from plots import *
from netCDFaccess import *
from interferometry import *
from neutrals import *
from setup import *
from bapsflib import lapd   # TODO move to getIVsweep.py

# User global parameters
sample_sec = (100 / 16 * 1e6) ** (-1) * u.s                      # Note that this is small. 10^6 is in the denominator
probe_area = (1. * u.mm) ** 2                                    # From MATLAB code
core_region = 26. * u.cm                                         # From MATLAB code
ion_type = 'He-4+'
bimaxwellian = False                                             # TODO rely on metadata to determine if bimaxwellian when opening file; save with different names
# TODO compare bimaxwellian n_e, n_i, etc. with nonbimaxwellian to determine if we need to store under different filenames or just in same file with different diagnostic names
smoothing_margin = 0                                             # Optimal values in range 0-25
steady_state_start_plateau, steady_state_end_plateau = 5, 11     # From MATLAB code
diagnostics_plotted = ['T_e_cold', 'T_e']                        # String or list of strings
# End of global parameters

# User file path names
hdf5_path = "/Users/leo/Plasma code/HDF5/16-2000A-redo.hdf5"   # Path of chosen HDF5 file; available under repository Releases
interferometry_filename = hdf5_path           # Interferometry data stored in same HDF5 file
netcdf_subfolder_name = "netcdf"              # Subfolder to save and read netcdf files; set to "" to use current folder
# End of file path names

# User file options
""" Set the below variable to True to open an existing diagnostic dataset from a NetCDF file
    or False to create a new diagnostic dataset from the given HDF5 file. """
use_existing = False
""" Set the below variable to True to save diagnostic data to a NetCDF file if a new one is created from HDF5 data. """
save_diagnostics = True
# End of file options

if __name__ == "__main__":
    # Build paths to save and open netcdf files in specified subfolder
    netcdf_subfolder_path = ensure_netcdf_directory(netcdf_subfolder_name)

    lapd_file = lapd.File(hdf5_path)
    full_netcdf_path = netcdf_path(lapd_file.info['file'], netcdf_subfolder_path)
    save_diagnostic_path = open_diagnostic_path = full_netcdf_path
    print("Diagnostic dataset will be saved to or opened from the path", repr(full_netcdf_path))

    # Read LAPD experimental parameters
    experimental_parameters, experimental_parameters_rounded = setup_lapd(hdf5_path)
    print("Experimental parameters:", {key: str(value) for key, value in experimental_parameters_rounded.items()})
    bias, current, x, y = get_isweep_vsweep(hdf5_path)
    # TODO create position matrix in diagnostic dataset to avoid having to reload and recalculate position data?

    # Create the diagnostics dataset by generating new or opening existing NetCDF file
    diagnostics_dataset = read_netcdf(open_diagnostic_path) if use_existing else None  # desired dataset or None to use HDF5
    if not diagnostics_dataset:  # diagnostic dataset not loaded; create new from HDF5 file
        characteristics = characterize_sweep_array(bias, current, x, y, margin=smoothing_margin, sample_sec=sample_sec)
        diagnostics_dataset = plasma_diagnostics(characteristics, probe_area, ion_type,
                                                 experimental_parameters, bimaxwellian=bimaxwellian)
        print("Bimaxwellian, main.py: ", diagnostics_dataset.attrs['bimaxwellian'])
        if save_diagnostics:
            write_netcdf(diagnostics_dataset, save_diagnostic_path)
    else:
        print(diagnostics_dataset.attrs['bimaxwellian'])

    # Print list of diagnostics generated
    # print("Plasma diagnostics:", [key for key in diagnostics_dataset.keys()])

    # Plot chosen diagnostics by linear position and time, then time-averaged diagnostics for each position
    linear_diagnostic_plot(diagnostics_dataset, diagnostics_plotted, plot='contour')
    linear_diagnostic_plot(diagnostics_dataset, diagnostics_plotted, plot='line')

    # TODO somehow separate bimaxwellian and non-bimaxwellian diagnostic data because they can be different!

    """
    print({diagnostic: diagnostics_dataset[diagnostic][(30, 0, 7)].values for diagnostic in diagnostics_dataset.keys()})
    print("Done analyzing sample characteristic")
    # """

    # Perform interferometry calibration for electron density
    electron_density, density_scaling = interferometry_calibration(
        diagnostics_dataset['n_e'], interferometry_filename,
        steady_state_start_plateau, steady_state_end_plateau, core_region=core_region)

    # Plot electron density data that was calibrated using interferometry data
    # linear_diagnostic_plot(xr.Dataset({"n_e_cal": electron_density}), "n_e_cal", plot="contour")
    linear_diagnostic_plot(xr.Dataset({"n_e_cal": electron_density}), "n_e_cal", plot="line")
    # linear_diagnostic_plot(xr.Dataset({"n_e_hot": electron_density * diagnostics_dataset['hot_fraction']}), "n_e_hot", plot="contour")
    if bimaxwellian:
        linear_diagnostic_plot(xr.Dataset({"n_e_hot": electron_density * diagnostics_dataset['hot_fraction']}), "n_e_hot", plot="line")

    # Calculation for pressure; in the future, this will consider only the steady-state region
    # Why is the pressure about twice what is expected as given in Overleaf document?
    electron_temperature = diagnostics_dataset['T_e_avg'] if bimaxwellian else diagnostics_dataset['T_e']
    pressure = (3 / 2) * electron_temperature * electron_density * (1. * u.eV * u.m ** -3).to(u.Pa)
    linear_diagnostic_plot(xr.Dataset({"p_e": pressure}), "p_e", plot="contour")
    linear_diagnostic_plot(xr.Dataset({"p_e": pressure}), "p_e", plot="line")
    # TODO RAISE ISSUE OF RECIPROCAL TEMPERATURE FOR NON-BIMAXWELLIAN TEMPERATURE

    # Plot neutral ratios (NOTE: INCOMPLETE)
    # neutral_ratio(diagnostics_dataset['n_e'], experimental_parameters, steady_state_start_plateau, steady_state_end_plateau)

    # Note: Plot generation code and neutral fraction analysis is incomplete.
    # Note: The non-bimaxwellian electron temperature calculated using the PlasmaPy code seems to be
    #    the *reciprocal* of the correct value. I may raise an issue on the PlasmaPy GitHub page.
