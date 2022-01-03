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

# User global parameters
sample_sec = (100 / 16 * 1e6) ** (-1) * u.s                   # Note that this is small. 10^6 is in the denominator
probe_area = (1. * u.mm) ** 2                                 # From MATLAB code
core_region = 26. * u.cm                                      # From MATLAB code
ion_type = 'He-4+'
bimaxwellian = False
smoothing_margin = 0                                          # Optimal values in range 0-25
steady_state_start_plateau, steady_state_end_plateau = 5, 11  # From MATLAB code
# End of global parameters

# User file path names
hdf5_path = "/Users/leo/Plasma code/HDF5/9-4000A.hdf5"   # Path of chosen HDF5 file; available under repository Releases
interferometry_filename = hdf5_path           # Interferometry data stored in same HDF5 file
save_diagnostic_name = "diagnostic_dataset"   # File is saved to a subfolder (named below) to be created if necessary
open_diagnostic_name = save_diagnostic_name   # Write diagnostics to and read diagnostics from the same location
netcdf_subfolder_name = "netcdf"              # Subfolder to save and read netcdf files; set to "" to use current folder
# End of file path names

# User file options
""" Set the below variable to True to open an existing diagnostic dataset from a NetCDF file
    or False to create a new diagnostic dataset from the given HDF5 file. """
use_existing = True
""" Set the below variable to True to save diagnostic data to a NetCDF file if a new one is created from HDF5 data. """
save_diagnostics = True
# End of file options

if __name__ == "__main__":
    # Build paths to save and open netcdf files in specified subfolder
    netcdf_subfolder_path = ensure_netcdf_directory(netcdf_subfolder_name)
    save_diagnostic_path = netcdf_path(save_diagnostic_name, netcdf_subfolder_path, bimaxwellian)
    open_diagnostic_path = netcdf_path(open_diagnostic_name, netcdf_subfolder_path, bimaxwellian)

    # Read LAPD experimental parameters
    experimental_parameters, experimental_parameters_rounded = setup_lapd(hdf5_path)
    print("Experimental parameters:", {key: str(value) for key, value in experimental_parameters_rounded.items()})
    bias, current, x, y = get_isweep_vsweep(hdf5_path)

    # Create the diagnostics dataset by generating new or opening existing NetCDF file
    diagnostics_dataset = read_netcdf(open_diagnostic_path) if use_existing else None  # desired dataset or None to use HDF5
    if not diagnostics_dataset:  # diagnostic dataset not loaded; create new from HDF5 file
        characteristics = characterize_sweep_array(bias, current, x, y, margin=smoothing_margin, sample_sec=sample_sec)
        diagnostics_dataset = plasma_diagnostics(characteristics, probe_area, ion_type,
                                                 experimental_parameters, bimaxwellian=bimaxwellian)
        if save_diagnostics:
            write_netcdf(diagnostics_dataset, save_diagnostic_path)

    # Print list of diagnostics generated
    print("Plasma diagnostics:", [key for key in diagnostics_dataset.keys()])

    # Plot one chosen diagnostic against radial position and time
    radial_diagnostic_plot(diagnostics_dataset, diagnostic='T_e', plot='contour')

    """
    print({diagnostic: diagnostics_dataset[diagnostic][sample_indices].values for diagnostic in diagnostics_dataset.keys()})
    print("Done analyzing sample characteristic")
    # """

    # Perform interferometry calibration for electron density
    electron_density, density_scaling = interferometry_calibration(
        diagnostics_dataset['n_e'], interferometry_filename,
        steady_state_start_plateau, steady_state_end_plateau, core_region=core_region)

    # Plot electron density data that was calibrated using interferometry data
    electron_density.squeeze().plot.contourf(robust=True)
    plt.show()

    # Calculation for pressure; in the future, this will consider only the steady-state region
    # TODO use interferometry calibrated electron density
    if not bimaxwellian:
        pressure = (3 / 2) * diagnostics_dataset['n_e'] * diagnostics_dataset['T_e'] * (1. * u.eV * u.m ** -3).to(u.Pa)
        # TODO RAISE ISSUE OF RECIPROCAL TEMPERATURE FOR NON-BIMAXWELLIAN TEMPERATURE
        # TODO add bimaxwellian single-temperature data to diagnostic_dataset and use to calculate pressure
        both = xr.ufuncs.logical_and
        plateau = pressure.coords['plateau']  # rename variables for comprehensibility

        steady_state_pressure = pressure.where(
            both(plateau >= steady_state_start_plateau, plateau <= steady_state_end_plateau))
        steady_state_pressure.squeeze(dim='y').assign_attrs({"units": str(u.Pa)}).plot.contourf(x='time', y='x',
                                                                                                robust=True)
        plt.title("Steady state pressure")
        plt.show()
    else:
        print("Pressure plotting not yet supported for bimaxwellian temperature distributions.")

    # Plot neutral ratios (NOTE: INCOMPLETE)
    neutral_ratio(diagnostics_dataset['n_e'], experimental_parameters, steady_state_start_plateau, steady_state_end_plateau)

    # Note: Plot generation code and neutral fraction analysis is incomplete.
    # Note: The non-bimaxwellian electron temperature calculated using the PlasmaPy code seems to be
    #    the *reciprocal* of the correct value. I may raise an issue on the PlasmaPy GitHub page.
