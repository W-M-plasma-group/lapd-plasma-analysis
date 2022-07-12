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

# TODO remove manual mar2022, detect from bapsflib configurations
mar2022 = False

# User global parameters
sample_sec = 12.5e6 ** (-1) * u.s if mar2022 else (100 / 16 * 1e6) ** (-1) * u.s      # Manually reported by SV. - check for each port
probe_area = (1. * u.mm) ** 2                                    # From MATLAB code
core_region = 26. * u.cm                                         # From MATLAB code
ion_type = 'He-4+'
bimaxwellian = False                                              # TODO rely on metadata to determine if bimaxwellian when opening file; save with different names
# TODO compare bimaxwellian n_e, n_i, etc. with nonbimaxwellian to determine if need to store under different filenames
#   or just in same file with different diagnostic names
smoothing_margin = 15                                            # Optimal values in range 0-25
steady_state_plateaus = (6, 13) if mar2022 else (5, 11)          # From MATLAB code
diagnostics_plotted = ['T_e_avg', 'T_e', 'n_e']                  # String or list of strings
# End of global parameters

# User file path names
# Path of chosen HDF5 file; available under repository Releases
if mar2022:
    # hdf5_path = "/data/BAPSF_Data/Particle_Transport/March_2022/01_line_valves85V_7400A.hdf5"
    # hdf5_path = "/Users/leo/lapd-data/10_line_valves95V_7500A.hdf5"
    hdf5_path = "/Users/leo/lapd-data/March_2022/01_line_valves85V_7400A.hdf5"
else:
    hdf5_path = "/Users/leo/lapd-data/April_2018/10_radial_line_25press_4500A_redo.hdf5"

interferometry_filename = hdf5_path           # Interferometry data stored in same HDF5 file
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

    # TODO user prompt file options
    lapd_file = lapd.File(hdf5_path)
    full_netcdf_path = netcdf_path(lapd_file.info['file'], netcdf_subfolder_path, bimaxwellian)
    save_diagnostic_path = open_diagnostic_path = full_netcdf_path

    print("Mar2022 variable is", "ON" if mar2022 else "OFF")

    # Check if users want to load saved netCDF or create new
    new_diagnostics_choice = ''
    while new_diagnostics_choice not in ('l', 'c'):
        new_diagnostics_choice = input("Load saved diagnostic data ('l') or calculate new diagnostics ('c')? ").lower()
    load_diagnostics = (new_diagnostics_choice == 'l')
    if load_diagnostics:
        netcdf_files = search_netcdf()
        file_choice = choose_list(netcdf_files, kind="netCDF file", location="current working directory", add_new=True)
        if file_choice == 0:
            # return None
            pass
        # dataset = xr.open_dataset(netcdf_files[file_choice - 1])

    """
    Plan 
    - Check if users want to load a saved netCDF file or create a new one
    # We want the most common action to be loading a saved netcdf file! 
    #    - Make processing multiple HDF5 files in the same run without further input is easy
    #    - Store x, y data with nc files so HDF5 files can be unnecessary
    - If load
        - Check available .nc files and ask user to choose or create new 
    - If create new
        - User should have specified *hdf5 directory* in code 
        - Print HDF5 files and ask users to select (can select multiple)
        - Allow a single bimax/nonbimax/both choice
    #    - TODO check if bimax/nonbimax density, etc. are different enough to warrant separation!
    #    - TODO explicitly make .nc file system based on hdf5 file
    - List possible diagnostics to plot and have user select
    # For now: keep default plot as contour unless changed
    """

    """
    New plan
    1) Should be able to process and provide graphs for a whole directory of  / a few HDF5 files
    2) Should be able to allow users to process a few/a directory of HDF5 files into NetCDF files
    - Ask user to hard-code in HDF5 *directory* 
    - Allow user to select some or all of HDF5 files to process
    - - Make NetCDF file an essential/opt-out intermediate step for graph-making?
    """

    # lapd_file.overview.print()

    print("Diagnostic dataset will be saved to or opened from the path", repr(full_netcdf_path))

    # Read LAPD experimental parameters
    experimental_parameters, experimental_parameters_rounded = setup_lapd(hdf5_path)
    print("Experimental parameters:", {key: str(value) for key, value in experimental_parameters_rounded.items()})
    bias, current, x, y = get_isweep_vsweep(hdf5_path, mar2022)
    # TODO create position matrix in diagnostic dataset to avoid having to reload and recalculate position data?

    # Create the diagnostics dataset by generating new or opening existing NetCDF file
    diagnostics_dataset = read_netcdf(open_diagnostic_path) if use_existing else None  # desired dataset or None to use HDF5
    if not diagnostics_dataset:  # diagnostic dataset not loaded; create new from HDF5 file
        characteristics = characterize_sweep_array(bias, current, x, y, margin=smoothing_margin, sample_sec=sample_sec)
        diagnostics_dataset = plasma_diagnostics(characteristics, probe_area, ion_type,
                                                 experimental_parameters, bimaxwellian=bimaxwellian)
        # print("Bimaxwellian, main.py: ", diagnostics_dataset.attrs['bimaxwellian'])
        if save_diagnostics:
            write_netcdf(diagnostics_dataset, save_diagnostic_path)
    else:
        print(diagnostics_dataset.attrs['bimaxwellian'])

    # Print list of diagnostics generated
    # print("Plasma diagnostics:", [key for key in diagnostics_dataset.keys()])

    # Plot chosen diagnostics by linear position and time, then time-averaged diagnostics for each position
    linear_diagnostic_plot(diagnostics_dataset, diagnostics_plotted, plot='contour', steady_state=steady_state_plateaus)
    # linear_diagnostic_plot(diagnostics_dataset, diagnostics_plotted, plot='line', steady_state=steady_state_plateaus)


    # print({diagnostic: diagnostics_dataset[diagnostic][(30, 0, 7)].values for diagnostic in diagnostics_dataset.keys()})
    # print("Done analyzing sample characteristic")

    # Perform interferometry calibration for electron density
    # TODO mar2022 detected by word "fringes" in HDF5 metadata?
    calibrated_electron_density = interferometry_calibration(diagnostics_dataset['n_e'], interferometry_filename,
                                                             steady_state_plateaus, core_region)

    # Plot electron density data that was calibrated using interferometry data
    # linear_diagnostic_plot(xr.Dataset({"n_e_cal": calibrated_electron_density}), "n_e_cal", plot="contour")
    linear_diagnostic_plot(xr.Dataset({"n_e_cal": calibrated_electron_density}), "n_e_cal", plot="line",
                           steady_state=steady_state_plateaus)
    if bimaxwellian:
        linear_diagnostic_plot(xr.Dataset({"n_e_hot": calibrated_electron_density * diagnostics_dataset['hot_fraction']}),
                               "n_e_hot", plot="line", steady_state=steady_state_plateaus)

    # Calculation for pressure; in the future, this will consider only the steady-state region
    # TODO Why is the pressure about twice what is expected as given in Overleaf document?
    electron_temperature = diagnostics_dataset['T_e_avg'] if bimaxwellian else diagnostics_dataset['T_e']
    pressure = (3 / 2) * electron_temperature * calibrated_electron_density * (1. * u.eV * u.m ** -3).to(u.Pa)
    # linear_diagnostic_plot(xr.Dataset({"p_e": pressure}), "p_e", plot="contour", steady_state=steady_state_plateaus)
    linear_diagnostic_plot(xr.Dataset({"p_e": pressure}), "p_e", plot="line", steady_state=steady_state_plateaus)

    # Plot neutral ratios (NOTE: INCOMPLETE)
    # neutral_ratio(diagnostics_dataset['n_e'], experimental_parameters, *steady_state_plateaus)

    # Note: Plot generation code and neutral fraction analysis is incomplete.
