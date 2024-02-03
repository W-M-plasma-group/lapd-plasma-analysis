"""
The lapd-plasma-analysis repository was written by Leo Murphy based on code
written in MATLAB by Conor Perks (MIT) and using the PlasmaPy and bapsflib libraries.
Comments are added inline. A separate documentation page is not yet complete.
"""

from helper import *
from file_access import choose_multiple_list, ask_yes_or_no, ensure_directory
from load_datasets import load_datasets, interferometry_calibrate_datasets, save_datasets
from diagnostics import detect_steady_state_ramps
from plots import multiplot_line_diagnostic, plot_line_diagnostic, get_title

""" End directory paths with a slash """
# hdf5_folder = "/Users/leomurphy/lapd-data/April_2018/"
# hdf5_folder = "/Users/leomurphy/lapd-data/March_2022/"
# hdf5_folder = "/Users/leomurphy/lapd-data/November_2022/"
hdf5_folder = "/Users/leomurphy/lapd-data/January_2024/"

langmuir_nc_folder = hdf5_folder + "lang_nc/"

""" Set to False or equivalent if interferometry calibration is not desired """
interferometry_folder = False       # TODO set to False to avoid interferometry calibration
# interferometry_folder = hdf5_folder
# interferometry_folder = "/Users/leomurphy/lapd-data/November_2022/uwave_288_GHz_waveforms/"

""" User parameters """
isweep_choice = [[1, 0, 0, 0], [0, 0, 1, 0]]                # TODO user choice for probe or linear combination to use;
#                                                             see isweep_selector in helper.py for brief explanation
bimaxwellian = False                                        # TODO perform both and store in same NetCDF file?
plot_tolerance = np.nan   # was 2                           # Optimal values are np.nan (plot all points) or >= 0.5
core_radius = 26. * u.cm                                    # From MATLAB code


# Diagram of LAPD
"""
       <- ~18m plasma length -> 
  ____________________________________       
  |    |                      '      |       A       
  |    |                      '      |       |   ~75 cm plasma diameter
  |____|______________________'______|       V
      (a)                    (b)    (c)
        +z direction (+ports) ==>
                  plasma flow ==>
            magnetic field B0 ==>

a) LaB6 electron beam cathode
b) downstream mesh anode
c) downstream cathode
"""


if __name__ == "__main__":

    interferometry_calibrate = bool(interferometry_folder)
    if not interferometry_calibrate:
        print("\nInterferometry calibration is OFF. "
              "Interferometry-calibrated electron density ('n_e_cal') is not available.")

    print("Current HDF5 directory path:           \t",   repr(hdf5_folder),
          "\nCurrent NetCDF directory path:         \t", repr(langmuir_nc_folder),
          "\nCurrent interferometry directory path: \t", repr(interferometry_folder),
          "\nLinear combinations of isat sources:   \t", repr(isweep_choice),
          "\nThese can be changed in main.py.")
    input("Enter any key to continue: ")

    netcdf_folder = ensure_directory(langmuir_nc_folder)  # Create folder to save NetCDF files if not yet existing

    datasets, generate_new = load_datasets(hdf5_folder, netcdf_folder, interferometry_folder, isweep_choice,
                                           bimaxwellian)

    # Get ramp indices for beginning and end of steady state period in plasma; TODO hardcoded
    steady_state_plateaus_runs = ([(16, 24) for dataset in datasets] if "january" in hdf5_folder.lower()
                                  else [detect_steady_state_ramps(dataset['n_e'], core_radius) for dataset in datasets])

    # If new diagnostics were generated from HDF5 files, calibrate electron densities using interferometry data
    if generate_new:
        datasets = interferometry_calibrate_datasets(datasets, interferometry_folder, steady_state_plateaus_runs)

    # Save diagnostics datasets to folder
    save_datasets(datasets, netcdf_folder, bimaxwellian)

    # Get possible diagnostics and their full names, e.g. "n_e" and "Electron density"
    diagnostic_name_dict = {key: get_title(key)
                            for key in set.intersection(*[set(dataset) for dataset in datasets])}

    # Ask users for list of diagnostics to plot
    print("The following diagnostics are available to plot: ")
    diagnostics_sort_indices = np.argsort(list(diagnostic_name_dict.keys()))
    diagnostics_to_plot_ints = choose_multiple_list(np.array(list(diagnostic_name_dict.values())
                                                             )[diagnostics_sort_indices],
                                                    "diagnostic", null_action="end")
    diagnostic_to_plot_list = [np.array(list(diagnostic_name_dict.keys()))[diagnostics_sort_indices][choice]
                               for choice in diagnostics_to_plot_ints]
    print("Diagnostics selected:", diagnostic_to_plot_list)

    """Plot chosen diagnostics for each individual dataset"""
    if ask_yes_or_no("Generate contour plot of selected diagnostics over time and radial position? (y/n) "):
        for plot_diagnostic in diagnostic_to_plot_list:
            for i in range(len(datasets)):
                plot_line_diagnostic(isweep_selector(datasets[i], isweep_choice), plot_diagnostic, 'contour',
                                     steady_state_plateaus_runs[i], tolerance=plot_tolerance)

    """
    Plot radial profiles of diagnostic (steady-state time average), with color corresponding to first attribute
        and plot position on multiplot corresponding to second attribute
    """
    if ask_yes_or_no("Generate line plot of selected diagnostics over radial position? (y/n) "):
        for plot_diagnostic in diagnostic_to_plot_list:
            multiplot_line_diagnostic(datasets, plot_diagnostic, isweep_choice, steady_state_plateaus_runs,
                                      tolerance=plot_tolerance)

# TODO Not all MATLAB code has been transferred (e.g. neutrals, ExB)
