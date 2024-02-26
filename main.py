"""
The lapd-plasma-analysis repository was written by Leo Murphy based on code
written in MATLAB by Conor Perks (MIT) and using the PlasmaPy and bapsflib libraries.
Comments are added inline. A separate documentation page is not yet complete.
"""

from helper import *
from file_access import choose_multiple_list, ask_yes_or_no
from load_datasets import setup_datasets
from plots import multiplot_line_diagnostic, plot_line_diagnostic

""" End directory paths with a slash """
# hdf5_folder = "/Users/leomurphy/lapd-data/April_2018/"
# hdf5_folder = "/Users/leomurphy/lapd-data/March_2022/"
# hdf5_folder = "/Users/leomurphy/lapd-data/November_2022/"
hdf5_folder = "/Users/leomurphy/lapd-data/January_2024/"

langmuir_nc_folder = hdf5_folder + "lang_nc/"

""" Set to False or equivalent if interferometry calibration is not desired """
interferometry_folder = False                               # TODO set to False to avoid interferometry calibration
# interferometry_folder = hdf5_folder
# interferometry_folder = "/Users/leomurphy/lapd-data/November_2022/uwave_288_GHz_waveforms/"

""" User parameters """
isweep_choice = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]                # TODO user choice for probe or linear combination to use;
# isweep_choice = [1, 0]                                      # see isweep_selector in helper.py for brief explanation
# isweep_choice = [0]
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

    # TODO list of hardcoded parameters
    #    (16, 24) for January_2024 steady state period (main.py)

    datasets, steady_state_plateaus_runs, diagnostic_name_dict = setup_datasets(
        langmuir_nc_folder, hdf5_folder, interferometry_folder, isweep_choice, bimaxwellian)

    # Ask users for list of diagnostics to plot
    print("The following diagnostics are available to plot: ")
    diagnostics_sort_indices = np.argsort(list(diagnostic_name_dict.keys()))
    diagnostics_to_plot_ints = choose_multiple_list(np.array(list(diagnostic_name_dict.values())
                                                             )[diagnostics_sort_indices],
                                                    "diagnostic", null_action="end")
    diagnostic_to_plot_list = [np.array(list(diagnostic_name_dict.keys()))[diagnostics_sort_indices][choice]
                               for choice in diagnostics_to_plot_ints]
    print("Diagnostics selected:", diagnostic_to_plot_list)

    # Plot chosen diagnostics for each individual dataset
    if ask_yes_or_no("Generate contour plot of selected diagnostics over time and radial position? (y/n) "):
        for plot_diagnostic in diagnostic_to_plot_list:
            for i in range(len(datasets)):
                plot_line_diagnostic(isweep_selector(datasets[i], isweep_choice), plot_diagnostic, 'contour',
                                     steady_state_plateaus_runs[i], tolerance=plot_tolerance)

    # Plot radial profiles of diagnostic (steady-state time average), with color corresponding to first attribute
    #    and plot position on multiplot corresponding to second attribute
    if ask_yes_or_no("Generate line plot of selected diagnostics over radial position? (y/n) "):
        for plot_diagnostic in diagnostic_to_plot_list:
            multiplot_line_diagnostic(datasets, plot_diagnostic, isweep_choice, steady_state_plateaus_runs,
                                      tolerance=plot_tolerance)

    # TODO Shot plot! Multiplot line diagnostics at specific time, with x axis = x pos and curve color = shot #
    """if ask_yes_or_no("Generate line shot plot of selected diagnostics over radial position? (y/n) "):
        for plot_diagnostic in diagnostic_to_plot_list:
            multiplot_line_diagnostic(datasets, plot_diagnostic, isweep_choice, steady_state_plateaus_runs,
                                      tolerance=plot_tolerance)
    """

    for lang_ds in datasets:
        print(lang_ds.attrs['nu_ei'])

# TODO Not all MATLAB code has been transferred (e.g. neutrals, ExB)
# QUESTION: can we calibrate both Langmuir probes using an interferometry ratio depending only on one of them?
# TODO calculate collision freq/collisionality
