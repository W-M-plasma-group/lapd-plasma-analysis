"""
The lapd-plasma-analysis repository was written by Leo Murphy based on code
written in MATLAB by Conor Perks (MIT) and using the PlasmaPy and bapsflib libraries.
Comments are added inline. A separate documentation page is not yet complete.
"""

from helper import *
from file_access import ask_yes_or_no
from load_datasets import setup_datasets
from plots import multiplot_line_diagnostic, plot_line_diagnostic

import matplotlib

""" End directory paths with a slash """
# hdf5_folder = "/Users/leomurphy/lapd-data/April_2018/"
hdf5_folder = "/Users/leomurphy/lapd-data/March_2022/"
# hdf5_folder = "/Users/leomurphy/lapd-data/November_2022/"
# hdf5_folder = "/Users/leomurphy/lapd-data/January_2024/January_2024_all_working/"
# hdf5_folder = "/Users/leomurphy/lapd-data/all_lang_nc/"

langmuir_nc_folder = hdf5_folder + ("lang_nc/" if hdf5_folder.endswith("/") else "/lang_nc/")

""" Set to False or equivalent if interferometry calibration is not desired """
interferometry_folder = False                               # TODO set to False to avoid interferometry calibration
# interferometry_folder = hdf5_folder
# interferometry_folder = "/Users/leomurphy/lapd-data/November_2022/uwave_288_GHz_waveforms/"

""" User parameters """
# TODO isweep_choice is user choice for probe or linear combination to use; see isweep_selector in helper.py for more
# isweep_choice = [[1, 0, 0, 0], [0, 0, 1, 0]]
# isweep_choice = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
isweep_choice = [[1, 0], [0, 1]]
# isweep_choice = [0]
bimaxwellian = False                                        # TODO perform both and store in same NetCDF file?
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
    #    (16, 24) for January_2024 steady state period (detect_steady_state_period in diagnostics.py)

    datasets, steady_state_plateaus_runs, diagnostic_name_dict, diagnostics_to_plot_list = setup_datasets(
        langmuir_nc_folder, hdf5_folder, interferometry_folder, isweep_choice, bimaxwellian)
    print("Diagnostics selected:", diagnostics_to_plot_list)

    # Plot chosen diagnostics for each individual dataset
    if ask_yes_or_no("Generate contour plot of selected diagnostics over time and radial position? (y/n) "):
        for plot_diagnostic in diagnostics_to_plot_list:
            for i in range(len(datasets)):
                plot_line_diagnostic(isweep_selector(datasets[i], isweep_choice), plot_diagnostic, 'contour',
                                     steady_state_plateaus_runs[i], tolerance=plot_tolerance)

    # Plot radial profiles of diagnostic (steady-state time average), with color corresponding to first attribute
    #    and plot position on multiplot corresponding to second attribute
    if ask_yes_or_no("Generate line plot of selected diagnostics over radial position? (y/n) "):
        for plot_diagnostic in diagnostics_to_plot_list:
            multiplot_line_diagnostic(datasets, plot_diagnostic, isweep_choice, 'x',
                                      steady_state_by_runs=steady_state_plateaus_runs,
                                      core_rad=core_radius, tolerance=plot_tolerance)

    if ask_yes_or_no("Generate parallel pressure plot? (y/n) "):
        from diagnostics import in_core
        from plots import steady_state_only

        plt.rcParams['figure.figsize'] = (10, 4)
        color_map = matplotlib.colormaps["plasma"](np.linspace(0, 0.9, len(datasets)))
        for i in range(len(datasets)):
            isweep_choices = (0, 2) if datasets[i].attrs['Exp name'] == "January_2024" else (0, 1)
            pressures = []
            ports = []

            pressure_means = core_steady_state_mean(datasets[i]['P_e'],         core_radius,
                                                    steady_state_plateaus_runs[i],       dims_to_keep=("isweep",))
            collision_freq_means = core_steady_state_mean(datasets[i]['nu_ei'], core_radius,
                                                          steady_state_plateaus_runs[i], dims_to_keep=("isweep",))

            for j in range(len(isweep_choices)):
                isweep_choice = isweep_choices[j]
                pressures += [pressure_means[{"isweep": isweep_choice}].item()]   # median??
                ports += [-pressure_means[{"isweep": isweep_choice}].coords['z'].item()]
            collision_freq_mean = collision_freq_means[{"isweep": 0}].mean().item()
            plt.plot(ports, pressures, marker="o", label=f"{datasets[i].attrs['Run name']}:     "
                                                         f"{collision_freq_mean:.2E} Hz",
                     color=color_map[i])
        plt.title("Pressure versus port (z-position)")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.show()

    # (UNFINISHED) Shot plot: multiplot line diagnostics at specific time, with x-axis = x pos and curve color = shot #
    """if ask_yes_or_no("Generate line shot plot of selected diagnostics over radial position? (y/n) "):
        for plot_diagnostic in diagnostic_to_plot_list:
            multiplot_line_diagnostic(datasets, plot_diagnostic, isweep_choice, steady_state_plateaus_runs,
                                      tolerance=plot_tolerance)
    """

# TODO Not all MATLAB code has been transferred (e.g. neutrals, ExB)
# QUESTION: can we calibrate both Langmuir probes using an interferometry ratio depending only on one of them?
