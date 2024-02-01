"""
The lapd-plasma-analysis repository was written by Leo Murphy based on code
written in MATLAB by Conor Perks (MIT) and using the PlasmaPy and bapsflib libraries.
Comments are added inline. A separate documentation page is not yet complete.
"""

from getIVsweep import *
from characterization import *
from diagnostics import *
from plots import *
from file_access import *
from interferometry import *
from neutrals import *
from experimental import *
from preconfiguration import *
from characteristic_view import *

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
probes_choice = [1, 0, 0, 0]                                  # TODO user choice for probe or linear combination to use
# probes_choice = [1, 0]
bimaxwellian = False                                    # TODO perform both and store in same NetCDF file?
smoothing_margin = 1 * 200                                  # Optimal values in range 100-400 if "median" smoothing method
plot_tolerance = np.nan   # was 2                       # Optimal values are np.nan (plot all points) or >= 0.5

# QUESTION: can we calibrate both Langmuir probes using an interferometry ratio depending only on one of them?
core_radius = 26. * u.cm                                         # From MATLAB code


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
        print("Interferometry calibration is OFF. "
              "Interferometry-calibrated electron density ('n_e_cal') is not available.")

    print("Current HDF5 directory path:\t\t\t", repr(hdf5_folder),
          "\nCurrent NetCDF directory path:\t\t\t", repr(langmuir_nc_folder),
          "\nCurrent interferometry directory path:\t", repr(interferometry_folder),
          "\nThese can be changed in main.py.")
    input("Enter any key to continue: ")

    netcdf_folder = ensure_directory(langmuir_nc_folder)  # Create folder to save NetCDF files if not yet existing

    print("\nThe following NetCDF files were found in the NetCDF folder (specified in main.py): ")
    nc_paths = sorted(search_folder(netcdf_folder, 'nc', limit=52))
    nc_paths_to_open_ints = choose_multiple_list(nc_paths, "NetCDF file",
                                                 null_action="perform diagnostics on HDF5 files")

    if len(nc_paths_to_open_ints) > 0:
        datasets = [xr.open_dataset(nc_paths[choice]) for choice in nc_paths_to_open_ints]
    else:
        print("\nThe following HDF5 files were found in the HDF5 folder (specified in main.py): ")
        hdf5_paths = sorted(search_folder(hdf5_folder, "hdf5", limit=52))
        hdf5_chosen_ints = choose_multiple_list(hdf5_paths, "HDF5 file")
        hdf5_chosen_list = [hdf5_paths[choice] for choice in hdf5_chosen_ints]

        sweep_view_mode = (len(hdf5_chosen_list) == 1) and ask_yes_or_no("Use raw sweep plotting mode? (y/n) ")
        chara_view_mode = (len(hdf5_chosen_list) == 1) and ask_yes_or_no("Use characteristic plotting mode? (y/n) ")

        datasets = []
        for hdf5_path in hdf5_chosen_list:

            print("\nOpening file", repr(hdf5_path), "...")

            exp_params_dict = get_exp_params(hdf5_path)  # list of experimental parameters
            ion_type = get_ion(exp_params_dict['Run name'])
            config_id = get_config_id(exp_params_dict['Exp name'])
            vsweep_board_channel = get_vsweep_bc(config_id)

            langmuir_probes = get_langmuir_config(hdf5_path, config_id)  # TODO print face strings in outputs
            voltage_gain = get_voltage_gain(config_id)
            orientation = get_orientation(config_id)

            # get current and bias data from Langmuir probe
            bias, currents, positions, dt, ports = get_isweep_vsweep(hdf5_path, vsweep_board_channel,
                                                                     langmuir_probes, voltage_gain, orientation)

            if sweep_view_mode:
                display_raw_sweep(bias, currents, positions, ports, exp_params_dict, dt)

            # organize sweep bias and current into an array of Characteristic objects
            characteristics, ramp_times = characterize_sweep_array(bias, currents, smoothing_margin, dt)

            if chara_view_mode:
                display_characteristics(characteristics, positions, ports, ramp_times, exp_params_dict,
                                        diagnostics=True, areas=langmuir_probes['area'],
                                        ion=ion_type, bimaxwellian=bimaxwellian)

            diagnostics_dataset = langmuir_diagnostics(characteristics, positions, ramp_times, ports,
                                                       langmuir_probes['area'], ion_type, bimaxwellian=bimaxwellian)
            diagnostics_dataset = diagnostics_dataset.assign_attrs(exp_params_dict)
            datasets.append(diagnostics_dataset)

    # TODO this is extremely hardcoded
    if "january" in hdf5_folder.lower():
        steady_state_plateaus_runs = [(16, 24) for dataset in datasets]
    else:
        steady_state_plateaus_runs = [detect_steady_state_ramps(dataset['n_e'], core_radius) for dataset in datasets]

    if len(nc_paths_to_open_ints) == 0:
        # External interferometry calibration for electron density
        for i in range(len(datasets)):
            if interferometry_calibrate:
                try:
                    calibrated_electron_density = interferometry_calibration(datasets[i]['n_e'].copy(),
                                                                             datasets[i].attrs,          # exp params
                                                                             interferometry_folder,
                                                                             steady_state_plateaus_runs[i],
                                                                             core_radius=core_radius)
                    datasets[i] = datasets[i].assign({"n_e_cal": calibrated_electron_density})
                except (IndexError, ValueError, TypeError, AttributeError, KeyError) as e:
                    print(f"Error in calibrating electron density: \n{str(e)}")
                    calibrated_electron_density = datasets[i]['n_e'].copy()
            else:
                calibrated_electron_density = datasets[i]['n_e'].copy()

            datasets[i] = datasets[i].assign({'P_e': get_pressure(datasets[i], calibrated_electron_density,
                                                                  bimaxwellian)})
            datasets[i] = datasets[i].assign_attrs({"Interferometry calibrated": interferometry_calibrate})

            save_diagnostic_path = make_path(netcdf_folder, datasets[i].attrs['Run name'] + "_lang", "nc")
            write_netcdf(datasets[i], save_diagnostic_path)

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
                plot_line_diagnostic(isweep_selector(datasets[i], probes_choice), plot_diagnostic, 'contour',
                                     steady_state_plateaus_runs[i], tolerance=plot_tolerance)

    """
    Plot radial profiles of diagnostic (steady-state time average), with color corresponding to first attribute
        and plot position on multiplot corresponding to second attribute
    """
    if ask_yes_or_no("Generate line plot of selected diagnostics over radial position? (y/n) "):
        for plot_diagnostic in diagnostic_to_plot_list:
            multiplot_line_diagnostic(datasets, plot_diagnostic, probes_choice, steady_state_plateaus_runs,
                                      tolerance=plot_tolerance)

# TODO Not all MATLAB code has been transferred (e.g. neutrals, ExB)
