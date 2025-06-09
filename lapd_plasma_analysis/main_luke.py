from lapd_plasma_analysis.file_access import ask_yes_or_no, choose_multiple_from_list, ensure_directory
from lapd_plasma_analysis.fluctuations.interface_with_main import ask_about_plots
from lapd_plasma_analysis.fluctuations.analysis import get_isat_vf

from lapd_plasma_analysis.langmuir.configurations import get_config_id
from lapd_plasma_analysis.langmuir.analysis import (get_langmuir_datasets, get_diagnostics_to_plot, save_datasets_nc,
                                                    print_user_file_choices)

from lapd_plasma_analysis.langmuir.plots import *

from lapd_plasma_analysis.mach.analysis import get_mach_datasets, get_velocity_datasets

import os
import xarray as xr

from lapd_plasma_analysis.main import hdf5_folder, mach_nc_folder, flux_nc_folder
from obtain_plots.Functions_used_in_main_luke_plots import *

'''
Goals of this main: 
1. Check to see if necessary folders exist and if not, create them
2. Streamline the process of creating netCDF5 files
3. Get individual IV sweep curves to be able to perform eyeball analysis 
    and compare with what Leo gets from his analysis. Is our data actually bad
    or are we just throwing out too many data points
4. Perform a dimensionless comparison between hydrogen and helium plasmas
'''
# User parameters in original main that are useful as global variables
hdf5_folder = "/Users/lukec/Downloads/HTF5 file test/"
# TODO make a GUI so the user can select the folder in their directory?
assert hdf5_folder.endswith("/")

# Other user parameters
bimaxwellian = False
core_radius = 21. * u.cm                                                # TODO user can adjust (26 cm in MATLAB code)
plot_tolerance = np.nan  # 0.25                                         # TODO user can adjust
velocity_plot_unit = u.km / u.s         # TODO not yet working          # TODO adjust
display_core_steady_state_lines = True                                  # user can adjust

# Interferometry & Mach access modes. Options are "skip", "append", "overwrite"; recommended is "append".
interferometry_mode = "skip"                                            # TODO user adjust
mach_velocity_mode = "skip"                                           # not fully implemented
#TODO Error in 09 x = 27, y = 0, shot = 2. Why?

if __name__ == "__main__":

    # Check to see if these folders exist and if not creates them.
    # Returns a string of the file path name to those folders
    langmuir_nc_folder = ensure_directory(hdf5_folder + "lang_nc/")
    mach_nc_folder = ensure_directory(hdf5_folder + "mach_nc/")
    flux_nc_folder = ensure_directory(hdf5_folder + "flux_nc/")

    prompt_filetype = ["Convert HDF5 files to NetCDF files",
                        "Create plots from HDF5 files",
                       "Obtain plasma parameters from netCDF files"]
    user_choice_to_do = choose_multiple_from_list(prompt_filetype, 'action',null_action= "end main")
    # Gives a list of what the user wants to do

    # If the user chooses convert HDF5 files to NetCDF files
    if 0 in user_choice_to_do:
        print('in progress')

    # If the user chooses get plots from HDF5 files
    if 1 in user_choice_to_do:

        # Choose hdf5 files to read
        hdf5_list = [f for f in os.listdir(hdf5_folder) if f.endswith(".hdf5")]
        hdf5_choice = choose_multiple_from_list(hdf5_list,'HDF5 file',
                                                null_action="not retrieve data from HDF5 files.")
        if any(i >= len(hdf5_list) or i < 0 for i in hdf5_choice):
            hdf5_choice = []
            print("Invalid input")
        # Returns the indices in hdf5_list of the selected hdf5 files in a list format

        # Allow the user to select what they want to plot
        # Allow the user to select if they would like to save the plots
        if hdf5_choice:

            IV_plots_prompt = ["Plot bias voltage vs time for a position-shot combination",
                               "Plot current vs time for a position-shot combination",
                               "Plot individual raw IV sweeps for a position-shot combination",
                               "Plot log plot of IV sweeps for a position-shot combination",
                               "Plot Ion saturation current vs time for a position-shot combination (best in core region)"
                               ]
            IV_plots_choice = []
            while IV_plots_choice == [] or any(i >= len(IV_plots_prompt) or i < 0 for i in IV_plots_choice):
                IV_plots_choice = choose_multiple_from_list(IV_plots_prompt, 'parameter plot')
            save_plots = ask_yes_or_no("Do you want to save the plots? (Will be saved in a directory labelled by the run name)"
                                   " (y/n) ")
            # print('IV_plots_choice: ',IV_plots_choice)
            # Allow the user to determine how many plots they want to see
            how_many_plots = 0
            valid_input = False
            while (2 in IV_plots_choice or 3 in IV_plots_choice) and not valid_input:
                try:
                    how_many_plots = int(input("How many IV sweeps would you like to see? "))
                    valid_input = True
                except ValueError:
                    print("")

        # Create lists pf path names corresponding to the user's chosen hdf5 files
        hdf5_pathname_list = []
        for i in range(len(hdf5_choice)):
            hdf5_pathname_list.append(hdf5_folder + hdf5_list[hdf5_choice[i]])


        for hdf5_pathname in hdf5_pathname_list:
            # Obtain parameters to be used in IV sweep curves
            exp_params_dict,vsweep_bc,langmuir_configs,config_id,voltage_gain,orientation,current_bc \
                =(IV_parameters(hdf5_pathname))
            # Create IV curves for each face of the langmuir probe
            bias, dt = get_sweep_voltage(hdf5_pathname, vsweep_bc,voltage_gain)

            # Determine how many IV sweeps there are
            ramp_bounds = isolate_ramps(bias)
            # Determine the time each IV sweep was conducted
            ramp_times = ramp_bounds[:, 1] * dt.to(u.ms)

            # Reconfigure the bias and current data to be a 3D array giving the respective variable for
            # a specific (position, shot, frame)
            for i in range(len(langmuir_configs)):
                current, motor_data = get_sweep_current(hdf5_pathname, langmuir_configs[i], orientation)

                # ensure "hardcoded" ports listed in configurations.py match those listed in HDF5 file
                assert motor_data.info['controls']['6K Compumotor']['probe']['port'] == langmuir_configs[i]['port']

                position_array, num_positions, shots_per_position, selected_shots = get_shot_positions(motor_data)

                # Drop some shots from the data because they don't fit into a 3D structure
                if len(bias.shape) == 2:  # already selected certain shots in bias data
                    bias = bias[selected_shots, ...]
                current = current[selected_shots, ...]

                # Make bias and current 3D (position, shot_at_a_certain_position, frame) arrays
                #    as opposed to 2D (shot number, frame) arrays
                bias = bias.reshape(num_positions, shots_per_position, -1)
                current = current.reshape(num_positions, shots_per_position, -1)
                # Dimensions of bias and current arrays:   position, shot, frame   (e.g. (71, 15, 55296))

                # Grab the data for all given shots and positions so it can easily be extrapolated to other plots
                bias_to_plot,current_to_plot,loc_shot,filepath = (
                    obtain_data(hdf5_folder, bias, current, position_array, langmuir_configs[i], exp_params_dict, save_plots))
                port_face_string = f"{langmuir_configs[i]['port']}{langmuir_configs[i]['face'] if langmuir_configs[i]['face'] else ''}"
                for h in range(len(bias_to_plot)):
                    if 2 in IV_plots_choice or 3 in IV_plots_choice:
                        plot_iv_sweep(filepath, bias_to_plot[h], current_to_plot[h], port_face_string, IV_plots_choice,
                                    save_plots,how_many_plots, ramp_times, exp_params_dict,loc_shot[h],dt)

                    # If the user wants to plot the ion saturation current vs time
                    if 4 in IV_plots_choice:
                        plot_ion_isat_vs_time(dt,ramp_times,bias_to_plot[h],current_to_plot[h],exp_params_dict,
                                            port_face_string,loc_shot[h],save_plots)




    # If the user chooses Obtain plasma parameters from NetCDF files
    if 2 in user_choice_to_do:
        print('in progress')



