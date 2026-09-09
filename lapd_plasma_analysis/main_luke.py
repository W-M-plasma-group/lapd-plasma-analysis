from lapd_plasma_analysis.fluctuations.interface_with_main import ask_about_plots
from lapd_plasma_analysis.fluctuations.analysis import get_isat_vf

from lapd_plasma_analysis.obtain_plots.presentation_plots import *

from lapd_plasma_analysis.langmuir.analysis import (get_langmuir_datasets, get_diagnostics_to_plot, save_datasets_nc,
                                                    print_user_file_choices)
from lapd_plasma_analysis.fluctuations.luke_main_fluct_helpers import *

from lapd_plasma_analysis.mach.luke_main_mach_helpers import *
from lapd_plasma_analysis.mach.configurations import *
from lapd_plasma_analysis.mach.analysis import *
from lapd_plasma_analysis.mach.velocity import *
from lapd_plasma_analysis.mach.getMachIsat import *

import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import time
import re
import pandas as pd
from matplotlib.lines import Line2D

from lapd_plasma_analysis.main import hdf5_folder, plot_save_folder
from lapd_plasma_analysis.net_cdf_infrastructure import *

# from lapd_plasma_analysis.main import hdf5_folder, mach_nc_folder, flux_nc_folder
from bapsflib import lapd

from obtain_plots.Functions_used_in_main_luke_plots import *
from lapd_plasma_analysis.net_cdf_infrastructure.Build_netcdf import *
from lapd_plasma_analysis.obtain_plots.xarray_plots import *
from lapd_plasma_analysis.Read_hdf5.primary_functions import *
from lapd_plasma_analysis.Read_hdf5.read_metadata import *

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
# hdf5_folder = "C:\LAPD_HDF5_files/"

# TODO make a GUI so the user can select the folder in their directory?
assert hdf5_folder.endswith("/")


# Other user parameters
bimaxwellian = False
core_radius = 21. * u.cm                                                # TODO user can adjust (26 cm in MATLAB code)
plasma_length = 19.7 * u.m
plot_tolerance = np.nan  # 0.25                                         # TODO user can adjust
velocity_plot_unit = u.km / u.s         # TODO not yet working          # TODO adjust
display_core_steady_state_lines = True                                  # user can adjust
default_fig_height = 4.8
default_fig_width = 6.4
plt.rcParams.update({
                'font.size': 24,  # Global font size
                'axes.labelsize': 24,  # x and y labels
                'axes.titlesize': 24,  # Title size
                'xtick.labelsize': 24,  # x-axis tick labels
                'ytick.labelsize': 24,  # y-axis tick labels
                'legend.fontsize': 24,  # Legend text
                'legend.title_fontsize': 24,  # Legend title (if any)
                'axes.formatter.use_mathtext': True,  # Use LaTeX style math font
                'lines.linewidth': 3,  # Global line width (default is 1.5)
                'lines.markersize': 8,  # Global marker size (default is 6.0)
                'errorbar.capsize': 5  # Global error bar cap width (default is 0.0)
            })


# Interferometry & Mach access modes. Options are "skip", "append", "overwrite"; recommended is "append".
interferometry_mode = "skip"                                            # TODO user adjust
mach_velocity_mode = "skip"                                           # not fully implemented

if __name__ == "__main__":
    # Check to see if these folders exist and if not creates them.
    # Returns a string of the file path name to those folders
    langmuir_nc_folder = ensure_directory(hdf5_folder + "lang_nc/")
    mach_nc_folder = ensure_directory(hdf5_folder + "mach_nc/")
    flux_nc_folder = ensure_directory(hdf5_folder + "flux_nc/")
    figure_folder = ensure_directory(hdf5_folder + "figures/")
    csv_folder = ensure_directory(hdf5_folder + "csv/")



    # Prompt the user to choose what they would like to do
    prompt_filetype = ["Convert HDF5 files to NetCDF files",
                       "Convert HDF5 files to NetCDF files with tanh fit for T_e",
                       "Create plots from HDF5 files",
                       "Check NaNs",
                       "Obtain plasma parameters from netCDF files",
                       "Get HDF5 metadata",
                       "Obtain Fluctuations from HDF5",
                       "Build Mach Datasets",
                       'Do stuff with Mach datasets']

    # # Returns a list of what the user wants to do indexed by the location in the prompt_filetype list
    # user_choice_to_do = choose_multiple_from_list(prompt_filetype, 'action',null_action= "end main")
    #
    # # Returns options using the string from prompt_filetype rather than the index. This makes it easier to delete
    # # options
    # chosen_options = [prompt_filetype[choice] for choice in user_choice_to_do]

    chosen_options = int_choose_multiple_from_list(prompt_filetype, 'action', null_action= "end main")
    print('Chosen options: ' + str(chosen_options))


    # If the user chooses convert HDF5 files to NetCDF files or get plots from HDF5 files - loading data is the same.
    # Initial loading of HDF5 files is the same as Leo's main.
    if ('Convert HDF5 files to NetCDF files' in chosen_options or
            'Convert HDF5 files to NetCDF files with tanh fit for T_e' in chosen_options or
            'Create plots from HDF5 files' in chosen_options or
            'Plasma Py HDF5 to NetCDF' in chosen_options or
            'Get HDF5 metadata' in chosen_options):
        # Choose hdf5 files to read
        hdf5_list = sorted([f for f in os.listdir(hdf5_folder) if f.endswith(".hdf5")])
        # print(hdf5_list)

        # hdf5_choice returns the indices in hdf5_list associated with the files that the user wants to see.
        hdf5_choice = int_choose_multiple_from_list(hdf5_list, 'HDF5 file',
                                                    null_action="not retrieve data from HDF5 files.")

        # So long as the user selects a file to view it will run through this section
        if hdf5_choice:
            # Specific: User wants to look at individual sweep data
            if 'Create plots from HDF5 files' in chosen_options:

                # Allow the user to select what they want to plot
                IV_plots_prompt = ["Plot bias voltage vs time for a position-shot combination",
                                   "Plot current vs time for a position-shot combination",
                                   "Plot individual raw IV sweeps for a position-shot combination",
                                   "Plot log plot of IV sweeps for a position-shot combination",
                                   "Plot Ion saturation current vs time for a position-shot combination (best in core region)"
                                   ]


                plot_choices = int_choose_multiple_from_list(IV_plots_prompt, "parameter plot")

                # Allow the user to select if they would like to save the plots that are created
                save_plots = ask_yes_or_no("Do you want to save the plots? (Will be saved in a directory labelled by the run name)"
                                       " (y/n) ")

        # Create lists of path names corresponding to the user's chosen hdf5 files
        hdf5_pathname_list = []
        for choice in hdf5_choice:
            hdf5_pathname_list.append(hdf5_folder + choice)

        pathname_index = 0
        data_dict = {}
        for hdf5_pathname in hdf5_pathname_list:
            if 'Get HDF5 metadata' in chosen_options:
                with lapd.File(hdf5_pathname) as hdf5_file:
                    # Isolate the description from the .info file
                    # print(hdf5_file.info)
                    # print(list(hdf5_file.keys()))
                    desc = hdf5_file.info['run description']
                    magnet_fields = get_bfield_dict(desc)
                    cathode_currents = get_cathode_current(desc)
                    gas_puffs = get_gas_puff_voltage(desc)
                    gas_desc = get_gas_type(desc)
                    params_dict = metadata_dict(desc, hdf5_file.info['exp name'], hdf5_file.info['file'])

                    gp_voltage_values = []
                    for k, v in params_dict.items():
                        if k.startswith("GPV"):
                            try:
                                gp_voltage_values.append(float(v))
                            except (TypeError, ValueError):
                                pass  # skip non-convertible values

                    gp_voltage = sum(gp_voltage_values) / len(gp_voltage_values) if gp_voltage_values else None

                    # b_field_value = params_dict['Magenta']

                    print('gp_voltage ', gp_voltage)
                    # print('b_field_value ', b_field_value)

                    print(params_dict)

                continue

            data_dict[hdf5_pathname] = {}

            # Most of the following functions require the file to be opened so this just opens it once and
            # feeds it through
            with (lapd.File(hdf5_pathname) as hdf5_file):

                params_dict = metadata_dict(hdf5_file.info['run description'],
                                            hdf5_file.info['exp name'],
                                            hdf5_file.info['file'])


                # From Leo code for Isweep choices
                # isweep_choice is user choice for probe or linear combination to plot; see isweep_selector in helper.py for more
                # # e.g. coefficients are for [[p1f1, p1f2], [p2f1, p2f2]]
                # if ('jan' in params_dict['Exp name'].lower()
                #     and '24' in params_dict['Exp name'].lower()):
                #     isweep_choices = [[[1, 0], [0, 0]],  # . 1st combination to plot: 1 * (first face on first probe)
                #                       [[0, 0], [1, 0]]]  # . 2nd combination to plot: 1 * (first face on second probe)
                # else:
                #     isweep_choices = [[[1, 0], [-1, 0]]]  # . combination to plot: 1 * (face 1 on probe 1) - 1 * (face 1 on probe 2)

                # Which probes do we want to plot -- corresponds with the indices in langmuir.configurations
                if ('jan' in params_dict['Exp name'].lower()
                    and '24' in params_dict['Exp name'].lower()):
                    valid_probes = [0,2]
                else:
                    valid_probes = [0,1]

                # Obtain parameters to be used in IV sweep curves
                # TODO Merge exp_params_dict with params_dict
                exp_params_dict, vsweep_bc, langmuir_configs, config_id, voltage_gain, orientation, current_bc \
                     = (n_IV_parameters(hdf5_file, hdf5_pathname))

                # Obtain the voltage data for the associated file
                bias, dt = n_get_sweep_voltage(hdf5_file, vsweep_bc, voltage_gain)
                # Determine how many IV sweeps there are
                ramp_bounds = isolate_ramps(bias)
                # Determine the times each IV sweep was conducted
                ramp_times = ramp_bounds[:, 1] * dt.to(u.ms)
                for probe_num in valid_probes:
                    # All probes should share the same bias sweep so we don't need to recompute it for every probe

                    probe_bias = bias.copy()

                    probe_current, motor_data = n_get_sweep_current(hdf5_file,
                                                                  langmuir_configs[probe_num],
                                                                  orientation)
                    # ensure "hardcoded" ports listed in configurations.py match those listed in HDF5 file
                    assert (motor_data.info['controls']['6K Compumotor']['probe']['port'] ==
                            langmuir_configs[probe_num]['port'])

                    data_dict[hdf5_pathname][probe_num] = {}
                    data_dict[hdf5_pathname][probe_num]['bias'] = probe_bias
                    data_dict[hdf5_pathname][probe_num]['current'] = probe_current
                    data_dict[hdf5_pathname][probe_num]['motor data'] = motor_data

            # Reconfigure the bias and current data to be a 3D array giving the respective variable for
            # a specific (position, shot, frame)
            if  ('Convert HDF5 files to NetCDF files' in chosen_options or
                    'Convert HDF5 files to NetCDF files with tanh fit for T_e' in chosen_options or
                    'Plasma Py HDF5 to NetCDF' in chosen_options):
                bias_list = []
                current_list = []
                position_list = []

            # For each probe that sweeps we can get each individual data set
            for key in data_dict[hdf5_pathname].keys():
                probe_bias = data_dict[hdf5_pathname][key]['bias']
                probe_current = data_dict[hdf5_pathname][key]['current']
                probe_motor_data = data_dict[hdf5_pathname][key]['motor data']

                (probe_position_array,
                 num_positions,
                 shots_per_position,
                 selected_shots) = get_shot_positions(probe_motor_data)

                data_dict[hdf5_pathname][key]['positions'] = probe_position_array

                # Drop some shots from the data because they don't fit into a 3D structure
                if len(probe_bias.shape) == 2:  # already selected certain shots in bias data
                    probe_bias = probe_bias[selected_shots, ...]
                probe_current = probe_current[selected_shots, ...]

                # Make bias and current 3D (position, shot_at_a_certain_position, frame) arrays
                #    as opposed to 2D (shot number, frame) arrays
                probe_bias = probe_bias.reshape(num_positions, shots_per_position, -1)
                probe_current = probe_current.reshape(num_positions, shots_per_position, -1)
                # Dimensions of bias and current arrays:   position, shot, frame   (e.g. (71, 15, 55296))

                data_dict[hdf5_pathname][key]['bias'] = probe_bias
                data_dict[hdf5_pathname][key]['current'] = probe_current

                # If the user wants to plot the HDF5 file data
                if 'Create plots from HDF5 files' in chosen_options:
                    # Grab the data for all given shots and positions so it can easily be extrapolated to other plots
                    bias_to_plot, current_to_plot, loc_shot, filepath = (
                        obtain_data(hdf5_folder, probe_bias, probe_current, probe_position_array, langmuir_configs[key],
                                    exp_params_dict, save_plots))
                    port_face_string = (f"{langmuir_configs[key]['port']}"
                                        f"{langmuir_configs[key]['face'] if langmuir_configs[key]['face'] else ''}")
                    if "Plot bias voltage vs time for a position-shot combination" in plot_choices:
                        plot_bias_vs_time(bias_to_plot, loc_shot, exp_params_dict, dt, port_face_string)
                    if "Plot current vs time for a position-shot combination" in plot_choices:
                        plot_current_vs_time(bias_to_plot, current_to_plot, loc_shot, exp_params_dict, dt, port_face_string)
                    if "Plot the ratio between Plasma Py and v_f, v_p line temperature calculations" in plot_choices:
                        mean_slope_ratio_array = []
                        std_slope_ratio_array = []
                    for h in range(len(bias_to_plot)):
                        if ("Plot individual raw IV sweeps for a position-shot combination" in plot_choices
                                or "Plot log plot of IV sweeps for a position-shot combination" in plot_choices):
                            plot_iv_sweep(filepath, bias_to_plot[h], current_to_plot[h], port_face_string,
                                          plot_choices, save_plots, ramp_times, exp_params_dict, loc_shot[h], dt,
                                          figure_folder = figure_folder, csv_folder = csv_folder, make_csv = True)
                        # If the user wants to plot the ion saturation current vs time
                        if ("Plot Ion saturation current vs time for a position-shot combination (best in core region)"
                                in plot_choices):
                            plot_ion_isat_vs_time(dt, ramp_times, bias_to_plot[h], current_to_plot[h], exp_params_dict,
                                                  port_face_string, loc_shot[h], save_plots, filepath)

            internal_data_dict = data_dict[hdf5_pathname]

            # If the user selects to build the NETCDF files
            if ('Convert HDF5 files to NetCDF files' in chosen_options or
                    'Convert HDF5 files to NetCDF files with tanh fit for T_e' in chosen_options):

                # Create a list of bias, current data, and positions seperated out in the first dimension by probe
                bias_list = [internal_data_dict[probe]['bias'] for probe in internal_data_dict.keys()]
                current_list = [internal_data_dict[probe]['current'] for probe in internal_data_dict.keys()]
                position_list = [internal_data_dict[probe]['positions'] for probe in internal_data_dict.keys()]

                #TODO this gives the location in the langmuir_config file of the probe. Implement in build xarray?
                probe_list = [probe for probe in internal_data_dict.keys()]

                # Add a probe dimension to the bias and current we input to the dataset function
                stacked_bias = np.stack(bias_list, axis=0)
                stacked_current = np.stack(current_list, axis=0)

                # Make sure we have equal dimensions across the same x-y positions and if so passes just one of the
                # position arrays to the dataset builder
                assert all(np.array_equal(position_list[0], pa) for pa in position_list)
                shared_positions = position_list[0]

                # Determine whether we are dealing with a Hydrogen or Helium plasma
                ion_type = exp_params_dict['Ion type']

                if 'Convert HDF5 files to NetCDF files with tanh fit for T_e' in chosen_options:
                    ds = build_xarrays_tanh(stacked_bias, stacked_current, shared_positions, ramp_times, dt, langmuir_configs,
                                       ion_type, params_dict)
                else:
                    # Input data into the x-arrays function and receive a compacted x_array back
                    ds = build_xarrays(stacked_bias, stacked_current, shared_positions, ramp_times, dt, langmuir_configs,
                                       ion_type, params_dict)

                # Get the experiment parameters to be included in the saved filename
                exp_name = shortened_exp_name(ds.attrs['Exp name'])
                run_num = ds.attrs['Run number']
                gpv = ds.attrs['GP Voltage'].replace(' ','')
                b_field = ds.attrs['B-field'].replace(' ','')
                cath_curr = ds.attrs['Cathode Current'].replace(' ','')
                ion_type = ds.attrs['ion_type']

                if 'Convert HDF5 files to NetCDF files with tanh fit for T_e' in chosen_options:
                    nc_filename = (exp_name + '_' + run_num + '_' + gpv + '_' + b_field + '_' + cath_curr + '_' +
                                   ion_type + '_tanh')
                else:
                    nc_filename = exp_name + '_' + run_num + '_' + gpv + '_' + b_field + '_' + cath_curr + '_' + ion_type

                # Save the resulting x-array as a NetCDF file
                nc_save_path = os.path.join(langmuir_nc_folder, nc_filename + ".nc")
                ds.to_netcdf(nc_save_path)
                print('file saved to: ', nc_save_path)


    if 'Check NaNs' in chosen_options:
        nc_list = [f for f in os.listdir(langmuir_nc_folder) if f.endswith(".nc")]
        # nc_choice = choose_multiple_from_list(nc_list, 'NetCDF file',
        #                                       null_action="not retrieve data from NetCDF files.")
        nc_name_choice = int_choose_multiple_from_list(nc_list, 'NetCDF file',
                                                       null_action="not retrieve data from NetCDF files.")
        datasets = []
        steady_state_times_runs = []
        for choice in nc_name_choice:
            ds = xr.load_dataset(os.path.join(langmuir_nc_folder, choice))
            nan_summary(ds)

    updated_nc_folder = ensure_directory(langmuir_nc_folder + 'updated/')
    if 'Obtain plasma parameters from netCDF files' in chosen_options:
        nc_list = sorted([f for f in os.listdir(langmuir_nc_folder) if f.endswith(".nc")])
        # nc_choice = choose_multiple_from_list(nc_list, 'NetCDF file',
        #                                       null_action="Get data from updated nc files")

        nc_name_choice = int_choose_multiple_from_list(nc_list, 'NetCDF file',
                                              null_action="Get data from updated nc files")

        # Initialize our master list of paths
        selected_file_paths = []

        # Convert chosen indices into absolute paths
        if nc_name_choice:
            for choice in nc_name_choice:
                selected_file_paths.append(os.path.join(langmuir_nc_folder, choice))
        updated = ask_yes_or_no('Retrieve data from updated nc files? (y/n) ')
        updated_nc_list = None
        updated_nc_choice = None
        if updated:
            updated_nc_list = sorted([f for f in os.listdir(updated_nc_folder) if f.endswith(".nc")])
            # updated_nc_choice = choose_multiple_from_list(updated_nc_list, 'NetCDF file',
            #                                       null_action="Don't make any plots")

            updated_name_nc_choice = int_choose_multiple_from_list(updated_nc_list, 'NetCDF file',
                                                          null_action="Don't make any plots")
            # Convert chosen indices into absolute paths and add to the master list
            if updated_name_nc_choice:
                for choice in updated_name_nc_choice:
                    selected_file_paths.append(os.path.join(updated_nc_folder, choice))

        possible_plots = ['contour',
                          'contour_subplots - Only for 0 probe',
                          'Michael gradient plot',
                          'Show Isat time series',
                          'Show steady state',
                          'Isat radial plot',
                          'Radial plot',
                          'Overlapping Radial plot',
                          'Plot center of gradients vs experiment parameters',
                          'Dimensionless comparison'
                          ]

        # idx_plot_choices = choose_multiple_from_list(possible_plots, 'action', null_action='not plot data')
        # plot_choices = []
        # for idx in idx_plot_choices:
        #     plot_choices.append(possible_plots[idx])
        #
        plot_choices = int_choose_multiple_from_list(possible_plots, 'action', null_action='not plot data')

        if ('Show steady state' in plot_choices or
                'Isat radial plot' in plot_choices or
                'Radial plot' in plot_choices or
                'Overlapping Radial plot' in plot_choices or
                'Dimensionless comparison' in plot_choices):

            a_datasets = []
            for i in range(len(selected_file_paths)):
                pathname = selected_file_paths[i]
                try:
                    ds = xr.open_dataset(pathname, engine="netcdf4")
                except FileNotFoundError:
                    ds = xr.open_dataset(pathname, engine="netcdf4")

                run_identifier = f_run_identifier(ds=ds)

                need_to_append = False
                for probe in range(ds.sizes['probe']):
                    if (f'steady state start probe {probe}' not in ds.attrs or
                            f'steady state end probe {probe}' not in ds.attrs):
                        need_to_append = True

                if 'Show steady state' in plot_choices and not need_to_append:
                    redo_ss = ask_yes_or_no('Redo steady state (y/n)? ')
                else:
                    redo_ss = False

                ds.close()  # Close the read-only session

                if need_to_append or redo_ss:
                    # Open in read only mode with the option of appending
                    ds = xr.open_dataset(pathname, mode='r+', engine="netcdf4")

                    probe_dict = {}
                    for probe in range(ds.sizes['probe']):
                        print(f"\n--- Processing Probe {probe} ---")

                        probe_dict[probe] = {}
                        mean_data = ds['t_e'].sel(probe=probe).mean('shot')
                        std_data = ds['t_e'].sel(probe=probe).std('shot')

                        t_e_filtered_data = filter_data(mean_data, std_data)
                        where_nans = t_e_filtered_data.isnull()
                        n_e_filtered_data = ds['n_e'].sel(probe=probe).mean('shot').where(~where_nans)
                        nu_ei_filtered_data = ds['nu_ei'].sel(probe=probe).mean('shot').where(~where_nans)

                        # We still need to call it once here just to get range_tot for the data arrays
                        fig_temp, _ , range_tot = plot_time_series(ds, probe, run_identifier, return_range=True)
                        plt.close(fig_temp)  # Immediately close it, find_steady_state will draw it properly

                        zero_index = range_tot.index(0)
                        if len(range_tot) >= 5:
                            search_range = range_tot[(zero_index - 2): (zero_index + 3)]
                        elif len(range_tot) >= 3:
                            search_range = range_tot[(zero_index - 1): (zero_index + 2)]
                        else:
                            search_range = range_tot[zero_index]

                        t_e_data_arrays = [t_e_filtered_data.sel(x=x_val, y=0) for x_val in search_range]
                        n_e_data_arrays = [n_e_filtered_data.sel(x=x_val, y=0) for x_val in search_range]

                        # Extract previous attributes if they exist
                        prev_start = ds.attrs.get(f'steady state start probe {probe}', None)
                        prev_end = ds.attrs.get(f'steady state end probe {probe}', None)

                        # Pass everything into find_steady_state
                        min_time, max_time = find_steady_state(
                            t_e_data_arrays,
                            n_e_data_arrays,
                            ds=ds,
                            probe=probe,
                            run_identifier=run_identifier,
                            prev_start=prev_start,
                            prev_end=prev_end
                        )

                        ds.attrs[f'steady state start probe {probe}'] = min_time
                        ds.attrs[f'steady state end probe {probe}'] = max_time
                        ds.attrs[f'total range probe {probe}'] = range_tot

                    ds.to_netcdf(pathname, mode='a', engine='netcdf4')
                    ds.close()

        datasets = []
        for i in range(len(selected_file_paths)):
            pathname = selected_file_paths[i]
            # Select data sets to plot from saved .nc files in the selected folder and load them in read only mode
            # because nothing might need to be appended
            ds = xr.load_dataset(pathname)
            datasets.append(ds)

        # TODO Make this a function and input it into individual if statements below
        if ('contour' in plot_choices or
            'contour_subplots - Only for 0 probe' in plot_choices or
            'Michael gradient plot' in plot_choices):
            # What variables are in common between all the datasets selected
            common_vars = set.intersection(*[set(dataset.data_vars.keys()) for dataset in datasets])
            diagnostic_name_dict = {var: datasets[0][var].attrs.get("long_name",var) for var in common_vars}


            diagnostics_to_plot_list = get_diagnostics_to_plot(diagnostic_name_dict)

        if 'contour' in plot_choices:
            i = 0
            show_plot = ask_yes_or_no('See plots? (y/n) ')
            save_plot = ask_yes_or_no('Save plots? (y/n) ')

            check_shots = ask_yes_or_no('Plot shots individually? (y/n) ')
            if check_shots:
                shot_nums = datasets[0]['shot'].values
                min_shot = min(shot_nums)
                max_shot = max(shot_nums)
                prompt = 'Select which shots to plot '
                shots_to_plot = allow_only_ints(prompt, min_condition=min_shot, max_condition=max_shot, accept_empty=True)
            else:
                shots_to_plot = None

            for dataset in datasets:
                for plot_diagnostic in diagnostics_to_plot_list:
                    print(f'Plotting diagnostic: {plot_diagnostic}')
                    if not check_shots:
                        plot_std = ask_yes_or_no("Also plot standard deviation? (y/n) ")
                        filt_data = ask_yes_or_no("Filter data? (y/n) ") if plot_std else False
                    else:
                        plot_std = False
                        filt_data = False
                    for probe in range(ds.sizes['probe']):

                        run_identifier = f_run_identifier(ds = dataset)

                        contour_plot(dataset, plot_diagnostic, probe, run_identifier, figure_folder=figure_folder,
                                     plot_std = plot_std, filt_data = filt_data,
                                     check_shots = check_shots, shots_to_plot = shots_to_plot,
                                     show_plot = show_plot, save_plots = save_plot)
                i += 1

        if 'contour_subplots - Only for 0 probe' in plot_choices:
            for diagnostic_to_plot in diagnostics_to_plot_list:
                contour_subplots(datasets,diagnostic_to_plot)

        if 'Show Isat time series' in plot_choices:
            plot_ion_sat_curr_vs_time(datasets, figure_folder)

        if 'Show steady state' in plot_choices:
            for i, dataset in enumerate(datasets):
                run_identifier = f_run_identifier(ds=dataset)
                for probe in range(dataset.sizes['probe']):
                    show_steady_state(dataset, probe, run_identifier)

        if 'Dimensionless comparison' in plot_choices:
            a = core_radius.to(u.m)
            L = plasma_length.to(u.m)
            data_to_plot_dicts = []
            probe_num_dict = []
            # Normalized to Jan 2024 run 9
            norm_ds_filename = 'Jan2024_16_60.0V_1.0kG_3350.0A_H+_tanh.nc'
            norm_ds_path = os.path.join(langmuir_nc_folder, norm_ds_filename)
            norm_ds = xr.load_dataset(norm_ds_path)

            # Datasets to check dictionary
            i = 0
            # list_run_identifiers = []
            filename_list = []
            for idx, dataset in enumerate(datasets):
                run_identifier = f_run_identifier(ds = dataset)
                # print(run_identifier)
                # list_run_identifiers.append(run_identifier)
                filename = selected_file_paths[idx].split('/')[-1]


                ion_mass, z_eff, e_charge, b_field = dim_num_params(filename, dataset)
                probe_dict = compute_dimesionless_plots(dataset, ion_mass, z_eff, e_charge, b_field, a, L,
                                                        run_identifier)
                probe_num_dict.append(len(probe_dict.keys()))
                data_to_plot_dicts.append(probe_dict)
                i += 1

            # Normalized dataset dictionary
            norm_run_identifier = f_run_identifier(ds = norm_ds, filename = norm_ds_filename)
            norm_ion_mass, norm_z_eff, norm_e_charge, norm_b_field = dim_num_params(norm_ds_filename,norm_ds)
            norm_probe_dict = compute_dimesionless_plots(norm_ds, norm_ion_mass,
                                                    norm_z_eff, norm_e_charge,
                                                    norm_b_field, a, L,
                                                    norm_run_identifier)
            norm_num_probes = len(norm_probe_dict.keys())
            # At the moment, to get normalized values -- one probe must be true

            j = 0
            fig, ax = plt.subplots(2,2,figsize=(10,12))
            ax = ax.flatten()
            old_nu = False
            one_probe = True
            # Get dimensionless matches visually
            vis_plot = False
            save_csv = True


            clor, mark = determine_colors(datasets, one_probe, filename_list)
            if old_nu:
                clor_old = clor[::-1]

            # Gets the first probe in the normalizing dataset
            norm_min_probe = [min([int(k) for k in norm_probe_dict.keys()])][0]
            normalizing_rhostar = norm_probe_dict[norm_min_probe]['rhostar']
            normalizing_nu_ei = norm_probe_dict[norm_min_probe]['nu_eff']
            norm_x_vals = norm_probe_dict[norm_min_probe]['x_vals']

            avg_norm_rhostar_dict = {}
            avg_norm_nu_ei_dict = {}
            combined_val_dict = {}
            abs_combined_val_dict = {}
            for data in data_to_plot_dicts:
                if one_probe:
                    probes_to_plot = [min([int(k) for k in data.keys()])]
                else:
                    probes_to_plot = data.keys()
                for probe in probes_to_plot:
                    rhostar = data[probe]['rhostar']
                    nu_ei = data[probe]['nu_eff']
                    x_values = data[probe]['x_vals']
                    run_identifier = data[probe]['run identifier']

                    if old_nu:
                        nu_ei_old = data[probe]['nu_eff_old']
                    if vis_plot:
                        x_values_mask = ((x_values >= -10) & (x_values <= 10))
                        x_vals_masked = x_values[x_values_mask]
                        masked_nu = nu_ei[x_values_mask]
                        masked_rhostar = rhostar[x_values_mask]

                        ax[0].plot(x_vals_masked ,masked_rhostar,color=clor[j], marker = mark[j], linestyle = 'None',
                                   label=f'{run_identifier} probe: {probe}')
                        # ax[1].plot(x_values, nu_ei, color=clor[j], marker = mark[j], linestyle = 'None',
                        #            label=f'{run_identifier} probe: {probe}')
                        ax[1].plot(x_vals_masked, masked_nu, color=clor[j], marker=mark[j], linestyle='None',
                                   label=f'{run_identifier} probe: {probe}')
                        if old_nu:
                            ax[1].plot(x_values, nu_ei_old, color=clor_old[j], marker = mark[j], linestyle = 'None',
                                       label=rf'{run_identifier} probe: {probe} OLD $\nu$')
                        ax[2].plot(0,0,color=clor[j], marker = mark[j], linestyle = 'None',
                                   label=f'{run_identifier} probe: {probe}')

                        if old_nu:
                            ax[3].plot(0,0, color = clor[j], marker = mark[j], linestyle = 'None',
                                        label=f'{run_identifier} probe: {probe}')
                            ax[3].plot(0,0, color = clor_old[j], marker = mark[j], linestyle = 'None',
                                        label= str(f'{run_identifier} probe: {probe} ' +
                                                   r'$\nu_{eff} = \frac{\nu_{ei}^2 a^2 m_i}{T_{e \text{ Joules}}}$'))
                    else:
                        # Build masks
                        same_x_values_mask = np.isin(x_values, norm_x_vals)
                        # Masked parameters so they are all the same length
                        len_x_values_masked = x_values[same_x_values_mask]
                        len_rhostar_masked = rhostar[same_x_values_mask]
                        len_nu_ei_masked = nu_ei[same_x_values_mask]

                        # Do the same to the norm
                        same_norm_x_mask = np.isin(norm_x_vals, x_values)
                        len_norm_x_vals_masked = norm_x_vals[same_norm_x_mask]
                        len_norm_rhostar_masked = normalizing_rhostar[same_norm_x_mask]
                        len_norm_nu_ei_masked = normalizing_nu_ei[same_norm_x_mask]

                        # Only consider the core 10 cm for dim matching
                        x_values_mask = ((x_values >= -5) & (x_values <= 5))
                        x_vals_masked = x_values[x_values_mask]
                        masked_nu = nu_ei[x_values_mask]
                        masked_rhostar = rhostar[x_values_mask]
                        masked_norm_rhostar = len_norm_rhostar_masked[x_values_mask]
                        masked_norm_nu_ei = len_norm_nu_ei_masked[x_values_mask]

                        # Perform the normalizing
                        normalized_rhostar = masked_rhostar / masked_norm_rhostar
                        normalized_nu_ei = masked_nu / masked_norm_nu_ei

                        # Compute single parameter values to make it easy to determine the best matches
                        avg_norm_rhostar = np.mean(normalized_rhostar)
                        avg_norm_rhostar_dict[run_identifier] = avg_norm_rhostar

                        avg_norm_nu_ei = np.mean(normalized_nu_ei)
                        avg_norm_nu_ei_dict[run_identifier] = avg_norm_nu_ei

                        combined_val = avg_norm_rhostar - avg_norm_nu_ei
                        combined_val_dict[run_identifier] = combined_val

                        abs_combined_val = np.abs(combined_val)
                        abs_combined_val_dict[run_identifier] = abs_combined_val

                        ax[0].plot(x_vals_masked, normalized_rhostar, color=clor[j], marker=mark[j], linestyle='None',
                                   label=f'{run_identifier} probe: {probe}')

                        ax[1].plot(x_vals_masked, normalized_nu_ei, color=clor[j], marker=mark[j], linestyle='None',
                                   label=f'{run_identifier} probe: {probe}')

                        ax[2].plot(0, 0, color=clor[j], marker=mark[j], linestyle='None',
                                   label=f'{run_identifier} probe: {probe}')

                        ax[3].plot(avg_norm_rhostar, avg_norm_nu_ei, color=clor[j], marker=mark[j], linestyle='None',
                                   label=f'{run_identifier} probe: {probe}')

                    j += 1

            # ax[0].legend(loc='best')
            ax[0].set_xlabel('x (cm)')
            if vis_plot:
                ax[0].set_ylabel(r'$\rho^* = \frac{\sqrt{m_i T_{e\text{ Joules}}}}{eBa}$')
            else:
                ax[0].set_ylabel(r'Normalized $\rho^* = \frac{\sqrt{m_i T_{e\text{ Joules}}}}{eBa}$')

            # ax[1].legend(loc='best')
            ax[1].set_xlabel('x (cm)')
            # ax[1].set_ylabel(r'$\nu_{eff} = \nu_{ei}L\sqrt{\frac{m_i}{T_{e\text{ Joules}}}}$')
            if vis_plot:
                ax[1].set_ylabel(r'$\nu_{eff} = \frac{2\pi}{L\nu_{ei}}\sqrt{\frac{T_{e\text{ Joules}}}{m_i}}$')
            else:
                ax[1].set_ylabel(r'$\nu_{eff} = \frac{2\pi}{L}\frac{3\sqrt{\pi}\cdot(4\pi \epsilon_0)^2m_e^2\cdot 2(\sqrt{2T_i/m_i + 2T_e/m_i})^3}{16\pi n_eZe^4\text{ln}(\Lambda)}\sqrt{\frac{T_{e\text{ Joules}}}{m_i}}$')

            ax[1].set_title('Howe Prescription')

            ax[2].legend(loc='best')

            if old_nu:
                ax[3].legend(loc='best')

            if not vis_plot:
                ax[3].set_xlabel(r'Normalized $\rho^*$')
                ax[3].set_ylabel(r'Normalized $\nu_{eff}$')

            if vis_plot:
                plt.suptitle(f'Dimensionless parameters')
            else:
                plt.suptitle(f'Dimensionless parameters \n normalized by {norm_run_identifier}')
            plt.tight_layout()
            plt.show()

            if save_csv:
                # Build CSV
                rows = []
                print(avg_norm_rhostar_dict.keys())

                for run in avg_norm_rhostar_dict.keys():
                    rows.append({
                        "run": run,
                        "normalized rhostar": avg_norm_rhostar_dict[run],
                        "normalized nu_eff": avg_norm_nu_ei_dict[run],
                        "norm rhostar - norm nu_eff": combined_val_dict[run],
                        "|norm rhostar - norm nu_eff|": abs_combined_val_dict[run]
                    })

                print(rows)

                # Create DataFrame
                df = pd.DataFrame(rows)

                df["|norm rhostar - norm nu_eff|"] = df["norm rhostar - norm nu_eff"].abs()
                # Optional: sort by best match (smallest absolute difference)
                df = df.sort_values("|norm rhostar - norm nu_eff|")

                # Save to CSV
                csv_filename = f"dimensionless_comparison_normalized_by_{norm_run_identifier}_{len(datasets)}_datasets.csv"

                csv_folder = ensure_directory(langmuir_nc_folder + 'dim_match_csv/')
                csv_path = os.path.join(csv_folder, csv_filename)

                df.to_csv(csv_path, index=False)
                print(f"Saved CSV to: {csv_path}")

        if 'Isat radial plot' in plot_choices:
            see_plots = ask_yes_or_no('See plots? (y/n) ')
            save_plots = ask_yes_or_no('Save plots? (y/n) ')

            build_isat_radial_plot(datasets, pathnames = selected_file_paths, figure_folder = figure_folder,
                                   see_plots = see_plots, save_plots = save_plots)
        if "Radial plot" in plot_choices:

            make_presentable = ask_yes_or_no('Make for presentation rather than for analysis? (y/n) ')
            see_intermediate_plots = ask_yes_or_no('See individual radial plots (T_e and n_e)? (y/n) ')
            save_plots = ask_yes_or_no('Save plots? (y/n) ')
            one_probe = ask_yes_or_no('One probe? (y/n) ')
            lined_grads = ask_yes_or_no('Plot radial plot with lines indicating where the gradients are? (y/n) ')
            if not lined_grads:
                shaded_grads = ask_yes_or_no(
                    'Make shaded regions along the outside edge (|x| >= |15|) based off of \n '
                    'temperature and density gradients? (y/n) ')
            else:
                shaded_grads = False
            fit_lines = ask_yes_or_no('Plot a linear fit of the data bounded by the gradient regions? (y/n) ')
            build_radial_plot(datasets, selected_file_paths,
                              figure_folder,
                              make_presentable=make_presentable, see_temp_and_dens_plots=see_intermediate_plots,
                              lines=lined_grads, shaded=shaded_grads,
                              plot_final_fits=fit_lines, save_plots=save_plots,
                              from_main=True, one_probe=one_probe, hdf5_folder=hdf5_folder,
                              updated_nc_folder = updated_nc_folder)

        if 'Overlapping Radial plot' in plot_choices:
            see_plots = ask_yes_or_no('See final overlapping radial plot? (y/n) ')
            save_plots = ask_yes_or_no('Save final overlapping radial plot? (y/n) ')
            overlapping_radial_plots(datasets, selected_file_paths, figure_folder,see_plots=see_plots, save_plots=save_plots,
                                     one_probe = True)



        if 'Plot center of gradients vs experiment parameters' in plot_choices:
            options_to_plot = [
                'B-field',
                'GP Voltage',
                'Cathode Current',
                'Gradient Centers',
                'Core width'
            ]
            # plot_list_idxs = choose_multiple_from_list(options_to_plot, 'plots')
            # plot_list = [options_to_plot[i] for i in plot_list_idxs]
            plot_list = options_to_plot
            save_plots = ask_yes_or_no('Save plots? (y/n) ')
            show_plots = ask_yes_or_no('Show plots? (y/n) ')
            make_presentable = ask_yes_or_no('Make for presentation rather than for analysis? (y/n) ')
            center_grads_vs_experimental_params(datasets, figure_folder=figure_folder, axes=None,
                                                save_plots=save_plots, show_plots=show_plots,
                                                make_presentable=make_presentable)

    if "Obtain Fluctuations from HDF5" in chosen_options:
        print("\n===== Flux probe analysis =====")
        fluctuations_nc_files = sorted([f for f in os.listdir(flux_nc_folder) if f.endswith(".nc")])
        files_in_flux_nc = os.listdir(flux_nc_folder)

        # combined_files = (os.listdir(march_folder + "flux_nc/") + os.listdir(november_folder + "flux_nc/") +
        #                   os.listdir(january_folder + "flux_nc/"))
        #
        # files_in_flux_nc = combined_files
        print("Choose one of the following NetCDF files to analyze,\n"
              "or press Enter to retrieve data from HDF5 files")
        # choice_indices = choose_multiple_from_list(fluctuations_nc_files, "Fluctuations NetCDF file",
        #                                            null_action="retrieve data from HDF5 files.")

        choice_names = int_choose_multiple_from_list(fluctuations_nc_files, "Fluctuations NetCDF file",
                                                   null_action="retrieve data from HDF5 files.")
        files_to_plot = []
        if choice_names:
            datasets = []
            for name in choice_names:
                # for folder in [march_folder, november_folder, january_folder]:
                #     try:
                #         flux_nc_folder = folder + "flux_nc/"
                try:
                    datasets.append(xr.open_dataset(flux_nc_folder + name))
                    files_to_plot.append(name.split('.nc')[0])
                except:
                    pass
            plot_choices = ['Plot time series for a single x for each dataset',
                            'Spectrogram for a single dataset',
                            'Dimensionless matches, radial plot, and spectrograms',
                            '\delta n/n vs x for multiple datasets',
                            '\delta n/n vs L_n for multiple datasets',
                            "Other plots from Michael's structure"]

            # plot_choice_idxs = choose_multiple_from_list(plot_choices, "figure types available to plot")
            # plots_to_make = [plot_choices[i] for i in plot_choice_idxs]
            plots_to_make = int_choose_multiple_from_list(plot_choices, "figure types available to plot")




            if ('Plot time series for a single x for each dataset' in plots_to_make
                    or 'Spectrogram for a single dataset' in plots_to_make
                    or 'Dimensionless matches, radial plot, and spectrograms' in plots_to_make
                    or '\delta n/n vs x for multiple datasets' in plots_to_make
                    or '\delta n/n vs L_n for multiple datasets' in plots_to_make):
                quantities = ['density', 'isat', 'vf', 'dvf']
                if len(plots_to_make) == 1 and ('\delta n/n vs x for multiple datasets' in plots_to_make or
                                                '\delta n/n vs L_n for multiple datasets' in plots_to_make):
                    # plot_type_idxs =[0]
                    plot_type_list = ['density']

                else:
                    # plot_type_idxs = choose_multiple_from_list(quantities, "Quantities to plot")
                    plot_type_list = int_choose_multiple_from_list(quantities, "Quantities to plot")
                default_fig_height = 6.4
                default_fig_width = 4.8
                # plot_type_list = [quantities[j] for j in plot_type_idxs]
                if ('\delta n/n vs x for multiple datasets' in plots_to_make or
                    '\delta n/n vs L_n for multiple datasets' in plots_to_make)  and 'density' not in plot_type_list:
                    plot_type_list.append('density')

                for quantity in plot_type_list:
                    print('Quantity to plot = ', quantity)
                    if 'Plot time series for a single x for each dataset' in plots_to_make:
                        print('time series for single x for each dataset')
                        save_plots = ask_yes_or_no('Save plots? (y/n) ')
                        see_plots = ask_yes_or_no('See plots? (y/n) ')
                        save_full_folder = ensure_directory(figure_folder + "Time series/")
                        valid_runs = generate_colors_valid_ds_ri(datasets, files_to_plot, langmuir_nc_folder, mach_nc_folder,
                                                                 make_presentable = False)

                        select_shots = ask_yes_or_no('Look at specific shots? '
                                                     'Choosing no averages over all shots (y/n) ')

                        all_same_x = ask_yes_or_no("Choose xs for all datasets? "
                                                   "\n y - Choose x's once"
                                                   "\n n - Choose x's for every dataset individually\n")

                        if select_shots:
                            all_same_shot = ask_yes_or_no("Choose shots for all datasets? "
                                                       "\n y - Choose shots once"
                                                       "\n n - Choose shots for every dataset individually \n")
                        else:
                            all_same_shot = True

                        reference_dict = valid_runs[0]
                        reference_lang_ds = reference_dict['lang_ds']

                        x_list = []
                        if all_same_x:
                            possible_xs = reference_lang_ds['x'].values
                            min_x = np.min(possible_xs)
                            max_x = np.max(possible_xs)
                            x_list = allow_only_ints('Select which x values to see the time series for',
                                                     min_condition = min_x, max_condition = max_x, accept_empty = False)

                        shot_list = []
                        posible_shots = reference_lang_ds['shot'].values
                        if select_shots and all_same_shot:
                            min_shot = np.min(posible_shots)
                            max_shot = np.max(posible_shots)
                            shot_list = allow_only_ints('Select which shot values to see the time series for, '
                                                        'selecting nothing averages over all shots',
                                                     min_condition=min_shot, max_condition=max_shot)
                        elif not select_shots and all_same_shot:
                            shot_list = [None]

                        for idx, run in enumerate(valid_runs):
                            lang_ds = run['lang_ds']
                            fluct_ds = run['fluct_ds']

                            if not all_same_x:
                                possible_xs = lang_ds['x'].values
                                min_x = np.min(possible_xs)
                                max_x = np.max(possible_xs)
                                x_list = allow_only_ints('Select which x values to see the time series for',
                                                         min_condition=min_x, max_condition=max_x, accept_empty = False)
                            if select_shots and not all_same_shot:
                                possible_shots = lang_ds['shot'].values
                                min_shot = np.min(possible_shots)
                                max_shot = np.max(possible_shots)
                                shot_list = allow_only_ints('Select which shot values to see the time series for'
                                                            ', selecting nothing averages over all shots',
                                                            min_condition=min_shot, max_condition=max_shot)
                            if shot_list == []:
                                shot_list = [None]
                            for z in fluct_ds.coords["z"].values:
                                lang_z = lang_ds.coords["z"].values
                                lang_z_idx = np.argmin(np.abs(lang_z - z))
                                lang_z_times = lang_ds.coords["time"].values
                                min_time = np.min(lang_z_times)
                                max_time = np.max(lang_z_times)
                                time = (min_time, max_time)
                                min_steady_state = round(lang_ds.attrs[f"steady state start probe {lang_z_idx}"])
                                max_steady_state = round(lang_ds.attrs[f"steady state end probe {lang_z_idx}"])
                                for x in x_list:
                                    for shot in shot_list:
                                        layout = [[1]]
                                        fig, axes, letters = build_subplots(layout)
                                        ax = axes[letters[0]]
                                        get_time_series(fluct_ds[quantity].sel(z=z), x=x, time = time, shot=shot, z=z,
                                                        plot=True, axis = ax)
                                        ax.axvspan(min_steady_state, max_steady_state, color='green', alpha=0.2,
                                                   label = 'Steady State')

                                        lgd = fig.legend(loc='lower center', bbox_to_anchor=(0.5, -.18), ncol=2)

                                        fig.canvas.draw()
                                        legend_height_inches = lgd.get_window_extent().height / fig.dpi
                                        standard_w, standard_h = fig.get_size_inches()
                                        fig.set_size_inches(standard_w, standard_h + legend_height_inches)
                                        calculated_bottom = legend_height_inches / (standard_h + legend_height_inches)
                                        fig.subplots_adjust(bottom=calculated_bottom, wspace=0.2)

                                        if save_plots:
                                            ri = f_run_identifier(lang_ds)
                                            run_save_folder = ensure_directory(save_full_folder + ri + f'_z_{z:.2f}/')
                                            save_folder = ensure_directory(run_save_folder + f'{quantity}/')
                                            fig_save_name = f'x_{x}_shot_{shot}' if shot is not None \
                                                else f'x_{x}_shot_ALL'
                                            plt.savefig(save_folder + fig_save_name + '.png', bbox_inches='tight')
                                            print(f'Saved plot to {save_folder}{fig_save_name}.png')
                                        if see_plots:
                                            plt.show()

                                        plt.close()

                    if 'Spectrogram for a single dataset' in plots_to_make:
                        print('Spectrogram for a single dataset')
                        j = 0
                        for dataset in datasets:
                            filename = files_to_plot[j]
                            if filename + '_tanh.nc' in os.listdir(langmuir_nc_folder):
                                lang_filename = langmuir_nc_folder + filename + '_tanh.nc'
                                lang_ds = xr.open_dataset(lang_filename)
                                # print(lang_ds)
                                for z in dataset.coords["z"].values:
                                    # print('z: ', z)
                                    lang_ds_zs = lang_ds['z'].values
                                    #   print('langmuir zs: ', lang_ds_zs)
                                    lang_z_idx = np.abs(lang_ds_zs - z).argmin()
                                    # print('langmuir z selected index: ', lang_z_idx)

                                    lang_probe_ss_start = round(lang_ds.attrs[f'steady state start probe {lang_z_idx}'])
                                    lang_probe_ss_end = round(lang_ds.attrs[f'steady state end probe {lang_z_idx}'])
                                    bin = ast.literal_eval(f'({lang_probe_ss_start}, {lang_probe_ss_end})')
                                    print('bin: ', bin)
                                    x = None
                                    shot = None
                                    get_radial_spectrogram(dataset[quantities[i]].sel(z=z), x=x, bin=bin, shot=shot,
                                                           z=z, plot=True, axis=None, filename= filename)
                            j += 1

                    if 'Dimensionless matches, radial plot, and spectrograms' in plots_to_make:
                        print('Dimensionless matches, radial plot, and spectrograms')
                        one_probe = ask_yes_or_no(
                            'Consider only one probe (to be phased out once steady state definition is fixed) (y/n) ')
                        make_presentable = ask_yes_or_no('Make for presentation rather than for analysis? (y/n) ')
                        save_plots = ask_yes_or_no('Save plots? (y/n) ')
                        see_plots = ask_yes_or_no('See plots? (y/n) ')
                        psd_analysis = ask_yes_or_no('Plot PSD? (y/n) ')

                        a = core_radius.to(u.m)
                        L = plasma_length.to(u.m)


                        valid_runs = generate_colors_valid_ds_ri(datasets, files_to_plot, langmuir_nc_folder, mach_nc_folder,
                                                                 make_presentable)
                        split_runs_lists = []

                        if ask_yes_or_no('Split datasets? (y/n) '):
                            valid_filenames = [run['filename'] for run in valid_runs]

                            # dataset_choices returns indices (e.g., [0, 1, 2])
                            # dataset_choices = choose_multiple_from_list(valid_filenames, 'Datasets in first round')
                            dataset_choices = int_choose_multiple_from_list(valid_filenames,
                                                                            'Datasets in first round', return_idxs=True)

                            # Use enumerate to check if the index (i) is in your chosen indices
                            group_1 = [run for i, run in enumerate(valid_runs) if i in dataset_choices]
                            group_2 = [run for i, run in enumerate(valid_runs) if i not in dataset_choices]

                            if group_1: split_runs_lists.append(group_1)
                            if group_2: split_runs_lists.append(group_2)
                        else:
                            split_runs_lists = [valid_runs]

                        # 5. Process Each Group
                        for group_idx, run_group in enumerate(split_runs_lists):
                            print(f'\n--- Processing Group {group_idx + 1} ---')
                            print('Run IDs: ', [run['run_id'] for run in run_group])

                            temp_x_turb = []
                            dens_x_turb = []

                            for run in run_group:
                                fluct_ds = run['fluct_ds']
                                lang_ds = run['lang_ds']

                                if one_probe:
                                    # Safely get the max z-probe
                                    z_vals = fluct_ds.coords["z"].values
                                    fluct_probes = [z_vals[np.argmax(z_vals)]]
                                else:
                                    fluct_probes = fluct_ds.coords["z"].values

                                for probe in fluct_probes:
                                    lang_ds_zs = lang_ds['z'].values
                                    lang_z_idx = np.abs(lang_ds_zs - probe).argmin()

                                    temp_x, dens_x = temp_dens_spectrogram(
                                        lang_ds, fluct_ds, quantity,
                                        run['lang_pathname'], figure_folder,
                                        dataset_color=run['color'],
                                        dataset_mark=run['marker'],
                                        make_presentable=make_presentable,
                                        # Note: using make_presentable instead of one_probe here, assuming that was a bug
                                        fluc_probez=probe,
                                        lang_z_idx=lang_z_idx,
                                        save_plots=save_plots,
                                        see_plots=see_plots,
                                        run_identifier=run['run_id']
                                    )
                                    if psd_analysis:
                                        print('psd_analysis ')
                                        choose_x = ask_yes_or_no('Choose x location for PSD Analysis? (y/n) ')
                                        if choose_x:
                                            while True:
                                                user_input = input('Enter the x location index for temperature '
                                                                   'gradient PSD (integer): ')
                                                try:
                                                    temp_selected_x = int(user_input)
                                                    break
                                                except ValueError:
                                                    print("Invalid input. Please enter a valid whole number.")
                                                    time.sleep(0.1)
                                            temp_x_turb.append(temp_selected_x)

                                            while True:
                                                user_input = input('Enter the x location index for density '
                                                                   'gradient PSD (integer): ')
                                                try:
                                                    dens_selected_x = int(user_input)
                                                    break
                                                except ValueError:
                                                    print("Invalid input. Please enter a valid whole number.")
                                                    time.sleep(0.1)
                                            dens_x_turb.append(dens_selected_x)
                                        else:
                                            temp_x_turb.append(temp_x)
                                            dens_x_turb.append(dens_x)

                            # Re-extract lists for the final plotting functions specific to this group
                            group_fluct_ds = [run['fluct_ds'] for run in run_group]
                            group_filenames = [run['filename'] for run in run_group]
                            group_colors = [run['color'] for run in run_group]
                            group_marks = [run['marker'] for run in run_group]
                            group_run_ids = [run['run_id'] for run in run_group]

                            dimless_plots(
                                group_fluct_ds, group_filenames, figure_folder, langmuir_nc_folder,
                                group_colors, group_marks, a, L,
                                make_presentable=make_presentable, save_plots=save_plots, see_plots=see_plots,
                                run_identifiers=group_run_ids
                            )

                            if psd_analysis:
                                psd_plot(
                                    group_fluct_ds, group_filenames, quantity, temp_x_turb, dens_x_turb, figure_folder,
                                    make_presentable=make_presentable,
                                    langmuir_nc_folder=langmuir_nc_folder, save_plots=save_plots, see_plots=see_plots,
                                    dataset_colors=group_colors, run_identifiers=group_run_ids
                                )
                        plt.close('all')

                    if ('\delta n/n vs x for multiple datasets' in plots_to_make or
                            '\delta n/n vs L_n for multiple datasets' in plots_to_make):
                        print('\delta n/n plot for multiple datasets')

                        if quantity == 'density':
                            plot_vs_x = '\delta n/n vs x for multiple datasets' in plots_to_make
                            plot_vs_L_n = '\delta n/n vs L_n for multiple datasets' in plots_to_make

                            dens_grads = ask_yes_or_no(
                                'Show density gradient locations? (y/n) ') if plot_vs_x else plot_vs_L_n
                            temp_grads = ask_yes_or_no('Show temperature gradients? (y/n) ') if plot_vs_x else False

                            one_probe = ask_yes_or_no('Consider only one probe (y/n) ')
                            make_presentable = ask_yes_or_no('Make for presentation rather than for analysis? (y/n) ')

                            plt.rcParams.update({
                                'font.size': 24, 'axes.labelsize': 24, 'axes.titlesize': 24,
                                'xtick.labelsize': 24, 'ytick.labelsize': 24, 'legend.fontsize': 15,
                                'axes.formatter.use_mathtext': True, 'lines.linewidth': 3,
                                'lines.markersize': 8, 'errorbar.capsize': 5
                            })
                            if make_presentable:
                                plt.rcParams.update(
                                    {'figure.facecolor': 'none', 'axes.facecolor': 'none', 'savefig.transparent': True})

                            save_plots = ask_yes_or_no('Save plots? (y/n) ')
                            see_plots = ask_yes_or_no('See plots? (y/n) ')
                            split_x = ask_yes_or_no(
                                'Split into the left and right hand side of LAPD (-x vs +x)? (y/n) ')

                            valid_runs = generate_colors_valid_ds_ri(datasets, files_to_plot, langmuir_nc_folder,
                                                                     mach_nc_folder, make_presentable)
                            ref_lang_ds = valid_runs[0]['lang_ds'] if len(valid_runs) > 0 else None

                            # --- USER INPUT FOR PER-DATASET CUSTOM X POSITIONS (FOR L_N SAMPLING) ---
                            run_target_x = {}
                            if plot_vs_L_n:
                                use_custom_x = ask_yes_or_no('Specify target x positions for L_n sampling? (y/n) ')
                                if use_custom_x:
                                    print(
                                        '\nEnter target x positions for each dataset (leave blank to use fallback median logic):')
                                    for run in valid_runs:
                                        ri = run['run_id']
                                        print(f'\n--- Dataset: {ri} ---')
                                        if split_x:
                                            raw_lhs = input('  Left side (-x, comma-separated e.g. -18.0, -15.0): ')
                                            raw_rhs = input('  Right side (+x, comma-separated e.g. 13.0, 17.0): ')

                                            lhs_parsed = [float(val.strip()) for val in raw_lhs.split(',') if
                                                          val.strip()]
                                            rhs_parsed = [float(val.strip()) for val in raw_rhs.split(',') if
                                                          val.strip()]

                                            run_target_x[ri] = {
                                                'lhs': lhs_parsed if lhs_parsed else None,
                                                'rhs': rhs_parsed if rhs_parsed else None
                                            }
                                        else:
                                            raw_x = input('  Target x positions (comma-separated e.g. -18.0, 18.0): ')
                                            x_parsed = [float(val.strip()) for val in raw_x.split(',') if val.strip()]
                                            run_target_x[ri] = {
                                                'lhs': x_parsed if x_parsed else None,
                                                'rhs': None
                                            }

                            # --- INITIALIZE SHARED FIGURES OUTSIDE RUN LOOP ---
                            plot_axes = {}
                            layout = [[2]] if split_x else [[1]]

                            if plot_vs_x:
                                fig_x, ax_x, letters_x = build_subplots(layout)
                                save_folder_x = figure_folder + 'deltan_n_vs_x/' if save_plots else None
                                if save_folder_x: ensure_directory(save_folder_x)
                                plot_axes['x'] = {'fig': fig_x, 'ax': ax_x, 'letters': letters_x,
                                                  'save_folder': save_folder_x}

                            if plot_vs_L_n:
                                fig_L, ax_L, letters_L = build_subplots(layout)
                                save_folder_L = figure_folder + 'deltan_n_vs_L_n/' if save_plots else None
                                if save_folder_L: ensure_directory(save_folder_L)
                                plot_axes['L_n'] = {'fig': fig_L, 'ax': ax_L, 'letters': letters_L,
                                                    'save_folder': save_folder_L}

                            legend_handles, legend_labels, runs_plotted_list = [], [], []
                            updated, dens_color, temp_color = False, None, None

                            all_figure_velocities = []
                            base_legend_handles, base_legend_labels = [], []
                            run_Ln_x_positions = {}  # Store L_n x-positions per run ID

                            for idx, run in enumerate(valid_runs):
                                fluct_ds = run['fluct_ds']
                                lang_ds = run['lang_ds']
                                ri = run['run_id']
                                color = run['color']
                                mark = run['marker']

                                # Extract dataset-specific target x coordinates
                                target_x_lhs = run_target_x.get(ri, {}).get('lhs', None)
                                target_x_rhs = run_target_x.get(ri, {}).get('rhs', None)

                                if 'updated' in run['filename']:
                                    updated = True
                                runs_plotted_list.append(ri.split(',')[1].strip() if ',' in ri else ri)

                                dot_handle = Line2D([], [], marker=mark, color='none', markerfacecolor=color,
                                                    markersize=8, linestyle='none')
                                base_legend_handles.append(dot_handle)
                                base_legend_labels.append(ri)
                                run_Ln_x_positions[ri] = []

                                z_vals = fluct_ds.coords["z"].values
                                fluct_probes = [z_vals[np.argmax(z_vals)]] if one_probe else z_vals

                                for key, p_data in plot_axes.items():
                                    for fluct_z in fluct_probes:
                                        t_col, d_col, v_list, x_list = get_deltan_over_n(
                                            run, fluct_z, quantity,
                                            axis_list=p_data['ax'],
                                            letters=p_data['letters'],
                                            x_ax_arg=key,
                                            split_x=split_x,
                                            b_dens_grads=dens_grads,
                                            b_temp_grads=temp_grads,
                                            target_x_lhs=target_x_lhs,
                                            target_x_rhs=target_x_rhs
                                        )
                                        if d_col is not None: dens_color = d_col
                                        if t_col is not None: temp_color = t_col
                                        all_figure_velocities.extend(v_list)

                                        # Only record x-positions for the L_n plot
                                        if key == 'L_n':
                                            run_Ln_x_positions[ri].extend(x_list)

                            runs_plotted = '_'.join(runs_plotted_list)

                            # --- STEP 2: FORMAT FIGURES, BUILD PER-FIGURE LEGENDS, & SAVE ---
                            for key, p_data in plot_axes.items():
                                fig, axis, letters = p_data['fig'], p_data['ax'], p_data['letters']
                                axes_list = [axis[let] for let in letters]

                                fig_legend_handles = list(base_legend_handles)
                                fig_legend_labels = []

                                # Build legend labels: include x-positions ONLY for 'L_n' plots
                                for ri in base_legend_labels:
                                    if key == 'L_n' and run_Ln_x_positions.get(ri):
                                        unique_x = sorted(list(set(np.round(run_Ln_x_positions[ri], 1))))
                                        u_str = ref_lang_ds.attrs.get("x_units", "cm") if ref_lang_ds else "cm"
                                        x_formatted = ", ".join([f"{x:+.1f}" for x in unique_x])
                                        fig_legend_labels.append(f"{ri} ($x = {x_formatted}$ {u_str})")
                                    else:
                                        fig_legend_labels.append(ri)

                                # Append gradient patches to legend ONLY for 'vs x' plots
                                if key == 'x':
                                    if dens_grads and dens_color is not None:
                                        fig_legend_handles.append(mpatches.Patch(color=dens_color, alpha=0.2))
                                        fig_legend_labels.append('Density Gradient')
                                    if temp_grads and temp_color is not None:
                                        fig_legend_handles.append(mpatches.Patch(color=temp_color, alpha=0.2))
                                        fig_legend_labels.append('Temperature Gradient')

                                # Colorbar for L_n plot
                                if key == 'L_n' and all_figure_velocities:
                                    v_min, v_max = min(all_figure_velocities), max(all_figure_velocities)
                                    if v_min == v_max: v_max += 1e-5
                                    norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
                                    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
                                    sm.set_array([])

                                    cbar = fig.colorbar(sm, ax=axes_list, location='right', pad=0.04, fraction=0.04)
                                    cbar.set_label(r'$V_{\parallel}$ [m/s]', rotation=270, labelpad=25)

                                # Axis labels & layout formatting
                                if split_x:
                                    lhs_axis, rhs_axis = axis[letters[0]], axis[letters[1]]

                                    # Synchronize Y-axis limits across panels
                                    y_min = min(lhs_axis.get_ylim()[0], rhs_axis.get_ylim()[0])
                                    y_max = max(lhs_axis.get_ylim()[1], rhs_axis.get_ylim()[1])
                                    lhs_axis.set_ylim(y_min, y_max)
                                    rhs_axis.set_ylim(y_min, y_max)

                                    lhs_axis.yaxis.set_major_formatter('{x:.2f}')
                                    rhs_axis.tick_params(labelleft=False)

                                    lhs_axis.set_title('Left side (-x)')
                                    rhs_axis.set_title('Right side (+x)')
                                    lhs_axis.set_ylabel(r'$\delta n_e/n_e$')

                                    if ref_lang_ds is not None:
                                        u_str = ref_lang_ds.attrs.get("x_units", "cm")
                                        lbl = rf'$r$ [{u_str}]' if key == 'x' else rf'$L_n$ [{u_str}]'
                                        lhs_axis.set_xlabel(lbl)
                                        rhs_axis.set_xlabel(lbl)

                                    lgd = fig.legend(handles=fig_legend_handles, labels=fig_legend_labels,
                                                     loc='lower center', bbox_to_anchor=(0.5, -.18), ncol=3)
                                else:
                                    top_axis = axis[letters[0]]
                                    top_axis.yaxis.set_major_formatter('{x:.2f}')
                                    top_axis.set_ylabel(r'$\delta n_e/n_e$')

                                    if ref_lang_ds is not None:
                                        u_str = ref_lang_ds.attrs.get("x_units", "cm")
                                        lbl = rf'$x$ [{u_str}]' if key == 'x' else rf'$L_n$ [{u_str}]'
                                        top_axis.set_xlabel(lbl)

                                    lgd = fig.legend(handles=fig_legend_handles, labels=fig_legend_labels,
                                                     loc='lower center', bbox_to_anchor=(0.5, -.18), ncol=2)

                                fig.canvas.draw()
                                legend_height_inches = lgd.get_window_extent().height / fig.dpi
                                standard_w, standard_h = fig.get_size_inches()
                                fig.set_size_inches(standard_w, standard_h + legend_height_inches)

                                fig.subplots_adjust(
                                    bottom=legend_height_inches / (standard_h + legend_height_inches),
                                    right=0.88 if key == 'L_n' else 0.95,
                                    wspace=0.3
                                )

                                if save_plots:
                                    save_name = f'split_x_runs_{runs_plotted}' if split_x else f'runs_{runs_plotted}'
                                    if updated: save_name += '_updated'
                                    fig.savefig(p_data['save_folder'] + save_name + '.png', bbox_extra_artists=(lgd,),
                                                bbox_inches='tight')
                                    print('Saved plot to', p_data['save_folder'] + save_name + '.png')

                            if see_plots:
                                plt.show()
                            plt.close('all')

            if "Other plots from Michael's structure" in plots_to_make:
                print("--------Other plots from Michael's structure--------")
                ask_about_plots(datasets, plot_save_folder=plot_save_folder, langmuir_folder = langmuir_nc_folder,
                                filenames = files_to_plot)
        if not choice_names:
            print("Choose one of the following HDF5 files to extract data from.\n"
                  "Press Enter to continue without processing data.")
            files_in_hdf5_folder = sorted([f for f in os.listdir(hdf5_folder) if f.endswith(".hdf5")])
            # choice_indices = choose_multiple_from_list(files_in_hdf5_folder, "HDF5 file",
            #                                            null_action="not retrieve data from "
            #                                                        "HDF5 files.")
            choice_names = int_choose_multiple_from_list(files_in_hdf5_folder, "HDF5 file",
                                                       null_action="not retrieve data from "
                                                                   "HDF5 files.")
            if choice_names:
                for name in tqdm(choice_names, desc="Processing data from fluctuation probes..."):
                    get_isat_vf(hdf5_folder + name, hdf5_folder, flux_nc_folder,
                                main_luke = True)



    if "Build Mach Datasets" in chosen_options:
        updated_lang_folder = langmuir_nc_folder + 'updated/'
        nc_list = sorted([f for f in os.listdir(langmuir_nc_folder) if f.endswith(".nc")])
        # nc_choice = choose_multiple_from_list(nc_list, 'NetCDF file',
        #                                       null_action="Continue to updated nc files for temperature (can do both)")
        nc_names = int_choose_multiple_from_list(nc_list, 'NetCDF file',
                                              null_action="Continue to updated nc files for temperature (can do both)")

        # Initialize our master list of paths and filenames
        selected_file_paths = []
        selected_filenames = []

        # Convert chosen indices into absolute paths
        if nc_names:
            for name in nc_names:
                selected_filenames.append(name)
                selected_file_paths.append(os.path.join(langmuir_nc_folder, name))
        updated = ask_yes_or_no('Retrieve data from updated nc files? (y/n) ')
        if updated:
            updated_nc_list = sorted([f for f in os.listdir(updated_nc_folder) if f.endswith(".nc")])
            # updated_nc_choice = choose_multiple_from_list(updated_nc_list, 'NetCDF file',
            #                                               null_action="Don't make any plots")

            updated_nc_names = int_choose_multiple_from_list(updated_nc_list, 'NetCDF file',
                                                          null_action="Don't make any plots")
            # Convert chosen indices into absolute paths and add to the master list
            if updated_nc_names:
                for name in updated_nc_names:
                    selected_filenames.append(name)
                    selected_file_paths.append(os.path.join(updated_nc_folder, name))

        for i, filename in enumerate(selected_filenames):
            nc_filepath = selected_file_paths[i]
            print('filename: ', filename)
            hdf5_filename, hdf5_filepath, config_id = connect_lang_nc_to_hdf5(filename, langmuir_nc_folder, hdf5_folder)
            if hdf5_filepath == '':
                print(f'Skipping {filename}, no matching HDF5 file found')
                continue

            lang_ds = xr.open_dataset(nc_filepath)
            mach_configs = get_mach_config(hdf5_filepath,config_id)
            isat = get_mach_isat(hdf5_filepath, mach_configs)
            mach_numbers = get_mach_numbers(isat)

            # Extract ion temperature from attributes and convert string to float
            t_i_val = float(lang_ds.attrs['ion_temperature'])
            t_i_unit_str = lang_ds.attrs['ion_temperature_units']
            t_i_quantity = t_i_val * u.Unit(t_i_unit_str)

            # Pass it to get_velocity
            velocity = get_velocity(mach_numbers, lang_ds['t_e'], lang_ds.attrs['ion_type'], t_i_quantity)
            velocity = velocity.assign_coords(port=mach_numbers['port'])

            # Convert 'isat' (which is a DataArray) into a Dataset so it can be merged
            isat_ds = isat.to_dataset(name='isat')

            print("--- Checking Port Coordinates ---")
            print("ISAT port:", isat_ds['port'].values, isat_ds['port'].attrs)
            print("MACH port:", mach_numbers['port'].values, mach_numbers['port'].attrs)
            print("VELOCITY port:", velocity['port'].values, velocity['port'].attrs)

            # Merge them all seamlessly into one master Dataset
            mach_dataset = xr.merge([isat_ds, mach_numbers, velocity])

            mach_dataset.attrs['description'] = f'Mach data for {filename}'

            save_name = filename.replace('.nc', '_mach.nc')
            save_path = os.path.join(mach_nc_folder, save_name)  # Ensure mach_nc_folder is defined

            print(f"Executing math and saving to {save_path}...")
            mach_dataset.to_netcdf(save_path)
            lang_ds.close()

    if 'Do stuff with Mach datasets' in chosen_options:
        list_of_plots = ['Plot radial plots of Mach numbers and Velocity',
                         'Check Mach number by looking at Isat upstream and downstream contour plots']
        # plot_idx_choice = choose_multiple_from_list(list_of_plots, 'Choose which plots you want')
        chosen_plots = int_choose_multiple_from_list(list_of_plots, 'Choose which plots you want')

        # chosen_plots = []
        # if plot_idx_names:
        #     for idx in plot_idx_choice:
        #         plot_choice = list_of_plots[idx]
        #         chosen_plots.append(plot_choice)

        mach_figures = ensure_directory(figure_folder + 'mach/')

        nc_list = sorted([f for f in os.listdir(mach_nc_folder) if f.endswith(".nc")])
        # nc_choice = choose_multiple_from_list(nc_list, 'NetCDF file',
        #                                       null_action="Continue")

        nc_names = int_choose_multiple_from_list(nc_list, 'NetCDF file',
                                              null_action="Continue")

        # Initialize our master list of paths and filenames
        selected_file_paths = []
        selected_filenames = []

        # Convert chosen indices into absolute paths
        if nc_names:
            for name in nc_names:
                selected_filenames.append(name)
                selected_file_paths.append(os.path.join(mach_nc_folder, name))

        if selected_file_paths:
            if 'Plot radial plots of Mach numbers and Velocity' in chosen_plots:
                with xr.open_dataset(selected_file_paths[0]) as sample_ds:
                    min_x = int(sample_ds['x'].min().item())
                    max_x = int(sample_ds['x'].max().item())

                # Call helper function to get a verified list of target x integers
                target_xs = allow_only_ints(
                    prompt="Enter the x position(s) you want to analyze if empty analyze all x on one plot",
                    min_condition=min_x,
                    max_condition=max_x,
                    accept_empty=True
                )

                if not target_xs:
                    for i in range(min_x, max_x + 1):
                        target_xs.append(i)
                    one_plot_per_var = True  # Full range -> Plot vs X
                else:
                    one_plot_per_var = False  # Specific x chosen -> Plot vs Time


                for i, filepath in enumerate(selected_file_paths):
                    filename = selected_filenames[i]
                    mach_ds = xr.open_dataset(filepath)
                    lang_filename, lang_filepath = connect_lang_nc_to_mach_nc(filename, langmuir_nc_folder)
                    lang_ds = xr.open_dataset(lang_filepath)
                    run_id = f_run_identifier(lang_ds)
                    lang_zs = lang_ds['z'].values

                    for z in mach_ds['z'].values:
                        mach_probe_idx = int(np.where(mach_ds['z'] == z)[0][0])
                        mach_one_probe_data = mach_ds.isel(probe=mach_probe_idx)
                        lang_probe_idx = np.argmin(abs(lang_zs - z))
                        min_time = round(lang_ds.attrs[f"steady state start probe {lang_probe_idx}"])
                        max_time = round(lang_ds.attrs[f"steady state end probe {lang_probe_idx}"])

                        # --- MODE A: FULL PROFILE (PLOTTED VS X) ---
                        if one_plot_per_var:
                            layout = [[1]]
                            v_para_fig, v_para_axes, v_para_letters = build_subplots(layout)
                            v_para_ax = v_para_axes[v_para_letters[0]]

                            M_para_fig, M_para_axes, M_para_letters = build_subplots(layout)
                            M_para_ax = M_para_axes[M_para_letters[0]]

                            xs_profile, m_means, m_stds = [], [], []
                            v_means, v_stds = [], []
                            v_adj_means, v_adj_stds = [], []

                            for user_x in target_xs:
                                mach_x_slice = mach_one_probe_data.sel(x=user_x, method='nearest')
                                xs_profile.append(user_x)

                                m_para_steady = mach_x_slice['M_para'].sel(time=slice(min_time, max_time))
                                m_means.append(m_para_steady.mean(dim=['shot', 'time']).item())
                                m_stds.append(m_para_steady.std(dim=['shot', 'time']).item())

                                v_para_steady = mach_x_slice['v_para'].sel(sweep=slice(min_time, max_time))
                                v_means.append(v_para_steady.mean(dim=['shot', 'sweep']).item())
                                v_stds.append(v_para_steady.std(dim=['shot', 'sweep']).item())

                            sort_idx = np.argsort(xs_profile)
                            sorted_xs = np.array(xs_profile)[sort_idx]

                            # Plotting v_para errorbars AFTER the loop
                            v_para_ax.errorbar(sorted_xs, np.array(v_means)[sort_idx],
                                               yerr=np.array(v_stds)[sort_idx], fmt='o', capsize=4,
                                               color='tab:blue', label=r'$\gamma_i = 0$')

                            v_para_ax.set_xlabel("X [cm]")
                            v_para_ax.set_ylabel(r"$v_{\parallel}$ [m/s]")
                            v_para_ax.legend(loc='best')
                            v_para_ax.set_title(f'Parallel velocity \n{run_id}\n'
                                                f'Mach z: {z}, Steady state time: {min_time} - {max_time} ms')

                            M_para_ax.errorbar(sorted_xs, np.array(m_means)[sort_idx],
                                               yerr=np.array(m_stds)[sort_idx], fmt='o', capsize=4, color='tab:blue')
                            M_para_ax.set_xlabel("X Position [cm]")
                            M_para_ax.set_ylabel(r"$M_{\parallel}$")
                            M_para_ax.set_title(f'Parallel Mach Number \n{run_id}\n'
                                                f'Mach z: {z}, Steady state time: {min_time} - {max_time} ms')

                            v_para_save_folder = ensure_directory(mach_figures + 'v_para_vs_x/')
                            run_save_folder = ensure_directory(v_para_save_folder + f'{run_id}/')
                            # if both_adj_normal:
                            #     save_name = f'v_para_z_{z:.1f}_both.png'
                            # elif adj_plot:
                            #     save_name = f'v_para_z_{z:.1f}_adj.png'
                            save_name = f'v_para_z_{z:.1f}.png'
                            v_para_fig.savefig(run_save_folder + save_name, bbox_inches='tight')
                            print(f"Saved figure for to {run_save_folder}{save_name}.png")

                            M_para_save_folder = ensure_directory(mach_figures + 'M_para_vs_x/')
                            run_save_folder = ensure_directory(M_para_save_folder + f'{run_id}/')
                            M_para_fig.savefig(run_save_folder + f'M_para_z_{z:.1f}.png', bbox_inches='tight')
                            print(f"Saved figure for to {run_save_folder}M_para_z_{z:.1f}.png")

                        # --- MODE B: USER SPECIFIED X (PLOTTED VS TIME / SWEEP) ---
                        else:
                            for user_x in target_xs:
                                layout = [[1]]
                                v_para_fig, v_para_axes, v_para_letters = build_subplots(layout)
                                v_para_ax = v_para_axes[v_para_letters[0]]

                                M_para_fig, M_para_axes, M_para_letters = build_subplots(layout)
                                M_para_ax = M_para_axes[M_para_letters[0]]

                                mach_x_slice = mach_one_probe_data.sel(x=user_x, method='nearest')

                                m_para_steady = mach_x_slice['M_para'].sel(time=slice(min_time, max_time))
                                v_para_steady = mach_x_slice['v_para'].sel(sweep=slice(min_time, max_time))

                                v_time_series = v_para_steady.mean(dim='shot')
                                m_time_series = m_para_steady.mean(dim='shot')

                                # Match respective coordinate dimensions for x-axes
                                v_times = v_time_series['sweep'].values
                                m_times = m_time_series['time'].values

                                v_para_ax.plot(v_times, v_time_series.values, 'o', label=f'x = {user_x} cm')
                                M_para_ax.plot(m_times, m_time_series.values, 'o', label=f'x = {user_x} cm')

                                v_para_ax.set_xlabel("Time [ms]")
                                v_para_ax.set_ylabel(r"$v_{\parallel}$ [m/s]")
                                v_para_ax.set_title(f'Parallel velocity \n{run_id}\nMach z: {z}, x: {user_x} cm')

                                v_para_save_folder = ensure_directory(mach_figures + 'v_para_vs_time/')
                                run_save_folder = ensure_directory(v_para_save_folder + f'{run_id}/')
                                probe_save_folder = ensure_directory(run_save_folder + f'z_{z}/')
                                v_para_fig.savefig(probe_save_folder + f'v_para_z_{z:.1f}_x_{user_x}.png',
                                                   bbox_inches='tight')

                                M_para_ax.set_xlabel("Time [ms]")
                                M_para_ax.set_ylabel(r"$M_{\parallel}$")
                                M_para_ax.set_title(f'Parallel Mach Number \n{run_id}\nMach z: {z}, x: {user_x} cm')

                                M_para_save_folder = ensure_directory(mach_figures + 'M_para_vs_time/')
                                run_save_folder = ensure_directory(M_para_save_folder + f'{run_id}/')
                                probe_save_folder = ensure_directory(run_save_folder + f'z_{z}/')
                                M_para_fig.savefig(probe_save_folder + f'M_para_z_{z:.1f}_x_{user_x}.png',
                                                   bbox_inches='tight')
                                print(f'Saved M_para_z_{z} to {probe_save_folder}M_para_z_{z:.1f}_x_{user_x}.png')

            if 'Check Mach number by looking at Isat upstream and downstream contour plots' in chosen_plots:
                for i, filepath in enumerate(selected_file_paths):
                    filename = selected_filenames[i]
                    mach_ds = xr.open_dataset(filepath)
                    lang_filename, lang_filepath = connect_lang_nc_to_mach_nc(filename, langmuir_nc_folder)
                    lang_ds = xr.open_dataset(lang_filepath)
                    run_id = f_run_identifier(lang_ds)
                    lang_zs = lang_ds['z'].values
                    # Extract Langmuir time limits
                    lang_min_time = lang_ds['time'].min().item()
                    lang_max_time = lang_ds['time'].max().item()

                    # Cut mach_ds down to match the lang_ds time window
                    mach_ds_cut = mach_ds.sel(time=slice(lang_min_time, lang_max_time))
                    save_figs = ensure_directory(mach_figures + 'isat_contours/')
                    check_isats_contour(mach_ds_cut, save_figs, title_identifier = run_id)
