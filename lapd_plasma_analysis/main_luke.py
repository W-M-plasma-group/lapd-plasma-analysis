# from pty import slave_open

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
import matplotlib.pyplot as plt
import numpy as np

from lapd_plasma_analysis.main import hdf5_folder, plot_save_folder
from lapd_plasma_analysis.obtain_dchars import *
from lapd_plasma_analysis.obtain_dchars.build_dchars import diagnostics_xarray

# from lapd_plasma_analysis.main import hdf5_folder, mach_nc_folder, flux_nc_folder
from obtain_plots.Functions_used_in_main_luke_plots import *
from lapd_plasma_analysis.obtain_dchars.Build_netcdf import *
from lapd_plasma_analysis.obtain_plots.xarray_plots import *

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

# isweep_choice is user choice for probe or linear combination to plot; see isweep_selector in helper.py for more
# e.g. coefficients are for [[p1f1, p1f2], [p2f1, p2f2]]
isweep_choices = [[[1, 0], [0, 0]],     # . 1st combination to plot: 1 * (first face on first probe)
                  [[0, 0], [1, 0]]]     # . 2nd combination to plot: 1 * (first face on second probe)
# isweep_choices = [[[1, 0], [-1, 0]]]  # .     combination to plot: 1 * (face 1 on probe 1) - 1 * (face 1 on probe 2)


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
                       "Obtain plasma parameters from netCDF files",
                       "Check NaNs",
                       "Obtain plasma parameters from netCDF files new",
                       "Plasma Py HDF5 to NetCDF",
                       "Convert NetCDF to usable form for fluctuations"]
    user_choice_to_do = choose_multiple_from_list(prompt_filetype, 'action',null_action= "end main")
    # Gives a list of what the user wants to do

    # If the user chooses convert HDF5 files to NetCDF files or get plots from HDF5 files - loading data is the same.
    # Initial loading of HDF5 files is the same as Leo's main.
    if 0 in user_choice_to_do or 1 in user_choice_to_do or 5 in user_choice_to_do:
        # Choose hdf5 files to read
        hdf5_list = [f for f in os.listdir(hdf5_folder) if f.endswith(".hdf5")]
        hdf5_choice = choose_multiple_from_list(hdf5_list,'HDF5 file',
                                                null_action="not retrieve data from HDF5 files.")
        if any(i >= len(hdf5_list) or i < 0 for i in hdf5_choice):
            hdf5_choice = []
            print("Invalid input")
        # Returns the indices in hdf5_list of the selected hdf5 files in a list format


        if hdf5_choice:
            # If the user wants to plot the HDF5 data
            if 1 in user_choice_to_do:
                # Allow the user to select what they want to plot
                # Allow the user to select if they would like to save the plots
                IV_plots_prompt = ["Plot bias voltage vs time for a position-shot combination",
                                   "Plot current vs time for a position-shot combination",
                                   "Plot individual raw IV sweeps for a position-shot combination",
                                   "Plot log plot of IV sweeps for a position-shot combination",
                                   "Plot Ion saturation current vs time for a position-shot combination (best in core region)",
                                   "Plot the ratio between Plasma Py and v_f, v_p line temperature calculations",
                                   "Plot the ratio between Plasma Py and v_f, v_p line temperature calculations for all "
                                   "sweeps for the first probe"
                                   ]
                IV_plots_choice = []
                while IV_plots_choice == [] or any(i >= len(IV_plots_prompt) or i < 0 for i in IV_plots_choice):
                    IV_plots_choice = choose_multiple_from_list(IV_plots_prompt, 'parameter plot')
                save_plots = ask_yes_or_no("Do you want to save the plots? (Will be saved in a directory labelled by the run name)"
                                       " (y/n) ")
                # print('IV_plots_choice: ',IV_plots_choice)b
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

        pathname_index = 0
        for hdf5_pathname in hdf5_pathname_list:
            # Obtain parameters to be used in IV sweep curves
            exp_params_dict,vsweep_bc,langmuir_configs,config_id,voltage_gain,orientation,current_bc \
                =(IV_parameters(hdf5_pathname))
            bias, dt = get_sweep_voltage(hdf5_pathname, vsweep_bc, voltage_gain)
            # Determine how many IV sweeps there are
            ramp_bounds = isolate_ramps(bias)
            # Determine the times each IV sweep was conducted
            ramp_times = ramp_bounds[:, 1] * dt.to(u.ms)

            # Reconfigure the bias and current data to be a 3D array giving the respective variable for
            # a specific (position, shot, frame)
            if 0 in user_choice_to_do or 5 in user_choice_to_do:
                bias_list = []
                current_list = []
                position_list = []
            for i in range(len(langmuir_configs)):
                # All probes should share the same bias sweep so we don't need to recompute it for every probe
                probe_bias = bias.copy()
                probe_current, motor_data = get_sweep_current(hdf5_pathname, langmuir_configs[i], orientation)


                # ensure "hardcoded" ports listed in configurations.py match those listed in HDF5 file
                assert motor_data.info['controls']['6K Compumotor']['probe']['port'] == langmuir_configs[i]['port']

                probe_position_array, num_positions, shots_per_position, selected_shots = get_shot_positions(motor_data)

                # Drop some shots from the data because they don't fit into a 3D structure
                if len(probe_bias.shape) == 2:  # already selected certain shots in bias data
                    probe_bias = probe_bias[selected_shots, ...]
                probe_current = probe_current[selected_shots, ...]

                # Make bias and current 3D (position, shot_at_a_certain_position, frame) arrays
                #    as opposed to 2D (shot number, frame) arrays
                probe_bias = probe_bias.reshape(num_positions, shots_per_position, -1)
                probe_current = probe_current.reshape(num_positions, shots_per_position, -1)
                # Dimensions of bias and current arrays:   position, shot, frame   (e.g. (71, 15, 55296))

                # If the user wants to plot the HDF5 file data
                if 1 in user_choice_to_do and 6 not in IV_plots_choice:
                    # Grab the data for all given shots and positions so it can easily be extrapolated to other plots
                    bias_to_plot, current_to_plot, loc_shot, filepath = (
                        obtain_data(hdf5_folder, probe_bias, probe_current, probe_position_array, langmuir_configs[i],
                                    exp_params_dict,save_plots))
                    port_face_string = f"{langmuir_configs[i]['port']}{langmuir_configs[i]['face'] if langmuir_configs[i]['face'] else ''}"
                    if 5 in IV_plots_choice:
                        mean_slope_ratio_array = []
                        std_slope_ratio_array = []
                    for h in range(len(bias_to_plot)):
                        if 2 in IV_plots_choice or 3 in IV_plots_choice:
                            plot_iv_sweep(filepath, bias_to_plot[h], current_to_plot[h], port_face_string,
                                          IV_plots_choice,
                                          save_plots, how_many_plots, ramp_times, exp_params_dict, loc_shot[h], dt)

                        # If the user wants to plot the ion saturation current vs time
                        if 4 in IV_plots_choice:
                            plot_ion_isat_vs_time(dt, ramp_times, bias_to_plot[h], current_to_plot[h], exp_params_dict,
                                                  port_face_string, loc_shot[h], save_plots, filepath)

                        if 5 in IV_plots_choice:
                                slope_ratio_mean, slope_ratio_std = compare_pp_vs_luke(bias_to_plot[h], current_to_plot[h],
                                                                                       exp_params_dict,port_face_string,
                                                                                       ramp_times, loc_shot[h], dt)
                                mean_slope_ratio_array.append(slope_ratio_mean)
                                std_slope_ratio_array.append(slope_ratio_std)
                                mean_means_slope_ratio = np.mean(mean_slope_ratio_array)
                                std_means_slope_ratio = np.std(std_slope_ratio_array)
                                print('Average slope ratio: ', mean_means_slope_ratio)
                                print('Standard deviation of standard deviations: ', std_means_slope_ratio)
                                print('Raw means of slope ratio for selected sweeps: ', mean_slope_ratio_array)
                                print('Raw standard deviation of slope ratio for selected sweeps: ', std_slope_ratio_array)

                if 0 in user_choice_to_do or 5 in user_choice_to_do:
                    bias_list.append(probe_bias)
                    current_list.append(probe_current)
                    position_list.append(probe_position_array)

            if 1 in user_choice_to_do and 6 in IV_plots_choice:
                print('made it')
                bias_to_plot, current_to_plot, loc_shot, filepath = (
                    obtain_data(hdf5_folder, probe_bias, probe_current, probe_position_array, langmuir_configs[0],
                                exp_params_dict, save_plots, select_sweeps = False))
                port_face_string = f"{langmuir_configs[0]['port']}{langmuir_configs[0]['face'] if langmuir_configs[0]['face'] else ''}"
                mean_slope_ratio_array = []
                std_slope_ratio_array = []
                for k in range(len(bias_to_plot)):
                    slope_ratio_mean, slope_ratio_std = compare_pp_vs_luke(bias_to_plot[k], current_to_plot[k],
                                                                           exp_params_dict, port_face_string,
                                                                           ramp_times, loc_shot[k], dt, plot = False)
                    if not np.isnan(slope_ratio_mean) and slope_ratio_mean < 10:
                        mean_slope_ratio_array.append(slope_ratio_mean)
                        std_slope_ratio_array.append(slope_ratio_std)

                plt.figure(figsize=(8, 5))
                plt.hist(mean_slope_ratio_array, bins=25, color='steelblue', edgecolor='black', alpha=0.8)
                plt.xlim(0, 8)
                plt.title("Histogram of Mean Slope Ratios", fontsize=14)
                plt.xlabel("Slope Ratio (pp_slope / vp_vf_slope)", fontsize=12)
                plt.ylabel("Number of Sweeps", fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.6)

                plt.tight_layout()
                plt.show()

                print('Average slope ratio: ', np.mean(mean_slope_ratio_array))
                print('Standard deviation of slope ratio: ', np.std(std_slope_ratio_array))


            # If the user selects to build the NETCDF files
            if 0 in user_choice_to_do:
                # Add a probe dimension to the bias and current we input to the dataset function
                # print(len(bias_list))
                stacked_bias = np.stack(bias_list, axis=0)
                stacked_current = np.stack(current_list, axis=0)
                # Make sure we have equal dimensions across the same x-y positions and if so passes just one of the
                # position arrays to the dataset builder
                assert all(np.array_equal(position_list[0],pa) for pa in position_list)
                shared_positions = position_list[0]

                ion_type = get_ion(exp_params_dict['Run name'])
                ds = build_xarrays(stacked_bias, stacked_current, shared_positions, ramp_times, dt, langmuir_configs, ion_type)
                # ds = diagnostics_xarray(stacked_bias,stacked_current,shared_positions,ramp_times,dt,langmuir_configs,ion_type)
                hdf5_filename = os.path.splitext(hdf5_list[hdf5_choice[pathname_index]])[0]
                nc_save_path = os.path.join(langmuir_nc_folder, 'n_' + hdf5_filename + "_curve_comparison.nc")
                ds.to_netcdf(nc_save_path)

            if 5 in user_choice_to_do:
                stacked_bias = np.stack(bias_list, axis=0)
                stacked_current = np.stack(current_list, axis=0)
                # Make sure we have equal dimensions across the same x-y positions and if so passes just one of the
                # position arrays to the dataset builder
                assert all(np.array_equal(position_list[0], pa) for pa in position_list)
                shared_positions = position_list[0]

                ion_type = get_ion(exp_params_dict['Run name'])
                ds = build_pp_xarray(stacked_bias, stacked_current, shared_positions, ramp_times, dt, langmuir_configs,
                                   ion_type)
                hdf5_filename = os.path.splitext(hdf5_list[hdf5_choice[pathname_index]])[0]
                nc_save_path = os.path.join(langmuir_nc_folder, hdf5_filename + "_pp.nc")
                ds.to_netcdf(nc_save_path)





    # If the user chooses Obtain plasma parameters from NetCDF files
    if 2 in user_choice_to_do:
        nc_list = [f for f in os.listdir(langmuir_nc_folder) if f.endswith(".nc")]
        nc_choice = choose_multiple_from_list(nc_list, 'NetCDF file',
                                                null_action="not retrieve data from NetCDF files.")
        datasets = []
        steady_state_times_runs =[]
        for i in range(len(nc_choice)):
            ds = xr.load_dataset(os.path.join(langmuir_nc_folder, nc_list[nc_choice[i]]))
            # TODO Update this to have a not hardcoded steady state period
            if "comparison" in nc_list[nc_choice[i]]:
                steady_state_times_runs.append([16,24] * u.ms)
                # TODO this assigns a face dimension. We will eventually want to do this in the dataset calculation
                num_faces = 2
                num_probes = 2
                total = num_probes * num_faces

                ds = ds.assign_coords(probe=np.arange(total))
                probe_ids = np.repeat(np.arange(num_probes), num_faces)
                faces = np.tile(np.arange(num_probes), num_faces)

                ds = ds.assign_coords(probe_id = ('probe', probe_ids), face = ('probe', faces))
                ds = ds.set_index(probe=['probe_id', 'face']).unstack('probe')
                ds = ds.rename({'probe_id': 'probe'})
                ds = ds.rename({'sweep': 'time'})

                # TODO Assign this in the dataset creation

                hdf5_list = [f for f in os.listdir(hdf5_folder) if f.endswith(".hdf5")]

                print(f"Choose the corresponding HDF5 file to assign attributes for dataset {nc_list[nc_choice[i]]}.")
                match_hdf5 = []
                while len(match_hdf5) != 1:
                    match_hdf5 = choose_multiple_from_list(hdf5_list, 'HDF5 file',)

                exp_params_dict, vsweep_bc, langmuir_configs, config_id, voltage_gain, orientation, current_bc \
                    = (IV_parameters(hdf5_folder + hdf5_list[match_hdf5[0]]))
                ds.attrs.update(exp_params_dict)
                # print('attributes', ds.attrs)
                bias, dt = get_sweep_voltage(hdf5_folder + hdf5_list[match_hdf5[0]], vsweep_bc, voltage_gain)
                # Determine how many IV sweeps there are
                ramp_bounds = isolate_ramps(bias)
                # Determine the times each IV sweep was conducted
                ramp_times = ramp_bounds[:, 1] * dt.to(u.ms).value
                ds = ds.assign_coords(time=('sweep', ramp_times))
                ds['time'].attrs['units'] = 'ms'
                ds['x'].attrs['units'] = "cm"
                ds['y'].attrs['units'] = "cm"
                ds['z'].attrs['units'] = "cm"
                # print('outside: ', ds.coords['time'].attrs)


            else:
                steady_state_times_runs.append(detect_steady_state_times(ds,core_radius))
            datasets.append(ds)
            plot_save_folder = ""


        diagnostic_name_dict = {key: get_title(key) for key in
                                set.intersection(*[set(dataset) for dataset in datasets])}
        diagnostics_to_plot_list = get_diagnostics_to_plot(diagnostic_name_dict)
        # Plot chosen diagnostics for each individual dataset
        if ask_yes_or_no("Generate contour plot of selected diagnostics over time and radial position? (y/n) "):
            for plot_diagnostic in diagnostics_to_plot_list:
                for i in range(len(datasets)):
                    plot_linear_diagnostic(datasets[i], isweep_choices, plot_diagnostic, 'contour',
                                           steady_state_times_runs[i],
                                           display_core_steady_state=True, core_radius=core_radius)

        # Plot radial profiles of diagnostic (steady-state time average), with color corresponding to first attribute
        #    and plot position on multiplot corresponding to second attribute
        if ask_yes_or_no("Generate line plot of selected diagnostics over radial position? (y/n) "):
            for plot_diagnostic in diagnostics_to_plot_list:
                multiplot_linear_diagnostic(datasets, plot_diagnostic, isweep_choices, 'x',
                                            steady_state_by_runs=steady_state_times_runs, core_rad=core_radius,
                                            tolerance=plot_tolerance, display_core_steady_state=True,
                                            save_directory=plot_save_folder)

        # Plot time profiles
        if ask_yes_or_no("Generate line plot of selected diagnostics over time? (y/n) "):
            for plot_diagnostic in diagnostics_to_plot_list:
                multiplot_linear_diagnostic(datasets, plot_diagnostic, isweep_choices, 'time',
                                            steady_state_by_runs=steady_state_times_runs, core_rad=core_radius,
                                            save_directory=plot_save_folder)

        # Split two steady state periods for jan_2024 data: (16, 24) and (27, 33) and plot with dotted
        available_marker_styles = ('D', 'o', '^', 's')  # markers for Apr_18, Mar_22, Nov_22, Jan_24
        marker_styles = [available_marker_styles[get_config_id(dataset.attrs['Exp name'])] for dataset in datasets]
        datasets_split = datasets.copy()
        for i in range(len(datasets)):
            if datasets[i].attrs[
                'Exp name'] == "January_2024":  # Add copies of Jan24 experiments at end for 2nd steady st.
                datasets_split += [datasets[i]]
                steady_state_times_runs += [(27, 33) * u.ms]
                marker_styles += ['x']

        # List that identifies probes and faces for 1) low-z/high-z and 2) midplane
        probes_faces_parallel = [((0, 0), (1, 0)) for dataset in datasets_split]
        probes_faces_midplane = [(1, 0) if dataset.attrs['Exp name'] == "January_2024" else (0, 0)
                                 for dataset in datasets_split]
        # format:  (probe,   face),  (probe,   face), ...; each tuple specifies one probe-face combination
        #    e.g.  (probe 1, face 0) for January 2024 midplane probe-face tuple

        if ask_yes_or_no(f"Generate parallel plot of selected diagnostics? (y/n) "):
            for plot_diagnostic in diagnostics_to_plot_list:
                plot_parallel_diagnostic(datasets_split, steady_state_times_runs,
                                         probes_faces_midplane, probes_faces_parallel,
                                         marker_styles, diagnostic=plot_diagnostic, operation="mean",
                                         core_radius=core_radius, line_style='-', save_directory=plot_save_folder)

        at_least_two_diagnostics = (len(diagnostics_to_plot_list) >= 2)
        if at_least_two_diagnostics and ask_yes_or_no(
                "Generate scatter plot of first two selected diagnostics? (y/n) "):
            scatter_plot_diagnostics(datasets_split, diagnostics_to_plot_list, steady_state_times_runs,
                                     probes_faces_midplane, marker_styles, operation="mean",
                                     core_radius=core_radius, save_directory=plot_save_folder)

        if ask_yes_or_no("Generate plot of gradient scale length by position for selected diagnostics? (y/n) "):
            for plot_diagnostic in diagnostics_to_plot_list:
                plot_parallel_inverse_scale_length(datasets_split, steady_state_times_runs, plot_diagnostic,
                                                   probes_faces_midplane, probes_faces_parallel,
                                                   marker_styles, "mean", core_radius, plot_save_folder,
                                                   scale_length_mode="exponential")  # 'linear' or 'exponential'

        if ask_yes_or_no("Generate grid line plots for selected diagnostics? (y/n) "):
            # time_unit = unit_safe(steady_state_times_runs[0])
            for x_dim in ("x", "time"):
                plot_grid(datasets, diagnostics_to_plot_list, steady_state_times_runs,
                          probes_faces_midplane, probes_faces_parallel, "mean", core_radius, x_dim,
                          num_rows=1, plot_save_folder=plot_save_folder)



    if 3 in user_choice_to_do:
        nc_list = [f for f in os.listdir(langmuir_nc_folder) if f.endswith(".nc")]
        nc_choice = choose_multiple_from_list(nc_list, 'NetCDF file',
                                              null_action="not retrieve data from NetCDF files.")
        datasets = []
        steady_state_times_runs = []
        for i in range(len(nc_choice)):
            ds = xr.load_dataset(os.path.join(langmuir_nc_folder, nc_list[nc_choice[i]]))
            nan_summary(ds)


    if 4 in user_choice_to_do:
        nc_list = [f for f in os.listdir(langmuir_nc_folder) if f.endswith(".nc")]
        nc_choice = choose_multiple_from_list(nc_list, 'NetCDF file',
                                              null_action="not retrieve data from NetCDF files.")

        datasets = []
        steady_state_times_runs = []

        # Select data sets to plot from saved .nc files in the selected folder
        for i in range(len(nc_choice)):
            ds = xr.load_dataset(os.path.join(langmuir_nc_folder, nc_list[nc_choice[i]]))
            print(os.path.join(langmuir_nc_folder, nc_list[nc_choice[i]]))
            datasets.append(ds)

        # What variables are in common between all the datasets selected
        common_vars = set.intersection(*[set(dataset.data_vars.keys()) for dataset in datasets])
        diagnostic_name_dict = {var: datasets[0][var].attrs.get("long_name",var) for var in common_vars}


        diagnostics_to_plot_list = get_diagnostics_to_plot(diagnostic_name_dict)

        possible_plots = ['contour',
                          'contour_subplots - Only for 0 probe',
                          'Michael gradient plot',
                          'Show steady state',
                          'Gradient plot'
                          ]
        idx_plot_choices = choose_multiple_from_list(possible_plots,'action', null_action = 'not plot data')
        plot_choices = []
        for idx in idx_plot_choices:
            plot_choices.append(possible_plots[idx])

        if 'contour' in plot_choices:
            i = 0
            for dataset in datasets:
                for plot_diagnostic in diagnostics_to_plot_list:
                    for probe in range(ds.sizes['probe']):
                        filename = nc_list[nc_choice[i]]

                        run_identifier = f_run_identifier(filename)

                        contour_plot(dataset, plot_diagnostic, probe, run_identifier)
                i += 1

        if 'contour_subplots - Only for 0 probe' in plot_choices:
            for diagnostic_to_plot in diagnostics_to_plot_list:
                contour_subplots(datasets,diagnostic_to_plot,nc_list, nc_choice)

        if 'Michael gradient plot' in plot_choices:
            i = 0
            for dataset in datasets:
                for plot_diagnostic in diagnostics_to_plot_list:
                    for probe in range(ds.sizes['probe']):
                        filename = nc_list[nc_choice[i]]


                        run_identifier = f_run_identifier(filename)

                        michael_density_plots(dataset, plot_diagnostic, probe, run_identifier)
                i += 1

        if 'Show steady state' in plot_choices:
            i = 0
            for dataset in datasets:
                for probe in range(ds.sizes['probe']):
                    filename = nc_list[nc_choice[i]]
                    run_identifier = f_run_identifier(filename)

                    show_steady_state(dataset, probe, run_identifier)
        if "Gradient plot" in plot_choices:
            gradient_times_dict = {}
            clor = ['#f55', '#a22', '#9c3', '#570', '#9bf', '#44c']
            mark = ['o','*','o','*','o','*']

            # --- Loop over datasets ---
            i = 0
            d_plots = {}
            for dataset in datasets:
                filename = nc_list[nc_choice[i]]
                run_identifier = f_run_identifier(filename)
                # print('run_identifier:', run_identifier)

                # create_gradient_plot returns updated gradient_times_dict and probe gradients
                gradient_times_dict, d_probe_gradients = create_gradient_plot(dataset, run_identifier,
                                                                              gradient_times_dict)
                num_probes = len(d_probe_gradients.keys())
                for j, key in enumerate(d_probe_gradients.keys()):
                    d_gradients = d_probe_gradients[key]
                    probe = int(key)
                    color = clor[(i * num_probes + j) % len(clor)]
                    # print('color: ', color)
                    # print(color)
                    d_plots[f'{color}'] = {}
                    label = f"{run_identifier} Probe {probe}"
                    d_plots[f'{color}']['label'] = label
                    d_plots[f'{color}']['x_vals_t'] = d_gradients['adj_x_vals']
                    d_plots[f'{color}']['x_vals_n'] = d_gradients['x_vals']
                    d_plots[f'{color}']['t_grad'] = []
                    d_plots[f'{color}']['n_grad'] = []
                    d_plots[f'{color}']['eta_e']= d_gradients['eta e']
                    d_plots[f'{color}']['normalized grad t'] = d_gradients['normalized grad t']
                    d_plots[f'{color}']['normalized grad n'] = d_gradients['normalized grad n']
                i += 1


            fig_te, ax_te = plt.subplots(figsize=(12, 8))
            fig_sub_te, ax_sub_te = plt.subplots(figsize=(24, 8), nrows=2, ncols=1)
            ax_sub_te=ax_sub_te.flatten()
            # threshold = 1000

            for i,color in enumerate(d_plots.keys()):
                pre_label = d_plots[f'{color}']['label']
                idx = pre_label.find("run")
                label = pre_label[idx:]
                marker = mark[i]
                if marker == 'o':
                    ax = ax_sub_te[0]
                else:
                    ax = ax_sub_te[1]
            #     for t_e_array_idx in range(len(d_plots[f'{color}']['x_vals_t'])):
            #         x_vals_t_grad = d_plots[f'{color}']['x_vals_t'][t_e_array_idx]
            #         t_grad = d_plots[f'{color}']['t_grad'][t_e_array_idx]
            #         plt.plot(x_vals_t_grad, t_grad, color=color)
            #         if t_e_array_idx == len(d_plots[f'{color}']['x_vals_t']) - 1:
            #             plt.plot(x_vals_t_grad, t_grad, color=color, label=label)


                y_ax = d_plots[f'{color}']['normalized grad t']
                y_ax_mask = (y_ax <= .15) & (y_ax >= -.15)
                y_ax = y_ax[y_ax_mask]
                x_ax = d_plots[f'{color}']['x_vals_t'][y_ax_mask]
                ax_te.plot(x_ax, y_ax, markerfacecolor=color,
                         markeredgecolor = color, label=label, marker=marker,linestyle='None', markersize=12)
                ax.plot(x_ax, y_ax, markerfacecolor=color,
                         markeredgecolor = color, label=label, marker=marker,linestyle='None', markersize=12)
            ax_te.set_xlabel("x (cm)",fontsize = 28)
            ax_te.set_ylabel(r"$\nabla T_e / T_e$", fontsize=28)
            ax_te.set_title(r"Normalized $T_e$ Gradient",fontsize=28)
            ax_te.legend(fontsize=20, loc='lower left')
            ax_te.tick_params(labelsize=20)

            ax_sub_te[0].set_xlabel("x (cm)", fontsize=28)
            ax_sub_te[0].set_ylabel(r"$\nabla T_e / T_e$", fontsize=28)
            ax_sub_te[0].set_title(r"Normalized $T_e$ Gradient, Probe: 0", fontsize=28)
            ax_sub_te[0].legend(fontsize=20, loc='lower left')
            ax_sub_te[0].tick_params(labelsize=20)

            ax_sub_te[1].set_xlabel("x (cm)", fontsize=28)
            ax_sub_te[1].set_ylabel(r"$\nabla T_e / T_e$", fontsize=28)
            ax_sub_te[1].set_title(r"Normalized $T_e$ Gradient, Probe: 1", fontsize=28)
            ax_sub_te[1].legend(fontsize=20, loc='lower left')
            ax_sub_te[1].tick_params(labelsize=20)
            fig_te.tight_layout()
            fig_sub_te.tight_layout()
            plt.show()

            fig_ne, ax_ne = plt.subplots(figsize=(12,8))
            fig_sub_ne, ax_sub_ne = plt.subplots(figsize=(12, 16), nrows=2, ncols=1)
            ax_sub_ne = ax_sub_ne.flatten()
            for i,color in enumerate(d_plots.keys()):
                pre_label = d_plots[f'{color}']['label']
                idx = pre_label.find("run")
                label = pre_label[idx:]
                marker = mark[i]
                if marker == 'o':
                    ax = ax_sub_ne[0]
                else:
                    ax = ax_sub_ne[1]
                # for n_e_array_idx in range(len(d_plots[f'{color}']['x_vals_n'])):
                #     x_vals_n_grad = d_plots[f'{color}']['x_vals_n'][n_e_array_idx]
                #     n_grad = d_plots[f'{color}']['n_grad'][n_e_array_idx]
                #     plt.plot(x_vals_n_grad, n_grad, color=color)
                #     if n_e_array_idx == len(d_plots[f'{color}']['x_vals_n']) - 1:
                #         plt.plot(x_vals_n_grad, n_grad, color=color, label=label)

                y_ax = d_plots[f'{color}']['normalized grad n']
                y_ax_mask = (y_ax <= .5) & (y_ax >= -.5)
                y_ax = y_ax[y_ax_mask]
                x_ax = d_plots[f'{color}']['x_vals_n'][y_ax_mask]
                ax_ne.plot(x_ax, y_ax, markerfacecolor = color,
                         markeredgecolor=color, label=label,marker=marker,linestyle='None',markersize=12)
                ax.plot(x_ax, y_ax, markerfacecolor = color,
                         markeredgecolor=color, label=label,marker=marker,linestyle='None',markersize=12)
            ax_ne.set_xlabel("x (cm)", fontsize=28)
            ax_ne.set_ylabel(r"$\nabla n_e / n_e$",fontsize = 28)
            ax_ne.set_title("Normalized $n_e$ Gradient",fontsize=28)
            ax_ne.tick_params(labelsize=20)
            ax_ne.legend(fontsize=20, loc='lower left')

            ax_sub_ne[0].set_xlabel("x (cm)", fontsize=28)
            ax_sub_ne[0].set_ylabel(r"$\nabla n_e / n_e$", fontsize=28)
            ax_sub_ne[0].set_title(r"Normalized $n_e$ Gradient, Probe: 0", fontsize=28)
            ax_sub_ne[0].tick_params(labelsize=20)
            ax_sub_ne[0].legend(fontsize=20, loc='lower left')

            ax_sub_ne[1].set_xlabel("x (cm)", fontsize=28)
            ax_sub_ne[1].set_ylabel(r"$\nabla n_e / n_e$", fontsize=28)
            ax_sub_ne[1].set_title(r"Normalized $n_e$ Gradient, Probe: 1", fontsize=28)
            ax_sub_ne[1].tick_params(labelsize=20)
            ax_sub_ne[1].legend(fontsize=20, loc='lower left')

            fig_ne.tight_layout()
            fig_sub_ne.tight_layout()
            plt.show()

            fig_eta, ax_eta = plt.subplots(figsize=(12, 8))
            fig_sub_eta, ax_sub_eta = plt.subplots(figsize=(12, 16), nrows=2, ncols=1)
            ax_sub_eta = ax_sub_eta.flatten()
            for i,color in enumerate(d_plots.keys()):
                pre_label = d_plots[f'{color}']['label']
                idx = pre_label.find("run")
                label = pre_label[idx:]
                marker = mark[i]
                if marker == 'o':
                    ax = ax_sub_eta[0]
                else:
                    ax = ax_sub_eta[1]
                # for eta_e_array_idx in range(len(d_plots[f'{color}']['x_vals_t'])):
                #     x_vals_eta = d_plots[f'{color}']['x_vals_t'][eta_e_array_idx]
                #     eta_e = d_plots[f'{color}']['eta_e'][eta_e_array_idx]
                #     plt.plot(x_vals_eta, eta_e, color=color)
                #     if eta_e_array_idx == len(d_plots[f'{color}']['x_vals_t']) - 1:
                #         plt.plot(x_vals_eta, eta_e, color=color, label=label)

                y_ax = d_plots[f'{color}']['eta_e']
                y_ax_mask = (y_ax <= 10) & (y_ax >= -10)
                y_ax = y_ax[y_ax_mask]
                x_ax = d_plots[f'{color}']['x_vals_t'][y_ax_mask]
                ax_eta.plot(x_ax, y_ax, markerfacecolor = color,
                         markeredgecolor=color, label=label, marker=marker,linestyle='None',markersize=12)
                ax.plot(x_ax, y_ax, markerfacecolor = color,
                         markeredgecolor=color, label=label, marker=marker,linestyle='None',markersize=12)
            ax_eta.set_xlabel("x (cm)",fontsize=28)
            ax_eta.set_ylabel(r"$\eta_e$",fontsize=28)
            ax_eta.tick_params(labelsize=20)
            ax_eta.set_title(r"$\eta_e = (\nabla T_e / T_e) / (\nabla n_e / n_e)$", fontsize=28)
            ax_eta.legend(fontsize=20, loc='lower left')

            ax_sub_eta[0].set_xlabel("x (cm)", fontsize=28)
            ax_sub_eta[0].set_ylabel(r"$\eta_e$", fontsize=28)
            ax_sub_eta[0].tick_params(labelsize=20)
            ax_sub_eta[0].set_title(r"$\eta_e = (\nabla T_e / T_e) / (\nabla n_e / n_e)$, Probe: 0", fontsize=28)
            ax_sub_eta[0].legend(fontsize=20, loc='lower left')

            ax_sub_eta[1].set_xlabel("x (cm)", fontsize=28)
            ax_sub_eta[1].set_ylabel(r"$\eta_e$", fontsize=28)
            ax_sub_eta[1].tick_params(labelsize=20)
            ax_sub_eta[1].set_title(r"$\eta_e = (\nabla T_e / T_e) / (\nabla n_e / n_e)$, Probe: 1", fontsize=28)
            ax_sub_eta[1].legend(fontsize=20, loc='lower left')

            fig_eta.tight_layout()
            fig_sub_eta.tight_layout()
            plt.show()


            figsub, axessub = plt.subplots(1, 2, figsize=(24, 8))
            axessub.flatten()
            for i,color in enumerate(d_plots.keys()):
                pre_label = d_plots[f'{color}']['label']
                idx = pre_label.find("run")
                label = pre_label[idx:]
                marker = mark[i]

                eta_y_ax = d_plots[f'{color}']['eta_e']
                eta_y_ax_mask = (eta_y_ax <= 10) & (eta_y_ax >= -10)
                eta_y_ax = eta_y_ax[eta_y_ax_mask]
                eta_x_ax = d_plots[f'{color}']['x_vals_t'][eta_y_ax_mask]
                axessub[0].plot(eta_x_ax, eta_y_ax, markerfacecolor = color,
                         markeredgecolor=color, label=label, marker=marker,linestyle='None',markersize=12)

                n_y_ax = d_plots[f'{color}']['normalized grad n']
                n_y_ax_mask = (n_y_ax <= .5) & (n_y_ax >= -.5)
                n_y_ax = n_y_ax[n_y_ax_mask]
                n_x_ax = d_plots[f'{color}']['x_vals_n'][n_y_ax_mask]
                axessub[1].plot(n_x_ax, n_y_ax, markerfacecolor=color,
                         markeredgecolor=color, label=label, marker=marker, linestyle='None', markersize=12)
            axessub[0].set_xlabel("x (cm)",fontsize=28)
            axessub[0].set_ylabel(r"$\eta_e$",fontsize=28)
            axessub[0].tick_params(labelsize=20)
            axessub[0].set_title(r"$\eta_e = (\nabla T_e / T_e) / (\nabla n_e / n_e)$", fontsize=28)
            axessub[0].legend(fontsize=20, loc='lower left')

            axessub[1].set_xlabel("x (cm)", fontsize=28)
            axessub[1].set_ylabel(r"$\nabla n_e / n_e$", fontsize=28)
            axessub[1].set_title("Normalized $n_e$ Gradient", fontsize=28)
            axessub[1].tick_params(labelsize=20)
            axessub[1].legend(fontsize=20, loc='lower left')
            plt.tight_layout(rect=[0, 0, 1, .9])
            plt.show()


    if 6 in user_choice_to_do:
        nc_list = [f for f in os.listdir(langmuir_nc_folder) if f.endswith(".nc")]
        nc_choice = choose_multiple_from_list(nc_list, 'NetCDF file',
                                              null_action="not retrieve data from NetCDF files.")

        datasets = []
        steady_state_times_runs = []

        # Select data sets to plot from saved .nc files in the selected folder
        for i in range(len(nc_choice)):
            ds = xr.load_dataset(os.path.join(langmuir_nc_folder, nc_list[nc_choice[i]]))
            datasets.append(ds)

        # Get temperature into a form where it is indexed by probe, x, y shot, time to be used in Michael's fluctuation
        # calculations
        for dataset in datasets:
            try:
                index = datasets.index(dataset)
                filename = nc_list[nc_choice[index]]
                filename = filename.split('_2024')[0]
                t_e = dataset['t_e']
                t_e = t_e.swap_dims({'sweep': 'time'})
                run_identifier = '2024_Jan_Run_' + filename + '_t_e.nc'
                fluctuations_nc_folder = ensure_directory(langmuir_nc_folder + "fluctuations_nc/")
                save_path = os.path.join(fluctuations_nc_folder, run_identifier)
                t_e.to_netcdf(save_path)
                print(f'Saved to {save_path}')
            except Exception as e:
                print(f'Failed to save dataset')



