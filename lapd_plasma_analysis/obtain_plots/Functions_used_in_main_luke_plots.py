# from pty import slave_open

import matplotlib.pyplot as plt
import numpy as np
import time
import re

from IPython.core.pylabtools import figsize
from plasmapy.analysis.swept_langmuir import find_ion_saturation_current

from lapd_plasma_analysis.file_access import *
from lapd_plasma_analysis.experimental import get_exp_params

from lapd_plasma_analysis.langmuir.helper import *
from lapd_plasma_analysis.langmuir.configurations import *
from lapd_plasma_analysis.langmuir.find_temperature_and_v_p import *
from lapd_plasma_analysis.obtain_plots.Auxillary_functions import *
from lapd_plasma_analysis.plasma_py_comparison import *
from lapd_plasma_analysis.obtain_plots.xarray_plots import f_run_identifier
from lapd_plasma_analysis.langmuir.getIVsweep import get_sweep_voltage, get_sweep_current, get_shot_positions
from lapd_plasma_analysis.langmuir.characterization import make_characteristic_array, isolate_ramps
from lapd_plasma_analysis.langmuir.preview import preview_raw_sweep, preview_characteristics
from lapd_plasma_analysis.langmuir.diagnostics import (langmuir_diagnostics, detect_steady_state_times, get_pressure,
                                                       get_electron_ion_collision_frequencies)
from lapd_plasma_analysis.langmuir.neutrals import get_neutral_density
from lapd_plasma_analysis.langmuir.interferometry import interferometry_calibration
from lapd_plasma_analysis.langmuir.plots import get_title
from lapd_plasma_analysis.langmuir.metadata_for_dataset import get_supplemental_metadata
from lapd_plasma_analysis.obtain_plots.xarray_plots import build_subplots
from lapd_plasma_analysis.file_access import default_fig_params
import pandas as pd


def IV_parameters(hdf5_path):
    """

        Parameters
        ----------
        hdf5_path : `str`
            The directory in which the HDF5 files are stored.
    """

    # Create a dictionary of experimental parameters (run name, experiment name, Discharge current\
        # Fill pressure, and peak magnetic field). Other parameters are dependent on the experiment
        # that was done and are described in the function in experimental.py
    exp_params_dict = get_exp_params(hdf5_path)

    # Determine if we are dealing with a hydrogen or helium plasma (returns a string H+ of He 4+)
        # also append to exp_params_dict
    ion_type = get_ion(exp_params_dict['Run name'])
    exp_params_dict = exp_params_dict | {"Ion type": ion_type}

    # Now we want to get the configuration ID (integer) (What experimental run was it Jan 2024 etc.) and what board/channel
        # the IV sweep was taken on tuple in the format (board,channel). Depending on the config ID
    config_id = get_config_id(exp_params_dict['Exp name'])

    # Langmuir configs are mostly hardcoded and return Langmuir probe xarray depending on config ID
    #   (board, channel, receptacle, port, face, resistance, area, gain)
    # Voltage gain is a hard coded dependent on config ID. We need to divide the bias by the voltage gain to get raw voltage data
    # Orientation is hard coded dependent on config ID. Tells us if the data is upright or inverted. Returns 1 or -1

    langmuir_configs = get_langmuir_config(hdf5_path, config_id)
    voltage_gain = get_voltage_gain(config_id)
    orientation = get_orientation(config_id)
    vsweep_bc = get_vsweep_bc(config_id)
    current_bc=[]
    for i in range(len(langmuir_configs)):
        current_bc.append((langmuir_configs['board'][i],langmuir_configs['channel'][i]))


    return exp_params_dict, vsweep_bc, langmuir_configs, config_id, voltage_gain, orientation,current_bc

# Adapted from Leo Murphy's preview raw sweep function in the langmuir.preview directory
def obtain_data(hdf5_path, bias, current, positions, langmuir_config, exp_params_dict, save_plots, select_sweeps = True):
    """
    Parameters
    ----------
    hdf5_path: String indicating the path to the HDF5 file.
    bias : 3D numpy array with
    current : `astropy.units.Quantity`
        2D array (dimensions: x-y position, shot at x-y position) of Langmuir sweep (collected) current
    positions: ndarray of x and y positions
    langmuir_config: Array yielding key parameters for langmuir analysis
    exp_params_dict: Dictionary of experiment parameters
    save_plots: Boolean - whether to save the figure
    select_sweeps: Boolean - whether the user wants to manually select sweeps

    Returns
    -------
    bias_to_plot - A list of list containing bias data for the user's chosen shot,position combinations
    current_to_plot - A list of lists containing current data for the user's chosen shot,position combinations
    loc_shot - A list of lists containing position and shot locations for the user's chosen combinations
    filepath - A string indicating the folder path to save the plots yet to be generated

    """
    if save_plots:
        ensure_directory(hdf5_path + f"{exp_params_dict['Exp name']}/")
        ensure_directory(hdf5_path + f"{exp_params_dict['Exp name']}/{exp_params_dict['Run name']}/")
        filepath = hdf5_path + f"{exp_params_dict['Exp name']}/{exp_params_dict['Run name']}/"
    else:
        filepath = ""

    x = np.unique(positions[:, 0])
    y = np.unique(positions[:, 1])
    bias_to_plot = []
    current_to_plot = []
    loc_shot = []
    if select_sweeps:
        print(f"\nPort {langmuir_config['port']}, face {langmuir_config['face']}")
        print(f"Dimensions of sweep bias and current array: "
              f"{len(x)} x-positions, "
              f"{len(y)} y-positions, and "
              f"{current.shape[1]} shots")
        print(f"  * x positions range from {min(x)} to {max(x)}",
              f"  * y positions range from {min(y)} to {max(y)}",
              f"  * shot indices range from 0 to {current.shape[1] - 1}.", sep="\n")
        # Build a 3 element list to collect x-position, y-position, shot information from the user
        x_y_shot_to_plot = [0, 0, 0]
        variables_to_enter = ["x position", "y position", "shot"]
        print("\nNotes: \tIndices are zero-based; choose an integer between 0 and n - 1, inclusive."
              "\n \t \tEnter a non-integer below (such as the empty string) to skip raw sweep preview mode "
              "for this isweep source and continue to diagnostics.")
        sweep_view_mode = True
        # Loops until the user enters the empty string or a non-numeric value

        while sweep_view_mode:
            for i in range(len(x_y_shot_to_plot)):
                try:
                    index_given = int(input(f"Enter a zero-based index for {variables_to_enter[i]}: "))
                except ValueError:
                    sweep_view_mode = False
                    break
                x_y_shot_to_plot[i] = index_given
            if not sweep_view_mode:
                break
            print()

            try:
                loc_x, loc_y = x[x_y_shot_to_plot[0]], y[x_y_shot_to_plot[1]]
                loc = (positions == [loc_x, loc_y]).all(axis=1).nonzero()[0][0]

                # Build a list of location shot combinations to use in plot titles
                loc_shot.append([loc_x, loc_y,x_y_shot_to_plot[2]])

                bias_to_plot.append(bias[(loc, *x_y_shot_to_plot[2:])])
                current_to_plot.append(current[(loc, *x_y_shot_to_plot[2:])])

                # String indicating port and face of sweep current source (e.g. "20L" or "27")


            except IndexError as e:
                print(e)
            continue
    else:
        # Loop over all (x, y, shot) combinations automatically
        for loc_x in x:
            for loc_y in y:
                loc = (positions == [loc_x, loc_y]).all(axis=1).nonzero()[0][0]

                for shot_index in range(current.shape[1]):
                    bias_to_plot.append(bias[(loc, shot_index)])
                    current_to_plot.append(current[(loc, shot_index)])
                    loc_shot.append([loc_x, loc_y, shot_index])


    return bias_to_plot,current_to_plot,loc_shot,filepath

def plot_bias_vs_time(bias_to_plot, loc_shot, exp_params_dict, dt, port_face_string):
    '''

    Parameters
    ----------
    bias_to_plot - Quantity List (V) - Full array with bias values associated with a location shot combination.
    Each indexed list corresponds to a different shot - position combination
    loc_shot - Three element array where idx 0 is the x-position, idx 1 is the y-position, idx 2 is the shot index
    exp_params_dict - Dictionary of experiment parameters see experimental for more details
    dt - Float - Time step for the bias array

    Returns
    -------

    '''
    i = 0
    for bias_array in bias_to_plot:
        bias_array = bias_array.value
        time_array = np.arange(len(bias_array)) * dt.to(u.ms).value

        plt.plot(time_array, bias_array)
        plt.xlabel("Time (ms)")
        plt.ylabel("Bias (V)")
        run_identifier = f_run_identifier(filename=exp_params_dict['Run name'])
        plt.title(f"{run_identifier}, x: {loc_shot[i][0]}, shot: {loc_shot[i][2]}, port: {port_face_string}")
        plt.tight_layout()
        # TODO add probe to the title
        plt.show()
        i += 1
def plot_current_vs_time(bias_to_plot, current_to_plot, loc_shot, exp_params_dict, dt, port_face_string):
    '''

    Parameters
    ----------
    bias_to_plot - Quantity List (V) - Full array with bias values associated with a location shot combination.
    Each indexed list corresponds to a different shot - position combination
    current_to_plot - Quantity Array (A) - Full array with current values associated with a location shot combination
    Works exactly the same way as bias_to_plot
    loc_shot - Three element array where idx 0 is the x-position, idx 1 is the y-position, idx 2 is the shot index
    exp_params_dict - Dictionary of experiment parameters. (See experimental for more details)
    dt - Float - Time step for the bias array

    Returns
    -------

    '''
    for i in range(len(bias_to_plot)):
        bias_array = bias_to_plot[i].value
        time_array = np.arange(len(bias_array)) * dt.to(u.ms).value
        current_array = current_to_plot[i].value

        plt.plot(time_array, current_array)
        plt.xlabel("Time (ms)")
        plt.ylabel("Current (A)")
        run_identifier = f_run_identifier(filename=exp_params_dict['Run name'])
        plt.title(f"{run_identifier}, x: {loc_shot[i][0]}, shot: {loc_shot[i][2]}, port: {port_face_string}")
        plt.tight_layout()
        # TODO add probe to the title
        plt.show()


def plot_iv_sweep(filepath, bias_to_plot, current_to_plot, port_face_string, plot_choices, save_plots,
                  ramp_times,exp_params_dict,loc_shot,dt, figure_folder = None, csv_folder = None, make_csv = False):
    '''

    Parameters
    ----------
    filepath - String - Location of hdf5 file on the user's device
    bias_to_plot - Quantity List (V) - Full array with bias values associated with a location shot combination.
    Each indexed list corresponds to a different shot - position combination
    current_to_plot Quantity Array (A) - Full array with current values associated with a location shot combination
    Works exactly the same way as bias_to_plot
    port_face_string - String - Indicates the port and the face for the probe that is currently being plotted.
    plots_choices - List of str - Indicates which plots the user wants to see
    save_plots - Boolean - Indicates whether to save the plot
    ramp_times - Quantity array (ms) - Array of times where the IV sweep ends
    exp_params_dict - Dictionary of experiment parameters. (See experimental for more details)
    loc_shot - Three element array where idx 0 is the x-position, idx 1 is the y-position, idx 2 is the shot index
    dt - Float - Time step for the bias array
    figure_folder - String - Folder where to save the figure
    csv_folder - String - Folder where to save a csv file of raw IV sweep data

    Returns
    -------

    Runs if the user selects that they want to see a logarithmic or a raw IV sweep
    '''

    default_fig_height, default_fig_width = default_fig_params()

    # Obtains the run number, Date and Ion type to be used in the plot title
    run_identifier = f_run_identifier(filename = exp_params_dict['Run name'])


    # Allows the user to choose whether they want to automatically choose evenly spaced sweeps or choose individual
    # sweeps to look at
    user_choice = ask_yes_or_no('Do you want to choose specific times (y/n)? ')
    if user_choice:
        # This loop prompts the user to select the time values of the sweeps they are interested.
        # It handles improper inputs and repeated values
        end_times = list(ramp_times[1:].to(u.ms).value)

        times_proper_input = False
        time_choice = None  # For Py Charm warning handling
        while not times_proper_input:
            try:
                time_choice = choose_multiple_from_list(end_times, 'Select ramp end times',
                                                        null_action= 'end selection')
                time_choice = list(set(time_choice))
                if (time_choice == [] or
                        any(i >= len(end_times) or i < 0 for i in time_choice)):
                    print('Invalid input - Ensure all selected letters correspond to a listed time')
                    time.sleep(1)
                    continue

                times_proper_input = True
            except ValueError:
                print('Invalid input - please input a LETTER associated with an end time in the list.')
                time.sleep(1)
        end_ramp_times_to_plot = [end_times[choice] * u.ms for choice in time_choice]

    else:
        # Allows user to select how many individual IV sweeps they want to look at
        how_many_plots = 0
        valid_input = False
        while not valid_input:
            try:
                how_many_plots = int(input("How many IV sweeps would you like to see? "))
                valid_input = True
            except ValueError:
                print("")
        end_ramp_times_to_plot = []
        for h in range(how_many_plots):
            end_ramp_times_to_plot.append(ramp_times[(h + 1) * int(len(ramp_times) / (how_many_plots + 1))])

    # Plot each individual sweep based off of the end times selected in the previous section
    time_array = np.arange(len(bias_to_plot)) * dt.to(u.ms).value
    for i in range(len(end_ramp_times_to_plot)):
        # Create a broad range to search for the sweep
        end_time = end_ramp_times_to_plot[i].to(u.ms)
        previous_end_time = ramp_times[np.where(ramp_times.to(u.ms) < end_time)[0][-1]].to(u.ms)

        search_times = ((time_array >= previous_end_time.value) &
                            (time_array <= end_time.value))

        # Narrow the time range to more accurately determine where the sweep begins and ends
        first_index, last_index = find_sweep_indices(time_array, end_ramp_times_to_plot[i],
                                                     search_times, bias_to_plot, dt)

        # Mask bias and current based off of the narrowed time for the sweep
        start_time = (first_index * dt.to(u.ms).value)
        end_time = end_ramp_times_to_plot[i].to(u.ms).value
        mask = ((time_array >= start_time) & (time_array <= end_time))
        sorted_bias = bias_to_plot[mask][np.argsort(bias_to_plot[mask])]
        sorted_current = current_to_plot[mask][np.argsort(bias_to_plot[mask])]

        # Build a title that can be used for both types of figures
        plot_title = (f"Run: {run_identifier}\n"
                      f"Probe port and face: {port_face_string}, "
                      f"x: {loc_shot[0]}, y: {loc_shot[1]}, shot: {loc_shot[2]}\n\n"
                      f"Sweep at [ms]: {dt.to(u.ms).value * first_index}")
        if make_csv:
            raw_data_csv_path = ensure_directory(csv_folder + 'raw_data/')

            # Replace newlines with spaces, colons with underscores, and remove commas
            safe_title = plot_title.replace('\n', '_').replace(':', '_').replace(',', '')

            # Strip out any remaining illegal characters for Windows/Mac/Linux
            safe_title = re.sub(r'[\\/*?"<>|\[\]]', "", safe_title)

            # Create the full file path
            csv_filename = f"{safe_title}.csv"
            full_csv_path = os.path.join(raw_data_csv_path, csv_filename)

            # Stack the 1D arrays as columns
            data_to_save = np.column_stack((sorted_bias.value, sorted_current.value))

            # Save the data
            np.savetxt(
                full_csv_path,
                data_to_save,
                delimiter=",",
                header="sorted_bias,sorted_current",
                comments=''  # Prevents numpy from adding a '#' before the header
            )
            print(f"Data saved successfully to: {full_csv_path}")

        # Find the floating potential
        v_f_bias, v_f_current, v_f_index = get_floating_potential(sorted_bias, sorted_current)

        if v_f_bias is not None:
            v_f_bias_val = v_f_bias.to(u.V).value
            ion_current = get_ion_current(sorted_bias, sorted_current, v_f_bias)
        else:
            v_f_bias_val = np.nan
            ion_current = np.nan

        t_e_spline, t_e_spline_int, vp_spline, esat_slope, esat_intercept, t_e_offset, spline_bias,_ = (
            get_t_e_spline(sorted_bias, sorted_current, v_f_bias))

        if t_e_spline is not None:
            t_e_spline_val = t_e_spline.to(u.eV).value
            vp_spline_val = vp_spline.to(u.V).value
            v_spline_mask = ((sorted_bias.to(u.V).value >= v_f_bias_val) &
                             (sorted_bias.to(u.V).value <= vp_spline_val))
            v_to_plot_spline = sorted_bias.to(u.V).value[v_spline_mask]
            ion_curr_match_spline = ion_current.to(u.A).value[v_spline_mask]

            log_spline_fit = (1 / t_e_spline_val * v_to_plot_spline) + t_e_spline_int
            i_e_spline_fit = np.exp(log_spline_fit) - t_e_offset
            i_total_fit_spline = i_e_spline_fit + ion_curr_match_spline
        else:
            i_total_fit_spline = None
            v_to_plot_spline = None
            t_e_spline_val = None
            vp_spline_val = None
            i_e_spline_fit = None

        if v_f_bias is not None:
            breakpoint_dict = get_t_e_breakpoint(sorted_bias, sorted_current, v_f_bias, ion_current, batch_mode=True)

            t_e_breakpoint = breakpoint_dict['Te']
        else:
            breakpoint_dict = None
            t_e_breakpoint = None

        if t_e_breakpoint is not None:
            vp_breakpoint = breakpoint_dict['V_p']
            t_e_breakpoint_value = t_e_breakpoint.to(u.eV).value
            vp_breakpoint_val = vp_breakpoint.to(u.V).value
            breakpoint_slope = breakpoint_dict['m_ret']
            breakpoint_intercept = breakpoint_dict['b_ret']

            # mask to the REPORTED segment's own voltage span (cold line if
            # single-population, hot line if two) instead of v_f -> V_p, so
            # the line is drawn only where the fit applies.
            seg_lo, seg_hi = breakpoint_dict['fit_region']['retarding_V']
            v_breakpoint_mask = ((sorted_bias.to(u.V).value >= seg_lo) &
                                 (sorted_bias.to(u.V).value <= seg_hi))
            v_to_plot_breakpoint = sorted_bias.to(u.V).value[v_breakpoint_mask]
            ion_curr_match_breakpoint = ion_current.to(u.A).value[v_breakpoint_mask]

            log_breakpoint_fit = (breakpoint_slope * v_to_plot_breakpoint) + breakpoint_intercept
            i_e_breakpoint_fit = np.exp(log_breakpoint_fit)
            i_total_fit_breakpoint = i_e_breakpoint_fit + ion_curr_match_breakpoint
        else:
            i_total_fit_breakpoint = None
            v_to_plot_breakpoint = None
            t_e_breakpoint_value = None
            log_spline_fit = None

        if 'Plot individual raw IV sweeps for a position-shot combination' in plot_choices:
            if save_plots:
                raw_sweep_dir = ensure_directory(figure_folder + 'raw_IV_sweeps/')
                run_dir = ensure_directory(raw_sweep_dir + f'{run_identifier}/')
                port_specific_folder = ensure_directory(run_dir + f'{port_face_string}/')
            else:
                port_specific_folder = None
            raw_sweep_plot_choices = ['Plot raw IV sweeps',
                                      'Plot individual raw IV sweeps with old and new temperature fits',
                                      'Plot individual raw IV sweeps with new temperature fit (breakpoint)',
                                      'Plot breakpoint method step-by-step verification']

            raw_sweep_proper_input = False
            raw_sweep_choice = None  # For Py Charm warning handling
            while not raw_sweep_proper_input:
                try:
                    raw_sweep_choice = choose_multiple_from_list(raw_sweep_plot_choices, 'Select plots to see',
                                                            null_action='end selection')
                    raw_sweep_choice = list(set(raw_sweep_choice))
                    if (raw_sweep_choice == [] or
                            any(i >= len(raw_sweep_plot_choices) or i < 0 for i in raw_sweep_choice)):
                        print('Invalid input - Ensure all selected letters correspond to a listed plot')
                        time.sleep(1)
                        continue

                    raw_sweep_proper_input = True
                except ValueError:
                    print('Invalid input - please input a LETTER associated with a plot in the list.')
                    time.sleep(1)
            raw_sweep_to_plot = [raw_sweep_plot_choices[choice] for choice in raw_sweep_choice]

            if 'Plot raw IV sweeps' in raw_sweep_to_plot:
                # Set figure size
                plt.rcParams['figure.figsize'] = (12, 6)
                # Plot the raw IV sweep
                plt.scatter(sorted_bias, sorted_current, color='b')

                plt.title(f"Langmuir sweep I vs V, Run: {run_identifier}\n"
                          f"Probe port and face: {port_face_string}, "
                          f"x: {loc_shot[0]}, y: {loc_shot[1]}, shot: {loc_shot[2]}\n\n"
                          f"Sweep at [ms]: {dt.to(u.ms).value * first_index}")
                plt.xlabel("Voltage [V]")
                plt.ylabel("Current [A]")

                if save_plots:
                    plot_specific_folder = ensure_directory(port_specific_folder + 'raw_IV_sweeps/')
                    fig_name = f'x_{loc_shot[0]}_shot_{loc_shot[2]}_time{dt.to(u.ms).value * first_index:.2f}.svg'
                    plt.savefig(plot_specific_folder + fig_name, bbox_inches='tight')
                    print('Plot saved to: ', plot_specific_folder + fig_name)
                plt.show()
                plt.close()

            if ('Plot individual raw IV sweeps with old and new temperature fits' in raw_sweep_to_plot or
                    'Plot individual raw IV sweeps with new temperature fit (breakpoint)' in raw_sweep_to_plot or
                    'Plot breakpoint method step-by-step verification' in raw_sweep_to_plot):

                if 'Plot individual raw IV sweeps with old and new temperature fits' in raw_sweep_to_plot:
                    plt.rcParams['figure.figsize'] = (12, 6)

                    # Plot the raw IV sweep
                    plt.scatter(sorted_bias, sorted_current, color='k', label='Probe data')

                    if t_e_spline is not None:
                        # Plot the fit associated with the old T_e calculations
                        plt.plot(v_to_plot_spline, i_total_fit_spline, color='r',
                                 label=fr'Fit from tanh function, $T_e$ = {t_e_spline_val:.2f} eV')
                        plt.axvline(vp_spline_val, color='r', linestyle='--')
                        plt.axvline(v_f_bias_val, color='r', linestyle='--')
                    else:
                        plt.plot([], [], ' ', label=fr"Fit from tanh $T_e$ method not found")

                    if t_e_breakpoint is not None:
                        # Plot the fit associated with the new T_e calculations
                        plt.plot(v_to_plot_breakpoint, i_total_fit_breakpoint, color='b',
                                 label=fr'New Te method, $T_e$ = {t_e_breakpoint_value:.2f} eV')
                        plt.axvline(seg_lo, color='b', linestyle='--')
                        plt.axvline(seg_hi, color='b', linestyle='--')
                    else:
                        plt.plot([], [], ' ', label=fr"Fit from new $T_e$ method not found")



                    plt.title(f"Langmuir sweep I vs V, Run: {run_identifier}\n"
                              f"Probe port and face: {port_face_string}, "
                              f"x: {loc_shot[0]}, y: {loc_shot[1]}, shot: {loc_shot[2]}\n\n"
                              f"Sweep at [ms]: {dt.to(u.ms).value * first_index}")
                    plt.xlabel("Voltage [V]")
                    plt.ylabel("Current [A]")
                    plt.legend(loc='upper left')

                    if save_plots:
                        plot_specific_folder = ensure_directory(port_specific_folder + 'new_old_fits_raw_IV_sweeps/')
                        fig_name = f'x_{loc_shot[0]}_shot_{loc_shot[2]}_time{dt.to(u.ms).value * first_index:.2f}.svg'
                        plt.savefig(plot_specific_folder + fig_name, bbox_inches='tight')
                        print('Plot saved to: ', plot_specific_folder + fig_name)
                    plt.show()
                    plt.close()
                if 'Plot individual raw IV sweeps with new temperature fit (breakpoint)' in raw_sweep_to_plot:
                    plt.rcParams['figure.figsize'] = (12, 6)
                    # Plot the raw IV sweep
                    plt.scatter(sorted_bias, sorted_current, color='k', label='Probe data')
                    # Plot the fit associated with the new T_e calculations
                    if t_e_breakpoint is not None:
                        plt.plot(v_to_plot_breakpoint, i_total_fit_breakpoint, color='b',
                                 label=fr'New Te method, $T_e$ = {t_e_breakpoint_value:.2f} eV')
                        plt.axvline(seg_lo, color='b', linestyle='--')
                        plt.axvline(seg_hi, color='b', linestyle='--')
                    else:
                        plt.plot([], [], ' ', label=fr"Fit from new $T_e$ method not found")

                    plt.title(f"Langmuir sweep I vs V, Run: {run_identifier}\n"
                              f"Probe port and face: {port_face_string}, "
                              f"x: {loc_shot[0]}, y: {loc_shot[1]}, shot: {loc_shot[2]}\n\n"
                              f"Sweep at [ms]: {dt.to(u.ms).value * first_index}")
                    plt.xlabel("Voltage [V]")
                    plt.ylabel("Current [A]")
                    plt.legend(loc='upper left', fontsize='x-small')

                    if save_plots:
                        plot_specific_folder = ensure_directory(port_specific_folder + 'raw_IV_sweeps_with_fit/')
                        fig_name = f'x_{loc_shot[0]}_shot_{loc_shot[2]}_time{dt.to(u.ms).value * first_index:.2f}.svg'
                        plt.savefig(plot_specific_folder + fig_name, bbox_inches='tight')
                        print('Plot saved to: ', plot_specific_folder + fig_name)
                    plt.show()
                    plt.close()

                if 'Plot breakpoint method step-by-step verification' in raw_sweep_to_plot:
                    snr_k = 4.0  # must match get_t_e_breakpoint's snr_k
                    V_arr = sorted_bias.to(u.V).value
                    I_arr = sorted_current.to(u.A).value
                    Iion_arr = ion_current.to(u.A).value if ion_current is not None \
                        else np.zeros_like(I_arr)
                    Ie_arr = I_arr - Iion_arr
                    rms = breakpoint_dict['flags']['floor_rms']
                    fl = breakpoint_dict['flags']

                    layout = [[3]]
                    fig, axes, letters = build_subplots(layout, 1.5 * default_fig_width, 1.5 * default_fig_height)

                    # Step 1: ion-current removal in IV space
                    ax = axes[letters[0]]
                    ax.scatter(V_arr, I_arr, s=6, color='0.6', label='raw probe I')
                    ax.plot(V_arr, Iion_arr, color='tab:orange', lw=1.5, label='ion current')
                    ax.scatter(V_arr, Ie_arr, s=6, color='tab:blue', label=r'$I_e = I - I_{ion}$')
                    ax.axvline(v_f_bias_val, color='0.4', ls=':', label=r'$V_f$')
                    ax.axhline(0, color='k', lw=0.6)
                    ax.set_xlabel('Voltage [V]');
                    ax.set_ylabel('Current [A]')
                    ax.set_title('Step 1: ion-current removal')
                    ax.legend(loc='upper left', fontsize=16)

                    # Step 2: floor-RMS mask into log space
                    ax = axes[letters[1]]
                    pos = Ie_arr > 0
                    ax.scatter(V_arr[pos], np.log(Ie_arr[pos]), s=6, color='0.75',
                               label=r'all $\ln I_e>0$')
                    if breakpoint_dict['Te'] is not None:
                        fr = breakpoint_dict['fit_region']
                        ax.scatter(fr['V'], fr['lnI'], s=10, color='tab:blue',
                                   label=r'kept: $I_e > %.0f\cdot$RMS' % snr_k)
                    ax.axhline(np.log(snr_k * rms), color='tab:red', ls='--',
                               label=r'$\ln(%.0f\cdot$RMS$)$' % snr_k)
                    ax.set_xlabel('Voltage [V]')
                    ax.set_ylabel(r'$\ln I_e$')
                    ax.set_title('Step 2: floor-RMS mask')
                    ax.legend(loc='lower right', fontsize=16)

                    # Step 3: region fit + V_p
                    ax = axes[letters[2]]
                    if breakpoint_dict['Te'] is not None:
                        fr = breakpoint_dict['fit_region']
                        ax.scatter(fr['V'], fr['lnI'], s=10, color='k', label=r'masked $\ln I_e$')
                        Vp = vp_breakpoint_val
                        m_ret, b_ret = fr['m_ret'], fr['b_ret']
                        m_sat, b_sat = fr['esat_slope'], fr['esat_intercept']
                        rlo, rhi = fr['retarding_V']
                        slo, shi = fr['saturation_V']

                        # shaded fit-region spans (which data each line was fit on)
                        ax.axvspan(rlo, rhi, color='tab:green', alpha=0.12,
                                   label=r'$T_e$ fit region')
                        ax.axvspan(slo, shi, color='tab:red', alpha=0.10,
                                   label='E-sat fit region')

                        # retarding T_e line: solid over its own span, dotted
                        # extrapolation UP to the V_p intersection
                        xr = np.linspace(rlo, rhi, 50)
                        ax.plot(xr, m_ret * xr + b_ret, color='tab:green', lw=2,
                                label=r'$T_e$ line = %.2f eV' % breakpoint_dict['Te'].to(u.eV).value)
                        ax.plot([rhi, Vp], [m_ret * rhi + b_ret, m_ret * Vp + b_ret],
                                color='tab:green', lw=1, ls=':')

                        # e-sat line: solid over its own span, dotted
                        # extrapolation DOWN to the V_p intersection
                        xs = np.linspace(slo, shi, 50)
                        ax.plot(xs, m_sat * xs + b_sat, color='tab:red', lw=2, label='E-sat line')
                        ax.plot([Vp, slo], [m_sat * Vp + b_sat, m_sat * slo + b_sat],
                                color='tab:red', lw=1, ls=':')

                        if fr.get('n_populations') == 2 and 'cold_V' in fr:
                            clo, chi = fr['cold_V']
                            xc = np.linspace(clo, chi, 50)
                            ax.plot(xc, fr['cold_m_ret'] * xc + fr['cold_b_ret'],
                                    color='tab:cyan', lw=1.2,
                                    label=r'cold line = %.2f eV' % fr['cold_Te'])

                        # the intersection of the two lines IS V_p
                        y_int = m_ret * Vp + b_ret
                        ax.plot([Vp], [y_int], marker='o', ms=9, mfc='none',
                                mec='k', mew=1.8, label=r'intersection = $V_p$')
                        if fl.get('vp_refined', False):
                            ax.axvline(vp_breakpoint_val, color='k', ls='-',
                                       label=rf'$V_p$ = {vp_breakpoint_val} V')
                        else:
                            ax.axvline(vp_breakpoint_val, color='k', ls='-',
                                       label=rf'$V_p$ = {vp_breakpoint_val} V')
                        ax.set_title('Step 3: region fit  (n_pop=%d, %s)' %
                                     (fr['n_populations'], fl.get('quality', '?')))
                        ax.legend(loc='lower right', fontsize=16)
                    else:
                        ax.text(0.5, 0.5, 'No valid fit\n(%s)' % fl.get('fail', 'rejected'),
                                ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('Step 3: region fit — rejected')
                    ax.set_xlabel('Voltage [V]');
                    ax.set_ylabel(r'$\ln I_e$')

                    fig.suptitle(f"Breakpoint step verification, Run: {run_identifier}\n"
                                 f"Probe port and face: {port_face_string}, "
                                 f"x: {loc_shot[0]}, y: {loc_shot[1]}, shot: {loc_shot[2]}, "
                                 f"Sweep at [ms]: {dt.to(u.ms).value * first_index:.3f}",
                                 fontsize=20)
                    fig.tight_layout(rect=[0, 0, 1, 0.97])

                    if save_plots:
                        plot_specific_folder = ensure_directory(
                            port_specific_folder + 'breakpoint_step_verification/')
                        fig_name = f'x_{loc_shot[0]}_shot_{loc_shot[2]}_time{dt.to(u.ms).value * first_index:.2f}.svg'
                        fig.savefig(plot_specific_folder + fig_name, bbox_inches='tight')
                        print('Plot saved to: ', plot_specific_folder + fig_name)
                    plt.show()
                    plt.close(fig)

        if 'Plot log plot of IV sweeps for a position-shot combination' in plot_choices:
            if save_plots:
                log_sweep_dir = ensure_directory(figure_folder + 'log_IV_sweeps/')
                run_dir = ensure_directory(log_sweep_dir + f'{run_identifier}/')
                port_specific_folder = ensure_directory(run_dir + f'{port_face_string}/')
            else:
                port_specific_folder = None
            log_sweep_plot_choices = ['Plot log IV sweeps',
                                      'Plot individual log IV sweeps with old and new temperature fits',
                                      'Plot individual log IV sweeps with new temperature fit (breakpoint)']

            log_sweep_proper_input = False
            log_sweep_choice = None  # For Py Charm warning handling
            while not log_sweep_proper_input:
                try:
                    log_sweep_choice = choose_multiple_from_list(log_sweep_plot_choices, 'Select plots to see',
                                                            null_action='end selection')
                    raw_sweep_choice = list(set(log_sweep_choice))
                    if (raw_sweep_choice == [] or
                            any(i >= len(log_sweep_plot_choices) or i < 0 for i in raw_sweep_choice)):
                        print('Invalid input - Ensure all selected letters correspond to a listed plot')
                        time.sleep(1)
                        continue

                    log_sweep_proper_input = True
                except ValueError:
                    print('Invalid input - please input a LETTER associated with a plot in the list.')
                    time.sleep(1)
            log_sweep_to_plot = [log_sweep_plot_choices[choice] for choice in log_sweep_choice]

            if 'Plot log IV sweeps' in log_sweep_to_plot:
                plt.rcParams['figure.figsize'] = (12, 6)

                # electron current in log space: I_e = I - I_ion, keep positives only
                if ion_current is not None and hasattr(ion_current, 'to'):
                    i_e_data = sorted_current.to(u.A).value - ion_current.to(u.A).value
                else:
                    i_e_data = sorted_current.to(u.A).value
                v_all = sorted_bias.to(u.V).value
                log_mask = i_e_data > 0
                full_v = v_all[log_mask]
                full_i = np.log(i_e_data[log_mask])
                if v_f_bias is not None:
                    pre_floating_v_mask = np.where(full_v <= v_f_bias.value)[0]
                    pre_floating_i = full_i[pre_floating_v_mask]
                    lower_y_lim = np.median(pre_floating_i) - 1
                else:
                    lower_y_lim = np.min(np.log(i_e_data[log_mask])) + 2
                upper_y_lim = np.max(np.log(i_e_data[log_mask])) + .25

                plt.scatter(full_v, full_i, color='b')

                plt.title(f"Langmuir sweep ln(I_e) vs V, Run: {run_identifier}\n"
                          f"Probe port and face: {port_face_string}, "
                          f"x: {loc_shot[0]}, y: {loc_shot[1]}, shot: {loc_shot[2]}\n\n"
                          f"Sweep at [ms]: {dt.to(u.ms).value * first_index}")
                plt.xlabel("Voltage [V]")
                plt.ylabel(r"$\ln(I_e)$")
                plt.ylim(lower_y_lim, upper_y_lim)

                if save_plots:
                    plot_specific_folder = ensure_directory(port_specific_folder + 'log_IV_sweeps/')
                    fig_name = f'x_{loc_shot[0]}_shot_{loc_shot[2]}_time{dt.to(u.ms).value * first_index:.2f}.svg'
                    plt.savefig(plot_specific_folder + fig_name, bbox_inches='tight')
                    print('Plot saved to: ', plot_specific_folder + fig_name)
                plt.show()
                plt.close()

            if 'Plot individual log IV sweeps with old and new temperature fits' in log_sweep_to_plot:
                layout = [[2]]
                fig, axes, letters = build_subplots(layout,
                                                    fig_width=1.25 * default_fig_width, fig_height=1.25 * default_fig_height)
                full_ax = axes[letters[0]]
                zoomed_ax = axes[letters[1]]

                if ion_current is not None and hasattr(ion_current, 'to'):
                    i_e_data = sorted_current.to(u.A).value - ion_current.to(u.A).value
                else:
                    i_e_data = sorted_current.to(u.A).value
                v_all = sorted_bias.to(u.V).value
                log_mask = i_e_data > 0
                full_v = v_all[log_mask]
                full_i = np.log(i_e_data[log_mask])
                if v_f_bias is not None:
                    pre_floating_v_mask = np.where(full_v <= v_f_bias.value)[0]
                    pre_floating_i = full_i[pre_floating_v_mask]
                    lower_y_lim = np.median(pre_floating_i) - 1
                else:
                    lower_y_lim = np.min(np.log(i_e_data[log_mask])) + 2
                upper_y_lim = np.max(np.log(i_e_data[log_mask])) + .25

                full_ax.scatter(full_v, full_i,
                            color='k', label='Probe data')

                # --- Visual T_e comparison: draw BOTH methods as straight lines that
                # sit on the ln(I_e) data, over EACH method's OWN fit region, so their
                # SLOPES (= 1/T_e) can be compared by eye. Steeper line = colder T_e.
                # The breakpoint line is already ln(I_e) with slope 1/T_e_bp over its own
                # retarding segment [seg_lo, seg_hi]. For the tanh we use spline_bias --
                # the bias array get_t_e_spline returns for the points it fit -- and draw
                # a straight line of the tanh's slope (1/T_e_tanh) across that span. The
                # tanh was fit in ln(I_e+offset) space, so we don't have its ln(I_e)
                # intercept directly; we anchor the line vertically with a fixed-slope
                # least-squares fit to the ln(I_e) data over the tanh's retarding portion
                # (positive-current points above V_f), so it lies on the data honestly.

                # tanh: straight line of slope 1/T_e_tanh over its OWN returned region
                if t_e_spline is not None and spline_bias is not None and len(spline_bias) > 1:
                    sb_lo, sb_hi = float(np.min(spline_bias)), float(np.max(spline_bias))
                    m_th = 1.0 / t_e_spline_val
                    # anchor only on positive-current retarding data (drop sub-floor tail)
                    anchor = log_mask & (v_all >= max(v_f_bias_val, sb_lo)) & (v_all <= sb_hi)
                    if np.count_nonzero(anchor) > 1:
                        b_th = np.mean(np.log(i_e_data[anchor]) - m_th * v_all[anchor])
                    else:  # degenerate: fall back to full region
                        reg = log_mask & (v_all >= sb_lo) & (v_all <= sb_hi)
                        b_th = np.mean(np.log(i_e_data[reg]) - m_th * v_all[reg])
                    x_th = np.array([max(v_f_bias_val, sb_lo), sb_hi])
                    full_ax.plot(x_th, m_th * x_th + b_th, color='r', lw=2, ls='--',
                             label=fr'tanh method, $T_e$ = {t_e_spline_val:.2f} eV')
                    full_ax.axvline(sb_lo, color='r', linestyle=':')
                    full_ax.axvline(sb_hi, color='r', linestyle=':')

                else:
                    full_ax.plot([], [], ' ', label=fr"Fit from tanh $T_e$ method not found")
                    sb_lo = np.nan
                    sb_hi = np.nan
                    x_th = np.array([np.nan])
                    m_th = np.nan
                    b_th = np.nan

                # breakpoint: log_breakpoint_fit is already ln(I_e_fit), slope 1/T_e_bp
                if t_e_breakpoint is not None:
                    full_ax.plot(v_to_plot_breakpoint, log_breakpoint_fit,
                             color='b', lw=2, label=fr'New method, $T_e$ = {t_e_breakpoint_value:.2f} eV')
                    full_ax.axvline(seg_lo, color='b', linestyle='--')
                    full_ax.axvline(seg_hi, color='b', linestyle='--')

                else:
                    full_ax.plot([], [], ' ', label=fr"Fit from new $T_e$ method not found")
                    seg_lo = np.nan
                    seg_hi = np.nan


                if (t_e_spline is not None or
                    t_e_breakpoint is not None):
                    min_bias = np.nanmin([sb_lo, seg_lo])
                    max_bias = np.nanmax([sb_hi, seg_hi])

                    zoom_v_mask = np.where((full_v <= max_bias) & (full_v >= min_bias))
                    zoom_v = full_v[zoom_v_mask]
                    zoom_i = full_i[zoom_v_mask]
                    zoomed_ax.scatter(zoom_v, zoom_i, color='k')

                    zoomed_ax.plot(x_th, m_th * x_th + b_th, color='r', lw=2, ls='--')
                    zoomed_ax.axvline(sb_lo, color='r', linestyle=':')
                    zoomed_ax.axvline(sb_hi, color='r', linestyle=':')

                    zoomed_ax.plot(v_to_plot_breakpoint, log_breakpoint_fit,
                                   color='b', lw=2)
                    zoomed_ax.axvline(seg_lo, color='b', linestyle='--')
                    zoomed_ax.axvline(seg_hi, color='b', linestyle='--')

                    zoomed_ax.set_xlabel("Voltage [V]")
                    zoomed_ax.set_ylabel(r"$\ln(I_e)$")
                    zoomed_ax.set_ylim(lower_y_lim, upper_y_lim)
                else:
                    zoomed_ax.text(0.5, 0.5, 'No valid fit\n(%s)',
                            ha='center', va='center', transform=zoomed_ax.transAxes)
                    zoomed_ax.set_title('Rejected for both cases')


                fig.suptitle(f"Langmuir sweep ln(I_e) vs V, Run: {run_identifier}\n"
                          f"Probe port and face: {port_face_string}, "
                          f"x: {loc_shot[0]}, y: {loc_shot[1]}, shot: {loc_shot[2]}\n\n"
                          f"Sweep at [ms]: {dt.to(u.ms).value * first_index}")
                full_ax.set_xlabel("Voltage [V]")
                full_ax.set_ylabel(r"$\ln(I_e)$")
                full_ax.set_ylim(lower_y_lim, upper_y_lim)

                full_ax.legend(loc='upper left',fontsize='x-small')
                fig.tight_layout(rect=[0, 0, 1, 0.99])

                if save_plots:
                    plot_specific_folder = ensure_directory(port_specific_folder + 'new_old_fits_log_IV_sweeps/')
                    fig_name = f'x_{loc_shot[0]}_shot_{loc_shot[2]}_time{dt.to(u.ms).value * first_index:.2f}.svg'
                    plt.savefig(plot_specific_folder + fig_name, bbox_inches='tight')
                    print('Plot saved to: ', plot_specific_folder + fig_name)
                plt.show()
                plt.close()

            if 'Plot individual log IV sweeps with new temperature fit (breakpoint)' in log_sweep_to_plot:
                plt.rcParams['figure.figsize'] = (12, 6)

                if ion_current is not None and hasattr(ion_current, 'to'):
                    i_e_data = sorted_current.to(u.A).value - ion_current.to(u.A).value
                else:
                    i_e_data = sorted_current.to(u.A).value
                v_all = sorted_bias.to(u.V).value
                log_mask = i_e_data > 0
                full_v = v_all[log_mask]
                full_i = np.log(i_e_data[log_mask])
                if v_f_bias is not None:
                    pre_floating_v_mask = np.where(full_v <= v_f_bias.value)[0]
                    pre_floating_i = full_i[pre_floating_v_mask]
                    lower_y_lim = np.median(pre_floating_i) - 1
                else:
                    lower_y_lim = np.min(np.log(i_e_data[log_mask])) + 2
                upper_y_lim = np.max(np.log(i_e_data[log_mask])) + .25

                plt.scatter(full_v, full_i,
                            color='k', label='Probe data')

                if t_e_breakpoint is not None:
                    plt.plot(v_to_plot_breakpoint, log_breakpoint_fit,
                             color='b', label=fr'New Te method, $T_e$ = {t_e_breakpoint_value:.2f} eV')
                    plt.axvline(seg_lo, color='b', linestyle='--')
                    plt.axvline(seg_hi, color='b', linestyle='--')
                else:
                    plt.plot([], [], ' ', label=fr"Fit from new $T_e$ method not found")

                plt.title(f"Langmuir sweep ln(I_e) vs V, Run: {run_identifier}\n"
                          f"Probe port and face: {port_face_string}, "
                          f"x: {loc_shot[0]}, y: {loc_shot[1]}, shot: {loc_shot[2]}\n\n"
                          f"Sweep at [ms]: {dt.to(u.ms).value * first_index}")
                plt.xlabel("Voltage [V]")
                plt.ylabel(r"$\ln(I_e)$")
                plt.legend(loc='upper left', fontsize='x-small')
                plt.ylim(lower_y_lim, upper_y_lim)

                if save_plots:
                    plot_specific_folder = ensure_directory(port_specific_folder + 'log_IV_sweeps_with_fit/')
                    fig_name = f'x_{loc_shot[0]}_shot_{loc_shot[2]}_time{dt.to(u.ms).value * first_index:.2f}.svg'
                    plt.savefig(plot_specific_folder + fig_name, bbox_inches='tight')
                    print('Plot saved to: ', plot_specific_folder + fig_name)

                plt.show()
                plt.close()









def plot_ion_isat_vs_time(dt,ramp_times,bias_to_plot,current_to_plot,exp_params_dict,port_face_string,loc_shot,save_plots,
                          filepath):
    """

    Parameters
    ----------
    dt - Float - Time step for the bias array
    ramp_times - Quantity array (ms) - Array of times where the IV sweep ends
    bias_to_plot - Quantity List (V) - Full array with bias values associated with a location shot combination.
    current_to_plot - Quantity Array (A) - Full array with current values associated with a location shot combination
    exp_params_dict - Dictionary of experiment parameters. (See experimental for more details)
    port_face_string - String - Indicates the port and the face for the probe that is currently being plotted.
    loc_shot - Three element array where idx 0 is the x-position, idx 1 is the y-position, idx 2 is the shot index
    save_plots - Boolean - Indicates whether to save the plot

    Returns
    -------

    """

    end_ramp_times_to_plot = []

    for h in range(len(ramp_times)-1):
        end_ramp_times_to_plot.append(ramp_times[h + 1])

    # Plot each of these subsets
    time_array = np.arange(len(bias_to_plot)) * dt.to(u.ms).value
    i_ion_sat = []
    plot_times=[]
    for i in range(len(end_ramp_times_to_plot)):
        # Obtain the time slice values for the sweep
        search_times = ((time_array >= ramp_times[i].to(
            u.ms).value) &
                        (time_array <= end_ramp_times_to_plot[i].to(u.ms).value))

        # Find the indices where the sweep starts and ends
        first_index, last_index = find_sweep_indices(time_array, end_ramp_times_to_plot[i],
                                                     search_times, bias_to_plot, dt)
        # Build the time series that will be plotted against the bias
        start_time = (first_index * dt.to(u.ms).value)
        end_time = end_ramp_times_to_plot[i].to(u.ms).value
        mask = ((time_array >= start_time) & (time_array <= end_time))
        plot_times.append(start_time)

        # Get the ion saturation current
        i_ion_sat.append(get_ion_isat_min(bias_to_plot[mask], current_to_plot[mask]).to(u.A).value)
    # print("plot times: ", plot_times)
    # print("i_ion_sat: ", i_ion_sat)
    plt.rcParams['figure.figsize'] = (8, 3)
    # plot_times is a scalar, i_ion_sat is a quantity. We need the value of I_ion_sat.
    plt.scatter(plot_times, i_ion_sat)
    plt.title(f"Run: {exp_params_dict['Exp name']}, {exp_params_dict['Run name']}\n"
              f"Probe port and face: {port_face_string}, "
              f"x: {loc_shot[0]}, y: {loc_shot[1]}, shot: {loc_shot[2]}\n\n"
              f"Ion Saturation Current [A], \n")
    plt.xlabel("Time [ms]")
    plt.ylabel("Ion Saturation Current [A]")
    plt.tight_layout()

    if save_plots:
        ensure_directory(filepath + "Ion_sat_current_vs_time/")
        ensure_directory(filepath + "Ion_sat_current_vs_time/" + f"Port_{port_face_string}/")
        plt.savefig(filepath + "/Ion_sat_current_vs_time/" + f"Port_{port_face_string}/" +
                    f"x_{loc_shot[0]},y_{loc_shot[1]},shot_{loc_shot[2]},probe_{port_face_string}.png")

    plt.show()
