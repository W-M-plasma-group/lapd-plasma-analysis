from pty import slave_open

import matplotlib.pyplot as plt
import numpy as np
from plasmapy.analysis.swept_langmuir import find_ion_saturation_current

from lapd_plasma_analysis.file_access import *
from lapd_plasma_analysis.experimental import get_exp_params

from lapd_plasma_analysis.langmuir.helper import *
from lapd_plasma_analysis.langmuir.configurations import *
from lapd_plasma_analysis.obtain_plots.Auxillary_functions import *
from lapd_plasma_analysis.langmuir.getIVsweep import get_sweep_voltage, get_sweep_current, get_shot_positions
from lapd_plasma_analysis.langmuir.characterization import make_characteristic_array, isolate_ramps
from lapd_plasma_analysis.langmuir.preview import preview_raw_sweep, preview_characteristics
from lapd_plasma_analysis.langmuir.diagnostics import (langmuir_diagnostics, detect_steady_state_times, get_pressure,
                                                       get_electron_ion_collision_frequencies)
from lapd_plasma_analysis.langmuir.neutrals import get_neutral_density
from lapd_plasma_analysis.langmuir.interferometry import interferometry_calibration
from lapd_plasma_analysis.langmuir.plots import get_title
from lapd_plasma_analysis.langmuir.metadata_for_dataset import get_supplemental_metadata


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
def obtain_data(hdf5_path, bias, current, positions, langmuir_config, exp_params_dict, save_plots):
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
    bias_to_plot = []
    current_to_plot = []
    loc_shot=[]
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

    return bias_to_plot,current_to_plot,loc_shot,filepath

def plot_iv_sweep(filepath,bias_to_plot,current_to_plot,port_face_string,IV_plots_choice,save_plots,
                  how_many_plots,ramp_times,exp_params_dict,loc_shot,dt):
    # This runs if the user selects to plot logarithmic or raw IV sweeps
    # Choose sweeps to plot based on the integer how many plots that the user inputs
    # Set figure siz
    plt.rcParams['figure.figsize'] = (8, 3)
    end_ramp_times_to_plot = []

    for h in range(how_many_plots):
        end_ramp_times_to_plot.append(ramp_times[(h + 1) * int(len(ramp_times) / (how_many_plots + 1))])

    # Plot each of these subsets
    time_array = np.arange(len(bias_to_plot)) * dt.to(u.ms).value
    for i in range(len(end_ramp_times_to_plot)):
        # Obtain the time slice values for the sweep
        search_times = ((time_array >= ramp_times[(i + 1) * int(len(ramp_times) / (how_many_plots + 1)) - 1].to(
            u.ms).value) &
                        (time_array <= end_ramp_times_to_plot[i].to(u.ms).value))

        # Find the indices where the sweep starts and ends
        first_index, last_index = find_sweep_indices(time_array, end_ramp_times_to_plot[i],
                                                     search_times, bias_to_plot, dt)
        # Build the time series that will be plotted against the bias
        start_time = (first_index * dt.to(u.ms).value)
        end_time = end_ramp_times_to_plot[i].to(u.ms).value
        mask = ((time_array >= start_time) & (time_array <= end_time))

        # Get the ion saturation current
        i_ion_sat = get_ion_isat(bias_to_plot[mask], current_to_plot[mask])
        if 2 in IV_plots_choice:
            if i_ion_sat is not None:
                plt.plot(bias_to_plot[mask], [i_ion_sat.to(u.A).value] * len(bias_to_plot[mask]), 'r',
                         label="Ion Isat")

            plt.scatter(bias_to_plot[mask], current_to_plot[mask])
            plt.title(f"Run: {exp_params_dict['Exp name']}, {exp_params_dict['Run name']}\n"
                      f"Probe port and face: {port_face_string}, "
                      f"x: {loc_shot[0]}, y: {loc_shot[1]}, shot: {loc_shot[2]}\n\n"
                      f"Voltage [V] vs current [A], \n"
                      f"Sweep at [ms]: {dt.to(u.ms).value * first_index}")
            plt.xlabel("Voltage [V]")
            plt.ylabel("Current [A]")
            plt.legend()
            plt.tight_layout()

            if save_plots:
                ensure_directory(filepath + "IV_sweep_curves/")
                ensure_directory(filepath + "IV_sweep_curves/" + f"Port_{port_face_string}/")
                plt.savefig(filepath + "/IV_sweep_curves/" + f"Port_{port_face_string}/" +
                            f"x_{loc_shot[0]},y_{loc_shot[1]},shot_{loc_shot[2]},"
                            f"time_{end_ramp_times_to_plot[i]},"f"probe_{port_face_string}_raw_data.png")

            plt.show()

        if 3 in IV_plots_choice:
            # Plot the natural log of the current shifted up by the ion saturation current to get rid of the negative
            # values
            adjusted_current = np.log(current_to_plot[mask].to(u.A).value +
                                      abs(min(current_to_plot[mask].to(u.A).value)))
            # May want to add the addition of *10^-9 or so after the absolute value - but there are significant outliers

            plt.scatter(bias_to_plot[mask], adjusted_current)
            plt.title(f"Run: {exp_params_dict['Exp name']}, {exp_params_dict['Run name']}\n"
                      f"Probe port and face: {port_face_string}, "
                      f"x: {loc_shot[0]}, y: {loc_shot[1]}, shot: {loc_shot[2]}\n\n"
                      f"Voltage [V] vs current [A], \n"
                      f"Sweep at [ms]: {dt.to(u.ms).value * first_index}")
            plt.xlabel("Voltage [V]")
            plt.ylabel("ln(Current)")

            plt.tight_layout()

            if save_plots:
                ensure_directory(filepath + "IV_sweep_curves/")
                ensure_directory(filepath + "IV_sweep_curves/" + f"Port_{port_face_string}/")
                plt.savefig(filepath + "/IV_sweep_curves/" + f"Port_{port_face_string}/" +
                            f"x_{loc_shot[0]},y_{loc_shot[1]},shot_{loc_shot[2]},"f"time_{end_ramp_times_to_plot[i]},"
                            f"probe_{port_face_string}_log_plot.png")

            plt.show()

        # To test IV sweep start and end time accuracy

        # If we get none of the Ion saturation current show the whole thing
        if i_ion_sat is None:
            plt.scatter(time_array[search_times], bias_to_plot[search_times])
            plt.title(f"bias [V] vs time [ms] for the selected run (Testing)")
            plt.xlabel("time[ms]")
            plt.ylabel("Bias [V]")
            plt.tight_layout()
            plt.show()


def plot_ion_isat_vs_time(dt,ramp_times,bias_to_plot,current_to_plot,exp_params_dict,port_face_string,loc_shot,save_plots):
    """

    Parameters
    ----------
    dt
    ramp_times
    bias_to_plot
    current_to_plot
    exp_params_dict
    port_face_string
    loc_shot
    save_plots

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
        i_ion_sat.append(get_ion_isat(bias_to_plot[mask], current_to_plot[mask]).to(u.A).value)
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
    plt.show()




