from pty import slave_open

import matplotlib.pyplot as plt
import numpy as np
from pycparser.c_ast import Return

from lapd_plasma_analysis.file_access import *
from lapd_plasma_analysis.experimental import get_exp_params

from lapd_plasma_analysis.langmuir.helper import *
from lapd_plasma_analysis.langmuir.configurations import *
from lapd_plasma_analysis.langmuir.getIVsweep import get_sweep_voltage, get_sweep_current, get_shot_positions
from lapd_plasma_analysis.langmuir.characterization import make_characteristic_array, isolate_ramps
from lapd_plasma_analysis.langmuir.preview import preview_raw_sweep, preview_characteristics
from lapd_plasma_analysis.langmuir.diagnostics import (langmuir_diagnostics, detect_steady_state_times, get_pressure,
                                                       get_electron_ion_collision_frequencies)
from lapd_plasma_analysis.langmuir.neutrals import get_neutral_density
from lapd_plasma_analysis.langmuir.interferometry import interferometry_calibration
from lapd_plasma_analysis.langmuir.plots import get_title
from lapd_plasma_analysis.langmuir.metadata_for_dataset import get_supplemental_metadata


def find_sweep_indices(time_array,end_time,search_times,bias,dt):
    '''

    Parameters
    ----------
    time_array - A time sequence of all the time values corresponding to the bias measurements.
    end_time - u.Quantity The time value in ms where the IV sweep ends
    search_times - A mask of the time array starting at the end of the last sweep and ending at the end of the
    sweep of interest.
    bias - 1D array giving the bias values over time for a selected position-shot combination
    dt - u.Quantity timestep in ms

    Returns
    index - int indicating the first index in the time array of the sweep of interest
    last_index - int indicating the last index in the time array of the sweep of interest
    -------

    '''
    # Find the index of the peak of the IV sweep
    last_index = np.searchsorted(time_array, end_time, "right") - 1

    # Make the assumption that the Langmuir probe sweeps at an approximately constant rate and calculate that slope
    check_bias = int(.00008 * len(bias))
    slope = (bias[last_index] - bias[last_index - check_bias]) / (check_bias * dt)
    search_slope = 0

    # Obtain the time of the previous sweep
    first_index = np.argmax(search_times)
    j = 0
    index = first_index

    search_indices = np.where(search_times)[0]

    # There is a dip immediately after the previous sweep that has a large slope as it comes back up
    # however it doesn't indicate the start of the next sweep so the upper bound tries to mitigate that

    # Tolerance for defining "start" of a sweep
    slope_tolerance = 0.2
    # How many steps between checks of the slope
    skip = 10

    while abs((search_slope-slope)/slope) > slope_tolerance and (index + skip) < len(bias):
        index = first_index + j
        search_slope = (bias[index + skip] - bias[index]) / (skip * dt)
        j += skip

    # Define index as the start of the voltage sweep
    index = first_index + j - skip

    # Check to make sure we have an initial bias less than 0 because otherwise we are not getting a full sweep
    if bias[index] > 0:
        # How many check biases off the end where we are going to start looking
        start = 2
        while bias[index] > 0 and start * check_bias < int(.2*len(bias)):
            slope = (bias[last_index-start*check_bias] - bias[last_index -(start-1) * check_bias]) / (check_bias * dt)
            search_slope = 0
            first_index = np.argmax(search_times)
            slope_tolerance = .3
            h = 0
            n_index = first_index
            # How many steps between checks of the slope
            while abs((search_slope - slope) / slope) > slope_tolerance and (n_index + skip) < len(bias):
                n_index = first_index + h
                search_slope = (bias[n_index + skip] - bias[n_index]) / (skip * dt)
                h += skip
            index = first_index + h - skip
            start += 1


    # Now we want to create a bias threshold so that we don't get a
    # tail which is appearing in some plots for some reason
    voltage_range = bias[last_index] - bias[index]
    voltage_tolerance = voltage_range * .1
    for i in range(index,last_index):
        if bias[i] > bias[index]+voltage_tolerance:
            break
        else:
            i = index



    return i,last_index

def get_ion_isat(bias,current):
    '''

    Parameters
    ----------
    bias - 1D array of masked bias values corresponding to a single voltage sweep
    current - 1D array of masked current values corresponding to a single voltage sweep

    Returns
    -------

    '''

    # Sort the current from being arranged by lowest time to being arranged by lowest bias
    sorted_current = current[np.argsort(bias)]
    sorted_bias = bias[np.argsort(bias)]
    # Get the mean of the first 5 elements of the sorted current array
    avg_current = np.mean(sorted_current[:5])
    tolerance = .3
    step = 200
    i = step

    # Look for when the percent difference between the test average (mean) and the full current average is less than 5%
    while i+step < len(bias):
        # print(i)
        mean = np.mean(sorted_current[i:i+step])
        if abs((mean - avg_current)/avg_current) >= tolerance:
            i_ion_sat = avg_current
            return i_ion_sat
        else:
            avg_current = np.mean(sorted_current[:i+step])
            i += step

    print("Failed to find an ion Isat")
    return None


