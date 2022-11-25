import warnings

import astropy.units as u
import numpy as np
from scipy.signal import find_peaks
from plasmapy.diagnostics.langmuir import Characteristic

import sys
from tqdm import tqdm, trange


def characterize_sweep_array(bias, current, margin, sample_sec):
    r"""
    Function that processes bias and current data into a DataArray of distinct Characteristics.
    Takes in bias and current arrays, smooths them, divides them into separate ramp sections, 
    and creates a Characteristic object for each ramp at each unique x,y position.

    Parameters
    ----------
    :param bias: array, units of voltage
    :param current: array, units of current
    :param margin: int, positive
    :param sample_sec: float, units of time
    :return: 2D array of Characteristic objects by shot number and plateau number
    """

    validate_sweep_units(bias, current)
    bias, current = smooth(bias, current, margin=margin)
    ramp_bounds = isolate_plateaus(bias)
    # Need to trim off edges of sweep distorted by smoothing function (contaminated by quench time!); margin/2 each side
    ramp_bounds + [margin // 2, -margin // 2]

    ramp_times = ramp_bounds[:, 1] * sample_sec.to(u.ms)
    # NOTE: MATLAB code stores peak voltage time (end of plateaus), then only uses plateau times for very first position
    # This uses the time of the peak voltage for the average of all shots ("top of the average ramp")

    return characteristic_array(bias, current, ramp_bounds), ramp_times


def smooth(*arrays, margin=0):
    r"""
    Utility function to smooth ndarray using moving average along last dimension.

    Parameters
    ----------
    :param arrays: ndarray(s) to be smoothed
    :param margin: width of moving average window
    :return: smoothed ndarray(s) with last dimension length decreased by margin
    """

    if margin < 0:
        raise ValueError("Cannot smooth over negative number", margin, "of points")
    if margin == 0:
        return arrays

    smoothed_arrays = []
    for array in arrays:
        if array.shape[-1] <= margin:
            raise ValueError(f"Final dimension of size {array.shape[-1]} is too short to take {margin}-point mean over")
        # Find cumulative mean of each consecutive block of (margin + 1) elements per row
        array_sum = np.cumsum(np.insert(array, 0, 0, axis=-1), axis=-1, dtype=np.float64)
        smoothed_array = (array_sum[..., margin:] - array_sum[..., :-margin]) / margin
        smoothed_arrays.append(smoothed_array.astype(float))

    return smoothed_arrays


def isolate_plateaus(bias):
    r"""
    Find indices corresponding to the beginning and end of every bias ramp.

    Parameters
    ----------
    :param bias:
    :return: num_plateaus-by-2 array; start indices in first column, end indices in second
    """

    # Assume strictly that all bias ramps start and end at the same time after the start of the shot
    bias_avg = np.mean(bias, axis=[0, 1])  # mean across all positions and shots, preserving time

    # Report on how dissimilar the vsweep biases are and if they can be averaged together safely

    # Initial fit to guess number of peaks
    min_plateau_width = 500  # change as necessary
    guess_num_plateaus = len(find_peaks(bias_avg, height=0, distance=min_plateau_width)[0])
    guess_plateau_spacing = bias.shape[-1] // guess_num_plateaus

    # Second fit to find maximum bias frames
    peak_frames, peak_properties = find_peaks(bias_avg, height=0, distance=guess_plateau_spacing // 2,
                                              width=min_plateau_width, rel_height=0.97)  # 0.97 may be hardcoded

    # return np.stack((peak_properties['left_ips'].astype(int) + margin // 2, peak_frames - margin // 2), axis=-1)
    return np.stack((peak_properties['left_ips'].astype(int), peak_frames), axis=-1)


def validate_sweep_units(bias, current):
    try:
        assert (bias.unit == u.V)
    except (AttributeError, AssertionError):
        warnings.warn("Input bias array does not have units of Volts. Ensure that bias values are in real units.")
    try:
        assert (current.unit == u.A)
    except (AttributeError, AssertionError):
        warnings.warn("Input bias array does not have units of Amps. Ensure that current values are in real units.")


def characteristic_array(bias, current, plateau_ranges):
    # 4D: langmuir_probe by unique_position by shot_at_position by plateau_number

    currents = current  # "currents" has "probe" dimension in front; may have size 1
    num_pos = bias.shape[0]
    # num_plats = plateau_ranges.shape[0]

    plateau_slices = np.array([slice(plateau[0], plateau[1]) for plateau in plateau_ranges])

    # Mixed arbitrary/indexed list comprehension
    print(f"Creating characteristics ({currents.shape[0]} probes to analyze)...")
    warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress FutureWarnings to not break loading bar
    print("\t(plasmapy.langmuir.diagnostics pending deprecation FutureWarning suppressed)")
    return np.concatenate([np.array([[[Characteristic(bias[pos, shot, plateau], current[pos, shot, plateau])
                                       for plateau in plateau_slices]
                                      for shot in range(bias.shape[0])]
                                     for pos in trange(num_pos, unit="position", file=sys.stdout)])[np.newaxis, ...]
                           for current in currents])
    # TODO: use nested tqdm instead of outer tqdm only?
