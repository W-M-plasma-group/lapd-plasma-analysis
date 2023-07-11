import warnings

import astropy.units as u
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from plasmapy.diagnostics.langmuir import Characteristic

import sys
from tqdm import tqdm, trange

from helper import *


def characterize_sweep_array(bias, current, margin, sample_sec, position_bounds=None):
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
    print("about to smooth")
    bias, current = smooth(bias, current, margin=margin)
    print("about to isolate plateaus")
    ramp_bounds = isolate_plateaus(bias)
    # Need to trim off sweep edges bc contaminated by averaging with quench current/voltage; edge = margin/2 each side
    ramp_bounds + [margin // 2, -margin // 2]

    ramp_times = ramp_bounds[:, 1] * sample_sec.to(u.ms)
    # NOTE: MATLAB code stores peak voltage time (end of plateaus), then only uses plateau times for very first position
    # This uses the time of the peak voltage for the average of all shots ("top of the average ramp")

    print("About to characteristic_array")
    return get_characteristic_array(bias, current, ramp_bounds, position_bounds=position_bounds), ramp_times


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
        unit = unit_safe(array)
        # Find cumulative mean of each consecutive block of (margin + 1) elements per row
        # array_sum = np.cumsum(np.insert(array, 0, 0, axis=-1), axis=-1, dtype=np.float64)
        # smoothed_array = (array_sum[..., margin:] - array_sum[..., :-margin]) / margin
        # smoothed_arrays.append(smoothed_array.astype(float))
        smoothed_arrays.append(uniform_filter1d(array, size=margin, mode='nearest', axis=-1) * unit)

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
    bias_avg = np.mean(bias, axis=(0, 1))  # mean across all positions and shots, preserving time

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


def get_characteristic_array(bias, current, plateau_ranges, position_bounds=None):
    # 4D: langmuir_probe by unique_position by shot_at_position by plateau_number
    # TODO position_bounds, if not None, gives min and max position index to study

    currents = current  # "currents" has "probe" dimension in front; may have size 1
    num_pos = bias.shape[0]
    # num_plats = plateau_ranges.shape[0]
    if position_bounds is not None:
        num_pos = min(num_pos, max(position_bounds) - min(position_bounds))
    else:
        position_bounds = (0, num_pos)

    plateau_slices = np.array([slice(plateau[0], plateau[1]) for plateau in plateau_ranges])

    # Mixed arbitrary/indexed list comprehension
    print(f"Creating characteristics ({currents.shape[0]} probes to analyze)...")
    warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress FutureWarnings to not break loading bar
    print("\t(plasmapy.langmuir.diagnostics pending deprecation FutureWarning suppressed)")
    characteristic_arrays = [np.zeros((num_pos, bias.shape[1], len(plateau_slices)), dtype=object) for _ in currents]
    chars_per_probe = num_pos * bias.shape[1] * len(plateau_slices)
    for p in range(len(currents)):
        with tqdm(total=chars_per_probe, unit="characteristic", file=sys.stdout) as pbar:
            for l in range(num_pos):  # noqa
                for s in range(bias.shape[1]):
                    for r in range(len(plateau_slices)):
                        r_slice = plateau_slices[r]
                        l_adj = l + min(position_bounds)
                        chara = Characteristic(bias[l_adj, s, r_slice], currents[p][l_adj, s, r_slice])
                        characteristic_arrays[p][l, s, r] = chara
                        pbar.update(1)

    return np.concatenate([characteristic_array[np.newaxis, ...] for characteristic_array in characteristic_arrays])



