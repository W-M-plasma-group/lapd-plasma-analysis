import warnings

import astropy.units as u
import numpy as np
from scipy.signal import find_peaks
from plasmapy.diagnostics.langmuir import Characteristic

import sys
from tqdm import tqdm, trange


def characterize_sweep_array(unsmooth_bias, unsmooth_current, margin, sample_sec):
    r"""
    Function that processes bias and current data into a DataArray of distinct Characteristics.
    Takes in bias and current arrays, smooths them, divides them into separate ramp sections, 
    and creates a Characteristic object for each ramp at each unique x,y position.

    Parameters
    ----------
    :param unsmooth_bias: array, units of voltage
    :param unsmooth_current: array, units of current
    :param margin: int, positive
    :param sample_sec: float, units of time
    :return: 2D array of Characteristic objects by shot number and plateau number
    """

    validate_sweep_units(unsmooth_bias, unsmooth_current)
    bias, current = smooth_characteristic(unsmooth_bias, unsmooth_current, margin=margin)
    ramp_bounds = isolate_plateaus(bias, margin=margin)

    ramp_times = ramp_bounds[:, 1] * sample_sec.to(u.ms)
    # NOTE: MATLAB code stores peak voltage time (end of plateaus), then only uses plateau times for very first position
    # This uses the time of the peak voltage for the average of all shots ("top of the average ramp")

    return characteristic_array(bias, current, ramp_bounds), ramp_times


def smooth_characteristic(bias, current, margin):
    r"""
    Simple moving-average smoothing function for bias and current.

    Parameters
    ----------
    :param bias: ndarray
    :param current: ndarray
    :param margin: int, window length for moving average
    :return: smoothed bias, smoothed current
    """

    if margin < 0:
        raise ValueError("Cannot smooth over negative number", margin, "of points")
    if margin == 0:
        return bias, current
    if current.shape[-1] <= margin:
        raise ValueError("Last dimension length", current.shape[-1], "is too short to take", margin, "-point mean over")

    return smooth_array(bias, margin), smooth_array(current, margin)


def smooth_array(array, margin):
    r"""
    Utility function to smooth ndarray using moving average along last dimension.

    Parameters
    ----------
    :param array: ndarray to be smoothed
    :param margin: width of moving average window
    :return: smoothed ndarray with last dimension length decreased by margin
    """

    # Find cumulative mean of each consecutive block of (margin + 1) elements per row
    array_sum = np.cumsum(np.insert(array, 0, 0, axis=-1), axis=-1, dtype=np.float64)
    smoothed_array = (array_sum[..., margin:] - array_sum[..., :-margin]) / margin

    return smoothed_array.astype(float)


def isolate_plateaus(bias, margin=0):
    r"""
    Find indices corresponding to the beginning and end of every bias ramp.

    Parameters
    ----------
    :param bias:
    :param margin:
    :return: num_plateaus-by-2 array; start indices in first column, end indices in second
    """

    # Assume strictly that all plateaus start and end at the same time after the start of the shot as in any other shot
    axes_to_average = tuple(np.arange(bias.ndim)[:-1])
    bias_avg = np.mean(bias, axis=axes_to_average)  # mean across all positions and shots, preserving time

    # Report on how dissimilar the vsweep biases are and if they can be averaged together safely

    # Initial fit to guess number of peaks
    min_plateau_width = 500  # change as necessary
    guess_num_plateaus = len(find_peaks(bias_avg, height=0, distance=min_plateau_width)[0])
    guess_plateau_spacing = bias.shape[-1] // guess_num_plateaus

    # Second fit to find maximum bias frames
    peak_frames, peak_properties = find_peaks(bias_avg, height=0, distance=guess_plateau_spacing // 2,
                                              width=min_plateau_width, rel_height=0.97)  # 0.97 may be hardcoded

    return np.stack((peak_properties['left_ips'].astype(int) + margin // 2, peak_frames - margin // 2), axis=-1)


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
    # 3D: unique_position by shot by plateau_num

    currents = current  # "currents" has "probe" dimension in front; may have size 1
    num_pos = bias.shape[0]
    num_shot = bias.shape[1]

    plateau_slices = np.array([slice(plateau[0], plateau[1]) for plateau in plateau_ranges])

    # Mixed arbitrary/indexed list comprehension
    print(f"Creating characteristics ({currents.shape[0]} probes to analyze)...")
    warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress FutureWarnings to not break loading bar
    print("\t(plasmapy.langmuir.diagnostics pending deprecation FutureWarning suppressed)")
    return np.concatenate([np.array([[[Characteristic(bias[pos, shot, plateau_slice], current[pos, shot, plateau_slice])
                                       for plateau_slice in plateau_slices]
                                      for shot in range(num_shot)]
                                     for pos in range(num_pos)])[np.newaxis, ...]
                           for current in currents])
    # for pos in trange(num_pos, unit="position", file=sys.stdout)])[np.newaxis, ...]
