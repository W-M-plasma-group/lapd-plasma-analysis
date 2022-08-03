import warnings

import astropy.units as u
import numpy as np
import xarray as xr
from scipy.signal import find_peaks
from plasmapy.diagnostics.langmuir import Characteristic

import sys
from tqdm import tqdm, trange


def characterize_sweep_array(unadjusted_bias, unadjusted_current, margin, sample_sec):
    r"""
    Function that processes bias and current data into a DataArray of distinct Characteristics.
    Takes in bias and current arrays, smooths them, divides them into separate ramp sections, 
    and creates a Characteristic object for each ramp at each unique x,y position.

    Parameters
    ----------
    :param y:
    :param x:
    :param unadjusted_bias: array
    :param unadjusted_current: array
    :param margin: int, positive
    :param sample_sec: float, units of time
    :return: 3D xarray DataArray of Characteristic objects
    """
    # TODO fix docstring

    dc_current_offset = np.mean(unadjusted_current[..., -1000:], axis=-1, keepdims=True)
    bias, current = smooth_characteristic(unadjusted_bias, unadjusted_current - dc_current_offset, margin=margin)
    ramp_bounds = isolate_plateaus(bias, margin=margin)
    ramp_times = get_time_array(ramp_bounds, sample_sec)

    # characteristic_array = get_characteristic_array(bias, current, ramp_bounds, len(x), len(y))
    # characteristic_xarray = to_characteristic_xarray(characteristic_array, time_array, x, y)

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

    :param bias:
    :param margin:
    :return: num_plateaus-by-2 array; start indices in first column, end indices in second
    """

    # Assume strictly that all plateaus start and end at the same time after the start of the shot as in any other shot
    bias_avg = np.mean(bias, axis=(0, 1))

    # Report on how dissimilar the vsweep biases are and if they can be averaged together safely

    # Initial fit to guess number of peaks
    min_plateau_width = 500  # change as necessary
    guess_num_plateaus = len(find_peaks(bias_avg, height=0, distance=min_plateau_width)[0])
    guess_plateau_spacing = bias.shape[-1] // guess_num_plateaus

    # Second fit to find maximum bias frames
    peak_frames, peak_properties = find_peaks(bias_avg, height=0, distance=guess_plateau_spacing // 2,
                                              width=min_plateau_width, rel_height=0.97)  # 0.97 may be hardcoded

    print(len(peak_frames), "plateaus detected")
    return np.stack((peak_properties['left_ips'].astype(int) + margin // 2, peak_frames - margin // 2))


def create_ranged_characteristic(bias, current, start, end):
    # Takes in a one-dimensional bias and current list; returns a Characteristic object for specified index range

    # Error checks for input shapes and indices
    if len(bias.shape) > 1:
        raise ValueError("Multidimensional characteristic creation is no longer supported. Pass 1D sweep arrays.")
    if bias.shape != current.shape:
        raise ValueError("Bias and current must be of the same dimensions and shape")
    if start < 0:
        raise ValueError("Start index must be non-negative")
    if end > len(bias):
        raise ValueError("End index", end, "out of range of bias and current arrays of length", len(bias))

    # Check units on input arrays
    try:
        assert (bias.unit == u.V)
    except (AttributeError, AssertionError):
        warnings.warn("Input bias array does not have units of Volts. Ensure that bias values are in real units.")
    try:
        assert (current.unit == u.A)
    except (AttributeError, AssertionError):
        warnings.warn("Input bias array does not have units of Amps. Ensure that current values are in real units.")

    return Characteristic(bias[start:end], current[start:end])


def get_time_array(plateau_ranges, sample_sec):
    # NOTE: MATLAB code stores peak voltage time (end of plateaus), then only uses plateau times for very first position
    # This uses the time of the peak voltage for the average of all shots ("top of the average ramp")
    return plateau_ranges[1] * sample_sec


def characteristic_array(bias, current, plateau_ranges):  # , positions
    # 2D: shot_num by plateau

    num_shots = bias.shape[0]
    # num_plats = plateau_ranges.shape[0]

    # MOVE THIS LINE TO PLATEAU RANGES?
    plateau_slices = np.array([slice(plateau[0], plateau[1]) for plateau in plateau_ranges])

    # Mixed arbitrary/indexed list comprehension
    return np.array([[Characteristic(bias[shot, plateau], current[shot, plateau])
                      for plateau in plateau_slices]
                     for shot in trange(num_shots)])
    # TODO: tqdm on every level? also, tqdm(range(_))


"""
def get_characteristic_array(bias, current, plateau_ranges, num_x, num_y):

    num_plateaus = plateau_ranges.shape[-1]
    num_shots = len(bias)

    characteristic_array = np.empty((num_shots, num_plateaus), dtype=object)  # dims: shot, plateau
    print("Forming characteristics...")
    with tqdm(total=num_shots * num_plateaus, unit="characteristic", file=sys.stdout) as pbar:
        for s in range(num_shots):
            for p in range(num_plateaus):
                characteristic_array[]


    num_plateaus = plateau_ranges.shape[-1]
    characteristic_array = np.empty((num_x, num_y, num_plateaus), dtype=object)  # dims: x, y, plateau

    print("Creating characteristics...")  # (May take up to 60 seconds)
    # warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress FutureWarnings to not break loading bar
    # print("    Note: plasmapy.langmuir.diagnostics pending deprecation FutureWarning suppressed")
    num_pos = num_x * num_y * num_plateaus
    # TODO OR just have num_shots * num_plateaus, keep positions as parameters, DONT MAKE characteristic_array x*y!
    with tqdm(total=num_pos, unit="characteristic", file=sys.stdout) as pbar:
        for i in range(num_x):
            for j in range(num_y):
                for p in range(num_plateaus):
                    characteristic_array[i, j, p] = create_ranged_characteristic(
                        bias[i * num_x * num_y + j * num_plateaus],
                        current[i * num_y ], start=plateau_ranges[0, p], end=plateau_ranges[1, p])
                    pbar.update(1)
    return characteristic_array


def to_characteristic_xarray(characteristic_array, time_array, x, y):
    # Use hard-coded and calculated inputs to add coordinate and variable information to characteristic array

    time_array_ms = time_array.to(u.ms).value
    characteristic_xarray = xr.DataArray(characteristic_array, dims=['x', 'y', 'time'],
                                         coords=(('x', x, {'units': str(u.cm)}),
                                                 ('y', y, {'units': str(u.cm)}),
                                                 ('time', time_array_ms, {'units': str(u.ms)})))
    characteristic_xarray = characteristic_xarray.assign_coords(
        {'plateau': ('time', np.arange(characteristic_array.shape[2]) + 1)})
    # Average the plateau time coordinate for all x,y positions to make 1D coordinate, keeping plateau dimension

    return characteristic_xarray
"""