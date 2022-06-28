import warnings

import astropy.units as u
import numpy as np
import xarray as xr
from scipy.signal import find_peaks, butter, sosfilt
from plasmapy.diagnostics.langmuir import Characteristic

import sys
from tqdm import tqdm


def characterize_sweep_array(unadjusted_bias, unadjusted_current, x_round, y_round, margin, sample_sec):
    r"""
    Function that processes bias and current data into a DataArray of distinct Characteristics.
    Takes in bias and current arrays, smooths them, divides them into separate ramp sections, 
    and creates a Characteristic object for each ramp at each unique x,y position.

    Parameters
    ----------
    :param y_round:
    :param x_round:
    :param unadjusted_bias: array
    :param unadjusted_current: array
    :param margin: int, positive
    :param sample_sec: float, units of time
    :return: 3D xarray DataArray of Characteristic objects
    """

    dc_current_offset = np.mean(unadjusted_current[..., :1000], axis=-1, keepdims=True)
    bias, current = smooth_current_array(unadjusted_bias, unadjusted_current - dc_current_offset, margin=margin)
    ramp_bounds = isolate_plateaus(bias, margin=margin)
    time_array = get_time_array(ramp_bounds, sample_sec)

    # debug
    """
    # pprint(swept_probe_analysis(smooth_plateau, probe_area, 'He-4+', bimaxwellian=True, visualize=True, plot_EEDF=True))
    plt.plot(adjusted_bias[sample_indices[:2]], 'b-',
             plateau_ranges[sample_indices[0], sample_indices[1], :, 0], adjusted_bias[sample_indices[0], sample_indices[1], plateau_ranges[sample_indices[0], sample_indices[1], :, 0]], 'go',
             plateau_ranges[sample_indices[0], sample_indices[1], :, 1], adjusted_bias[sample_indices[0], sample_indices[1], plateau_ranges[sample_indices[0], sample_indices[1], :, 1]], 'yo')
    plt.show()
    # """

    characteristic_array = get_characteristic_array(bias, current, ramp_bounds)
    characteristic_xarray = to_characteristic_xarray(characteristic_array, time_array, x_round, y_round)
    return characteristic_xarray


def smooth_current_array(bias, current, margin):
    # Distorts shape of current, especially at ends of plateaus, but much faster than smoothing plateaus individually

    if margin < 0:
        raise ValueError("Cannot smooth over negative number", margin, "of points")
    if margin == 0:
        # warnings.warn("Zero-point smoothing is redundant")
        return bias, current
    if current.shape[-1] <= margin:
        raise ValueError("Last dimension length", current.shape[-1], "is too short to take", margin, "-point mean over")

    current_sum = np.cumsum(np.insert(current, 0, 0, axis=-1), axis=-1, dtype=np.float64)
    bias_sum = np.cumsum(np.insert(bias, 0, 0, axis=-1), axis=-1, dtype=np.float64)

    # Find cumulative mean of each consecutive block of (margin + 1) elements per row
    smooth_current_full = (current_sum[..., margin:] - current_sum[..., :-margin]) / margin
    # Smooth bias in the same way to get shorter bias array
    smooth_bias_full = (bias_sum[..., margin:] - bias_sum[..., :-margin]) / margin

    return smooth_bias_full.astype(float), smooth_current_full.astype(float)


def isolate_plateaus(bias, margin=0):
    r"""

    :param bias:
    :param margin:
    :return:
    """

    # Assume strictly that all plateaus start and end at the same time after the start of the shot as in any other shot
    bias_avg = np.mean(bias, axis=(0, 1))
    bias_std = np.std(bias, axis=(0, 1))
    # Report on how dissimilar the vsweep biases are and if they can be averaged together safely
    print("Langmuir average error between shots:", np.mean(bias_std))
    print("Langmuir variation in bias over time:", np.std(bias))

    # Low-pass filter?

    # Initial fit to guess number of peaks
    min_plateau_width = 500  # change as necessary
    guess_num_plateaus = len(find_peaks(bias_avg, height=0, distance=min_plateau_width)[0])
    guess_plateau_spacing = bias.shape[-1] // guess_num_plateaus

    # Second fit to find maximum bias frames
    peak_frames, peak_properties = find_peaks(bias_avg, height=0, distance=guess_plateau_spacing // 2,
                                              width=min_plateau_width, rel_height=0.97)  # 0.97 may be hardcoded

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


def get_time_array(plateau_ranges, sample_sec=(100 / 16 * 1e6) ** (-1) * u.s):
    # NOTE: MATLAB code stores peak voltage time (end of plateaus), then only uses plateau times for very first position
    # This uses the time of the peak voltage for the average of all shots ("top of the average ramp")
    return plateau_ranges[1] * sample_sec


def get_characteristic_array(bias, current, plateau_ranges):

    num_plateaus = plateau_ranges.shape[-1]
    characteristic_array = np.empty(bias.shape[:2] + (num_plateaus,), dtype=object)  # x, y, plateau

    print("Creating characteristic array...")  # (May take up to 60 seconds)
    warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress FutureWarnings to not break loading bar
    print("    Note: plasmapy.langmuir.diagnostics pending deprecation FutureWarning suppressed")
    num_pos = bias.shape[0] * bias.shape[1] * num_plateaus
    with tqdm(total=num_pos, unit="characteristic", file=sys.stdout) as pbar:
        for i in range(bias.shape[0]):
            for j in range(bias.shape[1]):
                for p in range(num_plateaus):
                    characteristic_array[i, j, p] = create_ranged_characteristic(
                        bias[i, j], current[i, j], start=plateau_ranges[0, p], end=plateau_ranges[1, p])
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
