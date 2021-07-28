import warnings
import numpy as np
import xarray as xr
import astropy.units as u
from plasmapy.diagnostics.langmuir import Characteristic


def characterize_sweep_array(unadjusted_bias, unadjusted_current, margin, sample_sec):
    r"""
    Function that processes bias and current data into a DataArray of distinct Characteristics.
    Takes in bias and current arrays, smooths them, divides them into separate ramp sections, 
    and creates a Characteristic object for each ramp at each unique x,y position.

    Parameters
    ----------
    :param unadjusted_bias: array
    :param unadjusted_current: array
    :param margin: int, positive
    :param sample_sec: float, units of time
    :return: 3D xarray DataArray of Characteristic objects
    """

    bias, current = smooth_current_array(unadjusted_bias, unadjusted_current, margin=margin)
    plateau_ranges = isolate_plateaus(bias, margin=margin)
    time_array = get_time_array(plateau_ranges, sample_sec)

    # debug
    """
    # pprint(swept_probe_analysis(smooth_plateau, probe_area, 'He-4+', bimaxwellian=True, visualize=True, plot_EEDF=True))
    plt.plot(adjusted_bias[sample_indices[:2]], 'b-',
             plateau_ranges[sample_indices[0], sample_indices[1], :, 0], adjusted_bias[sample_indices[0], sample_indices[1], plateau_ranges[sample_indices[0], sample_indices[1], :, 0]], 'go',
             plateau_ranges[sample_indices[0], sample_indices[1], :, 1], adjusted_bias[sample_indices[0], sample_indices[1], plateau_ranges[sample_indices[0], sample_indices[1], :, 1]], 'yo')
    plt.show()
    # """

    characteristic_array = get_characteristic_array(bias, current, plateau_ranges)
    characteristic_xarray = to_characteristic_xarray(characteristic_array, time_array)
    # (debug) print(characteristic_xarray)
    return characteristic_xarray


def smooth_current_array(bias, current, margin):
    # This still distorts the shape of the current, especially at the ends of each plateau, but is much faster

    if margin < 0:
        raise ValueError("Cannot smooth over negative number", margin, "of points")
    if margin == 0:
        warnings.warn("Zero-point smoothing is redundant")
    if current.shape[-1] <= margin:
        raise ValueError("Last dimension length", current.shape[-1], "is too short to take", margin, "-point mean over")

    current_sum = np.cumsum(np.insert(current, 0, 0, axis=-1), axis=-1)
    smooth_current_full = (current_sum[..., margin:] - current_sum[..., :-margin]) / margin

    adjusted_bias = bias[..., (margin - 1) // 2:-(margin - 1) // 2]  # Aligns bias with new, shorter current array

    return adjusted_bias, smooth_current_full


def isolate_plateaus(bias, margin=0):

    r"""
    Function to identify start and stop frames of every ramp section within each plateau.
    Returns array containing frame indices for each shot.

    Parameters
    ----------
    :param bias: array
    :param margin: int, optional
    :return: array of ramp start and stop indices
    """

    quench_slope = -1  # "Threshold for voltage quench slope": MATLAB code comment
    quench_diff = 10  # Threshold for separating distinct voltage quench frames

    # Not in MATLAB code
    rise_slope = 0.4  # Threshold for increases in slope
    rise_diff = 100  # Threshold for separating distinct voltage ramp frames

    # The bias has three types of regions: constant low, increase at constant rate ("ramp"), and rapid decrease
    #    down to minimum value ("quench"). The ramp region is where useful Isweep-Vsweep data points are collected.
    # Since the bias changes almost linearly within each of these three regions, the slope (gradient) of the bias
    #    (normalized to be less than 1) can be used to divide the frames up into regions.
    # Note: Selecting ramps out of plateaus (the constant low bias and following ramp region) is not in the original
    #    MATLAB code, but is determined to be necessary for PlasmaPy diagnostics functions to work correctly.

    bias_gradient = np.gradient(bias, axis=-1)
    normalized_bias_gradient = bias_gradient / np.amax(bias_gradient, axis=-1, keepdims=True)

    # Previous efforts to create quench_frames solely using array methods (fastest! but harder) lie here
    # quench_frames = np.array((normalized_bias_gradient < quench_slope).nonzero())
    # quench_frames_by_position = frame_array[..., normalized_bias_gradient < quench_slope]

    # Using list comprehension, this line fills each x,y position in array with a list of quench frames
    quench_frames = np.array([[(same_xy < quench_slope).nonzero()[0]
                               for same_xy in same_x]
                              for same_x in normalized_bias_gradient], dtype=object)

    # Using list comprehension, this line creates an array storing significant quench frames (plus the last one, which
    #    should also be significant) for each x,y position
    # Define "significant"
    sig_quench_frames = np.array([[same_xy[(np.diff(same_xy) > quench_diff).tolist() + [True]]
                                   for same_xy in same_x]
                                  for same_x in quench_frames])

    # Using list comprehension, this line fills each x,y position in array with a list of pre-ramp frames
    ramp_frames = np.array([[(same_xy > rise_slope).nonzero()[0]
                             for same_xy in same_x]
                            for same_x in normalized_bias_gradient], dtype=object)

    # Using list comprehension, this line creates an array storing significant ramp start frames (plus the first one,
    #    which should also be significant) for each x,y position
    sig_ramp_frames = np.array([[same_xy[[True] + (np.diff(same_xy) > rise_diff).tolist()]
                                 for same_xy in same_x]
                                for same_x in ramp_frames])

    # print("This is the average of the non-quench normalized gradient array at the test indices:",
    #       np.mean(normalized_bias_gradient[test_indices +
    #                                        (normalized_bias_gradient[test_indices] > quench_slope).nonzero()]))

    # Is there a more efficient way to do this next code (such as with list comprehension)?
    # Using a for loop, these next lines identify the indices of the maximum bias within each plateau
    # Does being a ragged array mess this up at all? What about with sig_ramp_frames?
    max_bias_frames = np.full_like(sig_quench_frames, np.nan)

    for i in range(sig_quench_frames.shape[0]):
        for j in range(sig_quench_frames.shape[1]):
            for p in range(sig_quench_frames.shape[2]):
                start_ind = sig_ramp_frames[i, j, p]
                max_bias_frames[i, j, p] = np.argmax(bias[i, j, start_ind:sig_quench_frames[i, j, p]]) + start_ind

    pad = (margin - 1) // 2
    plateau_bounds = np.stack((sig_ramp_frames + pad, max_bias_frames - pad), axis=-1)

    return plateau_bounds


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
        assert(bias.unit == u.V)
    except (AttributeError, AssertionError):
        warnings.warn("Input bias array does not have units of Volts. Ensure that bias values are in real units.")
    try:
        assert(current.unit == u.A)
    except (AttributeError, AssertionError):
        warnings.warn("Input bias array does not have units of Amps. Ensure that current values are in real units.")

    # real_bias, real_current = to_real_units(bias[start:end], current[start:end])
    # return Characteristic(real_bias, real_current)
    return Characteristic(bias[start:end], current[start:end])


def get_time_array(plateau_ranges, sample_sec=(100 / 16 * 10 ** 6) ** (-1) * u.s):
    # Make more robust; is mean time during shot okay? Clean up, decide final form
    # x, y, time in milliseconds since start of that [average] shot using sample_sec in milliseconds

    # returns the time at the center of the ramp since the beginning of the shot
    return np.mean(plateau_ranges, axis=-1) * sample_sec


"""
    x_length = bias.shape[0]
    y_length = bias.shape[1]

    print("Splitting frames into plateaus...")
    max_bias_indices = np.nanargmax(np.arange(1), axis=-1)  # Is splitting more efficient to calculate the max indices?
"""


def get_characteristic_array(bias, current, plateau_ranges):
    # Make sure to store time information!

    characteristic_array = np.empty((plateau_ranges.shape[:3]), dtype=object)  # x, y, plateau
    # Address case where there are an irregular number of plateaus in a frame to begin with!
    #    This should be addressed by creating a secondary array (or list?) containing the indices of valid plateaus
    #    to analyze. Invalid ones should be skipped, but preserved in the array.

    print("Creating characteristic array... (May take up to 60 seconds)")
    for i in range(plateau_ranges.shape[0]):
        for j in range(plateau_ranges.shape[1]):
            for p in range(plateau_ranges.shape[2]):
                start_ind, stop_ind = plateau_ranges[i, j, p]
                characteristic_array[i, j, p] = create_ranged_characteristic(
                    bias[i, j], current[i, j], start_ind, stop_ind)

        print("Finished x position", i + 1, "/", plateau_ranges.shape[0])

    return characteristic_array

# Instead of having several separate methods "trade off", can have one overarching method that organizes things and
#    smaller helper functions that are called within overall method?


def to_characteristic_xarray(characteristic_array, time_array):
    # Use hard-coded and calculated inputs to add coordinate and variable information to characteristic array

    characteristic_xarray = xr.DataArray(characteristic_array, dims=['x', 'y', 'plateau'])

    if characteristic_xarray.sizes['x'] == 71:
        characteristic_xarray = characteristic_xarray.assign_coords({'x': np.arange(-30, 41)})
        characteristic_xarray.x.attrs['units'] = str(u.cm)
    characteristic_xarray = characteristic_xarray.assign_coords({'time': (('x', 'y', 'plateau'),
                                                                          time_array.to(u.ms).value)})
    characteristic_xarray.time.attrs['units'] = str(u.ms)

    """
    coords=[('x', np.arange(-30, 41), {'units': str(u.cm)})  # LAPD length
        if characteristic_array.shape[0] == 71
        else ('x', np.arange(characteristic_array.shape[0])),  # other length
        ('y', np.arange(characteristic_array.shape[1])),          # generic y
        ('time', time_array.to(u.ms).value, {'units': str(u.ms)})],  # in ms
    """

    return characteristic_xarray
