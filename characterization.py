import warnings
import numpy as np
import astropy.units as u
from plasmapy.diagnostics.langmuir import Characteristic


def characterize_sweep_array(unsmooth_bias, unsmooth_current, margin, sample_sec):
    # Function to create array of characteristics for bias and current data; make sure to put in real units!

    bias, current = smooth_current_array(unsmooth_bias, unsmooth_current, margin=margin)
    plateau_ranges = isolate_plateaus(bias, current, margin=margin)
    time_array = get_time_array(plateau_ranges, sample_sec)
    return get_characteristic_array(bias, current, plateau_ranges)


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

    adjusted_bias = bias[...,
                    (margin - 1) // 2:-(margin - 1) // 2]  # Shifts bias to align with new, shorter current array

    return adjusted_bias, smooth_current_full


def isolate_plateaus(bias, current=None, margin=0):  # Current is optional for maximum compatibility

    r"""
    Function to identify start and stop frames of every ramp section within each plateau.
    Returns array containing frame indices for each shot.

    Parameters
    ----------
    :param bias: array
    :param current: array, optional
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


def to_real_units(bias, current):
    r"""
    Parameters
    ----------
    :param bias: array
    :param current: array
    :return: bias and current array in real units
    """
    # The conversion factors from abstract units to real bias (V) and current values (A) are hard-coded in here.
    # Note that current is multiplied by -1 to get the "upright" traditional Isweep-Vsweep curve. Add to documentation?

    # Conversion factors taken from MATLAB code: Current = isweep / 11 ohms; Voltage = vsweep * 100
    gain = 100.  # voltage gain
    resistance = 11.  # current values from input current; implied units of ohms per volt since measured as potential

    return bias * gain * u.V, -1. * current / resistance * u.A


def create_ranged_characteristic(bias, current, start, end):
    # Returns a Characteristic object

    dimensions = len(bias.shape)
    if bias.shape != current.shape:
        raise ValueError("Bias and current must be of the same dimensions and shape")
    if start < 0:
        raise ValueError("Start index must be non-negative")
    if dimensions == 1:
        if end > len(bias):
            raise ValueError("End index", end, "out of range of bias and current arrays of length", len(bias))
        real_bias, real_current = to_real_units(bias[start:end], current[start:end])
        characteristic = Characteristic(real_bias, real_current)
    else:
        # Note: zero_indices is tuple of indices to access first position bias and current.
        #    In the future, multidimensional bias and current inputs should raise an error.
        print("Warning: multidimensional characteristic creation is unsupported. This function returns a characteristic"
              "with bias and current values only for the first position. Pass 1D arrays in the future to avoid this.")
        zero_indices = (0,) * (dimensions - 1)
        if end > len(bias[zero_indices]):
            raise ValueError("End index", end, "out of range of bias and current arrays of last-dimension length",
                             len(bias[zero_indices]))
        real_bias, real_current = to_real_units(bias[zero_indices + (slice(start, end),)],
                                                current[zero_indices + (slice(start, end),)])
        characteristic = Characteristic(real_bias, real_current)

    return characteristic


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
    # Still need to do plateau filtering
    # Make sure to store time information!

    characteristic_array = np.empty((plateau_ranges.shape[:3]), dtype=object)  # x, y, plateau?
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
