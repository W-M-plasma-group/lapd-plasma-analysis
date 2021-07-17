# Make sure to add code comments!
import warnings

import numpy as np
import astropy.units as u

from plasmapy.diagnostics.langmuir import Characteristic

from hdf5reader import *


# def isweep_vsweep_array


# def decompress_sweep_array


# def categorize_shots_xy


def get_xy(filename):
    file = open_hdf5(filename)

    motor_data = structures_at_path(file, '/Raw data + config/6K Compumotor')
    motion_path = (motor_data["Datasets"])[0]
    probe_motion = file[motion_path]
    # print("Data type of probe_motion dataset: ", probe_motion.dtype)

    print("Rounding position data...")

    places = 1
    x_round = np.round(probe_motion['x'], decimals=places)
    y_round = np.round(probe_motion['y'], decimals=places)
    x, x_loc = np.unique(x_round, return_inverse=True)
    y, y_loc = np.unique(y_round, return_inverse=True)
    x_length = len(x)
    y_length = len(y)

    # Act as soft warnings in case of limited x,y data
    if x_length == 1 and y_length == 1:
        print("Only one position value. No plots can be made")
    elif x_length == 1:
        print("Only one unique x value. Will only consider y position")
    elif y_length == 1:
        print("Only one unique y value. Will only consider x position")

    # Can these be rewritten as NumPy arrays?

    shot_list = tuple(probe_motion['Shot number'])
    num_shots = len(shot_list)
    print("Number of shots taken:", num_shots)

    # Creates an empty 2D array with a cell for each unique x,y position combination,
    #    then fills each cell with a list of indexes to the nth shot number such that
    #    each shot number is stored at the cell representing the x,y position where it was taken
    #    (for example, a shot might have been taken at the combination of the 10th unique
    #       x position and the 15th unique y position in the lists x and y)
    # For every shot index i, x_round[i] and y_round[i] give the position of the shot taken at that index
    print("Categorizing shots by x,y position...")
    xy_shot_ref = [[[] for _ in range(y_length)] for _ in range(x_length)]
    for i in range(num_shots):
        # noinspection PyTypeChecker
        xy_shot_ref[x_loc[i]][y_loc[i]].append(i)  # full of references to nth shot taken

    file.close()
    return xy_shot_ref

    # st_data = []
    # This part: list of links to nth smallest shot numbers (inside larger shot number array)
    # This part: (# unique x positions * # unique y positions) grid storing location? of shot numbers at that position
    # SKIP REMAINING X, Y POSITION DATA PROCESSING


def get_isweep_vsweep(filename):
    xy_shot_ref = get_xy(filename)
    file = open_hdf5(filename)

    # SIS crate data
    sis_group = structures_at_path(file, '/Raw data + config/SIS crate/')
    # print("Datasets in sis_data structure: " + str(sis_group["Datasets"]))

    # Add more code comments in general. In addition, keep a more detailed documentation outside of the code
    isweep_data_path = (sis_group['Datasets'])[2]
    isweep_headers_path = (sis_group['Datasets'])[3]
    isweep_data_raw = file[isweep_data_path]
    isweep_headers_raw = file[isweep_headers_path]

    vsweep_data_path = (sis_group['Datasets'])[4]
    vsweep_headers_path = (sis_group['Datasets'])[5]
    vsweep_data_raw = file[vsweep_data_path]
    vsweep_headers_raw = file[vsweep_headers_path]

    print("Shape of isweep data array:", isweep_data_raw.shape)

    isweep_raw_array = np.array(isweep_data_raw)
    vsweep_raw_array = np.array(vsweep_data_raw)

    print("Reading in scales and offsets from headers...")
    # Define: scale is 2nd index, offset is 3rd index
    # Can I skip some of the lists and turn data directly into arrays?
    isweep_scales = [header[1] for header in isweep_headers_raw]
    vsweep_scales = [header[1] for header in vsweep_headers_raw]
    isweep_offsets = [header[2] for header in isweep_headers_raw]
    vsweep_offsets = [header[2] for header in vsweep_headers_raw]

    isweep_scales_array = np.array(isweep_scales)
    isweep_offsets_array = np.array(isweep_offsets)
    vsweep_scales_array = np.array(vsweep_scales)
    vsweep_offsets_array = np.array(vsweep_offsets)

    # (SKIP AREAL PLOT CODE; GO TO RADIAL PLOT CODE)

    # Process (decompress) isweep, vsweep data; raw_size[0] should be number of shots, raw_size[1] should be number
    #   of measurements per shot ("frames")

    print("Decompressing raw data...")

    isweep_processed = isweep_scales_array[:, np.newaxis] * isweep_raw_array + isweep_offsets_array[:, np.newaxis]
    vsweep_processed = vsweep_scales_array[:, np.newaxis] * vsweep_raw_array + vsweep_offsets_array[:, np.newaxis]

    # Is the below necessary? Check the MATLAB code
    # To reflect MATLAB code, should I take (pointwise?) standard deviation for each across these shots too? (For error)
    # isweep_sumsq = np.ndarray((1065,), float)

    print("Finished decompressing compressed isweep and vsweep data")

    # Create 4D array: the first two dimensions correspond to all combinations of unique x and y positions,
    #    the third dimension represents the nth shot taken at that unique positions
    #    and the fourth dimensions lists all the frames in that nth shot.

    isweep_xy_shots_array = isweep_processed[xy_shot_ref]
    vsweep_xy_shots_array = vsweep_processed[xy_shot_ref]
    print("Shape of isweep_xy_shots_array:", isweep_xy_shots_array.shape)

    # Note: Further clean up comments, add other comments/documentation
    # Find the mean value of the current and voltage across all the shots taken at same position
    #    for each time (frame) in the shot, preserving the time axis
    # This creates an array of averages at each time, unique x pos, and unique y pos

    # Calculate means: "horizontally" average all shots taken at same position
    #    (average all corresponding frames into a single "average shot" with same number of frames)
    isweep_means = np.mean(isweep_xy_shots_array, 2)
    vsweep_means = np.mean(vsweep_xy_shots_array, 2)

    # Describe what vsweep and isweep values do within each shot for observers to understand

    """
    # print("Padded limit for characteristic:", characteristic.get_padded_limit(0.5))
    print("Characteristic V data:", characteristic.bias)
    print("Characteristic I data:", characteristic.current)
    """

    # Note: This function returns the bias values first, then the current
    file.close()
    return vsweep_means, isweep_means


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
    # Since the bias changes almost linearly within each of these three regions, by taking the slope (gradient)
    #    of the bias (and normalizing it to be less than 1) the bias can be used to divide the frames up into regions.
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


def smooth_current_array(bias, current, margin):
    # This may still distort the shape of the current (especially at the ends of each plateau), but is much faster

    if margin < 0:
        raise ValueError("Cannot smooth over negative number", margin, "of points")
    if margin == 0:
        warnings.warn("Zero-point smoothing is redundant")
    if current.shape[-1] <= margin:
        raise ValueError("Last dimension length", current.shape[-1], "is too short to take", margin, "-point mean over")

    # Note: Bias is not smoothed (only current is)
    current_sum = np.cumsum(np.insert(current, 0, 0, axis=-1), axis=-1)
    smooth_current_full = (current_sum[..., margin:] - current_sum[..., :-margin])/margin

    adjusted_bias = bias[..., (margin-1)//2:-(margin-1)//2]

    return adjusted_bias, smooth_current_full


# def get_time_array(shape_of_frames, sample_sec):  # Is this strictly necessary? All piles (pages) are identical anyway
def get_time_array(plateau_ranges, sample_sec=(100 / 16 * 10 ** 6) ** (-1) * u.s):
    # Make more robust; is mean time during shot okay? Clean up, decide final form
    # x, y, time in milliseconds since start of that [average] shot using sample_sec in milliseconds
    # return np.full(shape=shape_of_frames, fill_value=(np.arange(shape_of_frames[-1]) * sample_sec).to(u.ms))

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

# isolate_plateaus(get_isweep_vsweep('HDF5/8-3500A.hdf5'))
# get_isweep_vsweep('HDF5/09_radial_line_25press_4kA_redo.hdf5')

# Instead of having several separate methods "trade off", can have one overarching method that organizes things and
#    smaller helper functions that are called within overall method?
