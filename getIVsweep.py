# Make sure to add code comments and write documentation!
import warnings

import numpy as np

from hdf5reader import *


def get_isweep_vsweep(filename):
    r"""

    :param filename:
    :return:
    """

    file = open_hdf5(filename)
    x_round, y_round, shot_list = get_xy(file)
    xy_shot_ref = categorize_shots_xy(x_round, y_round, shot_list)

    isweep_data_raw, vsweep_data_raw, isweep_headers_raw, vsweep_headers_raw = get_sweep_data_headers(file)

    print("Reading in scales and offsets from headers...")
    # Define: scale is 2nd index, offset is 3rd index
    isweep_scales, isweep_offsets = get_scales_offsets(isweep_headers_raw, scale_index=1, offset_index=2)
    vsweep_scales, vsweep_offsets = get_scales_offsets(vsweep_headers_raw, scale_index=1, offset_index=2)

    print("Decompressing raw data...")
    isweep_processed = scale_offset_decompress(isweep_data_raw, isweep_scales, isweep_offsets)
    vsweep_processed = scale_offset_decompress(vsweep_data_raw, vsweep_scales, vsweep_offsets)

    # Can I convert isweep and vsweep arrays to real units here? Should do as long as numpy can handle astropy units

    # Is the below necessary? Check the MATLAB code
    # To reflect MATLAB code, should I take (pointwise?) standard deviation for each across these shots too? (For error)
    # isweep_sumsq = np.ndarray((1065,), float)

    # Create 4D array: the first two dimensions correspond to all combinations of unique x and y positions,
    #    the third dimension represents the nth shot taken at that unique positions
    #    and the fourth dimensions lists all the frames in that nth shot.
    isweep_xy_shots_array = isweep_processed[xy_shot_ref]
    vsweep_xy_shots_array = vsweep_processed[xy_shot_ref]
    print("Shape of isweep_xy_shots_array:", isweep_xy_shots_array.shape)  # Verify shape: (x, y, shot, frame)

    # Average all shots taken at same position into one shot. The vsweep and isweep values for each individual frame in
    #    the new "average shot" are the mean values for the corresponding frame in all other shots at the same position.
    # Creates a 3D array of sweep values for each unique x position, unique y position, and frame number.
    isweep_means = np.mean(isweep_xy_shots_array, 2)
    vsweep_means = np.mean(vsweep_xy_shots_array, 2)

    # Note: This function returns the bias values first, then the current
    file.close()
    return vsweep_means, isweep_means


def get_xy(file):
    r"""
    Reads the x, y, and shot data from a file. Returns rounded x and y data for each shot and a list of shots.

    Parameters
    ----------
    :param file: file object
    :return: tuple of three lists
    """
    motor_data = structures_at_path(file, '/Raw data + config/6K Compumotor')
    motion_path = (motor_data["Datasets"])[0]
    probe_motion = file[motion_path]
    # print("Data type of probe_motion dataset: ", probe_motion.dtype)

    print("Rounding position data...")

    places = 1
    x_round = np.round(probe_motion['x'], decimals=places)
    y_round = np.round(probe_motion['y'], decimals=places)

    shot_list = tuple(probe_motion['Shot number'])

    return x_round, y_round, shot_list


def categorize_shots_xy(x_round, y_round, shot_list):
    r"""
    Categorize shots by their x,y position. Returns a 3D list of shot numbers at each combination of unique x and y pos.

    Parameters
    ----------
    :param x_round: list
    :param y_round: list
    :param shot_list: list
    :return: list
    """

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

    # Creates an empty 2D array with a cell for each unique x,y position combination,
    #    then fills each cell with a list of indexes to the nth shot number such that
    #    each shot number is stored at the cell representing the x,y position where it was taken
    #    (for example, a shot might have been taken at the combination of the 10th unique
    #       x position and the 15th unique y position in the lists x and y)
    # For every shot index i, x_round[i] and y_round[i] give the x,y position of the shot taken at that index
    print("Categorizing shots by x,y position...")
    xy_shot_ref = [[[] for _ in range(y_length)] for _ in range(x_length)]
    for i in range(len(shot_list)):
        # noinspection PyTypeChecker
        xy_shot_ref[x_loc[i]][y_loc[i]].append(i)  # full of references to nth shot taken

    return xy_shot_ref

    # st_data = []
    # This part: list of links to nth smallest shot numbers (inside larger shot number array)
    # This part: (# unique x positions * # unique y positions) grid storing location? of shot numbers at that position
    # SKIP REMAINING X, Y POSITION DATA PROCESSING


def get_sweep_data_headers(file):
    r"""

    Function to read raw isweep and vsweep data and headers from a file.
    Returns four arrays arrays containing isweep and vsweep data and sweep headers in compressed format.

    Parameters
    ----------
    :param file: file object
    :return: arrays of raw isweep and vsweep data and headers
    """

    # SIS crate data
    sis_group = structures_at_path(file, '/Raw data + config/SIS crate/')
    # print("Datasets in sis_data structure: " + str(sis_group["Datasets"]))

    isweep_data_path = (sis_group['Datasets'])[2]
    isweep_headers_path = (sis_group['Datasets'])[3]
    vsweep_data_path = (sis_group['Datasets'])[4]
    vsweep_headers_path = (sis_group['Datasets'])[5]

    isweep_data_raw = np.array(file[isweep_data_path])
    isweep_headers_raw = file[isweep_headers_path]
    vsweep_data_raw = np.array(file[vsweep_data_path])
    vsweep_headers_raw = file[vsweep_headers_path]

    print("Shape of isweep data array:", isweep_data_raw.shape)

    return isweep_data_raw, vsweep_data_raw, isweep_headers_raw, vsweep_headers_raw


def get_scales_offsets(headers, scale_index, offset_index):
    r"""
    Unpack scales and offsets from headers to use in scale-offset decompression.

    Parameters
    ----------
    :param headers:
    :param scale_index:
    :param offset_index:
    :return:
    """

    scales = np.array([header[scale_index] for header in headers])
    offsets = np.array([header[offset_index] for header in headers])
    return scales, offsets


def scale_offset_decompress(data_raw, scales, offsets):
    r"""
    Decompress raw data using the specified arrays of scales and offsets.
    Scale and offset arrays must have first dimension corresponding to the length of the input data
    (for example, the number of shots taken).

    Parameters
    ----------
    :param data_raw: array
    :param scales: array
    :param offsets: array
    :return: decompressed data array
    """

    if len(data_raw.shape) > 2:
        raise ValueError("Only 2D arrays are currently supported for data decompression.")
    num_shots = data_raw.shape[0]

    return data_raw * scales.reshape(num_shots, 1) + offsets.reshape(num_shots, 1)
