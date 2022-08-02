import numpy as np
import astropy.units as u

from hdf5reader import *


def get_isweep_vsweep(filename, mar2022):
    r"""
    Reads all sweep data (V-sweep and I-sweep) from HDF5 file Langmuir code.
    :param filename: File path of HDF5 file from LAPD
    :return: bias, current, x, y: the relevant multi-dimensional sweep data and position data
    """

    """
    print("Loading HDF5 data...")
    hdf5_file = open_hdf5(filename)
    """
    # print("Categorizing shots by x,y position...")
    # xy_shot_ref, x, y = categorize_shots_xy(x_round, y_round, shot_list)

    lapd_file = lapd.File(filename)

    isweep_channel = 2  # 2 = port 27, 3 = port 43
    isweep_receptacles = {2: 1, 3: 2}
    # if mar2022:  # TODO replace with user menu?

    vsweep_data = lapd_file.read_data(1, 1, silent=True)
    isweep_data = lapd_file.read_data(1, isweep_channel, silent=True)
    vsweep_signal = vsweep_data['signal']
    isweep_signal = isweep_data['signal']
    signal_length = vsweep_signal.shape[-1]

    motor_data = lapd_file.read_controls([("6K Compumotor", isweep_receptacles[isweep_channel])], silent=True)
    num_shots = len(motor_data['shotnum'])
    position_data = np.round(motor_data['xyz'], 1)

    z_positions = position_data[:, 2]
    if np.amin(z_positions) != np.amax(z_positions):
        raise ValueError("Varying z-position when only x and/or y variation expected")
    # save z-position for later?

    positions = np.unique(position_data[:, :2], axis=0)
    num_positions = len(positions)
    if num_shots % num_positions != 0:
        raise ValueError("Number of shots " + str(num_shots) +
                         " does not evenly divide into " + str(num_positions) + " positions")
    shots_per_position = int(num_shots // num_positions)

    vsweep_signal = vsweep_signal.reshape((num_positions, shots_per_position, signal_length))
    isweep_signal = isweep_signal.reshape((num_positions, shots_per_position, signal_length))

    xy_at_positions = position_data[:, :2].reshape((num_positions, shots_per_position, 2))  # 2 for x, y (no z)
    if not (np.amax(xy_at_positions, axis=1) == np.amin(xy_at_positions, axis=1)).all():
        raise ValueError("Non-uniform positions values when grouping Langmuir probe data by position")

    vsweep_signal = vsweep_signal.mean(axis=1)
    isweep_signal = isweep_signal.mean(axis=1)

    # TODO welcome.py
    """
    PLAN
    - Ask user to select NetCDF file to use, or 0 to load HDF5
        - Ask user to select HDF5 file to use
        - Ask user if save new NetCDF file
    - Welcome.py
        - Search run description for isweep and vsweep
            - If ambiguous, ask user which to use
        - Search list of probes for common Langmuir types
            - If uncertain/ambiguous, ask user which to use
        ** Ask for HDF5 parameters "Vsweep location" and "List of Isweep locations for user to choose from"
        - Search for 96 GHz interferometer - first priority in interferometry?
        ** Ask for HDF5 parameters "preferred interferometer data path" and "interferometer type" (ex. 56 vs. 96 GHz)
    """

    # Note: I may be able to convert isweep/vsweep arrays to real units here, as long as numpy can handle astropy units

    # Note: I may take the standard deviation across shots to approximate error for sweep curves, as done in MATLAB code

    """
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
    """

    # hdf5_file.close()
    lapd_file.close()

    # Note: This function returns the bias values first, then the current
    # bias, current = to_real_sweep_units(vsweep_means, isweep_means, mar2022)
    # return bias, current, x, y

    bias, current = to_real_sweep_units(vsweep_signal, isweep_signal, mar2022)
    return bias, current, positions


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

    # May not be necessary
    """
    # Determine if data is areal, radial, or scalar
    if x_length == 1 and y_length == 1:
        print("Only one position value. No plots can be made")
    elif x_length == 1:
        print("Only one unique x value. Will only consider y position")
    elif y_length == 1:
        print("Only one unique y value. Will only consider x position")
    """

    # Can these lists be rewritten as NumPy arrays?

    # Creates an empty 2D array with a cell for each unique x,y position combination,
    #    then fills each cell with a list of indexes to the nth shot number such that
    #    each shot number is stored at the cell representing the x,y position where it was taken
    #    (for example, a shot might have been taken at the combination of the 10th unique
    #       x position and the 15th unique y position in the lists x and y)
    # For every shot index i, x_round[i] and y_round[i] give the x,y position of the shot taken at that index
    xy_shot_ref = [[[] for _ in range(y_length)] for _ in range(x_length)]
    for i in range(len(shot_list)):
        # noinspection PyTypeChecker
        xy_shot_ref[x_loc[i]][y_loc[i]].append(i)  # full of references to nth shot taken

    return xy_shot_ref, x, y

    # Some x,y position data processing in the MATLAB code was not translated.


def get_sweep_data_headers(file, mar2022):
    r"""

    Function to read raw isweep and vsweep data and headers from a file.
    Returns four arrays containing isweep and vsweep data and sweep headers in compressed format.

    Parameters
    ----------
    :param file: file object
    :return: arrays of raw isweep and vsweep data and headers
    """

    # SIS crate data
    sis_group = structures_at_path(file, '/Raw data + config/SIS crate/')
    print("Datasets in sis_data structure: ", *sis_group["Datasets"], sep="\n")

    # QUESTION: Can we use string-based Numpy field indexing to access these instead of hard-coding integer indices?
    isweep_data_path = (sis_group['Datasets'])[4 if mar2022 else 2]  # t 6 TODO RETURN TO 4 if - else 2
    isweep_headers_path = (sis_group['Datasets'])[5 if mar2022 else 3]  # t 7 TODO RETURN TO 5 if - else 3
    vsweep_data_path = (sis_group['Datasets'])[2 if mar2022 else 4]
    vsweep_headers_path = (sis_group['Datasets'])[3 if mar2022 else 5]

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

    # NOTE: We can access scales and offsets using dictionary string indexing

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


def to_real_sweep_units(bias, current, mar2022):
    r"""
    Parameters
    ----------
    :param bias: array
    :param current: array
    :return: bias and current array in real units
    """

    # The conversion factors from abstract units to real bias (V) and current values (A) are hard-coded in here.
    # Note that current is multiplied by -1 to get the "upright" traditional Isweep-Vsweep curve. This should be added to the documentation.

    # Conversion factors taken from MATLAB code: Current = isweep / 11 ohms; Voltage = vsweep * 100
    gain = 100.  # voltage gain
    resistance = 1.25 if mar2022 else 11.  # current values from input current; implied units of ohms per volt since measured as potential
    invert = 1 if mar2022 else -1

    return bias * gain * u.V, current / resistance * u.A * invert
