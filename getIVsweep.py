# Make sure to add code comments!

from hdf5reader import *


def get_xy(filename):
    file = open_HDF5(filename)

    motor_data = structures_at_path(file, '/Raw data + config/6K Compumotor')
    motion_path = (motor_data["Datasets"])[0]
    probe_motion = file[motion_path]
    # print("Data type of probe_motion dataset: ", probe_motion.dtype)

    print("Rounding position data...")
    """
    # print("Raw x data summary: ", probe_motion['x'])
    # print("Raw y data summary: ", probe_motion['y'])

    rnd = 0.10  # Change the x rounding step value here
    dec = 100  # Change number of decimal points to consider here
    prod = int(rnd * dec)
    x_round = tuple(round(dec * xin / prod) * prod / dec for xin in tuple(probe_motion['x']))
    y_round = tuple(round(dec * yin / prod) * prod / dec for yin in tuple(probe_motion['y']))
    x = sorted(tuple(set(x_round)))  # only the unique x position values
    y = sorted(tuple(set(y_round)))  # only the unique y position values
    """
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
    print("Total number of shots taken:", num_shots)

    # Creates an empty 2D array with a cell for each unique x,y position combination,
    #    then fills each cell with a list of indexes to the nth shot number such that
    #    each shot is stored at the cell representing the x,y position where it was taken
    #    (for example, a shot might have been taken at the combination of the 10th unique
    #       x position and the 15th unique y position in the lists x and y)
    # NOTE: This code used to be after "try all the categorizing needed down here" in get_isweep_vsweep
    # For every shot index i, x_round[i] and y_round[i] give the position of the shot taken at that index
    print("Categorizing shots by x,y position...")
    xy_shot_ref = [[[] for _ in range(y_length)] for _ in range(x_length)]
    for i in range(num_shots):
        # noinspection PyTypeChecker
        xy_shot_ref[x_loc[i]][y_loc[i]].append(i)  # full of references to nth shot taken
    #
    # print("xy shot refs:", xy_shot_ref)

    file.close()
    # return x, y
    return xy_shot_ref

    # print("List of shots performed: ", shotList)
    # st_data = []
    # This part: list of links to nth smallest shot numbers (inside larger shot number array)
    # This part: (# unique x positions * # unique y positions) grid storing location? of shot numbers at that position
    # SKIP REMAINING X, Y POSITION DATA PROCESSING


def get_isweep_vsweep(filename, sample_sec=(100 / 16 * 10 ** 6) ** (-1)):
    file = open_HDF5(filename)  # Should I make the previous function take a file as a parameter and not a filename?

    xy_shot_ref = get_xy(filename)

    # SIS crate data
    sis_group = structures_at_path(file, '/Raw data + config/SIS crate/')
    # print("Datasets in sis_data structure: " + str(sis_group["Datasets"]))

    # Add more code comments in general. In addition, keep a more detailed documentation outside of the code
    isweep_data_path = (sis_group['Datasets'])[2]
    isweep_headers_path = (sis_group['Datasets'])[3]
    isweep_data_raw = file[isweep_data_path]
    isweep_headers_raw = file[isweep_headers_path]

    # print("Shape of isweep dataset: "+str(isweep_data_raw.shape))

    vsweep_data_path = (sis_group['Datasets'])[4]
    vsweep_headers_path = (sis_group['Datasets'])[5]
    vsweep_data_raw = file[vsweep_data_path]
    vsweep_headers_raw = file[vsweep_headers_path]

    raw_size = (len(isweep_data_raw), len(isweep_data_raw[0]))  # shape of isweep_data
    print("Dimensions of isweep data array:", raw_size)
    num_shots = raw_size[0]

    isweep_raw_array = np.array(isweep_data_raw)
    vsweep_raw_array = np.array(vsweep_data_raw)

    print("Reading in scales and offsets from headers...")
    # Define: scale is 2nd index, offset is 3rd index
    isweep_scales = [header[1] for header in isweep_headers_raw]
    vsweep_scales = [header[1] for header in vsweep_headers_raw]
    isweep_offsets = [header[2] for header in isweep_headers_raw]
    vsweep_offsets = [header[2] for header in vsweep_headers_raw]

    isweep_scales_array = np.array(isweep_scales)
    isweep_offsets_array = np.array(isweep_offsets)
    vsweep_scales_array = np.array(vsweep_scales)
    vsweep_offsets_array = np.array(vsweep_offsets)

    # print("Finished reading in scales and offsets from headers")

    # (SKIP AREAL PLOT CODE; GO TO RADIAL PLOT CODE)

    # Process (decompress) isweep, vsweep data; raw_size[0] should be number of shots, raw_size[1] should be number
    #   of measurements per shot ("frames")

    print("Decompressing raw data...")

    isweep_processed = isweep_scales_array[:, np.newaxis] * isweep_raw_array + isweep_offsets_array[:, np.newaxis]
    vsweep_processed = vsweep_scales_array[:, np.newaxis] * vsweep_raw_array + vsweep_offsets_array[:, np.newaxis]
    # Is the below necessary? Check the MATLAB code
    # isweep_sumsq = np.ndarray((1065,), float)

    print("Finished decompressing compressed isweep and vsweep data")

    # Note: clean up and review some of the following comments
    # Take average of voltage and current across all shots at each unique position, preserving time dimension
    # To reflect MATLAB code, should I take (pointwise?) standard deviation for each across these shots too? (For error)

    # We want to try to create the mean value over all shots taken at same position for each time (frame) in the shot.
    # This creates an array of averages at each time, unique x pos, and unique y pos.
    # Try all the categorizing needed down here. (For example, store shot references in unique x, unique y grid)

    # Create 4D array: the first two dimensions correspond to all combinations of unique x and y positions,
    #    the third dimension represents the nth shot taken at that unique positions
    #    and the fourth dimensions lists all the frames in that nth shot.

    isweep_xy_shots_array = isweep_processed[xy_shot_ref]
    vsweep_xy_shots_array = vsweep_processed[xy_shot_ref]
    print("Shape of isweep_xy_shots_array:", isweep_xy_shots_array.shape)

    # Calculate means: "horizontally" average all shots taken at same position
    #    (average all corresponding frames into a single "average shot" with same number of frames)
    isweep_means = np.mean(isweep_xy_shots_array, 2)
    vsweep_means = np.mean(vsweep_xy_shots_array, 2)

    # Graph vsweep vs isweep for "average shot" (average of all 15 shots) in first unique x,y position
    # plt.plot(vsweep_means[0, 0], isweep_means[0, 0])
    # plt.plot(vsweep_means[0, 0, 12500:13100], np.multiply(isweep_means[0, 0, 12500:13100], -1))
    # plt.show()

    # Will have to note that a -1 factor is required to obtain upright Isweep-Vsweep curves

    """
    # print("Padded limit for characteristic:", characteristic.get_padded_limit(0.5))
    print("Characteristic V data:", characteristic.bias)
    print("Characteristic I data:", characteristic.current)
    """

    # Note: This function returns the bias values first, then the current
    file.close()
    return vsweep_means, isweep_means


def isolate_plateaus(bias, current):  # call near start of create_ranged_characteristic? Does this need current?

    quench_slope = -1  # "Threshold for voltage quench slope"
    quench_diff = 10  # Threshold for separating distinct voltage quench frames

    bias_gradient = np.gradient(bias, axis=-1)
    normalized_bias_gradient = bias_gradient / np.amax(bias_gradient, axis=-1, keepdims=True)

    # quench_frames = np.array((normalized_bias_gradient < quench_slope).nonzero())
    # print("Coordinates of quench frames:", quench_frames)
    # quench_frames_by_position = frame_array[..., normalized_bias_gradient < quench_slope]

    # Using list comprehension, this line fills each x,y position in array with a list of quench frames
    quench_frames = np.array([[(same_xy < quench_slope).nonzero()[0]
                               for same_xy in same_x]
                              for same_x in normalized_bias_gradient], dtype=object)
    # print("This is the quench frame array:", quench_frames)

    # Using list comprehension, this line creates an array storing significant quench frames (plus the last one, which
    #    should also be significant) for each x,y position
    sig_quench_frames = np.array([[same_xy[(np.diff(same_xy) > quench_diff).tolist() + [True]]
                                   for same_xy in same_x]
                                  for same_x in quench_frames])
    # print("This is the significant quench frame array:", sig_quench_frames)

    # Sample for first position (0,0); not needed in final function
    # plt.plot(bias[0, 0], 'b-', sig_quench_frames[0, 0], bias[0, 0, sig_quench_frames[0, 0]], 'ro')
    # plt.show()

    # max_num_plateaus = sig_quench_frames.shape[-1]
    # max_length_plateaus =
    # print("Maximum number of plateaus:", max_num_plateaus)

    # return sig_quench_frames, max_num_plateaus
    return sig_quench_frames


def create_ranged_characteristic(filename, start, end):
    bias, current = get_isweep_vsweep(filename)
    dimensions = len(bias.shape)
    # indices = tuple(np.zeros(dimensions - 1, dtype=int).tolist())
    zero_indices = (0,) * (dimensions - 1)
    zero_indices = (30, 0)
    # debug
    # print("Dimensions of incoming bias array:", bias.shape)
    # print("Zero coordinates to access first corner of bias and current:", zero_indices)

    if bias.shape != current.shape:
        raise ValueError("Bias and current must be of the same dimensions and shape")
    if start < 0:
        raise ValueError("Start index must be non-negative")
    if (dimensions == 1 and end > len(bias)) or (end > len(bias[zero_indices])):
        raise ValueError("End index", end, "out of range of bias and current arrays of last-dimension length",
                         len(bias[zero_indices]))

    # IMPORTANT: This function returns a characteristic with isweep values multiplied by a factor of -1. (As of 6/4/21)
    # Furthermore, these values have been converted into real quantities (Volts, Amps) from their abstract former units.
    #    (Conversion factors taken from MATLAB code: Current = isweep / 11 ohms; Voltage = vsweep * 100
    # Note that these values are hard-coded in
    # IMPORTANT: If there is more than one dimension to the bias and current arrays passed into the function,
    #    the function will only consider the first element of all the other dimensions. This will have to be addressed
    #    later, when the plateau function is added to separate shots into single plateaus. (Also see end-exception line)
    #    (This is done by the zero_indices variable. It is a tuple coordinate just for accessing the position (0, 0).)
    #    (In the future, having more than one dimension for bias or current arrays should raise error?)
    return Characteristic(u.Quantity(bias[zero_indices + (slice(start, end),)] * 100, u.V),
                          u.Quantity(current[zero_indices + (slice(start, end),)] * (-1. / 11.), u.A))
    # The addition involves adding to the (0,0) position accessor a slice in the last dimension from start to end frame.


def smooth_characteristic(characteristic, num_points_each_side):
    size = characteristic.bias.shape
    length = size[len(size) - 1]
    if num_points_each_side < 0:
        raise ValueError("Cannot smooth over negative number", num_points_each_side, "of points")
    if length < 2 * num_points_each_side:
        raise ValueError("Characteristic of", length, "data points is too short to take", num_points_each_side +
                         "-point average over")
    # smooth_bias = np.zeros(size)
    smooth_current = np.zeros(size)

    # Should the ends (which I set to be constant) just be chopped off instead? Probably yes.

    for i in range(length):
        if i < num_points_each_side:
            # smooth_bias[..., i] = np.mean(characteristic.bias[..., :2 * num_points_each_side])
            smooth_current[..., i] = np.mean(characteristic.current[..., :2 * num_points_each_side])
        elif i >= length - num_points_each_side:
            # smooth_bias[..., i] = np.mean(characteristic.bias[..., -2 * num_points_each_side - 1:])
            smooth_current[..., i] = np.mean(characteristic.current[..., -2 * num_points_each_side - 1:])
        else:
            # smooth_bias[..., i] = np.mean(
            #     characteristic.bias[..., i - num_points_each_side:i + num_points_each_side + 1])
            smooth_current[..., i] = np.mean(characteristic.current[...,
                                             i - num_points_each_side:i + num_points_each_side + 1])

    # return Characteristic(u.Quantity(smooth_bias, u.V), u.Quantity(smooth_current, u.A))
    return Characteristic(u.Quantity(characteristic.bias, u.V), u.Quantity(smooth_current, u.A))


def get_time_array(shape_of_frames, sample_sec):  # Is this strictly necessary? All piles (pages) are identical anyways

    # x, y, time in milliseconds since start of that average shot using sample_sec in milliseconds
    return np.full(shape=shape_of_frames, fill_value=(np.arange(shape_of_frames[-1]) * sample_sec).to(u.ms))


def split_plateaus(bias, current, sig_quench_frames):
    # Old: Return 4D array x,y,plateau number in shot,frame in plateau
    # New: Return (4D array: x, y, plateau number in shot, frame number in plateau; padded with zeros),
    #             (4D array: x, y, plateau number in shot, start & end significant frame in plateau)

    # Not in MATLAB code
    rise_slope = 0.5  # Threshold for increases in slope

    # print("Shapes: bias =", bias.shape, sig_quench_frames =", sig_quench_frames.shape)

    max_number_plateaus = sig_quench_frames.shape[-1]
    frames_per_plateau = np.diff(np.insert(sig_quench_frames, 0, 0, axis=-1), axis=-1)
    max_number_frames = np.amax(frames_per_plateau)
    # print("Frames per each plateau:", frames_per_plateau)
    # print("Max number of frames:", max_number_frames)

    x_length = bias.shape[0]
    y_length = bias.shape[1]

    split_bias = np.full((x_length, y_length, max_number_plateaus, max_number_frames), np.nan, dtype=float)
    split_current = np.full((x_length, y_length, max_number_plateaus, max_number_frames), np.nan, dtype=float)

    # Is there a better-performance way (for example, without a for loop) to do this? Note in documentation
    print("Splitting frames into plateaus...")
    for i in range(x_length):
        for j in range(y_length):
            split_bias_list = np.split(bias[i, j], sig_quench_frames[i, j])
            split_current_list = np.split(current[i, j], sig_quench_frames[i, j])
            for f in range(max_number_plateaus):
                split_bias[i, j, f, :frames_per_plateau[i, j, f]] = split_bias_list[f]
                split_current[i, j, f, :frames_per_plateau[i, j, f]] = split_current_list[f]
        # print("x position", i, "done")

    max_bias_indices = np.nanargmax(split_bias, axis=-1)

    # Should plateau_start_frames be a nested list?
    # normalized_bias_gradient =
    ramp_start_frames = np.zeros_like(max_bias_indices)

    print("Isolating rising segments...")
    for i in range(x_length):
        for j in range(y_length):
            for f in range(max_number_plateaus):
                plateau_before_max = split_bias[i, j, f, :max_bias_indices[i, j, f]]
                normalized_plateau_gradient = np.gradient(plateau_before_max) / np.max(np.gradient(plateau_before_max))
                ramp_start_frames[i, j, f] = np.amax((normalized_plateau_gradient < rise_slope).nonzero())

    """
    split_bias = np.array([[np.split(bias[i, j], sig_quench_frames[i, j], axis=-1)[:max_number_plateaus]
                            for j in range(y_length)]
                           for i in range(x_length)])
    split_current = np.array([[np.split(current[i, j], sig_quench_frames[i, j], axis=-1)[:max_number_plateaus]
                               for j in range(bias.shape[1])]
                              for i in range(bias.shape[0])])
    """

    # print("Shape of bias plateau array:", split_bias.shape, "; shape of current plateau array:", split_current.shape)

    # This part is not in the MATLAB code explicitly, but based on PlasmaPy's plasma potential calculation it is needed
    # Could I move/combine this with the isolate_plateaus function?
    # plt.plot(split_bias[30, 0, 7], split_current[30, 0, 7])

    # debug
    # test_gradient = np.gradient(split_bias[30, 0, 7, 0:max_bias_indices[30, 0, 7]], axis=-1)
    # plt.plot(test_gradient[ramp_start_frames/np.amax(bias_gradient, axis=-1, keepdims=True))
    # plt.plot(test_gradient / np.amax(test_gradient))
    # plt.show()
    #

    plateau_ranges = np.stack([ramp_start_frames, max_bias_indices], axis=-1)
    # print("Start and stop frames of plateaus:", plateau_ranges)

    # First corner plot
    # plt.plot(split_bias[30, 0, 7, :max_bias_indices[30, 0, 7]],
    #          split_current[30, 0, 7, :max_bias_indices[30, 0, 7]], 'bo')
    # plt.show()

    return split_bias, split_current, plateau_ranges

# isolate_plateaus(get_isweep_vsweep('HDF5/8-3500A.hdf5'))
# get_isweep_vsweep('HDF5/09_radial_line_25press_4kA_redo.hdf5')
