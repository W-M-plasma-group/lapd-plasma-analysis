# Make sure to add code comments!

from hdf5reader import *


def get_xy(filename):
    file = open_HDF5(filename)

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
    file = open_HDF5(filename)

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


def isolate_plateaus(bias, current=None):  # Current is optional for maximum compatibility

    quench_slope = -1  # "Threshold for voltage quench slope": MATLAB code comment
    quench_diff = 10  # Threshold for separating distinct voltage quench frames

    # The bias has three types of regions: constant low, increase at constant rate (ramp), and rapid decrease
    #    down to minimum value (quench). The ramp region is where useful Isweep-Vsweep data points are collected.
    # Since the bias changes almost linearly within each of these three regions, by taking the slope (gradient)
    #    of the bias (and normalizing it to be less than 1) the bias can be used to divide the frames up into regions.

    bias_gradient = np.gradient(bias, axis=-1)
    normalized_bias_gradient = bias_gradient / np.amax(bias_gradient, axis=-1, keepdims=True)

    # Previous efforts to create quench_frames solely using array methods (fastest! but harder) lie here
    # quench_frames = np.array((normalized_bias_gradient < quench_slope).nonzero())
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

    # Sample for first position (0,0), can be used for debugging; not needed in final function
    # plt.plot(bias[0, 0], 'b-', sig_quench_frames[0, 0], bias[0, 0, sig_quench_frames[0, 0]], 'ro')
    # plt.show()

    return sig_quench_frames


def to_real_units(bias, current):
    # The conversion factors from abstract units to real bias (V) and current values (A) are hard-coded in here.
    # Note that current is multiplied by -1 to get the "upright" traditional Isweep-Vsweep curve. Add to documentation?

    # Conversion factors taken from MATLAB code: Current = isweep / 11 ohms; Voltage = vsweep * 100
    gain = 100.  # voltage gain
    resistance = 11.  # current values from input current; implied units of ohms per volt since measured as potential

    return bias * gain * u.V, -1. * current / resistance * u.A


def create_ranged_characteristic(bias, current, start, end):
    dimensions = len(bias.shape)

    # debug
    # print("Dimensions of incoming bias array:", bias.shape)
    # print("Zero coordinates to access first corner of bias and current:", zero_indices)
    #

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


def smooth_characteristic(characteristic, num_points_each_side):

    # Note: smooth_characteristic changes (distorts) characteristic; use SLM-like fitting instead of smoothing later on?

    size = characteristic.bias.shape
    length = size[len(size) - 1]
    if num_points_each_side < 0:
        raise ValueError("Cannot smooth over negative number", num_points_each_side, "of points")
    if length < 2 * num_points_each_side:
        raise ValueError("Characteristic of", length, "data points is too short to take", num_points_each_side,
                         "-point average over")
    # smooth_bias = np.zeros(size)
    smooth_current = np.zeros(size)

    # Should the ends (which I set to be constant) just be chopped off instead? Yes.

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
    # New: Return (4D array: x, y, plateau number in shot, frame number in plateau; padded with nans),
    #             (4D array: x, y, plateau number in shot, start & end significant frame in plateau)

    # Not in MATLAB code
    rise_slope = 0.5  # Threshold for increases in slope

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

    # Note: The selection of ramps (increasing bias section only) from plateaus (constant and increasing bias sections)
    #    is not in the original MATLAB code, but is judged to be necessary based on the performance of PlasmaPy
    #    diagnostic functions. The isolated rising section ("ramp") is needed for the diagnostics to complete correctly.
    ramp_start_frames = np.zeros_like(max_bias_indices)

    # Can normalized_plateau_gradient/plateau_before_max be calculated otherwise? (array functions/list comprehension?)
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

    # Can I move/combine this with the isolate_plateaus function? Or would that not help, since gradient renormalized?

    plateau_ranges = np.stack([ramp_start_frames, max_bias_indices], axis=-1)

    # Clean up comments/debug cases when this function no longer being edited
    # debug
    # print("Start and stop frames of plateaus:", plateau_ranges)
    # plateau_lengths = plateau_ranges[..., 1] - plateau_ranges[..., 0]
    # print("Minimum characteristic length indices should be", np.argmin(plateau_lengths))
    # print("Length of plateaus:", plateau_lengths)
    # print("View of minimum-containing row?? of plateau_lengths:",
    #   plateau_lengths[0, 0])
    #       plateau_lengths[np.unravel_index(np.argmin(plateau_lengths), plateau_lengths.shape)])
    #

    # First corner plot
    # plt.plot(split_bias[30, 0, 7, :max_bias_indices[30, 0, 7]],
    #          split_current[30, 0, 7, :max_bias_indices[30, 0, 7]], 'bo')
    # plt.show()

    return split_bias, split_current, plateau_ranges


# isolate_plateaus(get_isweep_vsweep('HDF5/8-3500A.hdf5'))
# get_isweep_vsweep('HDF5/09_radial_line_25press_4kA_redo.hdf5')


def get_characteristic_array(split_bias, split_current, plateau_ranges):
    # Are split arrays needed? Use create_ranged_characteristic (original bias/current + start/stop indices for each?
    # Use ragged arrays????? Then can create zeros_like or something to match
    # Still need to do plateau filtering

    # characteristic_array = np.empty((split_bias.shape[:3]))  # x, y, plateau
    smooth_characteristic_array = np.empty((split_bias.shape[:3]), dtype=object)  # x, y, plateau; hard-coded in
    # Address case where there are an irregular number of plateaus in a frame to begin with!

    print("Creating characteristic array... (May take about 30 seconds)")
    for i in range(split_bias.shape[0]):
        for j in range(split_bias.shape[1]):
            for p in range(split_bias.shape[2]):
                # characteristic_array[i, j, p] = smooth_characteristic(create_ranged_characteristic(
                smooth_characteristic_array[i, j, p] = (smooth_characteristic(create_ranged_characteristic(
                    split_bias[i, j, p], split_current[i, j, p],
                    plateau_ranges[i, j, p, 0], plateau_ranges[i, j, p, 1]), 10)
                                                        if plateau_ranges[i, j, p, 1] - plateau_ranges[
                    i, j, p, 0] > 2 * 10 else None)

    return smooth_characteristic_array

# def extract_diagnostic_array(characteristic_array)
#
