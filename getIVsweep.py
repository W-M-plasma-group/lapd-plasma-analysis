# Make sure to add code comments!
import numpy
import numpy as np

from hdf5reader import *


def get_xy(filename):
    file = open_HDF5(filename)

    motor_data = structures_at_path(file, '/Raw data + config/6K Compumotor')
    motion_path = (motor_data["Datasets"])[0]
    probe_motion = file[motion_path]
    # print("Data type of probe_motion dataset: ", probe_motion.dtype)

    print("Rounding position data...")
    # print("Raw x data summary: ", probe_motion['x'])
    # print("Raw y data summary: ", probe_motion['y'])

    rnd = 0.10  # Change the x rounding step value here
    dec = 100  # Change number of decimal points to consider here
    prod = int(rnd * dec)
    x_round = tuple(round(dec * xin / prod) * prod / dec for xin in tuple(probe_motion['x']))
    y_round = tuple(round(dec * yin / prod) * prod / dec for yin in tuple(probe_motion['y']))
    x = sorted(tuple(set(x_round)))  # only the unique x position values
    y = sorted(tuple(set(y_round)))  # only the unique y position values
    x_length = len(x)
    y_length = len(y)

    # CAN I TURN these into np arrays too?

    # Act as soft warnings in case of limited x,y data
    if len(x) == 1 and len(y) == 1:
        print("Only one position value. No plots can be made")
    elif len(x) == 1:
        print("Only one unique x value. Will only consider y position")
    elif len(y) == 1:
        print("Only one unique y value. Will only consider x position")

    shot_list = tuple(probe_motion['Shot number'])
    num_shots = len(shot_list)
    print("Total number of shots taken:", num_shots)

    # Creates an empty 2D array with a cell for each unique x,y position combination,
    #    then fills each cell with a list of indexes to the nth shot number such that
    #    each shot is stored at the cell representing the x,y position where it was taken
    #    (for example, a shot might have been taken at the combination of the 10th unique
    #       x position and the 15th unique y position in the lists x and y)
    # NOTE: THIS CODE USED TO BE AFTER "try all the categorizing needed down here" in get_isweep_vsweep
    print("Categorizing shots by x,y position...")
    xy_shot_ref = [[[] for j in range(y_length)] for i in range(x_length)]
    for i in range(num_shots):
        xy_shot_ref[x.index(x_round[i])][y.index(y_round[i])].append(i)  # full of references to nth shot taken
    # print("xy shot refs:", xy_shot_ref)

    file.close()
    # return x, y
    return xy_shot_ref

    # print("List of shots performed: ", shotList)
    # st_data = []
    # This part: list of links to nth smallest shot numbers (inside larger shot number array)
    # This part: (# unique x positions * # unique y positions) grid storing location? of shot numbers at that position
    # SKIP REMAINING X, Y POSITION DATA PROCESSING


def get_isweep_vsweep(filename, sample_sec=(100 / 16 * 10 ** 6) ** (-1)):  # removed nodesmin parameter

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
    isweep_processed = np.ndarray((raw_size[0], raw_size[1]), float)
    vsweep_processed = np.ndarray((raw_size[0], raw_size[1]), float)
    # Is the below also necessary? Check the MATLAB code
    # isweep_sumsq = np.ndarray((1065,), float)

    for i in range(raw_size[0]):
        isweep_processed[i] = isweep_scales_array[i] * isweep_raw_array[i] + isweep_offsets_array[i]
        vsweep_processed[i] = vsweep_scales_array[i] * vsweep_raw_array[i] + vsweep_offsets_array[i]

    print("Finished decompressing compressed isweep and vsweep data")

    ### Note: Clean up these comments
    # Take average and standard deviation for isweep and vsweep values at each shot over time MAYBE I SHOULDN'T TAKE
    # AVERAGE FOR EACH SHOT! AFTER ALL, ISN'T EACH SHOT A SERIES OF _DIFFERENT_ BIASES? CONSULT MATLAB CODE! IF THEY
    # DON'T TAKE AVERAGE FOR EACH SHOT, I SHOULDN'T EITHER! I SHOULD NOT! DON'T TAKE AVERAGE OVER ALL TIMES/FRAMES
    # FOR EACH SHOT, BUT JUST CONDENSE SHOTS AT SAME LOCATION INTO EACH OTHER WHILE KEEPING TIMES/FRAMES SEPARATE!

    # We want to try to create the mean value over all shots taken at same position for each time (frame) in the shot.
    # This creates an array of averages at each time, unique x pos, and unique y pos.
    # Try all the categorizing needed down here. (For example, store shot references in unique x, unique y grid)

    # Creates 4D array! The first two dimensions correspond to all combinations of unique x and y positions,
    #    the third dimension represents the nth shot taken at that unique positions
    #    and the fourth dimensions lists all the frames in that nth shot.
    isweep_xy_shots = [
        [[isweep_processed[shot] for shot in range(len(xy_shot_ref[i][j]))] for j in range(len(xy_shot_ref[i]))]
        for i in range(len(xy_shot_ref))]
    isweep_xy_shots_array = np.array(isweep_xy_shots)
    vsweep_xy_shots = [
        [[vsweep_processed[shot] for shot in range(len(xy_shot_ref[i][j]))] for j in range(len(xy_shot_ref[i]))]
        for i in range(len(xy_shot_ref))]
    vsweep_xy_shots_array = np.array(vsweep_xy_shots)
    # print("Shape of isweep_xy_shots_array:", isweep_xy_shots_array.shape)

    # Graph vsweep vs isweep for all frames in one shot (namely the first shot in the first unique x,y position)
    # plt.plot(vsweep_xy_shots_array[0, 0, 0], isweep_xy_shots_array[0, 0, 0])
    # plt.plot(isweep_xy_shots_array[0, 0, 0])
    # plt.show()

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


def isolate_plateaus(filename):  # Change to taking bias&current, then call near start of create_ranged_characteristic?

    bias, current = get_isweep_vsweep(filename)

    max_num_plateaus = 1

    # Should plateau_start_frames be a nested list?
    # plateau_start_frames = np.ndarray()

    # "Threshold for voltage quench slope"
    quench_slope = -1

    # Threshold for separating distinct voltage quench frames
    quench_diff = 10

    bias_gradient = np.gradient(bias, axis=-1)

    normalized_bias_gradient = bias_gradient / np.amax(bias_gradient, axis=-1, keepdims=True)

    # quench_frames = np.array((normalized_bias_gradient < quench_slope).nonzero())
    # print("Coordinates of quench frames:", quench_frames)

    # This line should create an array with same shape as bias (x,y,frames), but frames are simply their own index
    #    instead of a bias or current value
    frame_array = np.full(bias.shape, np.arange(bias.shape[-1]))
    # print("All frame indices at each x,y position", frame_array)
    # quench_frames_by_position = frame_array[..., normalized_bias_gradient < quench_slope]

    # Using list comprehension, this line fills each x,y position in array with a list of quench frames
    quench_frames = np.array([[frame_array[i, j, normalized_bias_gradient[i, j] < quench_slope]
                               for j in range(normalized_bias_gradient.shape[1])]
                              for i in range(normalized_bias_gradient.shape[0])])

    # print("This is the quench frame array:", quench_frames)

    # Using list comprehension, this line creates an array storing significant quench frames (plus the last one, which
    #    should also be significant) for each x,y position
    sig_quench_frames = np.array([[same_xy[(np.diff(same_xy) > quench_diff).tolist() + [True]]
                                   for same_xy in same_y]
                                  for same_y in quench_frames])

    # print("This is the significant quench frame array:", sig_quench_frames)

    # Sample for first position (0,0)
    plt.plot(bias[0, 0], 'b-', sig_quench_frames[0, 0], bias[0, 0, sig_quench_frames[0, 0]], 'ro')
    plt.show()


def create_ranged_characteristic(filename, start, end):
    bias, current = get_isweep_vsweep(filename)
    dimensions = len(bias.shape)
    zero_indices = tuple(np.zeros(dimensions - 1, dtype=int).tolist())
    # debug
    # print("Dimensions of incoming bias array:", bias.shape)
    # print("Zero_indices:", zero_indices)

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
    # debug
    # print(size)
    length = size[len(size) - 1]
    if num_points_each_side < 0:
        raise ValueError("Cannot smooth over negative number", num_points_each_side, "of points")
    if length < 2 * num_points_each_side:
        raise ValueError("Characteristic of", length, "data points is too short to take", num_points_each_side +
                         "-point average over")
    smooth_bias = numpy.zeros(size)
    smooth_current = numpy.zeros(size)

    for i in range(length):
        if i < num_points_each_side:
            smooth_bias[..., i] = numpy.mean(characteristic.bias[..., :2 * num_points_each_side])
            smooth_current[..., i] = numpy.mean(characteristic.current[..., :2 * num_points_each_side])
        elif i >= length - num_points_each_side:
            smooth_bias[..., i] = numpy.mean(characteristic.bias[..., -2 * num_points_each_side - 1:])
            smooth_current[..., i] = numpy.mean(characteristic.current[..., -2 * num_points_each_side - 1:])
        else:
            smooth_bias[..., i] = numpy.mean(
                characteristic.bias[..., i - num_points_each_side:i + num_points_each_side + 1])
            smooth_current[..., i] = numpy.mean(characteristic.current[...,
                                                i - num_points_each_side:i + num_points_each_side + 1])

    return Characteristic(u.Quantity(smooth_bias, u.V), u.Quantity(smooth_current, u.A))


isolate_plateaus('HDF5/8-3500A.hdf5')
# get_isweep_vsweep('HDF5/09_radial_line_25press_4kA_redo.hdf5')
