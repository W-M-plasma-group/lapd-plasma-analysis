# Make sure to add code comments!
import numpy
from hdf5reader import *


def getIsweepVsweep(filename, nodesmin=-0.54, sample_sec=(100/16*10**6)**(-1)):

    file = open_HDF5(filename)

    motor_data = structures_at_path(file, '/Raw data + config/6K Compumotor')
    # print("Datasets in motor_data structure: "+str(motor_data["Datasets"]))

    probe_name = (motor_data["Datasets"])[0]
    probe_motion = file[probe_name]
    # print("Path to probe_motion dataset: "+probe_name)
    # print("Shape of probe_motion dataset: "+str(probe_motion.shape))
    print("Data type of probe_motion dataset: ", probe_motion.dtype)

    print("Rounding position data...")
    # print("Convert x field to list: ", list(probe_motion['x']))
    # print("Length of raw x data: ", len(probe_motion['x']))
    # print("Length of raw y data: ", len(probe_motion['y']))
    # print("Raw x data summary: ", probe_motion['x'])
    # print("Raw y data summary: ", probe_motion['y'])

    rnd = 0.100  # Change the x rounding step value here
    dec = 100
    num = int(rnd * dec)
    xround = tuple(round(dec*xin/num)*num/dec for xin in tuple(probe_motion['x']))
    yround = tuple(round(dec*yin/num)*num/dec for yin in tuple(probe_motion['y']))
    x = sorted(tuple(set(xround)))  # only the unique x position values
    y = sorted(tuple(set(yround)))  # only the unique y position values

    if len(x) == 1 and len(y) == 1:
        print("Only one position value. No plots can be made")
    elif len(x) == 1:
        print("Only one unique x value. Will only consider y position")
    elif len(y) == 1:
        print("Only one unique y value. Will only consider x position")

    # print("x: ", x)
    # print("y: ", y)

    xlength = len(x)
    ylength = len(y)

    print("Summary list of shots performed: ", probe_motion['Shot number'])
    shot_list = tuple(probe_motion['Shot number'])
    num_shots = len(shot_list)
    # print("List of shots performed: ", shotList)

    st_data = []

    # This part: list of links to nth smallest shot numbers (inside larger shot number array)

    # This part: (# unique x positions * # unique y positions) grid storing location? of shot numbers at that position



    # SKIP REMAINING X, Y POSITION DATA PROCESSING

    # SIS crate data
    sis_group = structures_at_path(file, '/Raw data + config/SIS crate/')
    print("Datasets in sis_data structure: " + str(sis_group["Datasets"]))

    isweep_data_path = (sis_group['Datasets'])[2]
    isweep_headers_path = (sis_group['Datasets'])[3]
    # print("Path to isweep data: " + isweep_data_path)
    # print("Path to isweep headers: " + isweep_headers_path)

    isweep_data_raw = file[isweep_data_path]
    isweep_headers_raw = file[isweep_headers_path]

    # print("Shape of isweep dataset: "+str(isweep_data_raw.shape))
    # print("Data type of isweep dataset: ", isweep_data_raw.dtype)
    # print("isweep dataset:", isweep_data_raw)
    # print("isweep headers:", isweep_headers_raw)
    # print("isweep data tuple: ", isweep_data_raw[:])
    # print("isweep headers tuple: ", isweep_headers_raw[:])

    vsweep_data_path = (sis_group['Datasets'])[4]
    vsweep_headers_path = (sis_group['Datasets'])[5]
    # print("Path to vsweep data: " + vsweep_data_path)
    # print("Path to vsweep headers: " + vsweep_headers_path)

    vsweep_data_raw = file[vsweep_data_path]
    vsweep_headers_raw = file[vsweep_headers_path]
    # print("vsweep dataset:", vsweep_data_raw)
    # print("vsweep headers:", vsweep_headers_raw)
    # print("vsweep data tuple: ", vsweep_data_raw[:])
    # print("vsweep headers tuple:", vsweep_headers_raw[:])

    raw_size = (len(isweep_data_raw), len(isweep_data_raw[0]))
    print("Raw size:", raw_size)

    isweep_raw_array = np.array(isweep_data_raw)
    vsweep_raw_array = np.array(vsweep_data_raw)

    # WHEN PROBE MOTION CODE IS COMMENTED OUT, USE THE FOLLOWING DEFINITION FOR NUMBER OF SHOTS TAKEN
    # num_shots = raw_size[0]

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

    print("Finished reading in scales and offsets from headers")

    # (SKIP AREAL PLOT CODE; GO TO RADIAL PLOT CODE)

    # Process isweep, vsweep data (decompress); raw_size[0] should be number of shots, raw_size[1] should be number
    #   of measurements per shot

    print("Decompressing raw data...")
    isweep_processed = np.ndarray((raw_size[0], raw_size[1]), float)
    vsweep_processed = np.ndarray((raw_size[0], raw_size[1]), float)
    # isweep_sums = np.ndarray((1065,), float)
    # isweep_sumsq = np.ndarray((1065,), float)

    for i in range(raw_size[0]):
        isweep_processed[i] = isweep_scales_array[i] * isweep_raw_array[i] + isweep_offsets_array[i]
        vsweep_processed[i] = vsweep_scales_array[i] * vsweep_raw_array[i] + vsweep_offsets_array[i]

    print("Finished decompressing compressed isweep and vsweep data")

    ###
    # Take average and standard deviation for isweep and vsweep values at each shot over time MAYBE I SHOULDN'T TAKE
    # AVERAGE FOR EACH SHOT! AFTER ALL, ISN'T EACH SHOT A SERIES OF _DIFFERENT_ BIASES? CONSULT MATLAB CODE! IF THEY
    # DON'T TAKE AVERAGE FOR EACH SHOT, I SHOULDN'T EITHER! I SHOULD NOT! DON'T TAKE AVERAGE OVER ALL TIMES/FRAMES
    # FOR EACH SHOT, BUT JUST CONDENSE SHOTS AT SAME LOCATION INTO EACH OTHER WHILE KEEPING TIMES/FRAMES SEPARATE!

    # isweep_means = np.divide(np.sum(isweep_processed, 1), np.full((raw_size[0],), raw_size[1]))
    # vsweep_means = np.divide(np.sum(vsweep_processed, 1), np.full((raw_size[0],), raw_size[1]))

    # We want to try to create the mean value over all shots taken at same position for each time (frame) in the shot.
    # This creates an array of averages at each time, unique x pos, and unique y pos.
    # Try all the categorizing needed down here. (For example, store shot references in unique x, unique y grid)

    # print("x:", x)
    # print("y:", y)
    # print("shot_list:", shot_list)
    # isweep_xy_shot_ref = [[[0]]*ylength]*xlength
    xy_shot_ref = [[[] for j in range(ylength)] for i in range(xlength)]
    for i in range(num_shots):
        xy_shot_ref[x.index(xround[i])][y.index(yround[i])].append(i)
    # print("xy shot refs:", xy_shot_ref)

    isweep_xy_shots = [
        [[isweep_processed[shot] for shot in range(len(xy_shot_ref[i][j]))] for j in range(len(xy_shot_ref[i]))]
        for i in range(len(xy_shot_ref))]
    isweep_xy_shots_array = np.array(isweep_xy_shots)
    vsweep_xy_shots = [
        [[vsweep_processed[shot] for shot in range(len(xy_shot_ref[i][j]))] for j in range(len(xy_shot_ref[i]))]
        for i in range(len(xy_shot_ref))]
    vsweep_xy_shots_array = np.array(vsweep_xy_shots)

    # Graph vsweep vs isweep for all frames in one shot (namely the first shot in the first unique x,y position)
    # plt.plot(vsweep_xy_shots_array[0, 0, 0], isweep_xy_shots_array[0, 0, 0])
    # plt.plot(isweep_xy_shots_array[0, 0, 0])
    # plt.show()
    #   plt.show(block=True)
    #   plt.interactive(False)

    isweep_means = np.mean(isweep_xy_shots, 2)
    vsweep_means = np.mean(vsweep_xy_shots, 2)

    # Graph vsweep vs isweep for average of all shots (15) in first unique x,y position?
    # plt.plot(vsweep_means[0, 0], isweep_means[0, 0])
    # plt.plot(vsweep_means[0, 0, 12500:13100], np.multiply(isweep_means[0, 0, 12500:13100], -1))
    # WHY THIS -1 TO GET CORRECT GRAPH ORIENTATION?
    # plt.show(block=True)
    # plt.interactive(False)

    """

    print("Mean isweep value for each shot:", isweep_means)
    print("Mean vsweep value for each shot:", vsweep_means)
    # isweep_sdevs = (sum(shot))

    # for i in range(raw_size[0]):
    #    for j in range(raw_size[1]):
    #        print(" ")
    """

    # TRY TO CREATE A CHARACTERISTIC OBJECT?
    # THIS RETURNS AN INVERTED-CURRENT CHARACTERISTIC! (As of 6/4/21)

    # THE RETURN FUNCTION HAS BEEN TEMPORARILY CHANGED TO RETURN ONLY A SLICE OF A CHARACTERISTIC AT ONE X,Y POSITION
    # MUST CONVERT TO REAL UNITS FIRST!!! OTHERWISE ARE IN ABSTRACT UNITS. MATLAB CODE GIVES VALUES:
    #    REAL CURRENT = I / 11 OHMS; REAL BIAS = V * 100

    characteristic = Characteristic(u.Quantity(vsweep_means[0, 0, 25000:25620] * 100, u.V),
                                    u.Quantity(isweep_means[0, 0, 25000:25620] * (-1./11.), u.A))
    # characteristic = Characteristic(u.Quantity(vsweep_means, u.V), u.Quantity(-1*isweep_means, u.A))

    """
    # characteristic.plot()
    # plt.plot([0, 1])
    plt.show(block=True)
    plt.interactive(False)

    # print("Padded limit for characteristic:", characteristic.get_padded_limit(0.5))
    print("Characteristic V data:", characteristic.bias)
    print("Characteristic I data:", characteristic.current)

    """

    file.close()
    return characteristic


def smooth_characteristic(characteristic, num_points_each_side):

    size = characteristic.bias.shape
    """LEO ADDITION TO FOLLOW"""
    print(size)
    length = size[len(size) - 1]
    if num_points_each_side < 0:
        raise ValueError("Cannot smooth over a negative number of points!")
    if length < 2*num_points_each_side:
        raise ValueError("Characteristic is too short to take average over!")
    smooth_bias = numpy.zeros(size)
    smooth_current = numpy.zeros(size)

    for i in range(length):
        if i < num_points_each_side:
            smooth_bias[..., i] = numpy.mean(characteristic.bias[..., :2*num_points_each_side])
            smooth_current[..., i] = numpy.mean(characteristic.current[..., :2*num_points_each_side])
        elif i >= length - num_points_each_side:
            smooth_bias[..., i] = numpy.mean(characteristic.bias[..., -2*num_points_each_side-1:])
            smooth_current[..., i] = numpy.mean(characteristic.current[..., -2*num_points_each_side-1:])
        else:
            smooth_bias[..., i] = numpy.mean(characteristic.bias[..., i-num_points_each_side:i+num_points_each_side+1])
            smooth_current[..., i] = numpy.mean(characteristic.current[...,
                                                i-num_points_each_side:i+num_points_each_side+1])

    return Characteristic(u.Quantity(smooth_bias, u.V), u.Quantity(smooth_current, u.A))

#getIsweepVsweep('HDF5/09_radial_line_25press_4kA_redo.hdf5')
