import numpy as np
import astropy.units as u
from bapsflib import lapd
from warnings import warn


def get_isweep_vsweep(filename, vsweep_bc, isweep_metadatas, voltage_gain, orientation):
    r"""
    Reads all sweep data (V-sweep and I-sweep) from HDF5 file Langmuir code.

    Parameters
    ----------
    :param filename: file path of HDF5 file from LAPD
    :param vsweep_bc: board and channel number of vsweep data in HDF5 file
    :param isweep_metadatas: structured array of board, channel, receptacle, port, face, resistance, and area
    for each isweep signal
    :param voltage_gain: numerical value of scaling constant for getting real bias voltages from abstract vsweep data
    :param orientation: +1 or -1, depending on if Isweep should be inverted before analysis
    :return: bias, currents, positions, dt: v_sweep array, i_sweeps array, position array, and timestep amount
    """

    with lapd.File(filename) as lapd_file:
        isweep_bcs = np.atleast_1d(isweep_metadatas[['board', 'channel']])

        vsweep = lapd_file.read_data(*vsweep_bc, silent=True)
        isweep = [lapd_file.read_data(*isweep_bc, silent=True) for isweep_bc in isweep_bcs]
        dt = vsweep.dt

        vsweep = vsweep['signal']
        isweep = np.concatenate([isweep_signal['signal'][np.newaxis, ...] for isweep_signal in isweep], axis=0)
        # Above: isweep_signal has one extra dimension "in front" compared to vsweep signal,
        #  to represent different probes or probe faces; ordered by (board, channel) as listed in isweep_metadatas
        signal_length = vsweep.shape[-1]

        # List of motor data about the probe associated with each isweep signal.
        #   Motor data may be repeated, for example if two isweep signals were taken using two faces on the same probe.
        motor_datas = [lapd_file.read_controls([("6K Compumotor", isweep_metadata['receptacle'])], silent=True)
                       for isweep_metadata in isweep_metadatas]

    # TODO allow isweep motor datas to be different or check
    #
    #  for now, assume identical and use only first motor data
    num_isweep = len(isweep_metadatas)
    positions, num_positions, shots_per_position, selected_shots = get_shot_positions(motor_datas[0])
    vsweep = vsweep[selected_shots,    ...]
    isweep = isweep[:, selected_shots, ...]

    vsweep = vsweep.reshape(num_positions,              shots_per_position, signal_length)
    isweep = isweep.reshape((num_isweep, num_positions, shots_per_position, signal_length))

    scale_shape = [num_isweep] + [1 for _ in range(len(isweep.shape) - 1)]
    resistances = np.reshape(isweep_metadatas['resistance'], scale_shape)
    gains = np.reshape(isweep_metadatas['gain'],             scale_shape)

    # Convert to real units (not abstract)
    bias = vsweep * voltage_gain * u.V
    currents = isweep / resistances / gains * u.A

    # Subtract out average of last thousand current measurements for each isweep signal,
    #   as this should be a while after the plasma has dissipated and thus be equal to zero.
    #   This eliminates any persistent DC offset current from the probe.
    currents -= np.mean(currents[..., -1000:], axis=-1, keepdims=True)

    # bias dimensions:               position, shot, frame   (e.g.    (71, 15, 55296))
    # currents dimensions:   isweep, position, shot, frame   (e.g. (1, 71, 15, 55296))

    # Up-down orientation of sweep is hardcoded for an entire experiment, e.g. November_2022, in configurations.py
    currents *= orientation

    ports = np.array([motor_data.info['controls']['6K Compumotor']['probe']['port'] for motor_data in motor_datas])
    assert np.all(ports == isweep_metadatas['port'])  # otherwise, ports in configurations.py do not match motor data

    return bias, currents, positions, dt


def get_shot_positions(isweep_motor_data):
    num_shots = len(isweep_motor_data['shotnum'])
    shot_positions = np.round(isweep_motor_data['xyz'], 1)

    z_positions = shot_positions[:, 2]
    if np.amin(z_positions) != np.amax(z_positions):
        raise ValueError("Varying z-position when only x and/or y variation expected")
    # Generate list of all unique (x, y) positions; save z-position for later?
    positions, inverse, counts = np.unique(shot_positions[:, :2], axis=0, return_inverse=True, return_counts=True)
    num_positions = len(positions)
    if num_shots % num_positions != 0:

        shots_per_position = np.min(counts)
        message = (f"{num_shots} shots do not evenly divide into {num_positions} unique positions. "
                   f"Only considering first {shots_per_position} shots per position; others are discarded.")
        warn(message)
        new_num_shots = num_positions * shots_per_position
        selected_shots = np.zeros((new_num_shots,), dtype=int)
        for i in range(len(positions)):
            # For each unique position, return shot indices located at that position
            shot_indices_for_position = np.all(shot_positions[:, :2] == positions[i], axis=1).nonzero()[0]

            # Produce indices of shots that exclude those in excess of the first shots_per_position at that position
            shot_indices_for_position = shot_indices_for_position[:shots_per_position]
            selected_shots[shots_per_position * i:shots_per_position * (i + 1)] = shot_indices_for_position

        shot_positions = shot_positions[selected_shots]

    else:
        shots_per_position = int(num_shots // num_positions)
        selected_shots = np.arange(num_shots)

    xy_at_positions = shot_positions[:, :2].reshape((num_positions, shots_per_position, 2))  # (x, y) at shots by pos.
    if not (np.max(xy_at_positions, axis=1) == np.min(xy_at_positions, axis=1)).all():
        raise ValueError("Non-uniform position values when grouping sweep data by position")

    return positions, num_positions, shots_per_position, selected_shots
