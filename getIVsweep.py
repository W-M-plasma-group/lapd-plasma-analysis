import numpy as np
import astropy.units as u
from bapsflib import lapd


def get_isweep_vsweep(filename, vsweep_bc, langmuir_probes):  # TODO get voltage_gain from metadata?
    r"""
    Reads all sweep data (V-sweep and I-sweep) from HDF5 file Langmuir code.

    Parameters
    ----------
    :param filename: File path of HDF5 file from LAPD
    :param vsweep_bc: Board and channel number of vsweep data in HDF5 file
    :param langmuir_probes: structure array of board, channel, receptacle, port, resistance, and area for each probe
    :return: bias, currents, x, y, dt: the relevant multidimensional v_sweep, i_sweeps, position, and timestep
    """

    lapd_file = lapd.File(filename)

    isweep_bcs = np.atleast_1d(langmuir_probes[['board', 'channel']])

    vsweep_data = lapd_file.read_data(*vsweep_bc, silent=True)
    isweep_datas = [lapd_file.read_data(*isweep_bc, silent=True) for isweep_bc in isweep_bcs]
    vsweep_signal = vsweep_data['signal']
    isweep_signal = np.concatenate([isweep_data['signal'][np.newaxis, ...] for isweep_data in isweep_datas], axis=0)
    signal_length = vsweep_signal.shape[-1]
    # Above: isweep_signal has one extra dimension "in front" compared to vsweep signal, to represent different *probes*

    motor_datas = [lapd_file.read_controls([("6K Compumotor", langmuir_probe['receptacle'])], silent=True)
                   for langmuir_probe in langmuir_probes]
    # TODO allow isweep motor datas to be different or check; for now, assume identical, and use only first motor data
    # for isweep_motor_data in motor_datas:
    positions, num_positions, shots_per_position = get_shot_positions(motor_datas[0])
    lapd_file.close()

    vsweep_signal = vsweep_signal.reshape(num_positions, shots_per_position, signal_length)
    isweep_signal = isweep_signal.reshape((-1, num_positions, shots_per_position, signal_length))

    ports = np.array([motor_data.info['controls']['6K Compumotor']['probe']['port'] for motor_data in motor_datas])
    resistances_shape = [len(ports)] + [1 for _ in range(len(isweep_signal.shape) - 1)]
    resistances = np.reshape([langmuir_probes['resistance'][langmuir_probes['port'] == port] for port in ports], resistances_shape)

    bias, currents = to_real_sweep_units(vsweep_signal, isweep_signal, resistances)
    currents_dc_offset = np.mean(currents[..., -1000:], axis=-1, keepdims=True)
    currents -= currents_dc_offset

    # bias dimensions:            position, shot, frame  (e.g.    (71, 15, 55296))
    # currents dimensions:  port, position, shot, frame  (e.g. (1, 71, 15, 55296))

    # Determine up/down orientation of sweep by finding median current at a central shot; should be negative
    invert = np.sign(np.median(currents[:, int(num_positions / 2), :, :]))
    currents *= -invert

    dt = vsweep_data.dt
    return bias, currents, positions, dt, ports


def detect_shot_orientation(mid_shot_currents):
    return np.mean(mid_shot_currents) < 0


def get_shot_positions(isweep_motor_data):

    num_shots = len(isweep_motor_data['shotnum'])
    shot_positions = np.round(isweep_motor_data['xyz'], 1)

    z_positions = shot_positions[:, 2]
    if np.amin(z_positions) != np.amax(z_positions):
        raise ValueError("Varying z-position when only x and/or y variation expected")
    # save z-position for later?
    positions = np.unique(shot_positions[:, :2], axis=0)  # list of all unique (x, y) positions
    num_positions = len(positions)
    if num_shots % num_positions != 0:
        raise ValueError("Number of shots " + str(num_shots) +
                         " does not evenly divide into " + str(num_positions) + " unique positions")
    shots_per_position = int(num_shots // num_positions)

    xy_at_positions = shot_positions[:, :2].reshape((num_positions, shots_per_position, 2))  # (x, y) at shots by pos.
    if not (np.amax(xy_at_positions, axis=1) == np.amin(xy_at_positions, axis=1)).all():
        raise ValueError("Non-uniform position values when grouping Langmuir probe data by position")

    return positions, num_positions, shots_per_position


def to_real_sweep_units(vsweep, isweep, resistances):
    r"""
    Convert raw sweep probe data into real bias and current measurements.

    Parameters
    ----------
    :param vsweep: array
    :param isweep: array
    :param resistances: array, resistance of each isweep probe
    :return: bias and current array in real units
    """

    # The conversion factors from abstract units to real bias (V) are hard-coded in here.
    # NOTE: 2018 current data is inverted, while 2022 data is not.

    # Conversion factors taken from MATLAB code: Current = isweep / 11 ohms; Voltage = vsweep * 100
    gain = 100.  # voltage gain                                         # TODO get from HDF5 metadata

    return vsweep * gain * u.V, isweep / resistances * u.A

