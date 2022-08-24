import numpy as np
import astropy.units as u
from bapsflib import lapd


def get_isweep_vsweep(filename, vsweep_bc, isweep_bcs, port_resistances, orientation):
    r"""
    Reads all sweep data (V-sweep and I-sweep) from HDF5 file Langmuir code.

    Parameters
    ----------
    :param filename: File path of HDF5 file from LAPD
    :param vsweep_bc:
    :param isweep_bcs:
    :param port_resistances:
    :param orientation:
    :return: bias, current, x, y, dt: the relevant multidimensional sweep, position, and timestep
    """

    lapd_file = lapd.File(filename)

    run_name = lapd_file.info['run name']

    # isweep_channel = 2  # 2 = port 27, 3 = port 43
    isweep_bcs = np.atleast_2d(isweep_bcs)
    isweep_receptacles = {2: 1, 3: 2}  # TODO hardcoded

    vsweep_data = lapd_file.read_data(*vsweep_bc, silent=True)
    isweep_datas = [lapd_file.read_data(*isweep_bc, silent=True) for isweep_bc in isweep_bcs]
    vsweep_signal = vsweep_data['signal']
    isweep_signal = np.concatenate([isweep_data['signal'][np.newaxis, ...] for isweep_data in isweep_datas], axis=0)
    signal_length = vsweep_signal.shape[-1]
    # Above: isweep_signal has one extra dimension "in front" than vsweep signal, to represent different *probes*

    motor_datas = [lapd_file.read_controls([("6K Compumotor", isweep_receptacles[isweep_channel])], silent=True)
                   for _, isweep_channel in isweep_bcs]
    # TODO allow isweep motor datas to be different or check; for now, assume identical, and use only first motor data
    # for isweep_motor_data in motor_datas:
    positions, num_positions, shots_per_position = get_shot_positions(motor_datas[0])

    vsweep_signal = vsweep_signal.reshape(num_positions, shots_per_position, signal_length)
    isweep_signal = isweep_signal.reshape((-1, num_positions, shots_per_position, signal_length))

    # average bias and current for all shots at same position
    vsweep_signal = vsweep_signal.mean(axis=-2)
    isweep_signal = isweep_signal.mean(axis=-2)
    # Note: I may take the standard deviation across shots to approximate error for sweep curves, as done in MATLAB code

    lapd_file.close()

    ports = np.array([motor_data.info['controls']['6K Compumotor']['probe']['port'] for motor_data in motor_datas])
    resistances_shape = [len(ports)] + [1 for _ in range(len(isweep_signal.shape) - 1)]
    resistances = np.reshape([port_resistances[port] for port in ports], resistances_shape)

    dt = vsweep_data.dt
    bias, currents = to_real_sweep_units(vsweep_signal, isweep_signal, resistances, orientation)

    return bias, currents, positions, dt, run_name, ports


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


def to_real_sweep_units(bias, current, resistances, orientation):
    r"""
    Parameters
    ----------
    :param orientation:
    :param bias: array
    :param current: array
    :param resistances: array
    :return: bias and current array in real units
    """

    # The conversion factors from abstract units to real bias (V) and current values (A) are hard-coded in here.
    # NOTE: 2018 current data is inverted, while 2022 data is not.

    # Conversion factors taken from MATLAB code: Current = isweep / 11 ohms; Voltage = vsweep * 100
    gain = 100.  # voltage gain                                         # TODO get from HDF5 metadata
    # resistance = np.array([[[2.10]], [[1.25]]]) if mar2022 else 11.
    invert = 1 if orientation else -1                                       # TODO get from HDF5 metadata?

    return bias * gain * u.V, current / resistances * u.A * invert
