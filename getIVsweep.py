import numpy as np
import astropy.units as u
from bapsflib import lapd


def get_isweep_vsweep(filename, vsweep_bc, langmuir_probes, voltage_gain=100):  # TODO get voltage_gain from metadata?
    r"""
    Reads all sweep data (V-sweep and I-sweep) from HDF5 file Langmuir code.

    Parameters
    ----------
    :param filename: File path of HDF5 file from LAPD
    :param vsweep_bc: Board and channel number of vsweep data in HDF5 file
    :return: bias, current, x, y, dt: the relevant multidimensional sweep, position, and timestep
    """

    lapd_file = lapd.File(filename)

    # isweep_bcs = np.atleast_2d(isweep_bcs)
    isweep_bcs = np.atleast_1d(langmuir_probes[['board', 'channel']])

    vsweep_data = lapd_file.read_data(*vsweep_bc, silent=True)
    isweep_datas = [lapd_file.read_data(*isweep_bc, silent=True) for isweep_bc in isweep_bcs]
    vsweep_signal = vsweep_data['signal']
    isweep_signal = np.concatenate([isweep_data['signal'][np.newaxis, ...] for isweep_data in isweep_datas], axis=0)
    signal_length = vsweep_signal.shape[-1]
    # Above: isweep_signal has one extra dimension "in front" than vsweep signal, to represent different *probes*

    # Detect digitizer receptacle for each probe (based on physical port window on LAPD)
    probe_configs = lapd_file.controls['6K Compumotor'].configs
    port_receptacles = {probe_configs[probe]['probe']['port']:
                        probe_configs[probe]['receptacle']
                        for probe in probe_configs}
    motor_datas = [lapd_file.read_controls([("6K Compumotor", port_receptacles[port])], silent=True)
                   for port in langmuir_probes['port']]

    # TODO allow isweep motor datas to be different or check; for now, assume identical, and use only first motor data
    # for isweep_motor_data in motor_datas:
    positions, num_positions, shots_per_position = get_shot_positions(motor_datas[0])
    lapd_file.close()

    vsweep_signal = vsweep_signal.reshape(num_positions, shots_per_position, signal_length)
    isweep_signal = isweep_signal.reshape((-1, num_positions, shots_per_position, signal_length))

    # DON'T average bias and current for all shots at same position
    # vsweep_signal = vsweep_signal.mean(axis=-2)
    # isweep_signal = isweep_signal.mean(axis=-2)
    # Note: I may take the standard deviation across shots to approximate error for sweep curves, as done in MATLAB code

    ports = np.array([motor_data.info['controls']['6K Compumotor']['probe']['port'] for motor_data in motor_datas])
    resistances = np.reshape([langmuir_probes['resistance'][langmuir_probes['port'] == port] for port in ports],
                             [len(ports)] + [1 for _ in range(len(isweep_signal.shape) - 1)])

    # Convert to real units (not abstract)
    bias = vsweep_signal * voltage_gain * u.V
    currents = isweep_signal / resistances * u.A

    currents_dc_offset = np.mean(currents[..., -1000:], axis=-1, keepdims=True)
    currents -= currents_dc_offset

    # Determine up/down orientation of sweep by finding median current at a central shot; should be negative
    invert = np.sign(np.median(currents[:, int(num_positions / 2), :]))
    currents *= -invert

    dt = vsweep_data.dt
    return bias, currents, positions, dt, ports


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
