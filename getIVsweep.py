import numpy as np
import astropy.units as u
from bapsflib import lapd


def get_isweep_vsweep(filename, mar2022):
    r"""
    Reads all sweep data (V-sweep and I-sweep) from HDF5 file Langmuir code.
    :param filename: File path of HDF5 file from LAPD
    :return: bias, current, x, y, dt: the relevant multidimensional sweep, position, and timestep
    """

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
    shot_positions = np.round(motor_data['xyz'], 1)

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

    vsweep_signal = vsweep_signal.reshape((num_positions, shots_per_position, signal_length))
    isweep_signal = isweep_signal.reshape((num_positions, shots_per_position, signal_length))

    xy_at_positions = shot_positions[:, :2].reshape((num_positions, shots_per_position, 2))  # (x, y) at shots by pos.
    if not (np.amax(xy_at_positions, axis=1) == np.amin(xy_at_positions, axis=1)).all():
        raise ValueError("Non-uniform position values when grouping Langmuir probe data by position")

    # average bias and current for all shots at same position
    vsweep_signal = vsweep_signal.mean(axis=1)
    isweep_signal = isweep_signal.mean(axis=1)
    # Note: I may take the standard deviation across shots to approximate error for sweep curves, as done in MATLAB code

    lapd_file.close()

    dt = vsweep_data.dt
    bias, current = to_real_sweep_units(vsweep_signal, isweep_signal, mar2022)

    return bias, current, positions, dt


def to_real_sweep_units(bias, current, mar2022):
    r"""
    Parameters
    ----------
    :param bias: array
    :param current: array
    :return: bias and current array in real units
    """

    # The conversion factors from abstract units to real bias (V) and current values (A) are hard-coded in here.
    # NOTE: 2018 current data is inverted, while 2022 data is not.

    # Conversion factors taken from MATLAB code: Current = isweep / 11 ohms; Voltage = vsweep * 100
    gain = 100.  # voltage gain             # TODO get from HDF5 metadata
    resistance = 1.25 if mar2022 else 11.   # TODO get from HDF5 metadata
    invert = 1 if mar2022 else -1           # TODO get from HDF5 metadata?

    return bias * gain * u.V, current / resistance * u.A * invert
