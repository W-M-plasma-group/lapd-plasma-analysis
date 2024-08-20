import numpy as np
import astropy.units as u
from bapsflib import lapd
from warnings import warn


def get_sweep_voltage(filename, vsweep_bc, voltage_gain):
    """
    Reads the voltage applied to Langmuir probes from an HDF5 file.
    Note that one single sweep voltage signal is applied to all Langmuir probes in each experiment.

    Parameters
    ----------
    filename : `str`
        File path of the HDF5 file (should end in '.hdf5')

    vsweep_bc : `tuple` or `list`

        The board and channel number for the V-sweep data. The format is
        (board number, channel number)-- this should be the output of
        `lapd_plasma_analysis.langmuir.configurations.get_vsweep_bc`

    voltage_gain : `float`
        Value of scaling constant for getting real bias voltage from
        V-sweep data (the output of `lapd_plasma_analysis.langmuir.configurations.get_voltage_gain`)

    Returns
    -------
    bias : `astropy.units.Quantity`
        Array of applied sweep voltage with dimensions of position, shot, and frame, e.g. of shape (71, 15, 55296).
    dt : `astropy.units.Quantity`
        Timestep in between individual Langmuir probe voltage and current measurements, sometimes referred to
        as "frames". One frame is only a tiny part of a single sweep curve, so the time in between Langmuir probe
        temperature and density measurements (the time in between voltage/current sweeps) is much larger.

    See Also
    --------
    lapd_plasma_analysis.langmuir.configurations.get_vsweep_bc
    lapd_plasma_analysis.langmuir.configurations.get_voltage_gain
    get_sweep_current
    """
    with lapd.File(filename) as lapd_file:
        vsweep = lapd_file.read_data(*vsweep_bc, silent=True)

    dt = vsweep.dt
    vsweep = vsweep['signal']

    # Convert to real units (not abstract)
    bias = vsweep * voltage_gain * u.V

    return bias, dt


def get_sweep_current(filename, isweep_metadata, orientation):
    """
    Reads the current collected by a Langmuir probe from an HDF5 file.
    Note that one sweep current signal is collected for every face on every Langmuir probe in each experiment.

    Parameters
    ----------
    filename : `str`
        file path of HDF5 file from LAPD (WIP)
    isweep_metadata : `numpy.ndarray` of `int` and `str`
        structured array of board, channel, receptacle, port, face, resistance, and area for each isweep signal.
        This should be the output of `lapd_plasma_analysis.langmuir.configurations.get_langmuir_config`.
    orientation : {+1, -1}
        +1 or -1, depending on if I_sweep should be inverted before analysis (WIP). This is 
        the output of `lapd_plasma_analysis.langmuir.configurations.get_orientation`.

    Returns
    -------
    bias, current, positions, dt: v_sweep array, i_sweep array, position array, and timestep amount (WIP)

    See Also
    --------
    lapd_plasma_analysis.langmuir.configurations.get_langmuir_config :
        (WIP)
    lapd_plasma_analysis.langmuir.configurations.get_orientation :
        (WIP)
    get_sweep_voltage
    """

    with lapd.File(filename) as lapd_file:
        # TODO revise
        #  isweep_metadata is a structured Numpy array storing the digitizer and motor information for Langmuir probes.
        #  It can have one or zero dimensions. Zero dimensions is possible if there is only one source of sweep current.

        isweep = lapd_file.read_data(isweep_metadata['board'], isweep_metadata['channel'], silent=True)['signal']

        # TODO revise List of motor data about the probe associated with each isweep signal.
        #   Motor data may be repeated, for example if two isweep signals were taken using two faces on the same probe.
        motor_data = lapd_file.read_controls([("6K Compumotor", isweep_metadata['receptacle'])], silent=True)

    # Convert to real units (not abstract)
    current = isweep / isweep_metadata['resistance'] / isweep_metadata['gain'] * u.A

    # Subtract out average of last thousand current measurements for each isweep signal,
    #   as this should be a while after the plasma has dissipated and thus be equal to zero.
    #   This eliminates any persistent DC offset current from the probe.
    current -= np.mean(current[..., -1000:], axis=-1, keepdims=True)

    # Up-down orientation of sweep is hardcoded for an entire experiment, e.g. November_2022, in configurations.py
    current *= orientation

    return current, motor_data


def get_shot_positions(isweep_motor_data):
    """
    Accesses probe position data. (WIP)

    Parameters
    ----------
    isweep_motor_data : Motor data object from one or more probes, retrieved from an HDF5 file using the
    `bapsflib.lapd.File.read_controls` function, for example as in `get_sweep_current`. (WIP)

    Returns
    -------
    positions : `numpy.ndarray` of `float`
        (WIP confirm)
        2D array that lists all unique (x, y) positions achieved during probe motion, in the order they were reached.
        The two columns store the x and y position of each unique (x, y) position.

    num_positions : `int`
        Number of unique (x, y) positions achieved during probe motion. This is the length of `positions` above.

    shots_per_position : `int`
        The number of shots that were taken at each unique (x, y) position. If the number of shots at each position
        differs by position, then this is the minimum number of shots taken at any unique position.
        For example, if 20 shots were conducted at position (x1, y1) and 8 shots at position (x2, y2),
        then `shots_per_position` is limited to 8.

    selected_shots : `numpy.ndarray` of `int`
        1D array of the indices of shots that are retained for future analysis.
        Length is `num_positions` * `shots_per_position`. If the number of shots at each position
        differs by position, then at each position, all shots after the first `shots_per_positions` are ignored.
        Therefore, each unique position in `positions` corresponds to a block of `shots_per_position` elements
        in `selected_shots`; each entry is an index for one shot taken at that unique position.

    Examples
    --------
    If five shots were conducted with (x, y) positions ((0, 0), (1, 2), (0, 0), (1, 2), (1, 2), (1, 2), (0, 0)), then
        - `positions` is ((0, 0), (1, 2)),
        - `num_positions` is 2,
        - `shots_per_position` is 3 (as there are 4 shots at (1, 2) but only 3 at (0, 0)), and
        - `selected_shots` is (0, 2, 6, 1, 3, 4), as
        the first block of `shots_per_position` elements of `selected_shots` gives indices for
        the shots at (0, 0), which is the first position in `positions`, and
        the second block of `shots_per_position` elements of `selected_shots` gives indices
        for the shots at (1, 2), which is the second position in `positions`.
    """
    num_shots = len(isweep_motor_data['shotnum'])
    shot_positions = np.round(isweep_motor_data['xyz'], 1)

    z_positions = shot_positions[:, 2]
    if np.min(z_positions) != np.max(z_positions):
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
            indices_of_shots_at_this_position = np.all(shot_positions[:, :2] == positions[i], axis=1).nonzero()[0]

            # Save indices of shots that exclude those in excess of the first shots_per_position at that position
            indices_of_shots_at_this_position = indices_of_shots_at_this_position[:shots_per_position]
            selected_shots[shots_per_position * i:shots_per_position * (i + 1)] = indices_of_shots_at_this_position

        shot_positions = shot_positions[selected_shots]

    else:
        shots_per_position = int(num_shots // num_positions)
        selected_shots = np.arange(num_shots)

    xy_at_positions = shot_positions[:, :2].reshape((num_positions, shots_per_position, 2))  # (x, y) at shots by pos.
    if not (np.max(xy_at_positions, axis=1) == np.min(xy_at_positions, axis=1)).all():
        raise ValueError("Non-uniform position values when grouping sweep data by position")

    return positions, num_positions, shots_per_position, selected_shots
