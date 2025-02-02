import numpy as np
from bapsflib import lapd
import astropy.units as u


def get_config_id(exp_name):
    r"""Gives the config ID for a given experiment.

    Associates each experiment to an integer (the config ID) which is used
    later to index hard-coded lists of parameters potentially unique to each experiment
    (e.g. the plasma ion species, board and channel numbers for different probes).

    Parameters
    ----------
    exp_name : `str`
        The month and year of the experiment (e.g. `'March_2022'`). Should be
        `lapd_plasma_analysis.experimental.get_exp_params(hdf5_path)['Exp name']` for
        the relevant file path `hdf5_path` (`str`) to an HDF5 file.

    Returns
    -------
    `int`
        The config ID (based off of the index of `exp_name` in a list)

    Raises
    ------
    `ValueError`
        If `exp_name` does not have an assigned config ID


    """
    valid_configs = ["April_2018", "March_2022", "November_2022", "January_2024"]

    try:
        return valid_configs.index(exp_name)
    except ValueError:
        raise NotImplementedError("Configuration data has not yet been added for experiment series " + repr(exp_name)
                                  + ". Acceptable configurations are " + str(valid_configs)
                                  + ". Please contact the maintainer of this code repository to update the experiment "
                                  + "configs or, if you like, you can manually update configurations.py.")


def get_vsweep_bc(config_id):
    r"""Obtains the board and channel number for the V-sweep data.

    The board and channel numbers indicate where the data collected from the
    Langmuir probes at a certain port was sent to to be recorded and stored. They
    uniquely determine from where in the HDF5 file data can be retrieved.

    Parameters
    ----------
    config_id : `int`
        The output of `lapd_plasma_analysis.langmuir.configurations.get_config_id` for
        the relevant experiment name

    Returns
    -------
    `tuple`
        (board number, channel number)

    """
    vsweep_bcs = [(1, 3),   # April_2018
                  (1, 1),   # March_2022
                  (1, 1),   # November_2022
                  (1, 1)]   # January_2024
    return vsweep_bcs[config_id]


def get_voltage_gain(config_id):
    r"""Gets the voltage gain for the V-sweep data.

    Parameters
    ----------
    config_id : `int`
        The output of 'lapd_plasma_analysis.langmuir.configurations.get_config_id' for
        the relevant experiment name

    Returns
    -------
    `int` or `float`
        Voltage gain

    """
    # TODO get from HDF5 metadata?
    return (100, 100, 100, 100)[config_id]
    # Note: here, gain refers to the inverse gain applied. Multiply by this gain to undo.


def get_langmuir_config(hdf5_path, config_id):
    r"""Obtains a dictionary of configuration parameters for a given experiment.

    Gets a list of configuration settings for the Langmuir probe corresponding to a
    specific dataset (`hdf5_path`) and experiment (specified by `config_id`). These
    settings are primarily hard-coded into this function based off of the config ID,
    except `receptacle` which is the output of
    `lapd_plasma_analysis.langmuir.configuration.get_ports_receptacles(hdf5_path)`.

    Parameters
    ----------
    hdf5_path : `str`
        Path to the relevant HDF5 file-- this should end with `'.hdf5'`

    config_id : `int`
        The output of 'lapd_plasma_analysis.langmuir.configurations.get_config_id' for
        the relevant experiment name

    Returns
    -------
    `numpy.array`
        `(board, channel, receptacle, port, face, resistance, area, gain)`.
        `board` and `channel` are the board and channel numbers for the Langmuir probes. These
            determine from where in the HDF5 file data can be retrieved.
        `receptacle`, `port`, and `face` together specify where the Langmuir probe was
            situated along LAPD (`port`) and which face on the probe corresponds to the data
            to be found at the given board and channel numbers `face`. `receptacle` is a unique
            identifier for every face of every probe used.
        `resistance`, `area`, and `gain` are the resistance, area, and gain of the probe as
            `float`, `astropy.units.Quantity`, and `float` respectively. The gain of the probe
            should divide the data to recover physical voltages.


    """

    # each list in tuple corresponds to an experiment series;
    # each tuple in list corresponds to configuration data for a single probe used in those experiments
    # -1 is placeholder; what each entry corresponds to is given in 'dtype' parameter below
    langmuir_probe_configs = (
                              [(1, 2, -1, 25, "", 11.,  1 * u.mm ** 2, 1)],    # April_2018

                              [(1, 2, -1, 27, "", 1.25, 1 * u.mm ** 2, 1),     # March_2022
                               (1, 3, -1, 43, "", 2.10, 1 * u.mm ** 2, 1)],

                              [(1, 2, -1, 29, "", 2.20, 2 * u.mm ** 2, 1),     # November_2022
                               (1, 3, -1, 35, "", 2.20, 2 * u.mm ** 2, 1)],

                              [(1, 2, -1, 20, "L", 1.,  2 * u.mm ** 2, 0.87 / 2),     # January_2024
                               (1, 3, -1, 20, "R", 1.,  2 * u.mm ** 2, 1.07 / 2),
                               (1, 4, -1, 27, "L", 1.,  4 * u.mm ** 2, 1.124 / 2),
                               (1, 7, -1, 27, "R", 1.,  4 * u.mm ** 2, 0.96 / 2)]
                              )

    langmuir_configs_array = np.array(langmuir_probe_configs[config_id], dtype=[('board', int),
                                                                                ('channel', int),
                                                                                ('receptacle', int),
                                                                                ('port', int),
                                                                                ('face', 'U10'),
                                                                                ('resistance', float),
                                                                                ('area', u.Quantity),
                                                                                ('gain', float)])  # see note below
    # Note: "gain" here refers to what was gained before saving data. Divide data by the gain to undo.
    # (End of hardcoded probe configuration data)

    ports_receptacles = get_ports_receptacles(hdf5_path)
    langmuir_configs_array['receptacle'] = [ports_receptacles[port] for port in langmuir_configs_array['port']]
    return langmuir_configs_array


def get_ports_receptacles(hdf5_path):
    """ Returns a map between port numbers and 6K Compumotor receptacles based on HDF5 file data. """
    with lapd.File(hdf5_path) as fi:
        configs = fi.controls['6K Compumotor'].configs
        return {configs[probe]['probe']['port']: configs[probe]['receptacle'] for probe in configs}


def get_ion(run_name: str):
    r"""Gives the ion species for a given plasma.

    Extracts the ion species used for a given run using its label `run_name` by
    determining if `run_name` contains the string `h2` corresponding to Deuterium
    or not.

    Parameters
    ----------
    run_name : `str`
        The `'Run name'` entry in the dictionary returned by
        `lapd_plasma_analysis.experimental.get_exp_params`

    Returns
    -------
    `str` :
        The ion species, either `'H+'` or `'He-4+'`

    """
    if "h2" in run_name.lower():
        print("Assuming fully dissociated hydrogen (H+)")
        return "H+"
    else:
        return "He-4+"


def get_orientation(config_id):
    r"""Finds the current inversion convention for a given experiment.

    Returns 1 if the data is already stored with current values negated, and
    returns -1 if the current data is non-inverted. Enables treatment of all
    current values with the negated convention. (double check that this is right)

    Parameters
    ----------
    config_id: `int`
        The output of 'lapd_plasma_analysis.langmuir.configurations.get_config_id' for
        the relevant experiment name

    Returns
    -------
   `int`
        Scaling constant, 1 or -1, to be used to invert current data if necessary

    """
    return (-1, 1, 1, -1)[config_id]
