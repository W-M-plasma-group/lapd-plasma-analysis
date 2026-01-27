from lapd_plasma_analysis.file_access import *
from lapd_plasma_analysis.experimental import *

from lapd_plasma_analysis.langmuir.helper import *
from lapd_plasma_analysis.langmuir.configurations import *

import astropy.units as u
import numpy as np

def n_IV_parameters(hdf5_file, hdf5_path):
    """

        Parameters
        ----------
        hdf5_path : `str`
            The directory in which the HDF5 files are stored.
        Returns
        ----------
        exp_params : `dict` Dictionary containing experimental parameters for the specific run.
        See experimental.get_exp_params() for more details
        v_sweep_bc : `tuple` of `int` indicating the board and channel of the voltage sweep
        langmuir_config : 'list' of 'tuple' indicating specific save locations of parameters
        config_id : `int` index location of experiment to be studied in the langmuir.configurations file
        voltage_gain : `float` Inverse voltage gain applied to the probe multiply to undo
        orientation : 'int' -1 or 1 used to invert current data if necessary
        current_bc : 'tuple' of 'int' Obtained from langmuir_config gives the board and channel of the current data
    """

    # Create a dictionary of experimental parameters (run name, experiment name, Discharge current\
        # Fill pressure, and peak magnetic field). Other parameters are dependent on the experiment
        # that was done and are described in the function in experimental.py
    exp_params_dict = n_get_exp_params(hdf5_file)

    # Make sure the above line returns a dictionary to make Py Charm warning handling happy
    assert isinstance(exp_params_dict, dict), "get_exp_params must return a dict"

    # Determine if we are dealing with a hydrogen or helium plasma (returns a string H+ of He 4+)
    # also append to exp_params_dict
    ion_type = get_ion(exp_params_dict['Run name'])
    exp_params_dict.update({'Ion type': ion_type})

    # Now we want to get the configuration ID (integer) (What experimental run was it Jan 2024 etc.) and what board/channel
    # the IV sweep was taken on tuple in the format (board,channel). Depending on the config ID
    config_id = exp_params_dict['Config ID']

    # Langmuir configs are mostly hardcoded and return Langmuir probe xarray depending on config ID
    #   (board, channel, receptacle, port, face, resistance, area, gain)
    langmuir_configs = get_langmuir_config(hdf5_path, config_id)

    # Voltage gain is a hard coded dependent on config ID.
    # We need to divide the bias by the voltage gain to get raw voltage data
    voltage_gain = get_voltage_gain(config_id)

    # Orientation is hard coded dependent on config ID.
    # Tells us if the data is upright or inverted. Returns 1 or -1
    orientation = get_orientation(config_id)

    # Returns the board and channel numbers for the Voltage sweep (tuple)
    vsweep_bc = get_vsweep_bc(config_id)
    current_bc=[]
    for i in range(len(langmuir_configs)):
        current_bc.append((langmuir_configs['board'][i],langmuir_configs['channel'][i]))


    return exp_params_dict, vsweep_bc, langmuir_configs, config_id, voltage_gain, orientation, current_bc

def n_get_exp_params(hdf5_file):
    """
    Returns a dictionary of LAPD experiment run parameter names and values.

    Parameters
    ----------
    hdf5_path : `str`
        The path to the HDF5 file

    Returns
    -------
    `dict`
        A dictionary of experimental parameters containing their name and
        corresponding value as a `str` or an `astropy.units.Quantity`.

    Notes
    _____
    Exactly which parameters are contained in the returned dictionary depends on the
    config ID, which obtained inside this function as it is implicit to `hdf5_path`.

    For all given files, the first five entries of the dictionary and the
    functions used to obtain them (all in the
    `lapd_plasma_analysis.experimental` module) are:

        'Run name' -------------------- `lapd_plasma_analysis.experimental.get_run_name`
        'Exp name' -------------------- `lapd_plasma_analysis.experimental.get_exp_name`
        'Discharge current' ----------- `lapd_plasma_analysis.experimental.get_discharge`
        'Fill pressure' --------------- `lapd_plasma_analysis.experimental.get_gas_pressure`
        'Peak magnetic field' --------- `lapd_plasma_analysis.experimental.get_magnetic_field`

    Afterward, the next dictionary entries vary depending on config ID. In each case,

    config_id == 0:

        'Nominal discharge' ----------- `lapd_plasma_analysis.experimental.get_nominal_discharge_03`
        'Nominal pressure' ------------ `lapd_plasma_analysis.experimental.get_nominal_pressure_0`

    config_id == 1 or config_id == 2:

        'Nominal discharge' ----------- `lapd_plasma_analysis.experimental.get_nominal_discharge_12`
        'Nominal gas puff' ------------ `lapd_plasma_analysis.experimental.get_nominal_gas_pump_12`

    config_id == 3:

        'Nominal magnetic field' ------ `lapd_plasma_analysis.experimental.get_nominal_magnetic_field`
        'Nominal discharge' ----------- `lapd_plasma_analysis.experimental.get_nominal_discharge_03`
        'Nominal gas puff' ------------ `lapd_plasma_analysis.experimental.get_nominal_gas_pump_3`

    See each of the functions for an explanation of the meaning of each parameter
    and the way in which it is obtained.

    """


    # The user can define these experimental control parameter functions
    exp_params_functions = [get_run_name,
                            get_exp_name,
                            get_discharge,
                            get_gas_pressure,
                            get_magnetic_field]
    # From configurations.py: 0 = April_2018, 1 = March_2022, 2 = November_2022, 3 = January_2024
    exp_params_functions_0 = [get_nominal_discharge_03,
                              get_nominal_pressure_0]
    exp_params_functions_12 = [get_nominal_discharge_12,
                               get_nominal_gas_puff_12]
    exp_params_functions_3 = [get_nominal_magnetic_field,
                              get_nominal_discharge_03,
                              get_nominal_gas_puff_3]
    # Units are given in MATLAB code
    exp_params_names_values = {}

    # .info gives a series of metadata about the file - Example metadata saved with Luke
    exp_name = hdf5_file.info['exp name']

    # Obtain the date of the experiment from the experiment name
    config_id = get_config_id(exp_name)
    if config_id == 0:
        exp_params_functions += exp_params_functions_0
    if config_id in (1, 2):
        exp_params_functions += exp_params_functions_12
    if config_id == 3:
        exp_params_functions += exp_params_functions_3

    # Run each function in the exp_params_functions list -- each function returns a key-value pair that is added
    # to the overall exp_params_names_values dictionary
    for exp_param_func in exp_params_functions:
        exp_params_names_values.update(exp_param_func(hdf5_file))

    # Ensure the config id is included in the experimental parameters dictionary so that we don't have to re-run
    # that function later
    exp_params_names_values.update({'Config ID': config_id})
    return exp_params_names_values

def n_get_sweep_voltage(lapd_file, vsweep_bc, voltage_gain):
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

    vsweep = lapd_file.read_data(*vsweep_bc, silent=True)

    dt = vsweep.dt
    vsweep = vsweep['signal']

    # Convert to real units (not abstract)
    bias = vsweep * voltage_gain * u.V

    return bias, dt

def n_get_sweep_current(lapd_file, isweep_metadata, orientation):
    """
    Reads the current collected by a Langmuir probe from an HDF5 file.
    Note that one sweep current signal is collected for every face on every Langmuir probe in each experiment.

    Parameters
    ----------
    lapd_file: `file`
        Opened hdf5 file
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