"""
Contains functions for extracting run parameters for LAPD experiments from LAPD HDF5 files.
For function names ending in a number, the number/numbers represent the configuration IDs
for which the function is valid. For example, `get_nominal_discharge_03` is valid
for Series 0 (April 2018) and Series 3 (January 2024) experiments.
"""

import numpy as np
import astropy.units as u
import re
from bapsflib import lapd

from lapd_plasma_analysis.langmuir.configurations import get_config_id


def get_exp_params(hdf5_path):
    """
    Returns a dictionary of important LAPD experiment run parameters and their values.

    Parameters
    ----------
    hdf5_path: `str`
        Path to HDF5 file of LAPD experimental data.

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
    with lapd.File(hdf5_path) as hdf5_file:
        exp_name = hdf5_file.info['exp name']
        config_id = get_config_id(exp_name)
        if config_id == 0:
            exp_params_functions += exp_params_functions_0
        if config_id in (1, 2):
            exp_params_functions += exp_params_functions_12
        if config_id == 3:
            exp_params_functions += exp_params_functions_3
        for exp_param_func in exp_params_functions:
            exp_params_names_values.update(exp_param_func(hdf5_file))
    return exp_params_names_values


def get_run_name(file):
    """
    Get run name of HDF5 file object, e.g. "01_line_valves90V_5000A"
    """
    return {"Run name": file.info['run name']}


def get_exp_name(file):
    """
    Get name of experiment series of HDF5 file object, e.g. "November_2022"
    """
    return {"Exp name": file.info['exp name']}


def get_discharge(file):
    return {"Discharge current": np.mean(file.read_msi("Discharge", silent=True)['meta']['peak current']) * u.A}
    # Future work: plotting the discharge current could give a really helpful
    #     visualization of the plasma heating over time


def get_gas_pressure(file):
    return {"Fill pressure": np.mean(file.read_msi("Gas pressure", silent=True)['meta']['fill pressure']) * u.Torr}


def get_magnetic_field(file):
    return {"Peak magnetic field": np.mean(file.read_msi("Magnetic field", silent=True)['meta']['peak magnetic field']
                                           ) * u.gauss}


def get_nominal_magnetic_field(file):
    magnetic_field = get_magnetic_field(file)["Peak magnetic field"]
    # Round magnetic field to nearest 500 Gauss
    nominal_magnetic_field = 500 * int(np.round(magnetic_field.to(u.gauss).value / 500)) * u.gauss  # round to 500s
    return {"Nominal magnetic field": nominal_magnetic_field}


def get_nominal_gas_puff_3(file):
    run_name = file.info['run name']
    voltage_phrase = re.search("[0-9]+V", run_name).group(0)  # search for "95V", for example
    nominal_gas_puff_voltage = float(re.search("[0-9]+", voltage_phrase).group(0))

    return {"Nominal gas puff": np.round(nominal_gas_puff_voltage, 0) * u.V}


def get_nominal_discharge_12(file):
    description = file.info['run description'].lower()
    current_phrase = re.search("(?<=idis=)[0-9]{4}", description).group(0)  # e.g. search for "7400" right after "Idis="
    return {"Nominal discharge": float(current_phrase) * u.A}


def get_nominal_gas_puff_12(file):
    description = file.info['run description'].lower()
    voltage_phrase = re.search(".*puffing([^0-9]*)([0-9.]*)(?= ?v)", description).group(2)  # eg. "105." after "puffing"
    return {"Nominal gas puff": float(voltage_phrase) * u.V}


def get_nominal_discharge_03(hdf5_file):
    run_name = hdf5_file.info['run name']
    current_phrase = re.search("[0-9]+k?A", run_name).group(0)  # search for "3500A" or "5kA", for example
    if "k" in current_phrase:
        current_digit = re.search("[0-9]+", current_phrase).group(0)
        nominal_discharge = float(current_digit) * 1000
    else:
        nominal_discharge = float(re.search("[0-9]+", current_phrase).group(0))

    return {"Nominal discharge": np.round(nominal_discharge, 0) * u.A}


def get_nominal_pressure_0(hdf5_file):
    run_name = hdf5_file.info['run name']
    pressure_phrase = re.search("[0-9]+(?=press)", run_name).group(0)
    nominal_pressure = float(pressure_phrase) * 1E-6

    return {"Nominal pressure": np.round(nominal_pressure, 8) * u.Torr}
