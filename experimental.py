import numpy as np
import astropy.units as u
import re

from bapsflib import lapd


def get_exp_params(hdf5_path):

    # The user can define these experimental control parameter functions
    exp_params_functions = [get_run_name,
                            get_exp_name,
                            get_discharge,
                            get_gas_pressure,
                            get_magnetic_field]
    exp_params_functions_2018 = [get_nominal_discharge_2018,
                                 get_nominal_pressure_2018]
    exp_params_functions_2022 = [get_nominal_discharge_2022,
                                 get_nominal_gas_puff_2022]
    # Units are given in MATLAB code
    exp_params_names_values = {}
    with lapd.File(hdf5_path) as hdf5_file:
        exp_name = hdf5_file.info['exp name']
        if exp_name == "April_2018":
            exp_params_functions += exp_params_functions_2018
        elif exp_name in ["March_2022", "November_2022"]:
            exp_params_functions += exp_params_functions_2022
        for exp_param_func in exp_params_functions:
            exp_params_names_values.update(exp_param_func(hdf5_file))
    return exp_params_names_values


def get_run_name(file):
    return {"Run name": file.info['run name']}


def get_exp_name(file):
    return{"Exp name": file.info['exp name']}


def get_discharge(file):
    return {"Discharge current": np.mean(file.read_msi("Discharge", silent=True)['meta']['peak current']) * u.A}
    # Future work: plotting the discharge current could give a really helpful
    #     visualization of the plasma heating over time


def get_gas_pressure(file):
    return {"Fill pressure": np.mean(file.read_msi("Gas pressure", silent=True)['meta']['fill pressure']) * u.Torr}


def get_magnetic_field(file):
    return {"Peak magnetic field": np.mean(file.read_msi("Magnetic field", silent=True)['meta']['peak magnetic field']
                                           ) * u.gauss}


def get_nominal_discharge_2022(file):
    description = str(file.info['run description'])

    dis_ind = description.index("Idis")
    start_ind = description[dis_ind:].index(next(filter(str.isnumeric, description[dis_ind:]))) + dis_ind
    end_ind = description[start_ind:].index(next(filter(lambda c: c in ("A", "k", " ", ","), description[start_ind:]))
                                            ) + start_ind
    return {"Nominal discharge": float(description[start_ind:end_ind]) * u.A}


def get_nominal_gas_puff_2022(file):
    description = str(file.info['run description']).lower()

    puff_ind = description.index("puffing")
    start_ind = description[puff_ind:].index(next(filter(str.isnumeric, description[puff_ind:]))) + puff_ind
    end_ind = description[start_ind:].index(next(filter(lambda c: c in ("v", " ", ","), description[start_ind:]))
                                            ) + start_ind

    return {"Nominal gas puff": float(description[start_ind:end_ind]) * u.V}


def get_nominal_discharge_2018(hdf5_file):
    # uh oh, regular expressions here
    run_name = hdf5_file.info['run name']
    current_phrase = re.search("[0-9]+k?A", run_name).group(0)  # search for "3500kA" or "5kA", for example
    if "k" in current_phrase:
        current_digit = re.search("[0-9]+", current_phrase).group(0)
        nominal_discharge = float(current_digit) * 1000
    else:
        nominal_discharge = float(re.search("[0-9]+", current_phrase).group(0))

    """
    dis_ind = description.index("kA")
    start_ind = dis_ind - description[dis_ind:].index(next(filter(str.isnumeric, description[dis_ind::-1])))
    end_ind = description[start_ind:].index(next(filter(lambda c: c in ("A", "k", " ", ","), description[start_ind:]))
                                            ) + start_ind
    return {"Nominal discharge": float(description[start_ind:end_ind]) * u.A}
    """

    return {"Nominal discharge": np.round(nominal_discharge, 0) * u.A}


def get_nominal_pressure_2018(hdf5_file):

    run_name = hdf5_file.info['run name']
    pressure_phrase = re.search("[0-9]+(?=press)", run_name).group(0)
    nominal_pressure = float(pressure_phrase) * 1E-6

    return {"Nominal pressure": np.round(nominal_pressure, 8) * u.Torr}
