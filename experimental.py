import numpy as np
import astropy.units as u

from bapsflib import lapd


def get_exp_params(hdf5_path):

    # The user can define these experimental control parameter functions
    exp_params_functions = [get_run_name,
                            get_discharge,
                            get_gas_pressure,
                            get_magnetic_field]
    exp_params_functions_optional = [get_nominal_discharge,
                                     get_nominal_gas_puff]
    # Units are given in MATLAB code
    exp_params_names_values = {}
    with lapd.File(hdf5_path) as hdf5_file:
        for exp_param_function in exp_params_functions:
            exp_params_names_values.update(exp_param_function(hdf5_file))
        for exp_param_function in exp_params_functions_optional:
            try:
                exp_params_names_values.update(exp_param_function(hdf5_file))
            except ValueError as e:
                print(e)
    return exp_params_names_values


def get_run_name(file):
    return {"Run name": file.info['run name']}


def get_nominal_discharge(file):
    description = str(file.info['run description'])

    dis_ind = description.index("Idis")
    start_ind = description[dis_ind:].index(next(filter(str.isnumeric, description[dis_ind:]))) + dis_ind
    end_ind = description[start_ind:].index(next(filter(lambda c: c in ("A", "k", " ", ","), description[start_ind:]))
                                            ) + start_ind
    return {"Nominal discharge": float(description[start_ind:end_ind]) * u.A}


def get_nominal_gas_puff(file):
    description = str(file.info['run description']).lower()

    puff_ind = description.index("puffing")
    start_ind = description[puff_ind:].index(next(filter(str.isnumeric, description[puff_ind:]))) + puff_ind
    end_ind = description[start_ind:].index(next(filter(lambda c: c in ("v", " ", ","), description[start_ind:]))
                                            ) + start_ind

    return {"Nominal gas puff": float(description[start_ind:end_ind]) * u.V}


def get_discharge(file):
    # return item_at_path(file, '/MSI/Discharge/Discharge summary/')
    return {"Discharge current": np.mean(file.read_msi("Discharge", silent=True)['meta']['peak current']) * u.A}
    # Future work: plotting the discharge current could give a really helpful
    #     visualization of the plasma heating over time


def get_gas_pressure(file):
    # return item_at_path(file, '/MSI/Gas pressure/Gas pressure summary/')
    return {"Fill pressure": np.mean(file.read_msi("Gas pressure", silent=True)['meta']['fill pressure']) * u.Torr}


def get_magnetic_field(file):
    # return item_at_path(file, '/MSI/Magnetic field/Magnetic field summary/')
    return {"Peak magnetic field": np.mean(file.read_msi("Magnetic field", silent=True)['meta']['peak magnetic field']
                                           ) * u.gauss}


