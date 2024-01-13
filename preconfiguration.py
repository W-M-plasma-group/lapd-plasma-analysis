import numpy as np
from bapsflib import lapd
import astropy.units as u
import xarray as xr


def get_config_id(exp_name):
    valid_configs = ["April_2018", "March_2022", "November_2022"]

    try:
        return valid_configs.index(exp_name)
    except ValueError:
        raise NotImplementedError("Configuration data has not yet been added for experiment series " + repr(exp_name)
                                  + ". Acceptable configurations are " + str(valid_configs)
                                  + ". Please contact the maintainer of this code repository to update the experiment "
                                  + "configs or, if you like, you can manually update preconfiguration.py.")


def get_vsweep_bc(config_id):  # return (board, channel) for vsweep data
    vsweep_bcs = [(1, 3),   # April_2018
                  (1, 1),   # March_2022
                  (1, 1)]   # November_2022
    return vsweep_bcs[config_id]


def get_voltage_gain(config_id):
    # TODO develop; get from HDF5 metadata someday
    return 100.


def get_langmuir_config(hdf5_path, config_id):

    # each list in tuple corresponds to an experiment series;
    # each tuple in list corresponds to configuration data for a single probe used in those experiments
    # -1 is placeholder; what each entry corresponds to is given in 'dtype' parameter below
    langmuir_probe_configs = ([(1, 2, 25, -1, 11., 1 * u.mm ** 2)],    # April_2018

                              [(1, 2, 27, -1, 1.25, 1 * u.mm ** 2),    # March_2022
                               (1, 3, 43, -1, 2.10, 1 * u.mm ** 2)],

                              [(1, 2, 29, -1, 2.20, 2 * u.mm ** 2),    # November_2022
                               (1, 3, 35, -1, 2.20, 2 * u.mm ** 2)]
                              )

    langmuir_configs_array = np.array(langmuir_probe_configs[config_id], dtype=[('board', int),
                                                                             ('channel', int),
                                                                             ('port', int),
                                                                             ('receptacle', int),
                                                                             ('resistance', float),
                                                                             ('area', u.Quantity)])
    # end of hardcoded probe configuration data

    ports_receptacles = get_ports_receptacles(hdf5_path)
    langmuir_configs_array['receptacle'] = [ports_receptacles[port] for port in langmuir_configs_array['port']]
    return langmuir_configs_array


def get_ports_receptacles(hdf5_path):
    with lapd.File(hdf5_path) as fi:
        configs = fi.controls['6K Compumotor'].configs
        return {configs[probe]['probe']['port']: configs[probe]['receptacle'] for probe in configs}


def get_ion(hdf5_path: str):
    if "h2" in hdf5_path.lower():
        print("Assuming fully dissociated hydrogen (H+)")
        return "H+"
    else:
        return "He-4+"
