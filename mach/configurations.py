import numpy as np
import astropy.units as u


mach_receptacles = [3, 4]
mach_face_resistances = [{2: 14.9, 5: 15.0},
                         {2: 14.9, 5: 15.0}]

def get_mach_config(hdf5_path, config_id):

    # each list in tuple corresponds to an experiment series;
    # each tuple in list corresponds to configuration data for a single probe used in those experiments
    # -1 is placeholder; what each entry corresponds to is given in 'dtype' parameter below
    mach_probe_configs = ([(1, 2, 25, -1, 0, 11.,  1 * u.mm ** 2, 1)],         # April_2018

                          [(3, 1, 29, -1, 2, 14.9, 8 * u.mm ** 2, 1),           # March_2022
                           (3, 3, 29, -1, 5, 15.0, 8 * u.mm ** 2, 1),
                           (3, 4, 45, -1, 2, 14.9, 8 * u.mm ** 2, 1),
                           (3, 5, 45, -1, 5, 15.0, 8 * u.mm ** 2, 1)],

                          [(2, 1, 27, -1, 2, 16.1, 8 * u.mm ** 2, 1 / 2),       # November_2022
                           (2, 2, 27, -1, 5, 16.0, 8 * u.mm ** 2, 1 / 2),
                           (2, 3, 33, -1, 2, 16.1, 8 * u.mm ** 2, 1 / 2),
                           (2, 4, 33, -1, 5, 16.1, 8 * u.mm ** 2, 1 / 2)],

                          [(2, 1, 18, -1, 2, 7.36, 2 * u.mm ** 2, 0.89),        # January_2024
                           (2, 2, 18, -1, 0, 7.32, 2 * u.mm ** 2, 0.95),        # every other is Upper (up) vs Lower (d)
                           (2, 3, 18, -1, 5, 7.51, 2 * u.mm ** 2, 0.96),        # first four are US, second four are DS
                           (2, 4, 18, -1, 0, 7.32, 2 * u.mm ** 2, 0.89),
                           (3, 1, 29, -1, 2, 5.26, 8 * u.mm ** 2, 0.89),
                           (3, 3, 29, -1, 0, 5.21, 8 * u.mm ** 2, 0.95),
                           (3, 5, 29, -1, 5, 5.21, 8 * u.mm ** 2, 0.96),
                           (3, 6, 29, -1, 0, 5.21, 8 * u.mm ** 2, 0.89)
                           ]
                          )

    mach_configs_array = np.array(mach_probe_configs[config_id], dtype=[('board', int),
                                                                        ('channel', int),
                                                                        ('port', int),
                                                                        ('receptacle', int),
                                                                        ('face', int),
                                                                        ('resistance', float),
                                                                        ('area', u.Quantity),
                                                                        ('gain', float)])  # see note below
    # Note: "gain" here refers to what was gained before saving data. Divide data by the gain to undo.
    # (End of hardcoded probe configuration data)

    ports_receptacles = get_ports_receptacles(hdf5_path)
    mach_configs_array['receptacle'] = [ports_receptacles[port] for port in mach_configs_array['port']]
    return mach_configs_array
