import numpy as np
import astropy.units as u

from lapd_plasma_analysis.langmuir.configurations import get_ports_receptacles


def get_mach_config(hdf5_path, config_id):
    r"""Obtains a dictionary of configuration parameters for a given experiment.

    Gets a list of configuration settings for Mach probe signals from a specific HDf5 file and experiment series.
    These settings are primarily hard-coded into this function based off of the config ID,
    except `receptacle` which is the output of
    `lapd_plasma_analysis.langmuir.configuration.get_ports_receptacles(hdf5_path)`.

    Parameters
    ----------
    hdf5_path : `str`
        Path to the relevant HDF5 file. This should end with ".hdf5"

    config_id : `int`
        The output of `lapd_plasma_analysis.langmuir.configurations.get_config_id` for
        the relevant experiment name

    Returns
    -------
    mach_configs_array : `numpy.ndarray`
    1D structured array of Mach configurations. Each element represents one Mach probe saturation current signal,
    for example from one probe face, and contains fields for
    board, channel, receptacle, port, face, resistance, probe area, and gain.
        - "board" is the digitizer board number for the Mach signal. Used to access the Mach signal in the HDF5 file.
        - "channel" is the digitizer channel number for the Mach signal.
          Used to access the Mach signal in the HDF5 file.
          The "board" and "channel" fields together uniquely identify a particular signal from a single Mach probe face.
        - "receptacle" indicates the motor receptacle for the Mach probe, which uniquely identifies a single Mach probe.
        - "port" is the LAPD port number where the Mach probe was connected. It also uniquely identifies one Mach probe.
        - "face" is a text label given to a face on a probe. It can be an empty string, for example,
          if only one face on each probe is connected, or it can indicate the left-side probe face with "L".
          The "probe" and "face" fields together also uniquely identify the signal from a specific Mach probe face.
        - "resistance" is the resistance in series with the Mach probe. Raw measurements are collected as voltage;
          these values must be divided by the series resistance to get the Mach probe saturation current.
        - "area" is a Quantity object giving the exposed area of the Mach probe face electrode.
          Collected current must be divided by area to get current density.
        - "gain" is the total gain of the system amplifiers and resistors that act on the Mach probe saturation current.
          The Mach probe signal should be divided by the gain to recover physical voltages.
    """

    # each list in tuple corresponds to an experiment series;
    # each tuple in list corresponds to configuration data for a single probe used in those experiments
    # -1 is placeholder; what each entry corresponds to is given in 'dtype' parameter below
    mach_probe_configs = ([(2, 1, 25, -1, 1, 300.,  1 * u.mm ** 2, 1),           # April_2018  # check area?
                           (2, 2, 25, -1, 2, 300.,  1 * u.mm ** 2, 1),
                           (2, 3, 25, -1, 3, 300.,  1 * u.mm ** 2, 1),
                           (2, 4, 25, -1, 4, 300.,  1 * u.mm ** 2, 1),
                           (2, 6, 25, -1, 5, 300.,  1 * u.mm ** 2, 1),
                           (2, 7, 25, -1, 6, 300.,  1 * u.mm ** 2, 1),],

                          [(3, 1, 29, -1, 2, 14.9,  8 * u.mm ** 2, 1),           # March_2022
                           (3, 3, 29, -1, 5, 15.0,  8 * u.mm ** 2, 1),
                           (3, 4, 45, -1, 2, 14.9,  8 * u.mm ** 2, 1),
                           (3, 5, 45, -1, 5, 15.0,  8 * u.mm ** 2, 1)],

                          [(2, 1, 27, -1, 2, 16.1,  8 * u.mm ** 2, 1 / 2),       # November_2022
                           (2, 2, 27, -1, 5, 16.0,  8 * u.mm ** 2, 1 / 2),
                           (2, 3, 33, -1, 2, 16.1,  8 * u.mm ** 2, 1 / 2),
                           (2, 4, 33, -1, 5, 16.1,  8 * u.mm ** 2, 1 / 2)],

                          [(2, 1, 18, -1, 2,  7.36, 2 * u.mm ** 2, 0.89),        # January_2024
                           (2, 2, 18, -1, -2, 7.32, 2 * u.mm ** 2, 0.95),       # every other is Upper (up) vs Lower (d)
                           (2, 3, 18, -1, 5,  7.51, 2 * u.mm ** 2, 0.96),        # first four are US, second four are DS
                           (2, 4, 18, -1, -5, 7.32, 2 * u.mm ** 2, 0.89),       # negative faces mean lower
                           (3, 1, 29, -1, 2,  5.26, 8 * u.mm ** 2, 0.89),
                           (3, 3, 29, -1, -2, 5.21, 8 * u.mm ** 2, 0.95),
                           (3, 4, 29, -1, 5,  5.21, 8 * u.mm ** 2, 0.96),
                           (3, 5, 29, -1, -5, 5.21, 8 * u.mm ** 2, 0.89)
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
