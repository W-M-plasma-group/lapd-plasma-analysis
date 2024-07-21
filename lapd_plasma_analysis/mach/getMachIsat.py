import numpy as np
import xarray as xr
import astropy.units as u
from bapsflib import lapd
from bapsflib.lapd.tools import portnum_to_z

from lapd_plasma_analysis.langmuir.getIVsweep import get_shot_positions

# Note: This code is based on getIVsweep.py in the lapd-plasma-analysis repository.
# TODO merge into getIVsweep.py ?


def get_mach_isat(filename, mach_configs):
    """

    :param filename:
    :param mach_configs:
    :return:
    """
    # TODO add function definition

    with lapd.File(filename) as lapd_file:
        run_name = lapd_file.info['run name']

        mach_bcs = np.atleast_1d(mach_configs[['board', 'channel']])

        isat_datas = [lapd_file.read_data(*mach_bc, silent=True) for mach_bc in mach_bcs]
        dt = isat_datas[0].dt

        isat = np.concatenate([isat_data['signal'][np.newaxis, ...] for isat_data in isat_datas], axis=0)
        del isat_datas

        mach_motor_datas = [lapd_file.read_controls([('6K Compumotor', receptacle)])
                            for receptacle in mach_configs['receptacle']]

    num_frames = isat.shape[-1]
    num_isweep = len(mach_configs)

    # NOTE: Assume mach motor datas from 6K Compumotor are identical, and only consider first one
    positions, num_positions, num_shots_per_position, selected_shots = get_shot_positions(mach_motor_datas[0])
    isat = isat[:, selected_shots, ...].reshape((num_isweep, num_positions, num_shots_per_position, num_frames))

    # scale_shape = [num_isweep] + [1 for _ in range(len(isat.shape) - 2)]
    # resistances = np.reshape(mach_configs['resistance'], scale_shape)
    # gains = np.reshape(mach_configs['gain'], scale_shape)

    for i in range(len(isat)):
        # assumes probes faces are all same area!  # * u.A
        isat[i] = isat[i] / mach_configs['resistance'][i] / mach_configs['gain'][i]

    ports = np.array([mach_motor_data.info['controls']['6K Compumotor']['probe']['port']
                      for mach_motor_data in mach_motor_datas])
    assert np.all(ports == mach_configs['port'])  # otherwise, ports in configurations.py do not match motor data

    # currents dimensions:   isweep, position, shot, frame   (e.g. (1, 70, 7, 55295))
    isat_da = to_mach_isat_da(isat, positions, num_shots_per_position, mach_configs, dt).rename(run_name)

    # Subtract out typical DC offset current on each face as average of last two thousand current measurements,
    #   as this should be a while after the plasma has dissipated and thus be equal to zero.
    #   This eliminates any persistent DC offset current from the probe.
    isat_offsets = isat_da[..., -2000:].quantile(0.25, dim="time"
                                                 ).drop_vars("quantile", errors="ignore")  # TODO should this be mean?
    isat_da -= isat_offsets
    # Drop negative values for current
    # Should not be negative in first place, so if needed for large areas, data may have been skewed/is unreliable
    isat_da = isat_da.where(isat_da > 0)

    return isat_da


"""
def get_shot_positions(isat_motor_data):
    num_shots = len(isat_motor_data['shotnum'])
    shot_positions = np.round(isat_motor_data['xyz'], 1)

    z_positions = shot_positions[:, 2]
    # if np.amin(z_positions) != np.amax(z_positions):
    #     raise ValueError("Varying z-position when only x and/or y variation expected")
    # save z-position for later? Shouldn't need to, because hard to accidentally vary port
    positions = np.unique(shot_positions[:, :2], axis=0)  # list of all unique (x, y) positions
    num_positions = len(positions)
    if num_shots % num_positions != 0:
        raise ValueError("Number of Mach measurements " + str(num_shots) +
                         " does not evenly divide into " + str(num_positions) + " unique positions")
    shots_per_position = int(num_shots // num_positions)

    xy_at_positions = shot_positions[:, :2].reshape((num_positions, shots_per_position, 2))  # (x, y) at shots by pos.
    if not (np.amax(xy_at_positions, axis=1) == np.amin(xy_at_positions, axis=1)).all():
        raise ValueError("Non-uniform position values when grouping Mach probe data by position")

    return positions, num_positions, shots_per_position
"""


def to_mach_isat_da(isat, positions, shots_per_position, mach_configs, dt):
    """

    :param isat:
    :param shots_per_position:
    :param positions:
    :param mach_configs:
    :param dt:
    :return:
    """

    ports_unique = np.unique([port for port in mach_configs['port']])
    faces_unique = np.unique([face for face in mach_configs['face']])

    x_pos = np.unique(positions[:, 0])
    y_pos = np.unique(positions[:, 1])

    # test_isat = isat_datas[0][list(isat_datas[0].keys())[0]]
    test_isat = isat[0]
    num_frames = test_isat.shape[-1]
    port_z = np.array([portnum_to_z(port).to(u.cm).value for port in ports_unique])

    # isat_signals = [np.concatenate([isat_data['signal'][np.newaxis, ...] for isat_data in isat_datas], axis=0)]
    isat_signals_shape = (len(x_pos), len(y_pos), shots_per_position, num_frames)
    isat_signals = [probe_face_isat.reshape(isat_signals_shape)
                    for probe_face_isat in isat]

    empty_isat_array = np.full((len(ports_unique), len(faces_unique), *isat_signals_shape), np.nan)
    isat_da = xr.DataArray(data=empty_isat_array,
                           dims=['probe', 'face', 'x', 'y', 'shot', 'time'],
                           coords=(('probe', np.arange(len(ports_unique))),
                                   ('face', faces_unique),
                                   ('x', x_pos, {"units": str(u.cm)}),
                                   ('y', y_pos, {"units": str(u.cm)}),
                                   ('shot', np.arange(shots_per_position)),  # + 1 (removed to match Langmuir)
                                   ('time', np.arange(num_frames) * dt.to(u.ms).value, {"units": str(u.ms)}))
                           ).assign_coords({'port': ('probe', ports_unique),
                                            'z': ('probe', port_z)})
    # for probe in range(len(isat_signals)):
    #     for face in isat_signals[probe]:
    #           isat_signals[probe][face]
    for i in range(len(isat)):
        isat_da.loc[{"probe": np.where(ports_unique == mach_configs['port'][i])[0][0],
                     "face": mach_configs['face'][i]}] = isat_signals[i]

    return isat_da
