import xarray as xr

import sys
import warnings
from tqdm import tqdm
from bapsflib.lapd.tools import portnum_to_z

from helper import *


def langmuir_diagnostics(characteristic_arrays, positions, ramp_times, ports, probe_area, ion_type, bimaxwellian=False):
    r"""
    Performs plasma diagnostics on a DataArray of Characteristic objects and returns the diagnostics as a Dataset.

    Parameters
    ----------
    :param characteristic_arrays: list of 3D NumPy arrays of Characteristics (dims: position #, shot, plateau)
    :param positions: list of coordinates for position of each shot
    :param ramp_times: list of time-based Quantities corresponding to time of each shot (peak vsweep)
    :param ports: list of port numbers corresponding to each probe
    :param probe_area: area Quantity or list of area Quantities
    :param ion_type: string corresponding to a Particle
    :param bimaxwellian: boolean
    :return: Dataset object containing diagnostic values at each position
    """

    x = np.unique(positions[:, 0])
    y = np.unique(positions[:, 1])
    num_plateaus = characteristic_arrays.shape[-1]
    num_shots = characteristic_arrays.shape[2]
    num_isweep = characteristic_arrays.shape[0]
    ports_z = np.array([portnum_to_z(port).to(u.cm).value for port in ports])

    probe_areas = np.atleast_1d(probe_area)
    if len(probe_areas) == 1:
        probe_areas = np.repeat(probe_areas, num_isweep)
    keys_units = get_diagnostic_keys_units(probe_areas[0], ion_type, bimaxwellian=bimaxwellian)

    # num_x * num_y * num_shots * num_plateaus template numpy_array
    templates = {key: np.full(shape=(num_isweep, len(x), len(y), num_shots, num_plateaus),
                              fill_value=np.nan, dtype=float)
                 for key in keys_units.keys()}
    diagnostics_ds = xr.Dataset({key: xr.DataArray(data=templates[key],
                                                   dims=['isweep', 'x', 'y', 'shot', 'time'],
                                                   coords=(('isweep', np.arange(num_isweep)),
                                                           ('x', x, {"units": str(u.cm)}),
                                                           ('y', y, {"units": str(u.cm)}),
                                                           ('shot', np.arange(num_shots)),
                                                           ('time', ramp_times.to(u.ms).value, {"units": str(u.ms)}))
                                                   ).assign_coords({'plateau': ('time', np.arange(num_plateaus) + 1),
                                                                    'port': ('isweep', ports),
                                                                    'z': ('isweep', ports_z, {"units": str(u.cm)})}
                                                                   ).assign_attrs({"units": keys_units[key]})
                                 for key in keys_units.keys()})

    num_characteristics = (diagnostics_ds.sizes['isweep'] * diagnostics_ds.sizes['x'] * diagnostics_ds.sizes['y']
                           * diagnostics_ds.sizes['shot'] * diagnostics_ds.sizes['time'])

    # TODO Leo debug!
    # """
    error_types = []
    error_chart = np.zeros(shape=(num_isweep, len(x), len(y), num_shots, num_plateaus))
    # """

    print(f"Calculating Langmuir diagnostics...")
    warnings.simplefilter(action='ignore')  # Suppress warnings to not break progress bar
    with tqdm(total=num_characteristics, unit="characteristic", file=sys.stdout) as pbar:
        for i in range(characteristic_arrays.shape[0]):  # isweep
            for l in range(characteristic_arrays.shape[1]):  # location  # noqa
                for s in range(characteristic_arrays.shape[2]):  # shot
                    for r in range(characteristic_arrays.shape[3]):  # ramp
                        characteristic = characteristic_arrays[i, l, s, r]
                        diagnostics = diagnose_char(characteristic, probe_areas[i], ion_type, bimaxwellian=bimaxwellian)
                        pbar.update(1)
                        if isinstance(diagnostics, str):  # error with diagnostics
                            # print(i, l, s, r, ": ", diagnostics)
                            if diagnostics not in error_types:
                                error_types.append(diagnostics)
                            error_chart[i,
                                        np.where(x == positions[l, 0])[0][0],
                                        np.where(y == positions[l, 1])[0][0], s, r] = error_types.index(diagnostics) + 1
                            continue
                        if bimaxwellian:
                            diagnostics = unpack_bimaxwellian(diagnostics)
                        for key in diagnostics.keys():
                            # REMOVED: crop diagnostics with "T_e" in name because otherwise skew averages/pressure data
                            val = value_safe(diagnostics[key])  # if "T_e" not in / crop_value(diagnostics[key], 0, 10)
                            diagnostics_ds[key].loc[i, positions[l, 0], positions[l, 1], s, ramp_times[r]] = val

    warnings.simplefilter(action='default')  # Restore warnings to default handling

    # Leo debug below: display plots showing types of errors
    """
    for s in range(error_chart.shape[3]):
        ax = plt.subplot()
        im = ax.imshow(error_chart[0, :, 0, s, :],
                       extent=(positions[:, 0].min(), positions[:, 0].max(),
                               ramp_times.value.min(), ramp_times.value.max()),
                       origin="lower")
        cbar = plt.colorbar(im)
        cbar.set_ticks(list(np.arange(len(error_types) + 1)))
        cbar.set_ticklabels(["No error"] + [error_types[t] for t in np.arange(len(error_types))])
        plt.title(f"Error types (s = {s})")
        plt.tight_layout()
        plt.show()
        # raise ValueError
    # """
    # end Leo Debug

    return diagnostics_ds


def filter_characteristic(characteristic) -> bool:
    bias = characteristic.bias
    current = characteristic.current

    # reject characteristic if V_F > 0
    if np.max(bias[current < 0]) > 0 * u.V:
        return False

    # reject characteristic if
    return True


def in_core(pos_list, core_rad):
    return [np.abs(pos) < core_rad.to(u.cm).value for pos in pos_list]


def get_pressure(lang_ds, calibrated_electron_density, bimaxwellian):
    r"""Calculate electron pressure from temperature and calibrated density"""
    pressure_unit = u.Pa
    electron_temperature = lang_ds['T_e_avg'] if bimaxwellian else lang_ds['T_e']
    pressure = (3 / 2) * electron_temperature * calibrated_electron_density * (1. * u.eV * u.m ** -3).to(pressure_unit)
    return pressure.assign_attrs({'units': str(pressure_unit)})


def detect_steady_state_ramps(density: xr.DataArray, core_rad):
    r"""Return start and end ramp indices for the steady-state period (density constant in time)"""
    # TODO hardcoded
    core_density = density.where(np.logical_and(*in_core([density.x, density.y], core_radius)), drop=True)
    core_density_time = core_density.isel(isweep=0).mean(['x', 'y', 'shot']).squeeze()
    threshold = 0.9 * core_density_time.max()
    start_index = (core_density_time > threshold).argmax().item() + 1
    end_index = core_density_time.sizes['time'] - np.nanargmax(
            core_density_time.reindex(time=core_density_time.time[::-1]) > threshold).item()
    return start_index, end_index


def diagnose_char(characteristic, probe_area, ion_type, bimaxwellian, indices=None):
    if characteristic is None:
        return "characteristic is None"
    threshold = 8 * u.mA  # TODO hardcoded
    if np.max(characteristic.current.to(u.A).value) < threshold.to(u.A).value:
        return f"Probe current is below {str(threshold)}, which can lead to unreliable results"
    try:
        diagnostics = swept_probe_analysis(characteristic, probe_area, ion_type, bimaxwellian=bimaxwellian)
    except (ValueError, TypeError, RuntimeError) as e:
        diagnostics = str(e)
    return diagnostics


def debug_char(characteristic, error_str, *pos):
    # A debug function to plot plateaus that cause errors
    # TODO update the below core-checker
    # if 5 < pos[-1] < 13 and 0.3 < pos[-2] / 71 < 0.7:  # max(characteristic.current) > 20 * u.mA
    #      pass
    #      tqdm.write("Plateau at position " + str(pos) + " is unusable")
    characteristic.plot()
    plt.title(f"{pos}\n{error_str}")
    plt.show()


def crop_value(diagnostic, minimum, maximum):  # discard diagnostic values (e.g. T_e) outside specified range
    value = value_safe(diagnostic)
    return value if minimum <= value <= maximum else np.nan
