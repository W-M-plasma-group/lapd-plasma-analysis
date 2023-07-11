import numpy as np
import xarray as xr

import astropy.units as u
import matplotlib.pyplot as plt
from plasmapy.diagnostics.langmuir import swept_probe_analysis, reduce_bimaxwellian_temperature, Characteristic

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
    num_ports = characteristic_arrays.shape[0]
    ports_z = np.array([portnum_to_z(port).to(u.cm).value for port in ports])

    probe_areas = np.atleast_1d(probe_area)
    if len(probe_areas) == 1:
        probe_areas = np.repeat(probe_areas, num_ports)
    keys_units = get_diagnostic_keys_units(probe_areas[0], ion_type, bimaxwellian=bimaxwellian)

    # num_x * num_y * num_shots * num_plateaus template numpy_array
    templates = {key: np.full(shape=(num_ports, len(x), len(y), num_shots, num_plateaus),
                              fill_value=np.nan, dtype=float)
                 for key in keys_units.keys()}
    diagnostics_ds = xr.Dataset({key: xr.DataArray(data=templates[key],
                                                   dims=['port', 'x', 'y', 'shot', 'time'],
                                                   coords=(('port', ports),
                                                           ('x', x, {"units": str(u.cm)}),
                                                           ('y', y, {"units": str(u.cm)}),
                                                           ('shot', np.arange(num_shots)),
                                                           ('time', ramp_times.to(u.ms).value, {"units": str(u.ms)}))
                                                   ).assign_coords({'plateau': ('time', np.arange(num_plateaus) + 1),
                                                                    'z': ('port', ports_z, {"units": str(u.cm)})}
                                                                   ).assign_attrs({"units": keys_units[key]})
                                 for key in keys_units.keys()})

    num_positions = (diagnostics_ds.sizes['x'] * diagnostics_ds.sizes['y']
                     * diagnostics_ds.sizes['shot'] * diagnostics_ds.sizes['time'])
    print(f"Calculating langmuir diagnostics ({len(ports)} probe(s) to analyze) ...")

    warnings.simplefilter(action='ignore')  # Suppress warnings to not break progress bar
    for p in range(characteristic_arrays.shape[0]):  # probe
        port = ports[p]
        with tqdm(total=num_positions, unit="characteristic", file=sys.stdout) as pbar:
            for l in range(characteristic_arrays.shape[1]):  # location  # noqa
                for s in range(characteristic_arrays.shape[2]):  # shot
                    for r in range(characteristic_arrays.shape[3]):  # ramp
                        characteristic = characteristic_arrays[p, l, s, r]
                        diagnostics = diagnose_char(characteristic, probe_areas[p], ion_type, bimaxwellian=bimaxwellian)
                        pbar.update(1)
                        if diagnostics in (1, 2):  # problem with diagnostics; otherwise diagnostics successful
                            # debug_char(characteristic, diagnostics, p, l, r)  # DEBUG
                            continue
                        if bimaxwellian:
                            diagnostics = unpack_bimaxwellian(diagnostics)
                        for key in diagnostics.keys():
                            # Crop diagnostics with "T_e" in name because otherwise skew averages, pressure data
                            val = value_safe(diagnostics[key]) if "T_e" not in key \
                                else crop_value(diagnostics[key], 0, 10)
                            diagnostics_ds[key].loc[port, positions[l, 0], positions[l, 1], s, ramp_times[r]] = val

    warnings.simplefilter(action='default')  # Restore warnings to default handling

    return diagnostics_ds


def in_core(pos_list, core_radius):
    return [abs(pos) < core_radius.to(u.cm).value for pos in pos_list]


def detect_steady_state_ramps(density: xr.DataArray, core_radius):
    core_density = density.where(np.logical_and(*in_core([density.x, density.y], core_radius)), drop=True)
    core_density_time = core_density.isel(port=0).mean(['x', 'y']).squeeze()
    threshold = 0.9 * core_density_time.max()
    start_index = (core_density_time > threshold).argmax().item() + 1
    end_index = core_density_time.sizes['time'] - np.nanargmax(
            core_density_time.reindex(time=core_density_time.time[::-1]) > threshold).item()
    return start_index, end_index


def diagnose_char(characteristic, probe_area, ion_type, bimaxwellian):
    # TODO save error messages/optionally print to separate log file
    try:
        diagnostics = swept_probe_analysis(characteristic, probe_area, ion_type, bimaxwellian=bimaxwellian)
    except ValueError:
        return 1
    except (TypeError, RuntimeError):
        return 2
    return diagnostics


def unpack_bimaxwellian(diagnostics):
    t_e_cold, t_e_hot = diagnostics['T_e']
    hot_frac = diagnostics['hot_fraction']
    t_e_avg = reduce_bimaxwellian_temperature([t_e_cold, t_e_hot], hot_frac)
    return diagnostics.update({'T_e_cold': t_e_cold,
                               'T_e_hot': t_e_hot,
                               'T_e_avg': t_e_avg}
                              ).pop('T_e')


def get_diagnostic_keys_units(probe_area, ion_type, bimaxwellian=False):
    # Perform diagnostic on some sample data to get all diagnostic names and units as dictionary of strings

    bias = np.arange(-20, 20, 1) * u.V
    current = ((bias.value / 100 + 0.2) ** 2 - 0.01) * u.A
    chara = Characteristic(bias, current)
    diagnostics = swept_probe_analysis(chara, probe_area, ion_type, bimaxwellian)
    if bimaxwellian:
        diagnostics = unpack_bimaxwellian(diagnostics)
    keys_units = {key: str(unit_safe(value)) for key, value in diagnostics.items()}
    keys_units.update({"n_e_cal": str(u.m ** -3)})
    keys_units.update({"P_e": str(u.Pa)})
    return keys_units


def debug_char(characteristic, error_code, *pos):
    # A debug function to plot plateaus that cause errors
    # tqdm.write(str(max(characteristic.current)))
    # tqdm.write(f"Pos: {pos}")
    if 5 < pos[-1] < 13 and 0.3 < pos[-2] / 71 < 0.7:  # max(characteristic.current) > 20 * u.mA
        pass
        # tqdm.write("Plateau at position " + str(pos) + " is unusable")
        characteristic.plot()
        plt.title(f"Plateau at position {pos} is unusable" if error_code == 1
                  else f"Unknown error at position {pos}")
        plt.show()


def crop_value(diagnostic, minimum, maximum):  # discard diagnostic values (e.g. T_e) outside specified range

    value = value_safe(diagnostic)
    return value if minimum <= value <= maximum else np.nan


def value_safe(quantity_or_scalar):  # Get value of quantity or scalar, depending on type

    try:
        val = quantity_or_scalar.value  # input is a quantity with dimension and value
    except AttributeError:
        val = quantity_or_scalar  # input is a dimensionless scalar with no value
    return val


def unit_safe(quantity_or_scalar):  # Get unit of quantity or scalar, if possible

    try:
        unit = quantity_or_scalar.unit
    except AttributeError:
        unit = None  # The input data is dimensionless
    return unit
