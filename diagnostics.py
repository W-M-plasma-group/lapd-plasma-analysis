import numpy as np
import xarray as xr

import astropy.units as u
import matplotlib.pyplot as plt
from plasmapy.diagnostics.langmuir import swept_probe_analysis, reduce_bimaxwellian_temperature, Characteristic

import sys
import warnings
from tqdm import tqdm


def langmuir_diagnostics(characteristic_array, positions, ramp_times, probe_area, ion_type,
                         bimaxwellian=False):  # lapd_parameters
    r"""
    Performs plasma diagnostics on a DataArray of Characteristic objects and returns the diagnostics as a Dataset.

    Parameters
    ----------
    :param ramp_times:
    :param positions:
    :param characteristic_array: 2D NumPy array of Characteristics
    :param probe_area: units of area
    :param ion_type: string corresponding to a Particle
    :param bimaxwellian: boolean
    :return: Dataset object containing diagnostic values at each position
    """

    keys_units = get_diagnostic_keys_units(probe_area, ion_type, bimaxwellian=bimaxwellian)

    x = np.unique(positions[:, 0])
    y = np.unique(positions[:, 1])
    num_plateaus = characteristic_array.shape[-1]

    # num_x * num_y * num_plateaus template numpy_array
    templates = {key: np.full(shape=(len(x), len(y), num_plateaus), fill_value=np.nan, dtype=float)
                 for key in keys_units.keys()}
    diagnostics_ds = xr.Dataset({key: xr.DataArray(data=templates[key],
                                                   dims=['x', 'y', 'time'],
                                                   coords=(('x', x, {"units": str(u.cm)}),
                                                           ('y', y, {"units": str(u.cm)}),
                                                           ('time', ramp_times, {"units": str(u.ms)}))
                                                   ).assign_coords({'plateau': ('time', np.arange(num_plateaus) + 1)}
                                                                   ).assign_attrs({"units": keys_units[key]})
                                 for key in keys_units.keys()})

    num_positions = diagnostics_ds.sizes['x'] * diagnostics_ds.sizes['y'] * diagnostics_ds.sizes['time']
    print("Calculating langmuir diagnostics...")

    warnings.simplefilter(action='ignore')  # Suppress warnings to not break progress bar
    with tqdm(total=num_positions, unit="characteristic", file=sys.stdout) as pbar:
        for l in range(characteristic_array.shape[0]):  # noqa
            for r in range(characteristic_array.shape[1]):
                characteristic = characteristic_array[l, r]
                diagnostics = diagnose_char(characteristic, probe_area, ion_type, bimaxwellian=bimaxwellian)
                if diagnostics in (1, 2):  # otherwise diagnostics successful
                    # debug_char(characteristic, diagnostics, l, r)
                    continue
                if bimaxwellian:
                    diagnostics = unpack_bimaxwellian(diagnostics)
                # Need to validate temperatures because otherwise skew averages, pressure data
                for key in diagnostics.keys():
                    # validate all diagnostics with "T_e" in name
                    val = value_safe(diagnostics[key]) if "T_e" not in key else crop_diagnostic(diagnostics[key], 0, 10)
                    diagnostics_ds[key].loc[positions[l, 0], positions[l, 1], ramp_times[r]] = val
                pbar.update(1)

    """
    diagnostic_dataset = diagnostic_dataset.assign_attrs(lapd_parameters)
    diagnostic_dataset = diagnostic_dataset.assign_attrs({"bimaxwellian": bimaxwellian})
    # print("Diagnostic dataset attributes:", diagnostic_dataset.attrs)
    """

    warnings.simplefilter(action='default')  # Restore warnings to default handling

    return diagnostics_ds


def diagnose_char(characteristic, probe_area, ion_type, bimaxwellian):
    # TODO save error messages/optionally print to separate log file
    try:
        diagnostics = swept_probe_analysis(characteristic, probe_area, ion_type, bimaxwellian=bimaxwellian)
    except ValueError as e:
        # tqdm.write(str(e))
        return 1
    except (TypeError, RuntimeError) as e:
        # tqdm.write(str(e))
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
    # Perform diagnostic on test data to get all diagnostic names and units as dictionary of strings

    bias = np.arange(-20, 20, 2) * u.V
    current = ((bias.value / 100 + 0.2) ** 2 - 0.01) * u.A
    chara = Characteristic(bias, current)
    diagnostics = swept_probe_analysis(chara, probe_area, ion_type, bimaxwellian)
    if bimaxwellian:
        diagnostics = unpack_bimaxwellian(diagnostics)
    keys_units = {key: str(unit_safe(value)) for key, value in diagnostics.items()}
    return keys_units


def debug_char(characteristic, error_code, *pos):
    tqdm.write(str(max(characteristic.current)))
    tqdm.write("^")
    if max(characteristic.current) > 100 * u.mA:
        pass
        tqdm.write("Plateau at position (" + str(pos) + ") is unusable")
        characteristic.plot()
        plt.title("Plateau at position (" + str(pos) + ") is unusable")
        plt.show()


def crop_diagnostic(diagnostic, minimum, maximum):  # discard diagnostic values (e.g. T_e) outside specified range

    return value_safe(diagnostic) if minimum <= value_safe(diagnostic) <= maximum else np.nan


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
