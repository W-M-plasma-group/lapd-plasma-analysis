import numpy as np
import xarray as xr
from plasmapy.diagnostics.langmuir import swept_probe_analysis


def plasma_diagnostics(characteristic_xarray, probe_area, ion_type, bimaxwellian=False):

    # take in xarray of characteristics, output xarray Dataset object

    number_of_diagnostics = 9 if bimaxwellian else 8

    xarray_list = [xr.full_like(characteristic_xarray, np.nan, dtype=float) for _ in range(number_of_diagnostics)]
    xarray_dict = {str(i): xarray_list[i] for i in range(number_of_diagnostics)}
    diagnostic_dataset = xr.Dataset(xarray_dict)

    diagnostic_names_assigned = False
    for i in range(characteristic_xarray.sizes['x']):
        for j in range(characteristic_xarray.sizes['y']):
            for p in range(characteristic_xarray.sizes['plateau']):
                characteristic = characteristic_xarray[i, j, p].item()
                diagnostics = verify_plateau(characteristic, probe_area, ion_type, bimaxwellian)
                if diagnostics == 1:
                    print("Plateau at position (", i, ",", j, ",", p, ") is unusable")
                    # characteristic_array[i, j, p].plot()
                elif diagnostics == 2:
                    print("Unknown error at position (", i, ",", j, ",", p, ")")
                    # characteristic_array[i, j, p].plot()
                else:
                    if not diagnostic_names_assigned:
                        diagnostic_dataset = diagnostic_dataset.rename(
                            {str(i): list(diagnostics.keys())[i] for i in range(len(diagnostics.keys()))})
                        for unit_key in diagnostics.keys():  # set units of results as attribute of each variable
                            try:
                                unit_string = str(diagnostics[unit_key].unit)
                            except AttributeError:
                                unit_string = None  # the data is dimensionless
                            diagnostic_dataset[unit_key].attrs['unit'] = unit_string
                        if bimaxwellian:  # the electron temperature value will be a two-element array
                            diagnostic_dataset['T_e'] = diagnostic_dataset['T_e'].expand_dims(
                                dim={"population": 2}, axis=-1).copy()
                        diagnostic_names_assigned = True

                    for key in diagnostics.keys():
                        diagnostic_value = value_safe(diagnostics[key])
                        if key == 'T_e':
                            if flag_electron_temperature(diagnostic_value, minimum=0, maximum=15):  # hard-coded range
                                diagnostic_value = np.nan
                            diagnostic_dataset[key][i, j, p] = diagnostic_value

    return diagnostic_dataset


def verify_plateau(characteristic, probe_area, ion_type, bimaxwellian):

    try:
        diagnostics = swept_probe_analysis(characteristic, probe_area, ion_type, bimaxwellian=bimaxwellian)
    except ValueError:
        return 1
    except (TypeError, RuntimeError):
        return 2
    return diagnostics


def flag_electron_temperature(temp, minimum, maximum):  # discard T_e values outside of specified range

    temp_1d = np.atleast_1d(temp)
    return (temp_1d < minimum).any() or (temp_1d > maximum).any()


def value_safe(quantity_or_scalar):  # Get value of quantity or scalar, depending on type

    try:
        val = quantity_or_scalar.value  # input is a quantity with dimension and value
    except AttributeError:
        val = quantity_or_scalar        # input is a dimensionless scalar with no value
    return val
