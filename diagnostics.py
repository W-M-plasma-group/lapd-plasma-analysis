import astropy.units as u
import numpy as np
import xarray as xr
from plasmapy.diagnostics.langmuir import swept_probe_analysis

# Write function in characterization.py to take in array of characteristics and output xarray !!! !!! !!! !!!
# Add time coordinates using get_time_array


def plasma_diagnostics(characteristic_array, probe_area, ion_type, bimaxwellian=False):

    # take in (x?)array of characteristics, output xarray Dataset object

    number_of_diagnostics = 9 if bimaxwellian else 8

    ndarray_list = [np.full_like(characteristic_array, np.nan, dtype=float) for _ in range(number_of_diagnostics)]
    xarray_list = [xr.DataArray(array, dims=['x', 'y', 'plateau']) for array in ndarray_list]
    xarray_dict = {str(i): xarray_list[i] for i in range(number_of_diagnostics)}
    diagnostic_dataset = xr.Dataset(xarray_dict)

    # Hard-coding in coordinates. Add time from get_time_array later
    diagnostic_dataset = diagnostic_dataset.assign_coords({'x': np.arange(-30, 41)})
    diagnostic_dataset['x'].attrs['unit'] = str(u.cm)

    diagnostic_names_assigned = False
    for i in range(diagnostic_dataset.sizes['x']):
        for j in range(diagnostic_dataset.sizes['y']):
            for p in range(diagnostic_dataset.sizes['plateau']):
                diagnostics = verify_plateau(characteristic_array[i, j, p], probe_area, ion_type, bimaxwellian)
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
                        try:
                            diagnostic_value = diagnostics[key].value
                            if key == 'T_e':
                                if bimaxwellian and (diagnostic_value > 10).any() or (not bimaxwellian) \
                                        and diagnostic_value > 10:  # discard T_e values above 100 eV
                                    diagnostic_value = np.nan
                                    print("Plateau at position (", i, ",", j, ",", p,
                                          ") produces an invalid electron temperature")
                            diagnostic_dataset[key][i, j, p] = diagnostic_value
                        except AttributeError:
                            diagnostic_dataset[key][i, j, p] = diagnostics[key]  # the data is dimensionless

    return diagnostic_dataset


def verify_plateau(characteristic, probe_area, ion_type, bimaxwellian):

    try:
        diagnostics = swept_probe_analysis(characteristic, probe_area, ion_type, bimaxwellian=bimaxwellian)
    except ValueError:
        return 1
    except (TypeError, RuntimeError):
        return 2
    return diagnostics
