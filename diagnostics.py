import numpy as np
import xarray as xr

from plasmapy.diagnostics.langmuir import swept_probe_analysis, reduce_bimaxwellian_temperature

import sys
import warnings
from tqdm import tqdm


def plasma_diagnostics(characteristic_xarray, probe_area, ion_type, lapd_parameters, bimaxwellian=False):
    r"""
    Performs plasma diagnostics on a DataArray of Characteristic objects and returns the diagnostics as a Dataset.

    Parameters
    ----------
    :param characteristic_xarray: DataArray
    :param probe_area: units of area
    :param ion_type: string corresponding to a Particle
    :param lapd_parameters: dictionary of LAPD experimental parameters
    :param bimaxwellian: boolean
    :return: Dataset object containing diagnostic values at each position
    """

    num_diagnostics = 11 if bimaxwellian else 8

    # Create a dataset with the given number of DataArrays, each with correct x, y, time(plat) dimension sizes but empty
    xarray_list = [xr.full_like(characteristic_xarray, np.nan, dtype=float) for _ in range(num_diagnostics)]
    xarray_dict = {str(i): xarray_list[i] for i in range(num_diagnostics)}
    diagnostic_dataset = xr.Dataset(xarray_dict)
    diagnostic_dataset.assign_attrs(lapd_parameters)
    diagnostic_dataset.assign_attrs({"bimaxwellian": bimaxwellian})

    """
    What to do
    - Get dictionary with plasma diagnostic names as keys
    - Save plasma diagnostic names as a set
    - If bimaxwellian
        - Replace T_e with T_e_avg
        - Add T_e_cold and T_e_hot as elements
    - Create dictionary matching string numbers to elements of the set
    - Rename DataArrays in datasets using rename function
    - Assign each DataArray a unit attribute from dictionary
    - Values
        - For normal (non-T_e) data points, assign to their DataArray element in Dataset
        - For T_e data points
            - Assign first element to T_e_cold, second to T_e_hot, and average temperature to T_e_avg
            - Flag each temperature for unrealistic values
    """

    print("Calculating plasma diagnostics...")  # (May take several minutes)
    diagnostic_names_assigned = False
    num_positions = characteristic_xarray.sizes['x'] * characteristic_xarray.sizes['y']
    warnings.simplefilter(action='ignore')  # Suppress warnings to not break progress bar
    with tqdm(total=num_positions, unit="position", file=sys.stdout) as pbar:
        for i in range(characteristic_xarray.sizes['x']):
            for j in range(characteristic_xarray.sizes['y']):
                for p in range(characteristic_xarray.sizes['time']):
                    characteristic = characteristic_xarray[i, j, p].item()  # Get characteristic @ x=i, y=j, plateau-1=p
                    diagnostics = verify_plateau(characteristic, probe_area, ion_type, bimaxwellian)
                    if diagnostics == 1:
                        # TODO print these two tqdm.write statements to a separate log file
                        pass
                        # tqdm.write("Plateau at position (" + str(i) + ", " + str(j) + ", " + str(p) + ") is unusable")
                        # characteristic_array[i, j, p].plot()
                    elif diagnostics == 2:
                        pass
                        # tqdm.write("Unknown error at position (" + str(i) + ", " + str(j) + "," + str(p) + ")")
                        # characteristic_array[i, j, p].plot()
                    else:
                        if not diagnostic_names_assigned:
                            diagnostic_names = {key: str(unit_safe(diagnostics[key])) for key in diagnostics.keys()}
                            if bimaxwellian:
                                diagnostic_names.pop('T_e')
                                temperature_unit = str(unit_safe(diagnostics['T_e']))
                                bimaxwellian_diagnostics = {'T_e_cold': temperature_unit,
                                                            'T_e_hot': temperature_unit,
                                                            'T_e_avg': temperature_unit}
                                diagnostic_names.update(bimaxwellian_diagnostics)
                            diagnostic_dataset = diagnostic_dataset.rename({str(i): list(diagnostic_names.keys())[i]
                                                                            for i in range(len(diagnostic_names))})
                            for key in diagnostic_names:
                                diagnostic_dataset[key].attrs['units'] = diagnostic_names[key]  # assign units
                            diagnostic_names_assigned = True

                        for key in diagnostics.keys():
                            if bimaxwellian and key == 'T_e':
                                t_e_cold = diagnostics[key][0]
                                t_e_hot = diagnostics[key][1]
                                diagnostic_dataset['T_e_cold'][i, j, p] = validate_diagnostic(t_e_cold, minimum=0, maximum=10)
                                diagnostic_dataset['T_e_hot'][i, j, p] = validate_diagnostic(t_e_hot, minimum=0, maximum=10)
                                diagnostic_dataset['T_e_avg'][i, j, p] = validate_diagnostic(reduce_bimaxwellian_temperature(
                                    diagnostics[key], diagnostics['hot_fraction']), minimum=0, maximum=10)
                                # remove unrealistic electron temperature values; hard-coded acceptable temp range
                            elif key == 'T_e':
                                diagnostic_dataset[key][i, j, p] = validate_diagnostic(value_safe(diagnostics[key]), minimum=0, maximum=10)
                            else:
                                diagnostic_dataset[key][i, j, p] = value_safe(diagnostics[key])

                pbar.update(1)

    warnings.simplefilter(action='default')  # Restore warnings to default handling

    # Calculate pressure and return as DataArray in diagnostic dataset
    # diagnostic_dataset['Pe'] = calculate_pressure(diagnostic_dataset)
    return diagnostic_dataset


def verify_plateau(characteristic, probe_area, ion_type, bimaxwellian):

    try:
        diagnostics = swept_probe_analysis(characteristic, probe_area, ion_type, bimaxwellian=bimaxwellian)
    except ValueError:
        return 1
    except (TypeError, RuntimeError):
        return 2
    return diagnostics


def validate_diagnostic(diagnostic, minimum, maximum):  # discard diagnostic values (e.g. T_e) outside specified range

    # print(diagnostic)
    return value_safe(diagnostic) if minimum <= value_safe(diagnostic) <= maximum else np.nan


def value_safe(quantity_or_scalar):     # Get value of quantity or scalar, depending on type

    try:
        val = quantity_or_scalar.value  # input is a quantity with dimension and value
    except AttributeError:
        val = quantity_or_scalar        # input is a dimensionless scalar with no value
    return val


def unit_safe(quantity_or_scalar):      # Get unit of quantity or scalar, if possible

    try:
        unit = quantity_or_scalar.unit
    except AttributeError:
        unit = None  # The input data is dimensionless
    return unit



