import numpy as np
import xarray as xr
from plasmapy.diagnostics.langmuir import swept_probe_analysis


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

    number_of_diagnostics = 9 if bimaxwellian else 8

    # Create a dataset with the given number of DataArrays, each with correct x, y, time(plat) dimension sizes but empty
    xarray_list = [xr.full_like(characteristic_xarray, np.nan, dtype=float) for _ in range(number_of_diagnostics)]
    xarray_dict = {str(i): xarray_list[i] for i in range(number_of_diagnostics)}
    diagnostic_dataset = xr.Dataset(xarray_dict)
    diagnostic_dataset.assign_attrs(lapd_parameters)

    print("Calculating plasma diagnostics... (May take several minutes)")
    diagnostic_names_assigned = False
    for i in range(characteristic_xarray.sizes['x']):
        for j in range(characteristic_xarray.sizes['y']):
            for p in range(characteristic_xarray.sizes['time']):
                characteristic = characteristic_xarray[i, j, p].item()  # Get characteristic at x=i, y=j, plateau-1=p
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
                        for unit_key in diagnostics.keys():
                            diagnostic_dataset[unit_key].attrs['units'] = str(unit_safe(diagnostics[unit_key]))
                        if bimaxwellian:
                            # electron temperature values broadcasted into array dimension of size two
                            # TODO store as separate diagnostics 'T_e_hot' and 'T_e_cold'? This could eliminate need
                            #     for two separate files (store in same) and for most contrivances in plots.py,
                            #     but should make sure that ex. old T_e data from a previous HDF5 file isn't used
                            diagnostic_dataset['T_e'] = diagnostic_dataset['T_e'].expand_dims(
                                dim={"population": ["cold", "hot"]}, axis=-1).copy()
                        diagnostic_names_assigned = True

                    for key in diagnostics.keys():
                        diagnostic_value = value_safe(diagnostics[key])
                        if key == 'T_e' and flag_diagnostic(diagnostic_value, minimum=0, maximum=10):
                            # remove unrealistic electron temperature values; hard-coded acceptable temperature range
                            diagnostic_value = np.nan
                        diagnostic_dataset[key][i, j, p] = diagnostic_value

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


def flag_diagnostic(diagnostic, minimum, maximum):  # discard T_e and other diagnostic values outside of specified range

    diagnostic_1d = np.atleast_1d(diagnostic)
    return (diagnostic_1d < minimum).any() or (diagnostic_1d > maximum).any()


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



