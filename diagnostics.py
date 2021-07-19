import numpy as np
import xarray as xr
from plasmapy.diagnostics.langmuir import swept_probe_analysis


def plasma_diagnostics(characteristic_array, probe_area, ion_type):
    # take in array of characteristics, output nd xarray

    diagnostic_xarray = xr.DataArray(np.full(characteristic_array.shape + (8,), np.nan, dtype=float),
                                     dims=['x', 'y', 'plateau', 'diagnostic'],
                                     coords={'x': np.arange(-30, 41)})
    # add xarray coordinates to x position at least
    diagnostic_names_assigned = False
    for i in range(characteristic_array.shape[0]):
        for j in range(characteristic_array.shape[1]):
            for p in range(characteristic_array.shape[2]):
                diagnostics = verify_plateau(characteristic_array[i, j, p], probe_area, ion_type)
                if diagnostics == 1:
                    print("Plateau at position (", i, ",", j, ",", p, ") is unusable")
                    # characteristic_array[i, j, p].plot()
                elif diagnostics == 2:
                    print("Unknown error at position (", i, ",", j, ",", p, ")")
                    # characteristic_array[i, j, p].plot()
                else:
                    diagnostic_xarray[i, j, p] = [var.value for var in diagnostics.values()]
                    if not diagnostic_names_assigned:
                        diagnostic_xarray = diagnostic_xarray.assign_coords(diagnostic=list(diagnostics.keys()))
                        diagnostic_names_assigned = True
                    # Make different diagnostic information into different DataArrays in one dataset?

    return diagnostic_xarray


def verify_plateau(characteristic, probe_area, ion_type):

    try:
        diagnostics = swept_probe_analysis(characteristic, probe_area, ion_type)
    except ValueError:
        return 1
    except (TypeError, RuntimeError):
        return 2
    return diagnostics
