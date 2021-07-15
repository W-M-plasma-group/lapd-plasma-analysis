from getIVsweep import *
import xarray as xr


def plasma_diagnostics(characteristic_array, probe_area, ion_type):
    # take in array of characteristics, output nd xarray?

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
                    print("Plateau at position (", i, ",", j, ",", p, ") was unusable")
                    # characteristic_array[i, j, p].plot()
                elif diagnostics == 2:
                    print("Something's up at position (", i, ",", j, ",", p, ")")
                    # characteristic_array[i, j, p].plot()
                else:
                    diagnostic_xarray[i, j, p] = [var.value for var in diagnostics.values()]
                    if not diagnostic_names_assigned:
                        diagnostic_xarray = diagnostic_xarray.assign_coords(diagnostic=list(diagnostics.keys()))
                        diagnostic_names_assigned = True
                    # Make diff diagnostic info different dataArrays in one dataset?

    # print(diagnostic_array)
    return diagnostic_xarray


def verify_plateau(characteristic, probe_area, ion_type):
    # MUST elaborate: this does no filtering, only a bare minimum non-defective check

    try:
        diagnostics = swept_probe_analysis(characteristic, probe_area, ion_type)
    except ValueError:
        return 1
    except (TypeError, RuntimeError):
        return 2
    return diagnostics


def extract_diagnostics(characteristic):
    return swept_probe_analysis(characteristic)


def get_diagnostic_names():
    pass
