from getIVsweep import *
import xarray as xr


def plasma_diagnostics(characteristic_array, probe_area, ion_type):
    # take in array of characteristics, output nd xarray?

    diagnostic_xarray = xr.DataArray(np.full(characteristic_array.shape + (8,), np.nan, dtype=u.Quantity),
                                     dims=['x', 'y', 'plateau', 'diagnostic'])
    for i in range(characteristic_array.shape[0]):
        for j in range(characteristic_array.shape[1]):
            for p in range(characteristic_array.shape[2]):
                diagnostics = None
                try:
                    diagnostics = swept_probe_analysis(characteristic_array[i, j, p], probe_area, ion_type)
                except ValueError:
                    print("Plateau at position (", i, ",", j, ",", p, ") was unusable")
                except (TypeError, RuntimeError):
                    print("Something's up at position (", i, ",", j, ",", p, ")")
                if diagnostics is not None:
                    # print(type(diagnostics))
                    # print(diagnostics.keys())
                    diagnostic_xarray[i, j, p, :] = diagnostics.values()
                    diagnostic_xarray = diagnostic_xarray.assign_coords(diagnostic=list(diagnostics.keys()))

    # print(diagnostic_array)
    return diagnostic_xarray


def verify_plateau(characteristic, probe_area, ion_type):
    # MUST elaborate: this does no filtering, only a bare minimum non-defective check

    diagnostics = swept_probe_analysis(characteristic, probe_area, ion_type)
    return diagnostics


def extract_diagnostics(characteristic):
    return swept_probe_analysis(characteristic)
