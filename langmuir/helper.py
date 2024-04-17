import numpy as np
import xarray as xr
import astropy.units as u
from plasmapy.diagnostics.langmuir import swept_probe_analysis, reduce_bimaxwellian_temperature, Characteristic

import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

anode_z = portnum_to_z(0).to(u.m)


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


def unpack_bimaxwellian(diagnostics):
    t_e_cold, t_e_hot = diagnostics['T_e']
    hot_frac = diagnostics['hot_fraction']
    t_e_avg = reduce_bimaxwellian_temperature([t_e_cold, t_e_hot], hot_frac)
    return diagnostics.update({'T_e_cold': t_e_cold,
                               'T_e_hot': t_e_hot,
                               'T_e_avg': t_e_avg}
                              ).pop('T_e')


def get_diagnostic_keys_units(probe_area=1.*u.mm**2, ion_type="He-4+", bimaxwellian=False):
    # Perform diagnostic on some sample data to get all diagnostic names and units as dictionary of strings

    bias = np.arange(-20, 20, 1) * u.V
    current = ((bias.value / 100 + 0.2) ** 2 - 0.01) * u.A
    chara = Characteristic(bias, current)
    diagnostics = swept_probe_analysis(chara, probe_area, ion_type, bimaxwellian)
    if bimaxwellian:
        diagnostics = unpack_bimaxwellian(diagnostics)
    keys_units = {key: str(unit_safe(value)) for key, value in diagnostics.items()}
    keys_units.update({"n_e_cal": str(u.m ** -3),
                       "n_i_cal": str(u.m ** -3),
                       "n_i_OML_cal": str(u.m ** -3),
                       "P_e": str(u.Pa),
                       "P_e_cal": str(u.Pa),
                       "nu_ei": str(u.Hz)})
    return keys_units


def probe_face_selector(ds, vectors):
    r"""
    Select an isweep signal, linear combination of isweep signals, or multiple such linear combinations from a
    diagnostic dataset. For example, on a dataset with two isweep signals (e.g. from 2 different probes or probe faces),
        [1,  0] would return the data from the first isweep signal (listed first in configurations.py)
        [1, -1] would return the parallel difference (first-listed minus second-listed)
        [[1, 0], [1, -1]] would return a list containing both of the above
    When multiple datasets are returned, they are placed on separate contour plots, but
    the same line plot with different line styles.
    :param ds: The Dataset of Langmuir data to select from
    :param vectors: "3D" nested list of linear combination of isweep signals to compute
    :return: Dataset containing data from the selected isweep signal or combination of isweep signals
    """

    manual_attrs = ds.attrs  # TODO raise xarray issue about losing attrs even with xr.set_options(keep_attrs=True):
    manual_sub_attrs = {key: ds[key].attrs for key in ds}
    if len(np.array(vectors).shape) != 3:
        raise ValueError(f"Expected '3D' nested list for probe_face_choices parameter, "
                         f"but got dimension {len(np.array(vectors).shape)}")
    ds_s = []
    for vector in vectors:
        ds_isweep_selected = 0 * ds.isel(isweep=0).copy()
        for i in range(ds.sizes['isweep']):
            ds_isweep_selected += vector[i] * ds.isel(isweep=i)
        for key in ds:
            ds_isweep_selected[key] = ds_isweep_selected[key].assign_attrs(manual_sub_attrs[key])
        ds_s += [ds_isweep_selected.assign_attrs(manual_attrs | {"facevector": str(vector)})]
    return ds_s


def array_lookup(array, value):
    return np.argmin(np.abs(array - value))


def in_core(pos_list, core_rad):
    return [np.abs(pos) < core_rad.to(u.cm).value for pos in pos_list]


# Make time based and not plateau based to improve compatibility with mach probe analysis?
def steady_state_only(diagnostics_dataset, steady_state_plateaus: tuple):

    # return diagnostics_dataset[{'time': slice(*steady_state_plateaus)}]
    return diagnostics_dataset.where(np.logical_and(diagnostics_dataset.plateau >= steady_state_plateaus[0],
                                                    diagnostics_dataset.plateau <= steady_state_plateaus[1]), drop=True)


def core_steady_state(da_input: xr.DataArray, core_rad=None, steady_state_plateaus=None, operation=None,
                      dims_to_keep=(None,)) -> xr.DataArray:
    r"""

    :param da_input: xarray DataArray with x and y dimensions
    :param core_rad: (optional) astropy Quantity convertible to centimeters giving radius of core
    :param steady_state_plateaus: (optional) tuple or list giving indices of start and end of steady-state period
    :param operation: (optional) Operation to perform on core/steady_state data
    :param dims_to_keep: (optional) optional list of dimensions not to calculate mean across
    :return: DataArray with dimensions dims_to_keep
    """

    da = da_input.copy()
    if core_rad is not None:
        da = da.where(np.logical_and(*in_core([da.x, da.y], core_rad)), drop=True)
    if steady_state_plateaus is not None:
        da = steady_state_only(da, steady_state_plateaus=steady_state_plateaus)

    dims_to_reduce = [dim for dim in da.dims if dim not in dims_to_keep]
    if operation is None:
        return da
    elif operation == "mean":
        return da.mean(dim=dims_to_reduce)
    elif operation == "median":
        return da.median(dim=dims_to_reduce)
    elif operation == "std":
        return da.std(dim=dims_to_reduce)
    elif operation == "std_error":
        # 95% (~two standard deviation) confidence interval
        da_std = da.std(dim=dims_to_reduce)
        non_nan_element_da = da.copy()
        non_nan_element_da[...] = ~np.isnan(da)
        effective_num_non_nan_per_std = non_nan_element_da.sum(dims_to_reduce)
        return da_std * 1.96 / np.sqrt(effective_num_non_nan_per_std)
    else:
        raise ValueError(f"Invalid operation {repr(operation)} when acceptable are None, 'mean', 'std', and 'std_err'")
