import numpy as np
import xarray as xr
import astropy.units as u
from plasmapy.diagnostics.langmuir import swept_probe_analysis, reduce_bimaxwellian_temperature, Characteristic
from bapsflib.lapd.tools import portnum_to_z

import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['figure.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

anode_z = portnum_to_z(0).to(u.m)
ion_temperature = 1 * u.eV


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

    bias = np.linspace(-20, 20, 200) * u.V
    current = ((1 + np.exp(-bias.value / 2)) ** (-1) - 0.1) * u.A
    chara = Characteristic(bias, current)
    diagnostics = swept_probe_analysis(chara, probe_area, ion_type, bimaxwellian)
    if bimaxwellian:
        diagnostics = unpack_bimaxwellian(diagnostics)
    keys_units = {key: str(unit_safe(value)) for key, value in diagnostics.items()}
    keys_units.update({"n_e_cal": str(u.m ** -3),
                       "n_i_cal": str(u.m ** -3),
                       "n_i_OML_cal": str(u.m ** -3),
                       "P_e": str(u.Pa),
                       "P_e_from_n_i_OML": str(u.Pa),
                       "P_e_cal": str(u.Pa),
                       "P_ei": str(u.Pa),
                       "P_ei_from_n_i_OML": str(u.Pa),
                       "P_ei_cal": str(u.Pa),
                       "nu_ei": str(u.Hz),
                       "v_para": str(u.m / u.s),
                       "v_perp": str(u.m / u.s)})
    return keys_units


def probe_face_selector(ds, vectors):
    """
    Select an isweep signal, linear combination of isweep signals, or multiple such linear combinations from a
    diagnostic dataset. For example, on a dataset with two isweep signals (e.g. from 2 different probes or probe faces),
    - [1,  0] would return the data from the first isweep signal (listed first in configurations.py)
    - [1, -1] would return the parallel difference (first-listed minus second-listed)
    - [[1, 0], [1, -1]] would return a list containing both of the above
    When multiple datasets are returned, they are placed on separate contour plots, but
    the same line plot with different line styles.

    Parameters
    ----------
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
        ds_isweep_selected = 0 * ds.isel(probe=0, face=0).copy()
        for p in range(ds.sizes['probe']):
            for f in range(ds.sizes['face']):
                ds_isweep_selected += vector[p][f] * ds.isel(probe=p, face=f)
            for key in ds:
                ds_isweep_selected[key] = ds_isweep_selected[key].assign_attrs(manual_sub_attrs[key])
        ds_s += [ds_isweep_selected.assign_attrs(manual_attrs | {"probe_face_vector": str(vector)})]
    return ds_s


def array_lookup(array, value):
    return np.argmin(np.abs(array - value))


def core_steady_state(da_input: xr.DataArray, core_rad=None, steady_state_times=None, operation=None,
                      dims_to_keep: list | tuple = (None,)) -> xr.DataArray:
    """
    Text

    Parameters
    ----------
    :param da_input: xarray DataArray with x and y dimensions
    :param core_rad: (optional) astropy Quantity convertible to centimeters giving radius of core
    :param steady_state_times: (optional) tuple or list giving indices of start and end of steady-state period
    :param operation: (optional) Operation to perform on core/steady_state data
    :param dims_to_keep: (optional) optional list of dimensions not to calculate mean across
    :return: DataArray with dimensions dims_to_keep
    """

    da = da_input.copy()
    if core_rad is not None:
        da = da.where(np.logical_and(np.abs(da.coords['x']) < core_rad.to(u.cm).value,
                                     np.abs(da.coords['y']) < core_rad.to(u.cm).value), drop=True)
    if steady_state_times is not None:
        steady_state_times_ms = steady_state_times.to(u.Unit(da.coords['time'].attrs['units'])).value
        da = da.where(np.logical_and(da.time >= steady_state_times_ms[0],
                                     da.time <= steady_state_times_ms[1]), drop=True)

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
        raise ValueError(f"Invalid operation {repr(operation)} "
                         f"when acceptable are None, 'mean', 'std', and 'std_error'")


def crunch_data(source_data: xr.DataArray | xr.Dataset,
                source_coord_name: str,
                destination_coord_da: xr.DataArray):
    """
    Group data along a specified dimension into bins determined by a destination coordinate and a step size,
    then return the mean of each bin with the dimensions and coordinates of the destination coordinate.

    Parameters
    ----------
    :param source_data: DataArray containing data to bin and average
    :param source_coord_name: string, dimension in data_array; used to bin data
    :param destination_coord_da: xarray DataArray, used as coordinate
    :return:
    """

    # "Crunch" interferometry data into the density data timescale by averaging all interferometry measurements
    # into a "bucket" around the closest matching density time coordinate (within half a time step)
    # [inter. ]   (*   *) (*) (*   *) (*) (*   *)   <-- average together all (grouped together) measurements
    # [density]   |__o__|__o__|__o__|__o__|__o__|   <-- measurements grouped by closest density measurement "o"
    # Take the mean of all interferometry measurements in the same "bucket" to match timescales

    step = (destination_coord_da[-1] - destination_coord_da[0]) / len(destination_coord_da)
    # Group input data "source_da" along the dimension specified by "source_coord_name"
    #    by the coordinate in the xarray "destination_coord_da", assumed to have regular spacing "step", and take means
    grouped_mean = source_data.groupby_bins(source_coord_name,
                                            np.linspace(destination_coord_da[0] - step / 2,
                                                        destination_coord_da[-1] + step / 2,
                                                        len(destination_coord_da) + 1
                                                        ), labels=destination_coord_da.data
                                            ).mean()

    # This result has only one dimension, the input data "dimension" + "_bins", labeled with the destination coordinate.
    #    We want to return a DataArray with all the dimensions and coordinates (in this case: time dimension,
    #    time dimension coordinate, plateau non-dimension coordinate) of the destination data.
    #    This involves renaming the "_bins" dimension to match the destination coordinate,
    #    creating a new coordinate identical to the destination coordinate's dimension coordinate,
    #    and swapping the two new coordinates to give the xarray the same dimension coordinate as the destination.

    destination_dimension = destination_coord_da.dims[0]  # The name of the dimension of the 1D destination coordinate
    destination_coordinate_name = destination_coord_da.name  # The name of the destination coordinate

    # Rename position-time-"_bins" dimension name to match destination coordinate, for example "x_time_bins" to "time"
    named_mean = grouped_mean.rename({source_coord_name + "_bins": destination_coordinate_name})
    # Add the destination dimension coordinate to the output xarray as a new coordinate
    named_mean = named_mean.assign_coords({destination_dimension: (destination_coordinate_name,
                                                                   destination_coord_da[destination_dimension].data)})
    # Make the new destination dimension coordinate the main (dimension) coordinate of the output as well
    named_mean = named_mean.swap_dims({destination_coordinate_name: destination_dimension})

    return named_mean
