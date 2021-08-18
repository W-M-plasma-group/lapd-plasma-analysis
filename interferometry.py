import warnings

import numpy as np
import astropy.units as u
import xarray as xr

from hdf5reader import *


def interferometry_calibration(density_xarray, interferometry_filename,
                               steady_state_start, steady_state_end, core_region=26. * u.cm):
    r"""
    Calibrates density data from sweep probe to measurements from interferometry probe.

    Parameters
    ----------
    :param density_xarray: DataArray of n_e diagnostic
    :param interferometry_filename: string, path to interferometry data file
    :param steady_state_start: number of first steady-state plateau, 1-based index
    :param steady_state_end: number of last steady-state plateau, 1-based index
    :param core_region: extent of core region, units of distance
    :return: tuple of (dictionary, tuple of (boolean, boolean), DataArray): x and y scaling constants,
        whether x and y dimensions exist, and calibrated electron density
    """

    # INTERFEROMETRY DATA #
    # ___________________ #

    # Read in interferometry data ("interferometry" abbreviated as "inter" in variable names)
    inter_file = open_hdf5(interferometry_filename)
    inter_raw = np.array(item_at_path(inter_file, '/MSI/Interferometer array/Interferometer [0]/Interferometer trace/'))

    print("Interferometry raw data shape:", inter_raw.shape)

    inter_means_abstract = np.mean(inter_raw, axis=0)
    inter_time_abstract = np.arange(len(inter_means_abstract))

    # Create interferometry DataArray with interferometry time as coordinates
    inter_values, inter_time = to_real_interferometry_units(inter_means_abstract, inter_time_abstract)
    inter_data = xr.DataArray(inter_values, coords=(('time', inter_time, {'units': str(u.ms)}),))

    # DENSITY DATA #
    # ____________ #

    # Calculate density line integrals
    density_data = density_xarray * (1 / u.m ** 3).to(1 / u.cm ** 3).value  # density data in units of 1 / cm^3; rename?

    has_x = density_data.sizes['x'] > 1
    has_y = density_data.sizes['y'] > 1

    if has_x and has_y:
        warnings.warn("Two-dimensional density interferometry calibration code is incomplete "
                      "and may give inaccurate or unexpected results.")

    # INTERFEROMETRY - DENSITY TIME ALIGNMENT #
    # _______________________________________ #

    # Find index of interferometry collapse; at point with most negative slope
    inter_collapse_time = inter_data.differentiate('time').idxmin('time')

    # Find average step in density (n_e) time coordinate by dividing total time elapsed by number of time measurements
    density_time_coord = density_data.coords['time']
    dt = (density_time_coord[-1] - density_time_coord[0]) / len(density_time_coord)  # time step for density time coord.

    both = xr.ufuncs.logical_and  # Rename function for readability

    # Select elements of density array with both x and y in the core position range (|x| < 26 cm & |y| < 26 cm)
    x_mask, y_mask = abs(density_data.x) < core_region.value, abs(density_data.y) < core_region.value
    core_density_array = density_data.where(both(x_mask, y_mask), drop=True)

    # Interpolate nan values linearly to allow trapezoidal integration over incomplete dimensions
    core_density_array = core_density_array.interpolate_na(dim='x', use_coordinate=True, max_gap=10.)  # 10 cm max gap

    density_scaling = dict()  # create empty dictionary to hold x and y scaling factors where necessary

    # TODO separate by radial or areal, not x or y dimensions. Implementation may be easier; or still try sep function?

    if has_x:
        # Line integral along x dimension
        x_integral = core_density_array.integrate('x')

        # Find time of density collapse in x; take average to get one single average x density collapse time (scalar)
        density_collapse_time_x = x_integral.idxmax('time').mean()

        # Align times so that interferometry data collapse matches x density data max
        aligned_x_time = inter_data.coords['time'] - inter_collapse_time + density_collapse_time_x
        inter_data = inter_data.assign_coords({'x_time': ('time', aligned_x_time.data)})

        # Average all interferometry measurements into data point with closest corresponding density time coordinate
        inter_avg_x_time = crunch_data(inter_data, 'x_time', density_data.coords['time'], dt)

        density_scaling['x'] = inter_avg_x_time / x_integral

    if has_y:
        # Line integral along y dimension
        y_integral = core_density_array.integrate('y')

        # Find time of density collapse in y; take average to get one single average y density collapse time (scalar)
        density_collapse_time_y = y_integral.idxmax('time').mean()

        # Align times so that interferometry data collapse matches y density data max
        aligned_y_time = {'y_time': ('time', inter_data.coords['time'] - inter_collapse_time + density_collapse_time_y)}
        inter_data = inter_data.assign_coords(aligned_y_time)

        # Average all interferometry measurements into data point with closest corresponding density time coordinate
        inter_avg_y_time = xr.DataArray([inter_data.where(both(inter_data.coords['y_time'] > t - dt / 2,
                                                               inter_data.coords['y_time'] < t + dt / 2)
                                                          ).mean() for t in density_data.coords['time']],
                                        dims=['time'],
                                        coords={'time':  density_data.coords['time'],
                                                'plateau': ('time', density_data.coords['plateau'])})
        density_scaling['y'] = inter_avg_y_time / y_integral

    # SCALING FACTOR CALCULATION #
    # __________________________ #

    # Average together x and y scaling factors, if they exist, to get scaling factor array in right number of dimensions
    density_scale_factor = (density_scaling['x'] if has_x else 0
                            + density_scaling['y'] if has_y else 0) / (has_x + has_y)  # average x and y scale factors

    # Return the calibrated electron temperature data only in the steady state region (given by plateau indices)
    steady_state_scale_factor = density_scale_factor.where(
        both(density_xarray.plateau >= steady_state_start, density_xarray.plateau <= steady_state_end))
    calibrated_density_xarray = density_xarray * steady_state_scale_factor

    print("Average interferometry calibration factor:", steady_state_scale_factor.mean().item())

    return calibrated_density_xarray, density_scaling


def to_real_interferometry_units(interferometry_data_array, interferometry_time_array):
    area_factor = 8. * 10 ** 13 / (u.cm ** 2)  # from MATLAB code
    time_factor = ((4.88 * 10 ** -5) * u.s).to(u.ms)  # from MATLAB code
    return interferometry_data_array * area_factor, interferometry_time_array * time_factor


def crunch_data(data_array, data_coordinate, destination_coordinate, step):
    # "Crunch" interferometry data into the density data timescale by averaging all interferometry measurements
    #     into a "bucket" around the closest matching density time coordinate (within half a time step)
    #     [inter. ]   (*   *) (*) (*   *) (*) (*   *)   <-- average together all (grouped together) measurements
    #     [density]   |__o__|__o__|__o__|__o__|__o__|   <-- measurements grouped by closest density measurement "o"
    # Take the mean of all interferometry measurements in the same "bucket" to match timescales
    r"""
    Group data along a specified dimension into bins determined by a destination coordinate and a step size,
    then return the mean of each bin with the dimensions and coordinates of the destination coordinate.

    Parameters
    ----------
    :param data_array: xarray DataArray
    :param data_coordinate: string, dimension in data_array
    :param destination_coordinate: xarray DataArray, used as coordinate
    :param step:
    :return:
    """

    # Group input data "data_array" along the dimension specified by "dimension" (dimension coordinate values used)
    #    by the coordinate in the xarray "destination_coordinate", assumed to have regular spacing "step" and take means
    grouped_mean = data_array.groupby_bins(data_coordinate, np.linspace(
        destination_coordinate[0] - step / 2, destination_coordinate[-1] + step / 2, len(destination_coordinate) + 1),
                                           labels=destination_coordinate.data).mean()

    # This result has only one dimension, the input data "dimension" + "_bins", labeled with the destination coordinate.
    #    We want to return an xarray with all the dimensions and coordinates (in this case: time dimension,
    #    time dimension coordinate, plateau non-dimension coordinate) of the destination data.
    #    This involves renaming the "_bins" dimension to match the destination coordinate,
    #    creating a new coordinate identical to the destination coordinate's dimension coordinate,
    #    and swapping the two new coordinates to give the xarray the same dimension coordinate as the destination.

    destination_dimension = destination_coordinate.dims[0]  # The name of the dimension of the 1D destination coordinate
    destination_coordinate_name = destination_coordinate.name  # The name of the destination coordinate

    # Rename position-time-"_bins" dimension name to match destination coordinate, for example "x_time_bins" to "time"
    named_mean = grouped_mean.rename({data_coordinate + "_bins": destination_coordinate_name})
    # Add the destination dimension coordinate to the output xarray as a new coordinate
    named_mean = named_mean.assign_coords({destination_dimension: (destination_coordinate_name,
                                                                   destination_coordinate[destination_dimension].data)})
    # Make the new destination dimension coordinate the main (dimension) coordinate of the output as well
    named_mean = named_mean.swap_dims({destination_coordinate_name: destination_dimension})

    return named_mean
