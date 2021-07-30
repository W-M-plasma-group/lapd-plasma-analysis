import warnings

import numpy as np
import astropy.units as u
import xarray as xr

from hdf5reader import *


def interferometry_calibration(density_xarray, interferometry_filename, bias, current,
                               steady_state_start, steady_state_end, core_region=26. * u.cm):
    r"""
    Calibrates density data from sweep probe to measurements from interferometry probe.

    Parameters
    ----------
    :param density_xarray:
    :param interferometry_filename:
    :param bias:
    :param current:
    :param steady_state_start:
    :param steady_state_end:
    :param core_region:
    :return:
    """
    #
    # INTERFEROMETRY #
    # ______________ #

    # Read in interferometry data ("interferometry" abbreviated as "inter" in variable names)
    inter_file = open_hdf5(interferometry_filename)
    inter_raw = np.array(item_at_path(inter_file, '/MSI/Interferometer array/Interferometer [0]/Interferometer trace/'))

    print("Interferometry raw data shape:", inter_raw.shape)

    inter_means_abstract = np.mean(inter_raw, axis=0)
    inter_time_abstract = np.arange(len(inter_means_abstract))

    # Create interferometry DataArray with interferometry time as coordinates
    inter_values, inter_time = to_real_units_interferometry(inter_means_abstract, inter_time_abstract)
    inter_data = xr.DataArray(inter_values, coords=(('time', inter_time, {'units': str(u.ms)}),))

    # DENSITY LINE INTEGRALS #
    # ______________________ #

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

    density_scaling = dict()

    if has_x:
        # Line integral along x dimension
        x_integral = core_density_array.integrate('x')

        # Find time of density collapse in x; take average to get one single average x density collapse time (scalar)
        density_collapse_time_x = x_integral.idxmax('plateau').coords['time'].mean()

        # Align times so that interferometry data collapse matches x density data max
        aligned_x_time = {'x_time': ('time', inter_data.coords['time'] - inter_collapse_time + density_collapse_time_x)}
        inter_data = inter_data.assign_coords(aligned_x_time)

        # "Crunch" interferometry data into the density data timescale by averaging all interferometry measurements
        #     into a "bucket" around the closest matching density time coordinate (within half a time step)
        #     [inter. ]   (*   *) (*) (*   *) (*) (*   *)   <-- average together all (grouped together) measurements
        #     [density]   |__o__|__o__|__o__|__o__|__o__|   <-- measurements grouped by closest density measurement "o"
        # Take the mean of all interferometry measurements in the same "bucket" to match timescales
        inter_avg_x_time = xr.DataArray([inter_data.where(both(inter_data.coords['x_time'] > t - dt / 2,
                                                               inter_data.coords['x_time'] < t + dt / 2)
                                                          ).mean() for t in density_data.coords['time']],
                                        dims=['plateau'],
                                        coords={'plateau': density_data.coords['plateau'],
                                                'time': ('plateau', density_data.coords['plateau'])})
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
                                        dims=['plateau'],
                                        coords={'plateau': density_data.coords['plateau'],
                                                'time': ('plateau', density_data.coords['plateau'])})
        density_scaling['y'] = inter_avg_y_time / y_integral

    # SCALING FACTOR CALCULATION #
    # __________________________ #

    density_scale_factor = (density_scaling['x'] if has_x else 0
                            + density_scaling['y'] if has_y else 0) / (has_x + has_y)  # average x and y scale factors
    # print("Density scale factor:", density_scale_factor)
    """
    scaled_density_xarray = density_xarray * (density_scaling['x'] if has_x else 0 +
                                              + density_scaling['y'] if has_y else 0) / (has_x + has_y)
    calibrated_density_xarray = scaled_density_xarray.where(both(scaled_density_xarray.plateau >= steady_state_start,
                                                                 scaled_density_xarray.plateau <= steady_state_end))
    """
    calibrated_density_xarray = (density_xarray * density_scale_factor).where(
        both(density_xarray.plateau >= steady_state_start, density_xarray.plateau <= steady_state_end))

    return (density_scaling['x'] if has_x else None,
            density_scaling['y'] if has_y else None), (has_x, has_y), calibrated_density_xarray


def to_real_units_interferometry(interferometry_data_array, interferometry_time_array):
    area_factor = 8. * 10 ** 13 / (u.cm ** 2)  # from MATLAB code
    time_factor = ((4.88 * 10 ** -5) * u.s).to(u.ms)  # from MATLAB code
    return interferometry_data_array * area_factor, interferometry_time_array * time_factor
