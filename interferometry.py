import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import xarray as xr

from hdf5reader import *


def interferometry_calibration(density_xarray, temperature_xarray, interferometry_filename, bias, current,
                               steady_state_start_plateau, steady_state_end_plateau, core_region=26. * u.cm):
    # INTERFEROMETRY #
    # ______________ #

    # Read in interferometry data ("interferometry" abbreviated as "inter" in variable names)
    # TODO reduce all unneeded intermediate variable names to make clearer what variables *should* be used
    # TODO keep all x and y data separate and optional, used only if corresponding dimension is not one element long
    inter_file = open_hdf5(interferometry_filename)
    inter_data_raw = item_at_path(inter_file, '/MSI/Interferometer array/Interferometer [0]/Interferometer trace/')
    inter_data_array = np.array(inter_data_raw)

    print("Interferometry data shape:", inter_data_array.shape)

    inter_means_abstract = np.mean(inter_data_array, axis=0)
    inter_time_abstract = np.arange(len(inter_means_abstract))

    # Create interferometry DataArray with interferometry time as coordinates
    inter_values, inter_time = to_real_units_interferometry(inter_means_abstract, inter_time_abstract)
    inter_data = xr.DataArray(inter_values, coords=(('time', inter_time, {'units': str(u.ms)}),))

    # print(inter_data)
    # DENSITY LINE INTEGRALS #
    # ______________________ #

    # Calculate density line integrals
    x_length, y_length, plateaus = density_xarray.sizes['x'], density_xarray.sizes['y'], density_xarray.sizes['plateau']
    density_xarray_cm = density_xarray * (1 / u.m ** 3).to(1 / u.cm ** 3).value

    has_x = True
    has_y = True

    # INTERFEROMETRY - DENSITY TIME ALIGNMENT #
    # _______________________________________ #

    # Find index of interferometry collapse; at point with most negative slope
    inter_collapse_time = inter_data.differentiate('time').idxmin('time')

    # Find average step in density (n_e) time coordinate by dividing total time elapsed by number of time measurements
    density_time_coord = density_xarray_cm.coords['plateau']
    dt = (density_time_coord[-1] - density_time_coord[0]) / len(density_time_coord)  # time step for density time coord.

    both = xr.ufuncs.logical_and  # Rename function for readability

    core_density_array = density_xarray_cm.copy()

    # Select elements of density array with both x and y in the core position range (|x| < 26 cm & |y| < 26 cm)
    x_mask, y_mask = abs(density_xarray_cm.x) < core_region.value, abs(density_xarray_cm.y) < core_region.value
    core_density_array = density_xarray_cm.where(both(x_mask, y_mask), drop=True)

    density_scaling = dict()

    if has_x:
        # Select density array with x coordinates in core region
        x_integral = core_density_array.integrate('x')
        density_collapse_time_x = x_integral.idxmax('plateau').mean()
        aligned_x_time = {'x_time': ('time', inter_data.coords['time'] - inter_collapse_time + density_collapse_time_x)}
        inter_data = inter_data.assign_coords(aligned_x_time)
        inter_avg_x_time = xr.DataArray([inter_data.where(both(inter_data.coords['x_time'] > t - dt / 2,
                                                               inter_data.coords['x_time'] < t + dt / 2)
                                                          ).mean() for t in density_xarray_cm.coords['plateau']],
                                        dims=['plateau'],
                                        coords={'plateau': density_xarray_cm.coords['plateau']})
        density_scaling['x'] = inter_avg_x_time / x_integral

    if has_y:
        y_integral = core_density_array.integrate('y')
        density_collapse_time_y = y_integral.idxmax('plateau').mean()
        aligned_y_time = {'y_time': ('time', inter_data.coords['time'] - inter_collapse_time + density_collapse_time_y)}
        inter_data = inter_data.assign_coords(aligned_y_time)
        inter_avg_y_time = xr.DataArray([inter_data.where(both(inter_data.coords['y_time'] > t - dt / 2,
                                                               inter_data.coords['y_time'] < t + dt / 2)
                                                          ).mean() for t in density_xarray_cm.coords['plateau']],
                                        dims=['plateau'],
                                        coords={'plateau': density_xarray_cm.coords['plateau']})
        density_scaling['y'] = inter_avg_y_time / y_integral

    # debug
    # print("x integral:", x_integral, "y integral:", y_integral, sep="\n")
    #

    # INTERFEROMETRY - DENSITY TIME ALIGNMENT #
    # _______________________________________ #

    # Find index of interferometry collapse; at point with most negative slope
    # debug
    # print("Interferometry collapse time:", inter_collapse_time.item())
    #

    # Find time of density collapses in x and y (average out multidimensional components)

    # debug
    # print("Density collapse times:", density_collapse_times)
    #

    # SCALING FACTOR CALCULATION #
    # __________________________ #

    # Find average interferometry value at each time (for each x-adjusted time and each y-adjusted time)
    # x_time_interferometry = for t in inter_data.coords['time']:

    # debug
    # print("\nInter x time data:", inter_avg_x_time, "\nInter y time data:", inter_avg_y_time, sep="\n")
    inter_data.plot()
    # inter_collapse_index.
    plt.show()
    #

    # density_scale_x =
    # density_scale_y =

    density_scale_factor = (density_scaling['x'] if has_x else 0
                            + density_scaling['y'] if has_y else 0) / (has_x + has_y)  # average x and y scaling factors
    print("Density scale factor:", density_scale_factor)
    calibrated_density_xarray = density_xarray * density_scale_factor

    return (density_scaling['x'] if has_x else None,
            density_scaling['y'] if has_y else None), (has_x, has_y), calibrated_density_xarray


def to_real_units_interferometry(interferometry_data_array, interferometry_time_array):
    area_factor = 8. * 10 ** 13 / (u.cm ** 2)  # from MATLAB code
    time_factor = ((4.88 * 10 ** -5) * u.s).to(u.ms)  # from MATLAB code
    return interferometry_data_array * area_factor, interferometry_time_array * time_factor
