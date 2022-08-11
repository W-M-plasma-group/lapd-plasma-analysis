import warnings

import numpy as np
import astropy.units as u
import xarray as xr

from hdf5reader import *
from bapsflib import lapd


def interferometry_calibration(density_xarray, interferometry_filename, steady_state, core_region=26. * u.cm):

    density_data = density_xarray * (1 / u.m ** 3).to(1 / u.cm ** 3).value  # density in 1/cm^3

    # Select core region in x and y
    x_mask, y_mask = abs(density_data.x) < core_region.value, abs(density_data.y) < core_region.value
    core_density_data = density_data.where(np.logical_and(x_mask, y_mask), drop=True)

    # Interpolate nan values linearly to allow trapezoidal integration over incomplete dimensions; 10 cm max gap
    core_density_data = core_density_data.interpolate_na(dim='x', use_coordinate=True, max_gap=10.)
    core_density_data = core_density_data.interpolate_na(dim='y', use_coordinate=True, max_gap=10.)

    # Detect spatial dimensions (1D or 2D)
    spatial_dimensions = (['x'] if core_density_data.sizes['x'] > 1 else []
                          + ['y'] if core_density_data.sizes['y'] > 1 else [])

    # Open file to decide to use new 96 GHz interferometry or old 56 GHz interferometry data
    inter_file = lapd.File(interferometry_filename)
    run_description = inter_file.info['run description']

    if "fringes" in run_description:
        print("Using high-frequency interferometer")  # March 2022

        num_fringes = find_fringes_metadata(run_description)

        density_scale_factor = interferometry_calibration_96ghz(core_density_data, num_fringes, spatial_dimensions)

        calibrated_density_xarray = density_xarray * density_scale_factor

        print("Interferometry calibration factor:", density_scale_factor.item())

    else:
        print("Using low-frequency interferometer")  # April 2018

        # Read in interferometry data ("interferometry" abbreviated as "inter" in variable names)
        with open_hdf5(interferometry_filename) as inter_file:
            inter_raw = np.array(
                item_at_path(inter_file, '/MSI/Interferometer array/Interferometer [0]/Interferometer trace/'))

            density_scale_factor = interferometry_calibration_56ghz(core_density_data, inter_raw, spatial_dimensions)

        # DENSITY SCALING #
        # _______________ #

        # Return the calibrated electron density data only in the steady state region (given by plateau indices)
        steady_state_scale_factor = density_scale_factor.where(np.logical_and(density_xarray.plateau >= steady_state[0],
                                                                              density_xarray.plateau <= steady_state[1]))

        calibrated_density_xarray = density_xarray * steady_state_scale_factor

        print("Average steady-state interferometry calibration factor:", steady_state_scale_factor.mean().item())

    return calibrated_density_xarray


def find_fringes_metadata(run_description: str) -> int:

    # Find number of fringes
    fringe_index = run_description.index("fringes")
    num_location = run_description.rindex(" ", 0, fringe_index - 1) + 1
    return int(run_description[num_location:fringe_index - 1])


def interferometry_calibration_96ghz(core_density_data, num_fringes, spatial_dimensions):
    print(num_fringes, "fringes in interferometry measurement")

    # PEAK INTERFEROMETRY DENSITY #
    # ___________________________ #

    # n*Diam = Nfringes * 1.88e13/cm^2   (HDF5 run description)
    peak_inter_line_density = num_fringes * 1.88e13 / u.cm ** 2   # / (2 * core_region) excluded bc already integ across
    # print("Peak interferometer density:", peak_inter_line_density)

    # PEAK LANGMUIR DENSITY #
    # _____________________ #

    integral = core_density_data
    for dim in spatial_dimensions:
        integral = integral.integrate(dim)

    peak_langmuir_line_density = integral.max('time', skipna=True)
    # print("Peak Langmuir density:", peak_langmuir_line_density.item() / u.cm ** 3)

    return peak_inter_line_density / peak_langmuir_line_density


def interferometry_calibration_56ghz(core_density_data, interferometry_data, spatial_dimensions):
    r"""
    Calibrates density data from sweep probe to measurements from interferometry probe.

    Parameters
    ----------
    :param core_density_data: DataArray of n_e in cm^-3, core region only
    :param interferometry_data: ndarray of raw interferometry data from HDF5 file
    :param spatial_dimensions: list of spatial dimensions of density data, e.g. ('x', 'y')
    :return: DataArray of dimensionless scale factors for density data
    """

    # INTERFEROMETRY DATA #
    # ___________________ #

    # print("Interferometry raw data shape:", interferometry_data.shape)

    inter_means_abstract = np.mean(interferometry_data, axis=0)
    inter_time_abstract = np.arange(len(inter_means_abstract))

    # Create interferometry DataArray with interferometry time as coordinates
    inter_values, inter_time = to_real_56ghz_interferometry_units(inter_means_abstract, inter_time_abstract)
    inter_data = xr.DataArray(inter_values, coords=(('time', inter_time, {'units': str(u.ms)}),))

    # DEBUG
    """
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.plot(inter_raw[0], 'r:', label="Row 1")
    plt.plot(inter_raw[1], 'b:', label="Row 2")
    plt.plot(inter_means_abstract, 'm-', label="Average")
    plt.title("Raw interferometer measurements, May 2018")
    plt.legend()
    plt.show()
    # """
    # print("Shape of processed inter_data array:", inter_data.shape)
    """
    inter_data.plot()
    plt.title("Data from interferometer, 2022 run")
    plt.show()
    # """
    #

    # DENSITY DATA #
    # ____________ #
    # TODO correct interferometry issues for bimaxwellian diagnostics

    if len(spatial_dimensions) > 2:
        warnings.warn("Two-dimensional density interferometry calibration code is incomplete "
                      "and may give inaccurate or unexpected results.")

    # INTERFEROMETRY - DENSITY TIME ALIGNMENT #
    # _______________________________________ #

    # Find index of interferometry collapse; at point with most negative slope
    inter_collapse_time = inter_data.differentiate('time').idxmin('time')

    # Find average step in density (n_e) time coordinate by dividing total time elapsed by number of time measurements
    density_time_coord = core_density_data.coords['time']
    dt = (density_time_coord[-1] - density_time_coord[0]) / len(density_time_coord)  # time step for density time coord.

    # DENSITY LINE INTEGRALS #
    # ______________________ #

    density_scales = dict()  # create empty dictionary to hold x and y scaling factors where necessary

    for dim in spatial_dimensions:
        # Line integral along x dimension
        integral = core_density_data.integrate(dim)

        # Find time of density collapse in dimension
        # Take average to get one single average dimension density collapse time (scalar)
        density_collapse_time = integral.idxmax('time').mean()

        # Align times so that interferometry data collapse matches x density data max
        aligned_time = inter_data.coords['time'] - inter_collapse_time + density_collapse_time
        inter_data = inter_data.assign_coords({dim+'_time': ('time', aligned_time.data)})

        # Average all interferometry measurements into data point with the closest corresponding density time coordinate
        inter_avg_time = crunch_data(inter_data, dim+'_time', core_density_data.coords['time'], dt)

        density_scales[dim] = inter_avg_time / integral

    # Average together x and y scale factors, if they exist, to get scale factor array in right number of dimensions
    return sum(density_scales[dim] for dim in spatial_dimensions) / len(spatial_dimensions)


def to_real_56ghz_interferometry_units(interferometry_data_array, interferometry_time_array):
    area_factor = 8e13 / (u.cm ** 2)            # from MATLAB code
    # TODO find out which correct
    time_factor = (4.88e-5 * u.s).to(u.ms)      # from MATLAB code
    # time_factor = (4.0e-5 * u.s).to(u.ms)     # from HDF5 header data
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
