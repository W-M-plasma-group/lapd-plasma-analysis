# Add comments
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np


def linear_diagnostic_plot(diagnostics_dataset, diagnostic='T_e', plot='contour'):
    # Plots the given diagnostic(s) from the dataset in the given style

    diagnostic_list = np.atleast_1d(diagnostic)

    if diagnostics_dataset.sizes['y'] == 1:
        linear_dimension = 'x'
    elif diagnostics_dataset.sizes['x'] == 1:
        linear_dimension = 'y'
    else:
        # if diagnostics_dataset.sizes['x'] > 1 and diagnostics_dataset.sizes['y'] > 1:
        raise ValueError("x and y dimensions have lengths " + str(diagnostics_dataset.shape[:2]) +
                         " both greater than 1. A linear plot cannot be made. Areal plots are not yet supported.")

    if diagnostics_dataset.sizes['time'] == 1:
        raise ValueError("Please pass a diagnostic dataset with multiple time measurements. "
                         "Single-time profiles are not yet supported.")
    linear_diagnostics_dataset = diagnostics_dataset.squeeze()

    """
    if np.any([choice not in linear_diagnostics_dataset for choice in diagnostic_list]):
        raise ValueError("The diagnostic choice input" + repr(diagnostic_list) + " contains invalid diagnostics. "
                         "Valid diagnostics are:" + repr(linear_diagnostics_dataset.keys()))
    """

    plot_types_1d = {'line'}
    plot_types_2d = {'contour', 'surface'}

    # Unsupported plot type
    if plot not in plot_types_1d and plot not in plot_types_2d:
        warn("The type of plot " + repr(plot) + " is not in the supported plot type list "
             + repr(plot_types_1d | plot_types_2d) + ". Defaulting to contour plot.")
        plot = 'contour'
    # 1D plot type
    if plot in plot_types_1d:
        diagnostics_dataset_1d = steady_state_time_only(linear_diagnostics_dataset,
                                                        steady_state_start_time=2,
                                                        steady_state_end_time=5).mean('time')
        for choice in diagnostic_list:
            if check_diagnostic(diagnostics_dataset_1d, choice):
                linear_plot_1d(diagnostics_dataset_1d[choice], plot, choice, linear_dimension)
    # 2D plot type
    elif plot in plot_types_2d:
        for choice in diagnostic_list:
            if check_diagnostic(linear_diagnostics_dataset, choice):
                linear_plot_2d(linear_diagnostics_dataset[choice], plot, choice, linear_dimension)


def linear_plot_1d(diagnostic_array_1d, plot_type, diagnostic_name, linear_dimension):
    diagnostic_array_1d.plot(x=linear_dimension)
    plt.title(get_title(diagnostic_name) + " " + plot_type + " plot")
    plt.show()


def linear_plot_2d(diagnostic_array, plot_type, diagnostic_name, linear_dimension):
    if plot_type == "contour":
        diagnostic_array.plot.contourf(x='time', y=linear_dimension, robust=True)
    elif plot_type == "surface":
        # TODO raise issue on xarray about surface plotting not handling np.nan properly in choosing colormap
        # crop outlier diagnostic values; make sure that this is acceptable data handling
        # note: only crops high values; cropping low and high would require numpy.logical_and, aka "both"
        q1, q3 = np.nanpercentile(diagnostic_array, [25, 75])
        cropped_var_xarray = diagnostic_array.where(diagnostic_array <= q3 + 1.5 * (q3 - q1))
        cropped_var_xarray.plot.surface(x='time', y=linear_dimension)  # , cmap='viridis'

    plt.title(get_title(diagnostic_name) + " " + plot_type + " plot (2D)")
    plt.show()


def check_diagnostic(diagnostic_dataset, choice):
    if choice in diagnostic_dataset:
        return True
    else:
        warn("The diagnostic " + repr(choice) + " is not in the list of acceptable diagnostics "
             + repr(list(diagnostic_dataset.keys())) + ". It will be ignored.")
        return False


def get_title(diagnostic):

    full_names = {'V_P': "Plasma potential",
                  'V_F': "Floating potential",
                  'I_es': "Electron saturation current",
                  'I_is': "Ion saturation current",
                  'n_e': "Electron density",
                  'n_i': "Ion density",
                  'T_e': "Electron temperature",
                  'n_i_OML': "Ion density (OML)",
                  'hot_fraction': "Hot electron fraction",
                  'T_e_cold': "Cold electron temperature",
                  'T_e_hot': "Hot electron temperature",
                  'T_e_avg': "Average bimaxwellian electron temperature"}

    if diagnostic in full_names:
        return full_names[diagnostic]
    else:
        return diagnostic


"""
def linear_time_diagnostic_plot(diagnostics_dataset, diagnostic='T_e', plot='contour'):
    # Plots the given diagnostic(s) from the dataset in the given style

    diagnostic_list = np.atleast_1d(diagnostic)

    if diagnostics_dataset.sizes['x'] > 1 and diagnostics_dataset.sizes['y'] > 1:
        raise ValueError("x and y dimensions have lengths " + str(diagnostics_dataset.shape[:2]) +
                         " both greater than 1. A linear plot cannot be made. Areal plots are not yet supported.")
    linear_diagnostics_dataset = diagnostics_dataset.squeeze()

    if np.any([choice not in diagnostics_dataset for choice in diagnostic_list]):
        # print([choice not in diagnostics_dataset.keys() for choice in diagnostic_list])
        raise ValueError("The diagnostic choice input" + repr(diagnostic_list) +
                         " contains invalid diagnostics. Valid diagnostics are:" + repr(diagnostics_dataset.keys()))

    allowed_plot_types = {'contour', 'surface'}
    if plot not in allowed_plot_types:
        warn("The type of plot " + repr(plot) + " is not in the supported plot type list " + repr(allowed_plot_types) +
             ". Defaulting to contour plot.")
        plot = 'contour'

    for choice in diagnostic_list:
        diagnostic_xarray = linear_diagnostics_dataset[choice]
        linear_time_crop_plot(diagnostic_xarray, plot_type=plot, diagnostic_name=choice)

        # check number of dimensions for validity?


def linear_time_crop_plot(pos_time_var_xarray, plot_type, diagnostic_name):

    full_names = {'V_P': "Plasma potential",
                  'V_F': "Floating potential",
                  'I_es': "Electron saturation current",
                  'I_is': "Ion saturation current",
                  'n_e': "Electron density",
                  'n_i': "Ion density",
                  'T_e': "Electron temperature",
                  'n_i_OML': "Ion density (OML)",
                  'hot_fraction': "Hot electron fraction",
                  'T_e_cold': "Cold electron temperature",
                  'T_e_hot': "Hot electron temperature",
                  'T_e_avg': "Average bimaxwellian electron temperature"}

    if plot_type == "contour":
        pos_time_var_xarray.plot.contourf(x='time', y='x', robust=True)
    elif plot_type == "surface":
        # TODO raise issue on xarray about surface plotting not handling np.nan properly in choosing colormap
        # crop outlier diagnostic values; make sure that this is acceptable data handling
        # note: only crops high values; cropping low and high would require numpy.logical_and, aka "both"
        q1, q3 = np.nanpercentile(pos_time_var_xarray, [25, 75])
        cropped_var_xarray = pos_time_var_xarray.where(pos_time_var_xarray <= q3 + 1.5 * (q3 - q1))
        color_min, color_max = np.nanmin(cropped_var_xarray), np.nanmax(cropped_var_xarray)
        cropped_var_xarray.plot.surface(x='time', y='x', cmap='viridis', vmin=color_min, vmax=color_max)

    if diagnostic_name in full_names:
        plt.title("Linear-time plot: " + full_names[diagnostic_name])
    plt.show()


def linear_diagnostic_plot(diagnostics_dataset, diagnostic='T_e', steady_state_start_time=2, steady_state_end_time=5):
  

    # Take mean along the time axis
    timeless_diagnostics = steady_state_time_only(diagnostics_dataset).mean('time') if 'time' in diagnostics_dataset.coords else diagnostics_dataset
    linear_diagnostics = timeless_diagnostics.squeeze()
    # print(linear_diagnostics)
    # print(diagnostic)
    linear_diagnostics[diagnostic].plot()
    plt.show()
"""


def steady_state_time_only(diagnostics_dataset, steady_state_start_time=2, steady_state_end_time=5):
    return diagnostics_dataset.where(np.logical_and(
        diagnostics_dataset.coords['time'] >= steady_state_start_time,
        diagnostics_dataset.coords['time'] <= steady_state_end_time), drop=True)


