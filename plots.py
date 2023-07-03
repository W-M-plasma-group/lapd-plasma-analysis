# Add comments
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy import visualization
from diagnostics import value_safe, unit_safe
# matplotlib.use('TkAgg')
# TODO investigate
matplotlib.use('QtAgg')


def plot_line_diagnostic_by(diagnostics_datasets: list, plot_diagnostic, port_selector, attribute, steady_state_by_runs,
                            tolerance=1/2, share_y=True):
    # diagnostics_datasets is a list of different HDF5 datasets

    attributes = np.atleast_1d(attribute)
    if len(attributes) > 2:
        raise ValueError("Cannot categorize line plots by more than two attributes")

    # outer_inverses = np.argsort([dataset[attributes[1]] for dataset in diagnostics_datasets]) if two_attrs else [None]
    # diagnostics_datasets_sorted = np.sort(diagnostics_datasets, order=np.flip(attribute))
    diagnostics_datasets_sorted = diagnostics_datasets  # not sorted yet
    for attr in attributes:
        try:
            diagnostics_datasets_sorted.sort(key=lambda d: d.attrs[attr])
        except KeyError:
            raise KeyError("Key error for key", repr(attr))
    outer_values = [dataset.attrs[attributes[1]] for dataset in diagnostics_datasets_sorted]
    outer_quants = u.Quantity([value_safe(value) for value in outer_values], unit_safe(outer_values[0]))
    outer_unique, outer_indexes = np.unique(outer_quants, return_index=True) if len(attributes) == 2 else ([None], [0])
    outer_bounds = np.append(outer_indexes, len(outer_values))

    visualization.quantity_support()
    plt.rcParams['figure.figsize'] = (4 + 4 * len(outer_indexes), 6)
    fig, axes = plt.subplots(1, len(outer_indexes), sharey="row")

    fig.suptitle(get_title(plot_diagnostic), size=18)
    for outer_index in range(len(outer_unique)):
        outer_val = outer_unique[outer_index]
        ax = np.atleast_1d(axes)[outer_index]

        datasets = diagnostics_datasets_sorted[outer_bounds[outer_index]:outer_bounds[outer_index + 1]]
        num_datasets = outer_bounds[outer_index + 1] - outer_bounds[outer_index]

        color_map = matplotlib.colormaps["plasma"](np.linspace(0, 1, num_datasets))
        for inner_index in range(num_datasets):
            dataset = port_selector(datasets[inner_index])  # TODO allow looping through multiple datasets returned
            inner_val = dataset.attrs[attributes[0]]

            linear_dimension = validate_dataset_dims(dataset.sizes)
            linear_da = dataset.squeeze()[plot_diagnostic]

            line_diagnostic = steady_state_only(linear_da,
                                                steady_state_plateaus=steady_state_by_runs[0])  # TODO finish this!
            line_diagnostic_mean = line_diagnostic.mean('time', keep_attrs=True)
            line_diagnostic_std = line_diagnostic.std('time', ddof=1)

            line_diagnostic_points = line_diagnostic_mean.where(
                line_diagnostic_std < np.abs(line_diagnostic_mean).mean() * tolerance)
            # TODO incorporate tolerance - maybe compare std to (std if you find two-element moving ave. over time)?

            ax.plot(line_diagnostic.coords[linear_dimension], line_diagnostic_points,
                    color=color_map[inner_index], label=str(inner_val))
            ax.tick_params(axis="y", left=True, labelleft=True)
        ax.title.set_text((attribute[1] + ": " + str(outer_val) if len(attribute) == 2 else "")
                          + "\nColor: " + attribute[0])
        ax.legend()
    plt.tight_layout()
    plt.show()


def line_time_diagnostic_plot(diagnostics_dataset, diagnostic, plot_type, steady_state, show=True, **kwargs):
    # Plots the given diagnostic(s) from the dataset in the given style

    try:
        run_name = diagnostics_dataset.attrs['Run name'] + "\n"
    except KeyError:
        run_name = ""

    linear_dimension = validate_dataset_dims(diagnostics_dataset.sizes)
    linear_diagnostics_dataset = diagnostics_dataset.squeeze()

    diagnostic_list = np.atleast_1d(diagnostic)

    """
    if np.any([choice not in linear_diagnostics_dataset for choice in diagnostic_list]):
        raise ValueError("The diagnostic choice input" + repr(diagnostic_list) + " contains invalid diagnostics. "
                         "Valid diagnostics are:" + repr(linear_diagnostics_dataset.keys()))
    """

    plot_types_1d = {'line'}
    plot_types_2d = {'contour', 'surface'}

    # Unsupported plot type
    if plot_type not in plot_types_1d and plot_type not in plot_types_2d:
        warn("The type of plot " + repr(plot_type) + " is not in the supported plot type list "
             + repr(plot_types_1d | plot_types_2d) + ". Defaulting to contour plot.")
        plot_type = 'contour'
    # 1D plot type
    if plot_type in plot_types_1d:
        linear_diagnostics_dataset_1d = steady_state_only(linear_diagnostics_dataset,
                                                          steady_state_plateaus=steady_state).mean('time')
        for choice in diagnostic_list:
            if check_diagnostic(linear_diagnostics_dataset_1d, choice):
                linear_plot_1d(linear_diagnostics_dataset_1d[choice], linear_dimension)
                plt.title(run_name + get_title(choice) + " " + plot_type + " plot")
                if show:
                    plt.show()
    # 2D plot type
    elif plot_type in plot_types_2d:
        for choice in diagnostic_list:
            if check_diagnostic(linear_diagnostics_dataset, choice):
                linear_plot_2d(linear_diagnostics_dataset[choice], plot_type, linear_dimension)
                plt.title(run_name + get_title(choice) + " " + plot_type + " plot (2D)")
                if show:
                    plt.show()


def validate_dataset_dims(diagnostics_dataset_sizes):

    if diagnostics_dataset_sizes['y'] == 1:
        linear_dimension = 'x'
    elif diagnostics_dataset_sizes['x'] == 1:
        linear_dimension = 'y'
    else:
        raise ValueError("x and y dimensions have lengths " + str(diagnostics_dataset_sizes[:2]) +
                         " both greater than 1. A linear plot cannot be made. Areal plots are not yet supported.")
    if diagnostics_dataset_sizes['time'] == 1:
        raise ValueError("Single-time profiles are not supported")

    return linear_dimension


def linear_plot_1d(diagnostic_array_1d, linear_dimension):
    diagnostic_array_1d.plot(x=linear_dimension)


def linear_plot_2d(diagnostic_array, plot_type, linear_dimension, trim_colormap=True):
    q1, q3, reasonable_max = np.nanpercentile(diagnostic_array, [25, 75, 98])
    trim_max = min(reasonable_max, q3 + 1.5 * (q3 - q1))
    plot_params = {'x': 'time', 'y': linear_dimension}
    if trim_colormap:
        plot_params = {**plot_params, 'vmax': trim_max}
    if plot_type == "contour":
        diagnostic_array.plot.contourf(**plot_params, robust=True)
    elif plot_type == "surface":
        trimmed_diagnostic_array = diagnostic_array.where(diagnostic_array <= trim_max)
        trimmed_diagnostic_array.plot.surface(**plot_params)
        # TODO raise issue on xarray about surface plotting not handling np.nan properly in choosing colormap
        """
        # crop outlier diagnostic values; make sure that this is acceptable data handling
        # note: only crops high values; cropping low and high would require numpy.logical_and, aka "both"
        """


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
                  'T_e_avg': "Average bimaxwellian electron temperature",
                  'P_e': "Electron pressure",
                  'n_e_cal': "Calibrated electron density"}

    if diagnostic in full_names:
        return full_names[diagnostic]
    else:
        return diagnostic


# TODO remove?
def steady_state_only(diagnostics_dataset, steady_state_plateaus: tuple):

    # return diagnostics_dataset[{'time': slice(*steady_state_plateaus)}]
    return diagnostics_dataset.where(np.logical_and(diagnostics_dataset.plateau >= steady_state_plateaus[0],
                                                    diagnostics_dataset.plateau <= steady_state_plateaus[1]), drop=True)
