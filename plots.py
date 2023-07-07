# Add comments
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import astropy.units as u
from astropy import visualization
from diagnostics import value_safe, unit_safe
# matplotlib.use('TkAgg')
# TODO investigate
matplotlib.use('QtAgg')


def multiplot_line_diagnostic(diagnostics_datasets: list[xr.Dataset], plot_diagnostic, port_selector,
                              steady_state_by_runs, attribute=None, tolerance=np.nan):
    # diagnostics_datasets is a list of different HDF5 datasets

    if attribute is None:
        attribute = [attr for attr in diagnostics_datasets[0].attrs if "Nominal" in attr]
    attributes = np.atleast_1d(attribute)
    if len(attributes) > 2:
        raise ValueError("Cannot currently categorize line plots by more than two attributes")

    # outer_inverses = np.argsort([dataset[attributes[1]] for dataset in diagnostics_datasets]) if two_attrs else [None]
    # diagnostics_datasets_sorted = np.sort(diagnostics_datasets, order=np.flip(attribute))
    diagnostics_datasets_sorted = diagnostics_datasets  # not sorted yet
    for attr in attributes:
        try:
            diagnostics_datasets_sorted.sort(key=lambda d: d.attrs[attr])
        except KeyError:
            raise KeyError("Key error for key " + repr(attr))
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

            linear_dimension = get_valid_linear_dimension(dataset.sizes)
            linear_da = dataset.squeeze()[plot_diagnostic]

            linear_da_steady_state_mean = steady_state_only(linear_da,
                                                            steady_state_plateaus=steady_state_by_runs[inner_index]
                                                            ).mean('time', keep_attrs=True)
            linear_da_mean = linear_da_steady_state_mean.mean('shot', keep_attrs=True)
            linear_da_std = linear_da_steady_state_mean.std('shot', ddof=1, keep_attrs=True)

            # Filter out certain points due to inconsistent data (likely random noise that skews average higher)
            if np.isfinite(tolerance):
                da_mean = linear_da_mean.mean()
                linear_da_mean = linear_da_mean.where(linear_da_std < tolerance * da_mean)

            ax.errorbar(linear_da_mean.coords[linear_dimension], linear_da_mean, yerr=linear_da_std,
                        fmt="-", color=color_map[inner_index], label=str(inner_val))
            # ax.plot(line_diagnostic.coords[linear_dimension], line_diagnostic_points,
            #         color=color_map[inner_index], label=str(inner_val))
            ax.set_xlabel(linear_da_mean.coords[linear_dimension].attrs['units'])
            ax.set_ylabel(linear_da_mean.attrs['units'])
        ax.tick_params(axis="y", left=True, labelleft=True)
        ax.title.set_text((attribute[1] + ": " + str(outer_val) if len(attribute) == 2 else "")
                          + "\nColor: " + attribute[0])
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_line_diagnostic(diagnostics_ds: xr.Dataset, diagnostic, plot_type, steady_state, show=True,
                         shot_mode="mean", tolerance=np.nan):
    # Plots the given diagnostic(s) from the dataset in the given style

    run_name = diagnostics_ds.attrs['Run name'] + "\n"

    linear_dimension = get_valid_linear_dimension(diagnostics_ds.sizes)
    pre_linear_ds = diagnostics_ds.squeeze()
    linear_ds_std = pre_linear_ds.std(dim='shot', ddof=1, keep_attrs=True)

    if shot_mode == "mean":
        linear_ds = pre_linear_ds.mean(dim='shot', keep_attrs=True).squeeze()
    elif shot_mode == "all":
        raise NotImplementedError("Shot handling mode 'all' not yet supported for plotting")
    else:
        raise NotImplementedError("Shot handling mode " + repr(shot_mode) + " not currently implemented for plotting")

    # Filter out certain points due to inconsistent data (likely random noise that skews average higher)
    if np.isfinite(tolerance):  # note: an insidious error was made here with (tolerance != np.nan)
        da_mean = linear_ds.mean()
        linear_ds = linear_ds.where(linear_ds_std < tolerance * da_mean)

    diagnostic_list = np.atleast_1d(diagnostic)

    plot_types_1d = {'line'}
    plot_types_2d = {'contour', 'surface'}

    # Unsupported plot type
    if plot_type not in plot_types_1d | plot_types_2d:
        warn("The type of plot " + repr(plot_type) + " is not in the supported plot type list "
             + repr(plot_types_1d | plot_types_2d) + ". Defaulting to contour plot.")
        plot_type = 'contour'
    # 1D plot type
    if plot_type in plot_types_1d:
        linear_ds_1d = steady_state_only(linear_ds, steady_state_plateaus=steady_state).mean('time')
        for key in diagnostic_list:
            # if check_diagnostic(linear_ds_1d, choice):
            linear_plot_1d(linear_ds_1d[key], linear_dimension)
            plt.title(run_name + get_title(key) + " " + plot_type + " plot")
            if show:
                plt.show()
    # 2D plot type
    elif plot_type in plot_types_2d:
        for key in diagnostic_list:
            # if check_diagnostic(linear_ds, choice):
            linear_plot_2d(linear_ds[key], plot_type, linear_dimension)
            plt.title(run_name + get_title(key) + " " + plot_type + " plot (2D)")
            if show:
                plt.show()


def get_valid_linear_dimension(diagnostics_dataset_sizes):

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
    # q1, q3, reasonable_max = np.nanpercentile(diagnostic_array, [25, 75, 98])
    q1, q3 = np.nanpercentile(diagnostic_array, [25, 75])
    # trim_max = min(reasonable_max, q3 + 1.5 * (q3 - q1))
    plot_params = {'x': 'time', 'y': linear_dimension}
    if trim_colormap:
        plot_params = {**plot_params, 'vmax': q3 + 1.5 * (q3 - q1)}  # trim_max
    if plot_type == "contour":
        diagnostic_array.plot.contourf(**plot_params, robust=True)
    elif plot_type == "surface":
        # trimmed_diagnostic_array = diagnostic_array.where(diagnostic_array <= trim_max); bad to cherry pick data?
        diagnostic_array.plot.surface(**plot_params)
        # TODO raise issue on xarray about surface plotting not handling np.nan properly in choosing colormap


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
