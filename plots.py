from warnings import warn

import matplotlib
from astropy import visualization

from helper import *

# matplotlib.use('TkAgg')
# matplotlib.use('QtAgg')


def multiplot_line_diagnostic(diagnostics_datasets: list[xr.Dataset], plot_diagnostic, isweep_choices, x_dim='x',
                              steady_state_by_runs=None, core_rad=None, attribute=None, tolerance=np.nan):
    r"""

    :param diagnostics_datasets: list of xarray Datasets
    :param plot_diagnostic: string identifying label of desired diagnostics
    :param isweep_choices:
    :param x_dim:
    :param steady_state_by_runs:
    :param core_rad:
    :param attribute:
    :param tolerance:
    :return:
    """
    # TODO generalize steady_state_by_runs, add curve_dimension to control what different colors represent

    linestyles = ("solid", "dotted", "dashed", "dashdot")
    x_dims = ['x', 'y', 'time']
    if x_dim not in x_dims:
        raise ValueError(f"Invalid dimension {repr(x_dim)} against which to plot diagnostic data. "
                         f"Valid x dimensions are {repr(x_dims)}")

    if attribute is None:
        attribute = [attr for attr in diagnostics_datasets[0].attrs if "Nominal" in attr]
    attributes = np.atleast_1d(attribute)
    if len(attributes) > 2:
        # TODO detect/fix
        # raise ValueError("Cannot currently categorize line plots that differ in more than two attributes")
        warn(f"Can currently only categorize line plots by two attributes. Selecting last two: {attributes[-2:]}")
        attributes = attributes[-2:]

    diagnostics_datasets_sorted = diagnostics_datasets  # not sorted yet
    for attr in attributes:
        try:
            diagnostics_datasets_sorted.sort(key=lambda d: d.attrs[attr])
        except KeyError:
            raise KeyError("Key error for key " + repr(attr))
    outer_values = [dataset.attrs[attributes[-1]] for dataset in diagnostics_datasets_sorted]
    outer_quants = u.Quantity([value_safe(value) for value in outer_values], unit_safe(outer_values[0]))
    outer_unique, outer_indexes = np.unique(outer_quants, return_index=True) if len(attributes) == 2 else ([None], [0])
    outer_bounds = np.append(outer_indexes, len(outer_values))

    visualization.quantity_support()
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (3 + 3 * len(outer_indexes), 4)
    fig, axes = plt.subplots(1, len(outer_indexes), sharey="row")

    fig.suptitle(get_title(plot_diagnostic), size=18)
    for outer_index in range(len(outer_unique)):
        outer_val = outer_unique[outer_index]
        ax = np.atleast_1d(axes)[outer_index]

        datasets = diagnostics_datasets_sorted[outer_bounds[outer_index]:outer_bounds[outer_index + 1]]
        num_datasets = outer_bounds[outer_index + 1] - outer_bounds[outer_index]

        color_map = matplotlib.colormaps["plasma"](np.linspace(0, 0.9, num_datasets))

        y_limits = []
        for inner_index in range(num_datasets):

            ds_s = isweep_selector(datasets[inner_index], isweep_choices)
            for i in range(len(ds_s)):  # isweep index
                dataset = ds_s[i]

                inner_val = dataset.attrs[attributes[0]]

                linear_dimension = get_valid_linear_dimension(dataset.sizes)
                linear_da = dataset.squeeze()[plot_diagnostic]

                linear_da_steady_state = steady_state_only(linear_da,
                                                           steady_state_plateaus=steady_state_by_runs[inner_index])
                linear_da_mean = linear_da_steady_state.mean(['shot', 'time'], keep_attrs=True)
                linear_da_std = linear_da_steady_state.std(['shot',   'time'], ddof=1, keep_attrs=True)
                # 95% (~two standard deviation) confidence interval
                # TODO fix this. Need to consider number of non-NaN entries only
                linear_da_error = linear_da_std * 1.96 / np.sqrt(linear_da_steady_state.sizes['shot']
                                                                 * linear_da_steady_state.sizes['time'])

                # Filter out certain points due to inconsistent data (likely random noise that skews average higher)
                if np.isfinite(tolerance):
                    da_median = linear_da_mean.median(keep_attrs=True)
                    linear_da_mean = linear_da_mean.where(linear_da_std < tolerance * da_median)  # TODO hardcoded

                if np.isfinite(da_mean).any():
                    ax.errorbar(da_mean.coords[x_dim], da_mean, yerr=linear_da_error,
                                color=color_map[inner_index], linestyle=linestyles[i], label=str(inner_val))
                ax.set_xlabel(da_mean.coords[x_dim].attrs['units'])
                ax.set_ylabel(da_mean.attrs['units'])

                # TODO a bit hardcoded
                """da_core_steady_state_max = steady_state_only(
                    da.where(np.logical_and(*in_core([da.x, da.y], core_rad)), drop=True),
                    steady_state_by_runs[inner_index]).max().item()
                """
                da_core_steady_state_max = core_steady_state(da, core_rad, steady_state_by_runs[inner_index]
                                                             ).max().item()
                y_limits += [np.min([1.1 * da_core_steady_state_max,
                                     2 * core_steady_state(da, core_rad=core_rad, operation="mean",
                                                           dims_to_keep=["isweep"]).max().item()])]
        ax.set_ylim(0, np.max(y_limits))
        ax.tick_params(axis="y", left=True, labelleft=True)
        ax.title.set_text(((str(attributes[1]) + ": " + str(outer_val)) if len(attributes) == 2 else '')
                          + f"\nColor: {attributes[0]}"
                          + f"\nIsweep styles: {linestyles}")
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_line_diagnostic(diagnostics_ds_s: list[xr.Dataset], diagnostic, plot_type, steady_state, show=True,
                         shot_mode="mean", tolerance=np.nan):
    # Plots the given diagnostic(s) from the dataset in the given style

    linear_ds_s = []
    linear_dimensions = []
    run_names = []
    for diagnostics_ds in diagnostics_ds_s:
        run_names += [f"{diagnostics_ds.attrs['Exp name']}, {diagnostics_ds.attrs['Run name']}"]

        linear_dimensions += [get_valid_linear_dimension(diagnostics_ds.sizes)]
        pre_linear_ds = diagnostics_ds.squeeze()
        linear_ds_std = pre_linear_ds.std(dim='shot', ddof=1, keep_attrs=True)

        if shot_mode == "mean":
            linear_ds = pre_linear_ds.mean(dim='shot', keep_attrs=True).squeeze()
        elif shot_mode == "all":
            raise NotImplementedError("Shot handling mode 'all' not yet supported for plotting")
        else:
            raise NotImplementedError(f"Shot handling mode {repr(shot_mode)} not currently implemented for plotting")

        # Filter out certain points due to inconsistent data (likely random noise that skews average higher)
        if np.isfinite(tolerance):  # note: an insidious error was made here with (tolerance != np.nan)
            da_mean = linear_ds.mean()
            linear_ds = linear_ds.where(linear_ds_std < tolerance * da_mean)
        linear_ds_s += [linear_ds]

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
        linear_ds_s_1d = [steady_state_only(linear_ds, steady_state_plateaus=steady_state).mean('time')
                          for linear_ds in linear_ds_s]
        for key in diagnostic_list:
            for d in range(len(linear_ds_s_1d)):
                linear_ds_1d = linear_ds_s_1d[d]
                linear_plot_1d(linear_ds_1d[key], linear_dimensions[d])
            plot_title = f"{run_names[0]}\n{get_title(key)} {plot_type} plot"
            if hasattr(linear_ds_s_1d[0], "facevector"):
                plot_title += f"\nLinear combination of faces: {linear_ds_s_1d[0].attrs['facevector']}"
            plt.title(plot_title)
            plt.tight_layout()
            if show:
                plt.show()
    # 2D plot type
    elif plot_type in plot_types_2d:
        for key in diagnostic_list:
            for d in range(len(linear_ds_s)):
                try:
                    linear_plot_2d(linear_ds_s[d][key], plot_type, linear_dimensions[d])
                    plot_title = f"{run_names[d]}\n{get_title(key)} {plot_type} plot (2D)"
                    if hasattr(linear_ds_s[0], "facevector"):
                        plot_title += f"\nLinear combination of faces: {linear_ds_s[d].attrs['facevector']}"
                    plt.title(plot_title)
                    plt.tight_layout()
                    if show:
                        plt.show()
                except ValueError as e:
                    print(f"Problem plotting {key} for {linear_ds_s[d].attrs['Run name']}:"
                          f"\n{repr(e)}")


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
    q1, q3, reasonable_max = np.nanpercentile(diagnostic_array, [25, 75, 98])
    trim_max = min(reasonable_max, q3 + 1.5 * (q3 - q1))
    # print(q1, q3, trim_max)  # debug
    plot_params = {'x': 'time', 'y': linear_dimension}
    if trim_colormap:
        plot_params = {**plot_params, 'vmax': trim_max}
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


def get_title(diagnostic: str) -> str:

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
                  'n_e_cal': "Calibrated electron density",
                  'nu_ei': "Electron-ion collision frequency"}

    for key in sorted(list(full_names.keys()), key=len, reverse=True):
        diagnostic = diagnostic.replace(key, full_names[key])

    return diagnostic
