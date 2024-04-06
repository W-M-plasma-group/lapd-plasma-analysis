from warnings import warn

import matplotlib
from astropy import visualization

from bapsflib.lapd.tools import portnum_to_z
from langmuir.helper import *

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# matplotlib.use('TkAgg')
# matplotlib.use('QtAgg')


def multiplot_line_diagnostic(diagnostics_datasets: list[xr.Dataset], plot_diagnostic, isweep_choices, x_dim='x',
                              steady_state_by_runs=None, core_rad=None, attribute=None, tolerance=np.nan,
                              save_directory=""):
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

    marker_styles = (".", "+", "x", "^")   # , "1", "2", "3", "4")
    x_dims = ['x', 'y', 'time']
    if x_dim not in x_dims:
        raise ValueError(f"Invalid dimension {repr(x_dim)} against which to plot diagnostic data. "
                         f"Valid x dimensions are {repr(x_dims)}")

    if attribute is None:
        attribute = [attr for attr in diagnostics_datasets[0].attrs if "Nominal" in attr]
    attributes = np.atleast_1d(attribute)
    if len(attributes) > 2:
        # TODO detect/fix
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
    plt.rcParams['figure.figsize'] = (3 + 3 * len(outer_indexes), 4.5)
    fig, axes = plt.subplots(1, len(outer_indexes), sharey="row")

    for outer_index in range(len(outer_unique)):    # gas puff voltage index
        outer_val = outer_unique[outer_index]
        ax = np.atleast_1d(axes)[outer_index]

        datasets = diagnostics_datasets_sorted[outer_bounds[outer_index]:outer_bounds[outer_index + 1]]
        num_datasets = outer_bounds[outer_index + 1] - outer_bounds[outer_index]

        color_map = matplotlib.colormaps["plasma"](np.linspace(0, 0.9, num_datasets))

        y_limits = [0]
        for inner_index in range(num_datasets):     # discharge current index

            ds_s = isweep_selector(datasets[inner_index], isweep_choices)
            for i in range(len(ds_s)):              # isweep index
                ds = ds_s[i]

                inner_val = ds.attrs[attributes[0]]

                da = ds[plot_diagnostic]
                if x_dim in ('x', 'y'):  # only consider steady state
                    da = steady_state_only(da, steady_state_by_runs[inner_index])  # TODO deprecate w core_steady_state?
                elif x_dim == 'time':  # only consider core region
                    da = da.where(np.logical_and(*in_core([da.x, da.y], core_rad)), drop=True)

                dims_to_average_out = ['shot'] + [dim for dim in x_dims if dim != x_dim]
                da_mean = da.mean(dims_to_average_out, keep_attrs=True)
                da_std = da.std(dims_to_average_out, ddof=1, keep_attrs=True)
                # both of the above should have only one dimension left?

                # 95% (~two standard deviation) confidence interval
                non_nan_element_da = da.copy()
                non_nan_element_da[...] = ~np.isnan(da)
                effective_num_non_nan_per_std = non_nan_element_da.sum(dims_to_average_out)
                linear_da_error = da_std * 1.96 / np.sqrt(effective_num_non_nan_per_std - 1)  # unbiased

                if np.isfinite(da_mean).any():
                    ax.errorbar(da_mean.coords[x_dim], da_mean, yerr=linear_da_error, linestyle="none",
                                color=color_map[inner_index], marker=marker_styles[i],
                                label=str(inner_val) + (f" ({isweep_choices[i]})" if len(isweep_choices) > 1 else ""))
                ax.set_xlabel(da_mean.coords[x_dim].attrs['units'])
                ax.set_ylabel(da_mean.attrs['units'])

                # TODO a bit hardcoded; TODO document fill_na!
                da_core_steady_state_max = core_steady_state(da.fillna(0), core_rad, steady_state_by_runs[inner_index]
                                                             ).max().item()
                y_limits += [np.nanmin([1.1 * da_core_steady_state_max,
                                        2.5 * core_steady_state(da, core_rad, steady_state_by_runs[inner_index],
                                                                operation="median", dims_to_keep=["isweep"]
                                                                ).max().item()])]
        if not np.nanmax(y_limits) > 0:
            warn("A plot was assigned a NaN upper axis limit")
        ax.set_ylim(0, np.nanmax(y_limits))
        ax.tick_params(axis="y", left=True, labelleft=True)
        ax.title.set_text((str(attributes[1]) + ": " + str(outer_val)) if len(attributes) == 2 else '')
        ax.legend(title=str(attributes[0]) + (f"\n    (sweep choice)" if len(isweep_choices) > 1 else ""))
    fig.suptitle(get_title(plot_diagnostic), size=18)
    plt.tight_layout()
    if save_directory:
        plt.savefig(save_directory + "multiplot_line_" + plot_diagnostic + ".pdf")
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
            raise ValueError(f"Shot handling mode {repr(shot_mode)} not supported for plotting")

        # This is still here for contour plots!
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


def plot_parallel_diagnostic(datasets_split, steady_state_plateaus_runs_split, isweep_choice_center_split,
                             marker_styles_split, diagnostic, operation="mean", core_radius=26 * u.cm,
                             save_directory=""):
    plt.rcParams['figure.figsize'] = (6.5, 3.5)
    plt.rcParams['figure.dpi'] = 300

    anode_z = portnum_to_z(0).to(u.m)

    collision_frequencies = extract_collision_frequencies(datasets_split, core_radius, steady_state_plateaus_runs_split,
                                                          isweep_choice_center_split, operation)
    collision_frequencies_log_normalized = normalize(np.log(collision_frequencies), 0, 0.9)
    color_map = matplotlib.colormaps["plasma"](collision_frequencies_log_normalized)

    diagnostic_units = ""
    for i in range(len(datasets_split)):
        isweep_choices = (0, 2) if datasets_split[i].attrs['Exp name'] == "January_2024" else (0, 1)
        diagnostic_values = []
        zs = []

        if diagnostic not in datasets_split[i]:  # TODO needs testing
            continue
        diagnostic_means = core_steady_state(datasets_split[i][diagnostic], core_radius,
                                             steady_state_plateaus_runs_split[i], operation, dims_to_keep=("isweep",))

        for isweep_choice in isweep_choices:
            diagnostic_values += [diagnostic_means[{"isweep": isweep_choice}].item()]
            zs += [diagnostic_means[{"isweep": isweep_choice}].coords['z'].item()]
        zs = anode_z - (zs * u.Unit(diagnostic_means.coords['z'].attrs['units'])).to(u.m)  # convert to meters
        diagnostic_units = datasets_split[i][diagnostic].attrs['units']

        plt.plot(zs, diagnostic_values, marker=marker_styles_split[i], color=color_map[i], linestyle='none')
    plt.title(get_title(diagnostic) + " ", y=0.9, loc='right')  # ({operation})
    plt.xlabel("z position [m]")
    plt.ylabel(f"[{diagnostic_units}]", rotation=0, labelpad=25)   # {get_title(diagnostic)}
    normalizer = matplotlib.colors.LogNorm(vmin=np.min(collision_frequencies),
                                           vmax=np.max(collision_frequencies))
    color_bar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalizer, cmap='plasma'), ax=plt.gca())
    color_bar.set_label("nu_ei [Hz]\n(midplane)", rotation=0, labelpad=30)
    # color_bar.set_label("Midplane electron-ion \ncollision frequency [Hz]", rotation=90, labelpad=10)
    plt.tight_layout()
    if save_directory:
        plt.savefig(f"{save_directory}parallel_plot_{diagnostic}.pdf")
    plt.show()


def scatter_plot_diagnostics(datasets_split, diagnostics_to_plot_list, steady_state_plateaus_runs_split,
                             isweep_choice_center_split, marker_styles_split, operation="mean", core_radius=26 * u.cm):
    plt.rcParams['figure.figsize'] = (8.5, 5.5)
    plt.rcParams['figure.dpi'] = 300

    collision_frequencies = extract_collision_frequencies(datasets_split, core_radius, steady_state_plateaus_runs_split,
                                                          isweep_choice_center_split, operation)
    collision_frequencies_log_normalized = normalize(np.log(collision_frequencies), 0, 0.9)
    color_map = matplotlib.colormaps["plasma"](collision_frequencies_log_normalized)

    diagnostics_points = []
    for i in range(len(datasets_split)):
        diagnostics_point = []
        for plot_diagnostic in diagnostics_to_plot_list[:2]:
            diagnostic_mean = core_steady_state(
                datasets_split[i][plot_diagnostic], core_radius, steady_state_plateaus_runs_split[i], operation,
                dims_to_keep=("isweep",))
            diagnostics_point += [diagnostic_mean[{"isweep": isweep_choice_center_split[i]}].item()]
        diagnostics_points += [diagnostics_point]

    scatter_points = np.array(diagnostics_points)
    for i in range(len(scatter_points)):
        plt.scatter(scatter_points[i, 0], scatter_points[i, 1], marker=marker_styles_split[i], color=color_map[i],
                    label=f"{datasets_split[i].attrs['Exp name'][:3]}, #{datasets_split[i].attrs['Run name'][:2]}"
                          f":  {collision_frequencies[i]:.2E} Hz")
        plt.annotate(f"{datasets_split[i].attrs['Exp name'][:3]}, #{datasets_split[i].attrs['Run name'][:2]}",
                     (scatter_points[i, 0], scatter_points[i, 1]), size="small")  # noqa

    if ("n_" in diagnostics_to_plot_list[0] or "n_" in diagnostics_to_plot_list[1]) \
            and ("T_e" in diagnostics_to_plot_list[0] or "T_e" in diagnostics_to_plot_list[1]):
        x_min, x_max = np.min(scatter_points[:, 0]), np.max(scatter_points[:, 0])
        y_min, y_max = np.min(scatter_points[:, 1]), np.max(scatter_points[:, 1])
        pressure_min = np.min(scatter_points[:, 0] * scatter_points[:, 1])
        pressure_max = np.max(scatter_points[:, 0] * scatter_points[:, 1])

        x_curve = np.linspace(x_min, x_max, 100)
        pressures = np.linspace(np.sqrt(pressure_min), np.sqrt(pressure_max), 8)[1:-1] ** 2
        for pressure in pressures:
            plt.plot(x_curve, pressure / x_curve, color='gray')
        plt.ylim(0, 1.1 * y_max)
        if "n_" in diagnostics_to_plot_list[0] and "T_e" in diagnostics_to_plot_list[1]:
            pass

    plt.xlabel(diagnostics_to_plot_list[0])
    plt.ylabel(diagnostics_to_plot_list[1])
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title("Scatter plot for selected runs at port ~27-29"
              "\nJan 2024 runs: x marker = post-gas puff"
              "\nColor map: ln(collision frequency at port 27/29)")
    plt.tight_layout()
    plt.show()


def plot_parallel_inverse_scale_length(datasets_split, steady_state_plateaus_runs_split, diagnostic,
                                       isweep_choice_center_split, marker_styles_split, operation, core_radius):
    plt.rcParams['figure.figsize'] = (7, 4.5)   # (8.5, 5.5)
    plt.rcParams['figure.dpi'] = 300

    # Get mean core-steady-state e-i collision frequencies for each dataset and store in list
    collision_frequencies = extract_collision_frequencies(datasets_split, core_radius, steady_state_plateaus_runs_split,
                                                          isweep_choice_center_split, operation)
    collision_frequencies_log_normalized = normalize(np.log(collision_frequencies), 0, 0.9)
    color_map = matplotlib.colormaps["plasma"](collision_frequencies_log_normalized)

    for i in range(len(datasets_split)):
        isweep_choices = (0, 2) if datasets_split[i].attrs['Exp name'] == "January_2024" else (0, 1)
        diagnostic_means = core_steady_state(datasets_split[i][diagnostic], core_radius,
                                             steady_state_plateaus_runs_split[i], operation, dims_to_keep=("isweep",))

        diagnostic_difference = (diagnostic_means[{"isweep": isweep_choices[1]}].item()
                                 - diagnostic_means[{"isweep": isweep_choices[0]}].item())
        diagnostic_mean = 0.5 * (diagnostic_means[{"isweep": isweep_choices[1]}].item()
                                 + diagnostic_means[{"isweep": isweep_choices[0]}].item())
        diagnostic_normalized_gradient = diagnostic_difference / diagnostic_mean
        z1 = -diagnostic_means[{"isweep": isweep_choices[1]}].coords['z'].item() / 100  # converts cm to m
        z0 = -diagnostic_means[{"isweep": isweep_choices[0]}].coords['z'].item() / 100  # converts cm to m
        z = 0.5 * (z1 + z0)
        diagnostic_scale_length = (z1 - z0) / diagnostic_normalized_gradient
        diagnostic_inverse_scale_length = 1 / diagnostic_scale_length

        plt.plot(z, diagnostic_scale_length, marker=marker_styles_split[i], color=color_map[i],
                 label=f"{datasets_split[i].attrs['Exp name'][:3]}, #{datasets_split[i].attrs['Run name'][:2]}"
                       f":  {collision_frequencies[i]:.2E} Hz")
    plt.title(f"Parallel gradient scale length (m) of {diagnostic} versus z"  # ({operation})
              f"\nColor map: ln(collision frequency at port ~27)")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
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
                  'P_e_cal': "Calibrated electron pressure",
                  'P_e_from_n_i': "Electron pressure (from ion density)",  # not added yet
                  'P_e_from_n_i_OML': "Electron pressure (from ion density (OML))",
                  'n_e_cal': "Calibrated electron density",
                  'n_i_cal': "Calibrated ion density",
                  'n_i_OML_cal': "Calibrated ion density (OML)",
                  'nu_ei': "Electron-ion collision frequency"}

    for key in sorted(list(full_names.keys()), key=len, reverse=True):
        diagnostic = diagnostic.replace(key, full_names[key])

    return diagnostic


def extract_collision_frequencies(datasets, core_radius, steady_state_plateaus_runs, isweep_choice_center, operation):
    r"""
    Finds typical core-steady-state electron-ion collision frequencies for each dataset.
    :param datasets:
    :param core_radius:
    :param steady_state_plateaus_runs:
    :param isweep_choice_center:
    :param operation:
    :return: tuple of (collision frequencies DataArray, log(collision frequencies) DataArray normalized to [0, 0.9])
    """

    collision_frequencies = []
    for i in range(len(datasets)):
        collision_frequencies_mean = core_steady_state(datasets[i]['nu_ei'], core_radius,
                                                       steady_state_plateaus_runs[i], operation,
                                                       dims_to_keep=("isweep",))
        collision_frequencies += [collision_frequencies_mean[{"isweep": isweep_choice_center[i]}].mean().item()]
    collision_frequencies = np.array(collision_frequencies)

    return collision_frequencies


def normalize(ndarray, lower=0., upper=1.):
    return lower + (upper - lower) * (ndarray - ndarray.min()) / (ndarray.max() - ndarray.min())


