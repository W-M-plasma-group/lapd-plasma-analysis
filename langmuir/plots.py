from warnings import warn

from astropy import visualization

from langmuir.helper import *
from langmuir.configurations import get_ion

# matplotlib.use('TkAgg')
# matplotlib.use('QtAgg')


def multiplot_linear_diagnostic(diagnostics_datasets: list[xr.Dataset], plot_diagnostic, probe_face_choices, x_dim='x',
                                steady_state_by_runs=None, core_rad=None, attribute=None, tolerance=np.nan,
                                save_directory=""):
    r"""

    :param diagnostics_datasets: list of xarray Datasets
    :param plot_diagnostic: string identifying label of desired diagnostics
    :param probe_face_choices:
    :param x_dim:
    :param steady_state_by_runs:
    :param core_rad:
    :param attribute:
    :param tolerance:
    :param save_directory:
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

    sort_indices = np.arange(len(diagnostics_datasets))         # not sorted yet
    for attr in attributes:
        try:
            sort_indices = sorted(sort_indices, key=lambda j: diagnostics_datasets[j].attrs[attr])
            # diagnostics_datasets_sorted.sort(key=lambda d: d.attrs[attr])
        except KeyError:
            raise KeyError("Key error for key " + repr(attr))

    diagnostics_datasets_sorted = [diagnostics_datasets[sort_index] for sort_index in sort_indices]
    steady_state_by_runs_sorted = [steady_state_by_runs[sort_index] for sort_index in sort_indices]

    outer_values = [dataset.attrs[attributes[-1]] for dataset in diagnostics_datasets_sorted]
    outer_quants = u.Quantity([value_safe(value) for value in outer_values], unit_safe(outer_values[0]))
    outer_unique, outer_indexes = np.unique(outer_quants, return_index=True) if len(attributes) == 2 else ([None], [0])
    outer_bounds = np.append(outer_indexes, len(outer_values))

    visualization.quantity_support()
    plt.rcParams['figure.figsize'] = (3 + 3 * len(outer_indexes), 4.5 + 0.1 * len(diagnostics_datasets))  # TODO hardcoded
    fig, axes = plt.subplots(1, len(outer_indexes), sharey="row", sharex="col", layout="constrained")

    for outer_index in range(len(outer_unique)):    # gas puff voltage index
        outer_val = outer_unique[outer_index]
        ax = np.atleast_1d(axes)[outer_index]

        datasets = diagnostics_datasets_sorted[outer_bounds[outer_index]:outer_bounds[outer_index + 1]]
        steady_state_times = steady_state_by_runs_sorted[outer_bounds[outer_index]:outer_bounds[outer_index + 1]]
        num_datasets = outer_bounds[outer_index + 1] - outer_bounds[outer_index]

        color_map = matplotlib.colormaps["plasma"](np.linspace(0, 0.9, num_datasets))

        y_limits = [0]
        for inner_index in range(num_datasets):     # discharge current index
            ports = datasets[inner_index].coords['port'].data
            faces = datasets[inner_index].coords['face'].data

            ds_s = probe_face_selector(datasets[inner_index], probe_face_choices)
            for i in range(len(ds_s)):              # probe/face linear combination index
                ds = ds_s[i]

                inner_val = ds.attrs[attributes[0]]

                da = ds[plot_diagnostic]

                core_steady_state_params = []
                # below: only consider steady state
                core_steady_state_params += [core_rad] if x_dim == 'time' else [None]
                core_steady_state_params += [steady_state_times[inner_index]] if x_dim in ('x', 'y') else [None]
                # above: only consider core region
                da = core_steady_state(da, *core_steady_state_params)

                dims_to_average_out = ['shot'] + [dim for dim in x_dims if dim != x_dim]
                da_mean = da.mean(dims_to_average_out, keep_attrs=True)
                da_std = da.std(dims_to_average_out, ddof=1, keep_attrs=True)
                # both of the above should have only one dimension left?

                # 95% (~two standard deviation) confidence interval TODO replace by std_error in core_steady_state func?
                non_nan_elements_da = da.copy()
                non_nan_elements_da[...] = ~np.isnan(da)
                effective_num_non_nan_per_std = non_nan_elements_da.sum(dims_to_average_out)
                linear_da_error = da_std * np.nan
                linear_da_error[effective_num_non_nan_per_std > 1
                                ] = (da_std * 1.96 / np.sqrt(effective_num_non_nan_per_std - 1)
                                     )[effective_num_non_nan_per_std > 1]  # unbiased

                if np.isfinite(tolerance):  # remove points with too much variation  # TODO hardcoded! document!
                    da_median = core_steady_state(da, core_rad, steady_state_times[inner_index], "median")
                    da_mean = apply_tolerance(da_mean, linear_da_error, da_median, tolerance)

                if np.isfinite(da_mean).any():
                    probe_face_eq_str = probe_face_choice_to_eq_string(probe_face_choices[i], ports, faces)
                    ax.errorbar(da_mean.coords[x_dim], da_mean, yerr=linear_da_error, linestyle="none",
                                color=color_map[inner_index], marker=marker_styles[i],
                                label=str(inner_val) + f" ({probe_face_eq_str})")
                ax.set_xlabel(da_mean.coords[x_dim].attrs['units'])
                ax.set_ylabel(da_mean.attrs['units'])

                # TODO very hardcoded; TODO document fill_na!
                da_small_core_steady_state_max = core_steady_state(da.fillna(0), core_rad / 2,
                                                                   steady_state_times[inner_index]
                                                                   ).max().item()
                y_limits += [np.nanmin([1.1 * da_small_core_steady_state_max,
                                        2.2 * core_steady_state(da, core_rad, steady_state_times[inner_index],  # TODO?
                                                                operation="median", dims_to_keep=["probe", "face"]
                                                                ).max().item()])]

        if not np.nanmax(y_limits) > 0:
            warn("A plot was assigned a NaN upper axis limit")
        ax.set_ylim(0, np.nanmax(y_limits))
        ax.tick_params(axis="y", left=True, labelleft=True)
        ax.title.set_text((str(attributes[1]) + ": " + str(outer_val)) if len(attributes) == 2 else '')
        ax.legend(title=f"\n{attributes[0]} (probe face)", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    fig.suptitle(get_title(plot_diagnostic), size=18)
    plt.tight_layout()
    if save_directory:
        plt.savefig(f"{save_directory}multiplot_line_{x_dim}_{plot_diagnostic}.pdf", bbox_inches="tight")
    plt.show()


def plot_linear_diagnostic(diagnostics_dataset: xr.Dataset, probe_face_coefficients, diagnostic, plot_type, steady_state,
                           shot_mode="mean", save_directory="", tolerance=np.nan):
    # Plots the given diagnostic(s) from the dataset in the given style

    linear_ds_s = []
    linear_dimensions = []
    run_names = []
    diagnostics_ds_s = probe_face_selector(diagnostics_dataset, probe_face_coefficients)
    for diagnostics_ds in diagnostics_ds_s:
        run_names += [get_exp_run_string(diagnostics_ds.attrs)]

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
            da_median = linear_ds.median()
            linear_ds = linear_ds.where(linear_ds_std < tolerance * da_median)
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
        for key in diagnostic_list:
            for d in range(len(linear_ds_s)):
                linear_ds_1d = core_steady_state(linear_ds_s[d], steady_state_times=steady_state, operation="mean",
                                                 dims_to_keep=linear_dimensions[d])
                linear_plot_1d(linear_ds_1d[key], linear_dimensions[d])
            # probe_eq_string = probe_face_choice_to_eq_string(probe_face_coefficients[d], ports, faces)
            plot_title = (f"{run_names[0]}"  # , {probe_eq_string}"
                          f"\n{get_title(key)} {plot_type} plot")
            # TODO change
            """
            if hasattr(linear_ds_s_1d[0], "facevector"):
                plot_title += f"\nLinear combination of faces: {linear_ds_s_1d[0].attrs['facevector']}"
            """
            plt.title(plot_title)
            plt.tight_layout()
            plt.show()
    # 2D plot type
    elif plot_type in plot_types_2d:
        for key in diagnostic_list:
            for d in range(len(linear_ds_s)):
                try:
                    linear_plot_2d(linear_ds_s[d][key], plot_type, linear_dimensions[d])
                    plot_title = f"{run_names[d]}"
                    try:
                        probe_eq_string = probe_face_choice_to_eq_string(probe_face_coefficients[d],
                                                                         diagnostics_dataset.coords['port'].data,
                                                                         diagnostics_dataset.coords['face'].data)
                        plot_title += f", {probe_eq_string}"
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
                        print(e)
                        pass
                    plot_title += f"\n{get_title(key)} {plot_type} plot (2D)"
                    # TODO change
                    """
                    if hasattr(linear_ds_s[0], "facevector"):
                        plot_title += f"\nLinear combination of faces: {linear_ds_s[d].attrs['facevector']}"
                    """
                    plt.title(plot_title)
                    plt.tight_layout()
                    if save_directory:
                        plt.savefig(save_directory + "2D_plot_" + diagnostic + ".pdf", bbox_inches="tight")
                    plt.show()
                except ValueError as e:
                    print(f"Problem plotting {key} for {linear_ds_s[d].attrs['Run name']}:"
                          f"\n{repr(e)}")


def plot_parallel_diagnostic(datasets, steady_state_times_runs, probes_faces_midplane, probes_faces_parallel,
                             marker_styles, diagnostic, operation="mean", core_radius=26 * u.cm,
                             save_directory=""):
    plt.rcParams['figure.figsize'] = (6.5, 3.5)

    collision_frequencies = extract_collision_frequencies(datasets, core_radius, steady_state_times_runs,
                                                          probes_faces_midplane, operation)
    collision_frequencies_log_normalized = normalize(np.log(collision_frequencies), 0, 0.9)
    color_map = matplotlib.colormaps["plasma"](collision_frequencies_log_normalized)

    diagnostic_units = ""
    max_diagnostic = 0
    for i in range(len(datasets)):
        diagnostic_values = []
        diagnostic_errors = []
        zs = []

        if diagnostic not in datasets[i]:  # TODO needs testing
            continue
        diagnostic_means = core_steady_state(datasets[i][diagnostic], core_radius,
                                             steady_state_times_runs[i], operation,
                                             dims_to_keep=("probe", "face"))
        diagnostic_std_errors = core_steady_state(datasets[i][diagnostic], core_radius,
                                                  steady_state_times_runs[i], "std_error",
                                                  dims_to_keep=("probe", "face"))

        for probe_face in probes_faces_parallel[i]:
            diagnostic_value = diagnostic_means[{"probe": probe_face[0],
                                                 "face": probe_face[1]}].item()
            diagnostic_values += [diagnostic_value]
            diagnostic_errors += [diagnostic_std_errors[{"probe": probe_face[0],
                                                         "face": probe_face[1]}].item()]
            zs += [diagnostic_means[{"probe": probe_face[0],
                                     "face": probe_face[1]}].coords['z'].item()]
            if diagnostic_value > max_diagnostic:
                max_diagnostic = diagnostic_value
        zs = anode_z - (zs * u.Unit(diagnostic_means.coords['z'].attrs['units'])).to(u.m)  # convert to meters
        diagnostic_units = datasets[i][diagnostic].attrs['units']

        plt.errorbar(zs, diagnostic_values, yerr=diagnostic_errors, marker=marker_styles[i],
                     color=color_map[i], linestyle='none')
    plt.title(f"{get_title(diagnostic)} [{get_diagnostic_keys_units()[diagnostic]}] ", y=0.9, loc='right')  # ({operation})
    plt.xlabel("z location [m]")
    # plt.ylabel(f"[{diagnostic_units}]", rotation=0, labelpad=25)   # {get_title(diagnostic)}
    normalizer = matplotlib.colors.LogNorm(vmin=np.min(collision_frequencies),
                                           vmax=np.max(collision_frequencies))
    color_bar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalizer, cmap='plasma'), ax=plt.gca())
    color_bar.set_label(r"$\nu_{ei}$" " [Hz]\n(midplane)", rotation=0, labelpad=30)
    # color_bar.set_label("Midplane electron-ion \ncollision frequency [Hz]", rotation=90, labelpad=10)
    plt.ylim(0, 1.05 * max_diagnostic)  # TODO hardcoded
    plt.tight_layout()
    if save_directory:
        plt.savefig(f"{save_directory}parallel_plot_{diagnostic}.pdf")
    plt.show()


def scatter_plot_diagnostics(datasets, diagnostics_to_plot_list, steady_state_times_runs,
                             probes_faces_midplane, marker_styles, operation="mean", core_radius=26 * u.cm,
                             save_directory=""):
    plt.rcParams['figure.figsize'] = (6, 4)

    collision_frequencies = extract_collision_frequencies(datasets, core_radius, steady_state_times_runs,
                                                          probes_faces_midplane, operation)
    collision_frequencies_log_normalized = normalize(np.log(collision_frequencies), 0, 0.9)
    color_map = matplotlib.colormaps["plasma"](collision_frequencies_log_normalized)

    diagnostics_points = []
    for i in range(len(datasets)):
        diagnostics_point = []
        for plot_diagnostic in diagnostics_to_plot_list[:2]:
            diagnostic_mean = core_steady_state(
                datasets[i][plot_diagnostic], core_radius, steady_state_times_runs[i], operation,
                dims_to_keep=("probe", "face"))
            diagnostics_point += [diagnostic_mean[{"probe": probes_faces_midplane[i][0],
                                                   "face": probes_faces_midplane[i][1]}].item()]
        diagnostics_points += [diagnostics_point]

    scatter_points = np.array(diagnostics_points)
    for i in range(len(scatter_points)):
        plt.scatter(scatter_points[i, 0], scatter_points[i, 1], marker=marker_styles[i], color=color_map[i])
        """
                    label=get_exp_run_string(datasets[i].attrs)
                          + f":  {collision_frequencies[i]:.2E} Hz")
        """
        plt.annotate(get_exp_run_string(datasets[i].attrs),
                     (scatter_points[i, 0], scatter_points[i, 1]), size="small")  # noqa

    if ("n_" in diagnostics_to_plot_list[0] or "n_" in diagnostics_to_plot_list[1]) \
            and ("T_e" in diagnostics_to_plot_list[0] or "T_e" in diagnostics_to_plot_list[1]):
        x_min, x_max = np.min(scatter_points[:, 0]), np.max(scatter_points[:, 0])
        y_min, y_max = np.min(scatter_points[:, 1]), np.max(scatter_points[:, 1])
        pressure_min = np.min(scatter_points[:, 0] * scatter_points[:, 1])
        pressure_max = np.max(scatter_points[:, 0] * scatter_points[:, 1])

        x_curve = np.linspace(x_min, x_max, 100)
        # pressures_pa = np.linspace(np.sqrt(pressure_min), np.sqrt(pressure_max), 8)[1:-1] ** 2
        pressures_pa = np.arange(1, 5)  # TODO very hardcoded
        pa_to_ev_m3 = (1 * u.Pa).to(u.eV * u.m ** -3).value * 1 ** -1  # formerly * (3 / 2) ** (-1)
        for pressure in pressures_pa:
            plt.plot(x_curve, pa_to_ev_m3 * pressure / x_curve, color='#bbbbbb', label=f"{pressure} Pa")
        plt.ylim(0, 1.1 * y_max)
        if "n_" in diagnostics_to_plot_list[0] and "T_e" in diagnostics_to_plot_list[1]:
            pass

    plt.xlabel(f"{get_title(diagnostics_to_plot_list[0])} [{get_diagnostic_keys_units()[diagnostics_to_plot_list[0]]}]")
    plt.ylabel(f"{get_title(diagnostics_to_plot_list[1])} [{get_diagnostic_keys_units()[diagnostics_to_plot_list[1]]}]")
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.legend()
    normalizer = matplotlib.colors.LogNorm(vmin=np.min(collision_frequencies),
                                           vmax=np.max(collision_frequencies))
    color_bar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalizer, cmap='plasma'), ax=plt.gca())
    color_bar.set_label(r"$\nu_{ei}$" " [Hz]\n(midplane)", rotation=0, labelpad=30)
    # plt.title("Midplane scatter plot")
    # "\nJan 2024 runs: x marker = post-gas puff")
    plt.tight_layout()
    if save_directory:
        plt.savefig(f"{save_directory}scatter_plot_{diagnostics_to_plot_list[0]}_{diagnostics_to_plot_list[1]}.pdf")
    plt.show()


def plot_parallel_inverse_scale_length(datasets, steady_state_times_runs, diagnostic, probes_faces_midplane,
                                       probes_faces_parallel, marker_styles, operation, core_radius, save_directory,
                                       scale_length_mode="linear"):
    plt.rcParams['figure.figsize'] = 6, 3.5  # (7, 4.5)   # (8.5, 5.5)

    anode_z = portnum_to_z(0).to(u.m).value

    # Get mean core-steady-state e-i collision frequencies for each dataset and store in list
    collision_frequencies = extract_collision_frequencies(datasets, core_radius, steady_state_times_runs,
                                                          probes_faces_midplane, operation)
    collision_frequencies_log_normalized = normalize(np.log(collision_frequencies), 0, 0.9)
    color_map = matplotlib.colormaps["plasma"](collision_frequencies_log_normalized)

    for i in range(len(datasets)):
        probes_faces = probes_faces_parallel[i]
        diagnostic_means = core_steady_state(datasets[i][diagnostic], core_radius,
                                             steady_state_times_runs[i], operation,
                                             dims_to_keep=("probe", "face"))

        diagnostic_value_0 = diagnostic_means[{"probe": probes_faces[0][0],
                                               "face": probes_faces[0][1]}].item()
        diagnostic_value_1 = diagnostic_means[{"probe": probes_faces[1][0],
                                               "face": probes_faces[1][1]}].item()
        z1 = anode_z - diagnostic_means[{"probe": probes_faces[0][0],
                                         "face": probes_faces[0][1]}].coords['z'].item() / 100  # converts cm to m
        z0 = anode_z - diagnostic_means[{"probe": probes_faces[1][0],
                                         "face": probes_faces[1][1]}].coords['z'].item() / 100  # converts cm to m
        z = 0.5 * (z1 + z0)

        if scale_length_mode == "linear":
            diagnostic_difference = diagnostic_value_1 - diagnostic_value_0
            diagnostic_value = diagnostic_value_0  # no longer the average of value_0 and value_1
            diagnostic_normalized_difference = diagnostic_difference / diagnostic_value
            diagnostic_inverse_scale_length = diagnostic_normalized_difference / (z1 - z0)
        elif scale_length_mode == "exponential":
            diagnostic_ratio = diagnostic_value_1 / diagnostic_value_0
            diagnostic_inverse_scale_length = np.log(diagnostic_ratio) / (z1 - z0)
        else:
            raise ValueError(f"Unknown parallel gradient scale length method {repr(scale_length_mode)}."
                             "Acceptable methods are 'linear' and 'exponential'.")

        diagnostic_scale_length_abs = 1 / np.abs(diagnostic_inverse_scale_length)

        plt.plot(z, diagnostic_scale_length_abs, marker=marker_styles[i], color=color_map[i],
                 label=(get_exp_run_string(datasets[i].attrs)
                        + f":  {collision_frequencies[i]:.2E} Hz"))
    plt.title(f"Parallel gradient scale length [m] \n\n{get_title(diagnostic)} ", y=0.9)  # loc='right'
    plt.xlabel("z location [m]")

    normalizer = matplotlib.colors.LogNorm(vmin=np.min(collision_frequencies),
                                           vmax=np.max(collision_frequencies))
    color_bar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalizer, cmap='plasma'), ax=plt.gca())
    color_bar.set_label(r"$\nu_{ei}$" " [Hz]\n(midplane)", rotation=0, labelpad=30)
    # color_bar.set_label("Midplane electron-ion \ncollision frequency [Hz]", rotation=90, labelpad=10)

    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.ylim(bottom=0)  # TODO hardcoded

    if save_directory:
        plt.savefig(f"{save_directory}parallel_gradient_scale_length_plot_{diagnostic}.pdf")
    plt.show()

    plt.show()


def plot_grid(datasets, diagnostics_to_plot_list, steady_state_times_runs, probes_faces_midplane,
              probes_faces_parallel, operation, core_radius, x_dim, num_rows=2, plot_save_folder=""):
    # Split into top plot row and bottom plot row
    num_cols = (len(datasets) - 1) // num_rows + 1
    tolerance = 0.5  # TODO note; set very high to only discard points with nan error bars
    port_marker_styles = {20: 'x', 27: '.', 29: '.', 35: '^', 43: '^'}
    save_directory = plot_save_folder
    x_dims = ['x', 'y', 'time']
    for plot_diagnostic in diagnostics_to_plot_list:
        """ plot_ab(datasets_split, steady_state_times_runs_split, plot_diagnostic, probes_faces_midplane_split,
                probes_faces_parallel_split, port_marker_styles, "mean", core_radius)
            def plot_ab(datasets, steady_state_times_runs, plot_diagnostic,
                    probes_faces_midplane, probes_faces_parallel,
                    port_marker_styles, operation, core_radius): """

        plt.rcParams['figure.figsize'] = (1 + 3 * num_cols, 3 * num_rows)  # TODO hardcoded
        fig, axes = plt.subplots(num_rows, num_cols, sharex='none', sharey='row', layout="constrained")  # sharey=all
        axes_2d = np.atleast_2d(axes)

        collision_frequencies = extract_collision_frequencies(datasets, core_radius, steady_state_times_runs,
                                                              probes_faces_midplane, operation)
        collision_frequencies_log_normalized = normalize(np.log(collision_frequencies), 0, 0.9)
        color_map = matplotlib.colormaps["plasma"](collision_frequencies_log_normalized)

        indices = np.arange(num_rows * (len(datasets) // num_rows)).reshape(num_rows, -1)  # e.g. 1 2 3; 4 5 6
        for row in range(num_rows):  # row of the plot grid
            max_val = 0
            for col in range(len(indices[row])):  # column of the plot grid
                index = indices[row][col]

                ax = axes_2d[row][col]
                dataset = datasets[index]
                # ports = dataset.coords['port'].data
                # faces = dataset.coords['face'].data
                """
                ds_s = [dataset[{"probe": probe_face[0],
                                 "face": probe_face[1]}]
                        for probe_face in probes_faces_parallel[index]]  # todo [0:1]  !
                """
                probe_face = probes_faces_midplane[index]
                ds_s = [dataset[{"probe": probe_face[0],
                                 "face": probe_face[1]}]]
                # """
                for ds in ds_s:  # ds for upstream (cathode) and downstream (anti-cathode)
                    da = ds[plot_diagnostic]

                    core_steady_state_params = []
                    # below: only consider steady state
                    core_steady_state_params += [core_radius] if x_dim == 'time' else [None]
                    core_steady_state_params += [steady_state_times_runs[index]] if x_dim in ('x', 'y') else [None]
                    # above: only consider core region
                    da = core_steady_state(da, *core_steady_state_params)
                    # print(f"{get_exp_run_string(ds.attrs)}: {steady_state_times_runs[index]}")

                    dims_to_average_out = ['shot'] + [dim for dim in x_dims if dim != x_dim]  # hardcoded
                    da_mean = da.mean(dims_to_average_out, keep_attrs=True)
                    da_median = da.median()
                    da_std = da.std(dims_to_average_out, ddof=1, keep_attrs=True)
                    # both of the above should have only one dimension left?

                    # 95% (~two standard deviation) confidence interval
                    non_nan_element_da = da.copy()
                    non_nan_element_da[...] = ~np.isnan(da)
                    effective_num_non_nan_per_std = non_nan_element_da.sum(dims_to_average_out)
                    linear_da_error = da_std * np.nan
                    linear_da_error[effective_num_non_nan_per_std > 1
                                    ] = (da_std * 1.96 / np.sqrt(effective_num_non_nan_per_std - 1)
                                         )[effective_num_non_nan_per_std > 1]  # unbiased

                    if np.isfinite(da_mean).any():
                        # probe_face_eq_str = probe_face_choice_to_eq_string(probe_face_choices[i], ports, faces)
                        port = ds.coords['port'].item()
                        z_real = anode_z.to(u.m).value - ds.coords['z'].item() / 100
                        marker_style = port_marker_styles[port]

                        # da_mean[~(linear_da_error < tolerance * da_median)] = np.nan  # TODO hardcoded! document!
                        da_mean = apply_tolerance(da_mean, linear_da_error, da_median, tolerance)

                        ax.errorbar(da_mean.coords[x_dim], da_mean, yerr=linear_da_error, linestyle="none",
                                    color=color_map[index], marker=marker_style,
                                    # label=f"{z_real:.1f}")  # str(port)
                                    label=get_exp_run_string(ds.attrs, mode="short"))
                        max_val = np.nanmax([max_val, np.nanmax(da_mean)])
                    ax.set_xlabel(f"{x_dim} [{da_mean.coords[x_dim].attrs['units']}]")
                    ax.set_ylabel(da_mean.attrs['units'])

                # ax.set_ylim(0, [30, 20][index])  # TODO extremely hardcoded
                ax.set_ylim(bottom=0)
                """ax.title.set_text(get_exp_run_string(dataset.attrs)
                                  + ", run_name " + dataset.attrs['Run name'][:2])"""

                # ax.legend(title=f"\n{attributes[0]} (probe face)", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
                # ax.legend(title="z [m]")
                ax.legend(title="Experiment")
                ax.set_ylim(top=1.05 * max_val)
        normalizer = matplotlib.colors.LogNorm(vmin=np.min(collision_frequencies),
                                               vmax=np.max(collision_frequencies))
        color_bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=normalizer, cmap='plasma'),
                                 ax=axes_2d.ravel().tolist())
        color_bar.set_label(r"$\nu_{ei}$" " [Hz]\n(midplane)", rotation=0, labelpad=32)

        fig.suptitle(get_title(plot_diagnostic), size=18)
        if save_directory:
            plt.savefig(f"{save_directory}grid_{plot_diagnostic}_{x_dim}.pdf", bbox_inches="tight")
        plt.show()


def plot_acceleration_vs_pressure_gradient(datasets, steady_state_times_runs, core_radius,
                                           probes_faces_midplane, marker_styles_split,
                                           operation="mean", plot_save_folder="", with_expectation=False):
    r""""""
    z_distances = [((anode_z - dataset['z'].isel(probe=1).item() * u.cm)
                    - (anode_z - dataset['z'].isel(probe=0).item() * u.cm)).to(u.m)
                   for dataset in datasets]

    parallel_velocity_average = [(dataset['v_para'].isel(probe=1) + dataset['v_para'].isel(probe=0)) / 2
                                 for dataset in datasets]
    parallel_velocity_gradient = [(datasets[i]['v_para'].isel(probe=1) - datasets[i]['v_para'].isel(probe=0)
                                   ) / z_distances[i] for i in range(len(datasets))]    # z is in m; v is in m/s
    parallel_acceleration = [parallel_velocity_average[i] * parallel_velocity_gradient[i]
                             for i in range(len(datasets))]

    parallel_pressure_gradient = [(datasets[i]['P_ei_from_n_i_OML'].isel(probe=1)
                                   - datasets[i]['P_ei_from_n_i_OML'].isel(probe=0)
                                   ) / z_distances[i] for i in range(len(datasets))]
    from plasmapy.particles import particle_mass
    parallel_mass_density_average = [particle_mass(get_ion(datasets[i].attrs['Run name']))
                                     * (datasets[i]['n_i_OML'].isel(probe=1)
                                     + datasets[i]['n_i_OML'].isel(probe=0)) / 2
                                     for i in range(len(datasets))]
    parallel_pressure_gradient_over_density = [-parallel_pressure_gradient[i] / parallel_mass_density_average[i]
                                               for i in range(len(datasets))]

    plt.rcParams['figure.figsize'] = (5.5, 3.5)
    collision_frequencies = extract_collision_frequencies(datasets, core_radius, steady_state_times_runs,
                                                          probes_faces_midplane, "mean")
    collision_frequencies_log_normalized = normalize(np.log(collision_frequencies), 0, 0.9)
    color_map = matplotlib.colormaps["plasma"](collision_frequencies_log_normalized)

    pressure_gradients = []
    accelerations = []
    for i in range(len(datasets)):
        pressure_gradient = core_steady_state(parallel_pressure_gradient_over_density[i], core_radius,
                                              steady_state_times_runs[i], operation)
        acceleration = core_steady_state(parallel_acceleration[i], core_radius,
                                         steady_state_times_runs[i], operation)
        pressure_gradients += [value_safe(pressure_gradient)]
        accelerations += [value_safe(acceleration)]
        plt.scatter(pressure_gradient, acceleration, color=color_map[i], marker=marker_styles_split[i])

    if with_expectation:
        min_pressure_gradient = np.min(pressure_gradients)
        max_pressure_gradient = np.max(pressure_gradients)
        min_acceleration = np.min(accelerations)
        max_acceleration = np.max(accelerations)

        x_all = np.linspace(0,
                            1.1 * max_pressure_gradient, 10)
        x_some = np.linspace(0.8 * min_pressure_gradient,
                             0.4 * max_pressure_gradient, 10)  # was 0.39
        plt.plot(x_all, 1 * x_all, color='black', linestyle='--',           # equality line (predicted)
                 label=r"$y = x$")  # r"$y = x$ (model)"
        plt.plot(x_all, 0 * x_all, color='gray', linestyle=':',             # zero line
                 label=r"$y = 0$")  # r"$y = 0$"
        plt.plot(x_some, x_some - 2.7e7, color='silver', linestyle='--',    # offset equality line
                 label=r"$y = x$, offset")  # r"$y = x - 2.7e7$"
        # label=r"$v_z \ \frac{\partial v_z}{\partial z} = - \frac{1}{\rho} \frac{\partial p}{\partial z}$")
        # label=r"$v_z \ \frac{\partial v_z}{\partial z} = 0$")
        # \ \left[ \frac{\text{m}}{\text{s}^2} \right]

        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.84))
        plt.xlim(left=0, right=1.05 * max_pressure_gradient)
        plt.ylim(bot=min_acceleration - 0.2 * (max_acceleration - min_acceleration), top=0.3 * max_pressure_gradient)


    normalizer = matplotlib.colors.LogNorm(vmin=np.min(collision_frequencies),
                                           vmax=np.max(collision_frequencies))
    color_bar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalizer, cmap='plasma'), ax=plt.gca())
    color_bar.set_label(r"$\nu_{ei}$" " [Hz]\n(midplane)", rotation=0, labelpad=30)

    plt.title(r"$v_z \ \frac{\partial v_z}{\partial z}$ [m/s$^2$]  ", y=0.85, loc='right')
    plt.xlabel(r"$- \frac{1}{\rho} \frac{\partial p}{\partial z}$ [m/s$^2$]")
    plt.tight_layout()
    if plot_save_folder:
        plt.savefig(f"{plot_save_folder}parallel_acceleration_vs_parallel_pressure_gradient"
                    f"{'_with_expectation' if with_expectation else ''}.pdf")


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
    full_names = {'V_P':            "Plasma potential",
                  'V_F':            "Floating potential",
                  'I_es':           "Electron saturation current",
                  'I_is':           "Ion saturation current",
                  'n_e':            "Electron density",
                  'n_i':            "Ion density",
                  'T_e':            "Electron temperature",
                  'n_i_OML':        "Ion density (OML)",
                  'hot_fraction':   "Hot electron fraction",
                  'T_e_cold':       "Cold electron temperature",
                  'T_e_hot':        "Hot electron temperature",
                  'T_e_avg':        "Average bimaxwellian electron temperature",
                  'P_e':            "Electron pressure",
                  'P_e_cal':        "Calibrated electron pressure",
                  'P_e_from_n_i':   "Electron pressure (from ion density)",             # not added yet
                  'P_e_from_n_i_OML': "Electron pressure (from ion density (OML))",
                  'P_ei':           "Plasma pressure",                                  # Electron + ion pressure
                  'P_ei_cal':       "Calibrated plasma pressure",
                  'P_ei_from_n_i':  "Plasma pressure (from ion density)",               # not added yet
                  'P_ei_from_n_i_OML': "Plasma pressure" " (from ion density (OML))",
                  'n_e_cal':        "Calibrated electron density",
                  'n_i_cal':        "Calibrated ion density",
                  'n_i_OML_cal':    "Calibrated ion density (OML)",
                  'nu_ei':          "Electron-ion collision frequency",
                  'M_para':         "Parallel Mach number",
                  'M_perp':         "Perpendicular Mach number",
                  'v_para':         "Parallel velocity",
                  'v_perp':         "Perpendicular velocity"}

    for key in sorted(list(full_names.keys()), key=len, reverse=True):
        diagnostic = diagnostic.replace(key, full_names[key])

    return diagnostic


def extract_collision_frequencies(datasets, core_radius, steady_state_times_runs, probe_face_midplane, operation):
    r"""
    Finds typical core-steady-state electron-ion collision frequencies for each dataset.
    :param datasets:
    :param core_radius:
    :param steady_state_times_runs:
    :param probe_face_midplane:
    :param operation:
    :return: tuple of (collision frequencies DataArray, log(collision frequencies) DataArray normalized to [0, 0.9])
    """

    collision_frequencies = []
    for i in range(len(datasets)):
        collision_frequencies_mean = core_steady_state(datasets[i]['nu_ei'], core_radius,
                                                       steady_state_times_runs[i], operation,
                                                       dims_to_keep=("probe", "face"))
        collision_frequencies += [collision_frequencies_mean[{"probe": probe_face_midplane[i][0],
                                                              "face": probe_face_midplane[i][1]}].mean().item()]
    collision_frequencies = np.array(collision_frequencies)
    return collision_frequencies


def normalize(ndarray, lower=0., upper=1.):
    return lower + (upper - lower) * (ndarray - ndarray.min()) / (ndarray.max() - ndarray.min())


def probe_face_choice_to_eq_string(probe_face_coefficient, ports, faces):
    eq_string = ""
    for p in range(len(probe_face_coefficient)):
        for f in range(len(probe_face_coefficient[p])):
            if probe_face_coefficient[p][f] != 0:
                port_face_string = str(ports[p]) + str(faces[f])   # (faces[f] if faces[f] else "")
                eq_string += (((probe_face_coefficient[p][f] + "*") if probe_face_coefficient[p][f] != 1 else '')
                              + str(port_face_string) + " + ")
    return eq_string[:-3]


def get_exp_run_string(attrs, mode="long"):
    if mode == "short":
        series_number = ("April_2018", "March_2022", "November_2022", "January_2024").index(attrs['Exp name']) # TODO hc
        return f"{series_number}-{attrs['Run name'][:2]}"
    else:
        return f"{attrs['Exp name'][:3]} {attrs['Exp name'][-2:]} #{attrs['Run name'][:2]}"


def apply_tolerance(da, da_error, da_reference_value, tolerance):
    da[~(da_error < tolerance * da_reference_value)] = np.nan  # TODO hardcoded! document!
    return da
