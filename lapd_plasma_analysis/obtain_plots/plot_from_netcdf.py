
from lapd_plasma_analysis.obtain_plots.xarray_plots import *
from lapd_plasma_analysis.obtain_plots.plot_from_netcdf_helpers.radial_plots import *

from lapd_plasma_analysis.obtain_plots.Auxillary_functions import filter_data

from plasmapy.particles import *
from astropy import constants as c
from astropy import units as u
from astropy.units import Unit, Quantity
import matplotlib.colors as mcolors



def  build_isat_radial_plot(datasets, pathnames = None, figure_folder = '', see_plots = False, save_plots = False,
                            clor = None, mark = None, one_probe = True, axes = None):

    show_density = ask_yes_or_no('Show density plot? (y/n) ')

    if save_plots:
        isat_radial_plot_folder = ensure_directory(figure_folder + 'isat_radial_plots/')
    else:
        isat_radial_plot_folder = None

    if clor is None:
        clor, mark = determine_colors(datasets)

    default_fig_height, default_fig_width = default_fig_params()

    for j, ds in enumerate(datasets):
        if pathnames is not None:
            pathname = pathnames[j]
        else:
            pathname = ''
        run_identifier = f_run_identifier(ds)
        if 'updated' in pathname.lower():
            run_identifier = f"{run_identifier}_updated"
        dataset_clor = clor[j]
        dataset_mark = mark[j]
        if one_probe:
            probes = [0]
        else:
            probes = [probe for probe in ds.coords['probe'].values]

        for probe in probes:
            if axes is None:
                # Subplots oriented as 1 plot on top of the other
                layout = [[1]]
                fig, ax, letters = build_subplots(layout, 1.2 * default_fig_width, 1.2 * default_fig_height,
                                                  sharex=True)
                ax = ax[letters[0]]
            else:
                ax = axes

            if show_density:
                t_e_to_plot_vals, t_e_std_to_plot_vals, _ = process_variable_data(ds, probe, var_name='t_e')
                layout_dens = [[1]]
                fig_dens, ax_dens, letters_dens = build_subplots(layout_dens, 1.2 * default_fig_width, 1.2 * default_fig_height,
                                                  sharex=True)
                ax_dens = ax_dens[letters_dens[0]]
            else:
                fig_dens = None
                ax_dens = None
                t_e_to_plot_vals = None
                t_e_std_to_plot_vals = None

            isat_name = ds['ion_isat'].attrs.get("long_name", 'ion_isat')
            ylabel = fr"$I_{{sat}}$ [$\mathregular{{{ds['ion_isat'].attrs.get('units', 'ion_isat')}}}$]"
            xlabel = f'x [{ds.attrs.get("x_units")}]'

            min_time = ds.attrs[f'steady state start probe {probe}']
            max_time = ds.attrs[f'steady state end probe {probe}']

            mean_data = ds['ion_isat'].sel(probe=probe).mean('shot')

            isat_mask = (mean_data['time'] >= min_time) & (mean_data['time'] <= max_time)

            isat_to_plot = mean_data.sel(sweep=ds['sweep'][isat_mask]).mean('sweep')
            isat_std_to_plot = mean_data.sel(sweep=ds['sweep'][isat_mask]).std('sweep')

            # Format everything for matplot.lib plotting
            x_vals = mean_data['x'].values
            isat_to_plot_vals = isat_to_plot.squeeze().values
            isat_std_to_plot_vals = isat_std_to_plot.squeeze().values

            ax.errorbar(x_vals, isat_to_plot_vals, yerr=isat_std_to_plot_vals, fmt=dataset_mark, capsize=3,
                          color=dataset_clor, label = run_identifier)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            probe_z = ds['z'].isel(probe=probe).item()
            ax.set_title(f"{run_identifier}\n "
                         f"{isat_name}\n "
                         f"z = {probe_z} \n "
                         f"Between times ({min_time:.2f},{max_time:.2f}) ms")

            ax.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.2),
                ncol=2,
                framealpha=0.8,
                fontsize=20
            )
            plt.tight_layout()

            if show_density:
                # Filter out NaNs and negative temperatures which will mess up the algorithm
                valid_mask = (
                        ~np.isnan(t_e_to_plot_vals) &
                        ~np.isnan(isat_to_plot_vals) &
                        (t_e_to_plot_vals > 0)
                )

                x_valid = x_vals[valid_mask]
                isat_valid = isat_to_plot_vals[valid_mask]
                te_valid = t_e_to_plot_vals[valid_mask]

                # Density without constants
                density_vals = -isat_valid / np.sqrt(te_valid)

                ax_dens.plot(
                    x_valid, density_vals, marker=dataset_mark, linestyle='None',
                    color=dataset_clor, label=run_identifier,
                )

                ax_dens.set_ylabel(r"$n_e \frac{0.61 e A_p}{\sqrt{m_i}}$")
                ax_dens.set_xlabel(xlabel)
                ax_dens.set_title(f"{run_identifier}\n Density Check\n z = {probe_z}"
                                  f"\n Between times ({min_time:.2f},{max_time:.2f}) ms")


                ax_dens.legend(
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.2),
                    ncol=2,
                    framealpha=0.8,
                    fontsize=20
                )

            fig = ax.get_figure()
            fig.tight_layout()
            if show_density:
                fig_dens.tight_layout()

            if save_plots:
                save_path = os.path.join(isat_radial_plot_folder, f"{run_identifier}_probe{probe}.png")
                fig.savefig(save_path, bbox_inches='tight')
                if show_density:
                    dens_folder = ensure_directory(isat_radial_plot_folder + 'densities/')
                    save_path_dens = os.path.join(dens_folder, f"{run_identifier}_probe{probe}.png")
                    fig_dens.savefig(save_path_dens, bbox_inches='tight')
            if see_plots:
                plt.show()

            if axes is None:
                plt.close(fig)

            if show_density:
                plt.close(fig_dens)


def build_radial_plot(
    datasets,
    pathnames="",
    figure_folder="",
    make_presentable=False,
    see_temp_and_dens_plots=False,
    lines=False,
    shaded=False,
    plot_final_fits=False,
    save_plots=False,
    clor=None,
    mark=None,
    from_main=False,
    one_probe=True,
    axes=None,
    dens_bottomx=True,
    temp_bottomx=False,
    press_bottomx=False,
    run_identifiers="",
    hdf5_folder=None,
    updated_nc_folder=None,
    time=None,
    core_width=False,
    by_shot=False,
    redo_grad_regions=None,
):
    # print('Building radial plots...')

    # 1. Standard base strings from dataset configuration (returns 6 values)
    (
        base_dens_reg,
        base_dens_slp,
        base_dens_int,
        base_temp_reg,
        base_temp_slp,
        base_temp_int,
    ) = xarray_gradient_strings()

    # Define base strings for pressure directly
    base_press_reg = base_dens_reg.replace("dens", "press")
    base_press_slp = base_dens_slp.replace("dens", "press")
    base_press_int = base_dens_int.replace("dens", "press")

    c_temp = "#999999"  # Light Gray
    c_dens = "#56B4E9"  # Okabe-Ito Sky Blue
    c_opp = "#009E73"  # Okabe-Ito Bluish Green
    c_same = "#F0E442"  # Okabe-Ito Yellow
    gradient_regions_colors = {
        "c_temp": c_temp,
        "c_dens": c_dens,
        "c_opp": c_opp,
        "c_same": c_same,
    }
    default_fig_height, default_fig_width = default_fig_params()

    if make_presentable:
        plt.rcParams.update({
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.transparent": True,
        })

    if clor is None:
        clor, mark = determine_colors(datasets)

    gradient_regions_dict = {}

    for j, dataset in enumerate(datasets):
        print('ds: ', j)
        ds_grad_dict = {}
        pathname = (
            pathnames[j] if isinstance(pathnames, (list, tuple)) else pathnames
        )
        run_identifier = f_run_identifier(dataset)

        # 1. Determine active probes
        probes = (
            [0]
            if one_probe
            else [int(p) for p in dataset.coords["probe"].values]
        )

        # 2. Determine time vector
        if core_width:
            if time is not None:
                time_vals = np.atleast_1d(time)
            elif "time" in dataset.coords:
                time_vals = dataset.coords["time"].values
            else:
                time_vals = [None]

            if redo_grad_regions is None:
                check_all = ask_yes_or_no(
                    f"Check/redo gradient regions across all probes/times for"
                    f" Dataset {j}? (y/n) "
                )
            else:
                check_all = redo_grad_regions
        else:
            time_vals = [time]
            check_all = False

        # 3. Determine shot vector if by_shot is True
        has_shot_dim = ("shot" in dataset.coords) or ("shot" in dataset.dims)
        shot_vals = (
            dataset.coords["shot"].values
            if (by_shot and has_shot_dim)
            else [None]
        )

        # 4. Iterate over Probes
        for probe in probes:
            if probe not in ds_grad_dict:
                ds_grad_dict[probe] = {}

            # 5. Iterate over Time Steps
            for t_val in time_vals:
                if core_width and by_shot and has_shot_dim:
                    if t_val not in ds_grad_dict[probe]:
                        ds_grad_dict[probe][t_val] = {}

                # 6. Iterate over Shots
                for s_val in shot_vals:
                    ds_curr = (
                        dataset.sel(shot=s_val)
                        if s_val is not None
                        else dataset
                    )

                    t_suffix = (
                        f"_t_{t_val}"
                        if (core_width and t_val is not None)
                        else ""
                    )
                    s_suffix = f"_shot_{s_val}" if s_val is not None else ""
                    full_suffix = f"{t_suffix}{s_suffix}"

                    p_temp_reg = f"probe_{probe}_{base_temp_reg}{full_suffix}"
                    p_temp_slp = f"probe_{probe}_{base_temp_slp}{full_suffix}"
                    p_temp_int = f"probe_{probe}_{base_temp_int}{full_suffix}"

                    p_dens_reg = f"probe_{probe}_{base_dens_reg}{full_suffix}"
                    p_dens_slp = f"probe_{probe}_{base_dens_slp}{full_suffix}"
                    p_dens_int = f"probe_{probe}_{base_dens_int}{full_suffix}"

                    p_press_reg = f"probe_{probe}_{base_press_reg}{full_suffix}"
                    p_press_slp = f"probe_{probe}_{base_press_slp}{full_suffix}"
                    p_press_int = f"probe_{probe}_{base_press_int}{full_suffix}"

                    # Check saved locations for temp and dens to prevent unnecessary interactive prompts
                    has_saved_location = (p_temp_reg in ds_curr.attrs) and (
                        p_dens_reg in ds_curr.attrs
                    )

                    if core_width:
                        redo_grad_regions = (
                            check_all if check_all else not has_saved_location
                        )
                    else:
                        if not has_saved_location:
                            redo_grad_regions = ask_yes_or_no(
                                f"Determine gradient regions for Dataset {j},"
                                f" Probe {probe}? (y/n) "
                            )
                        else:
                            redo_grad_regions = (
                                ask_yes_or_no(
                                    f"Redo gradient regions for Dataset {j},"
                                    f" Probe {probe}? (y/n) "
                                )
                                if from_main
                                else False
                            )

                    # Setup Subplots independently for each dataset if axes is None
                    if not core_width:
                        if axes is None:
                            layout = [[1], [1], [1]]  # 3 rows for Te, Ne, Pe
                            fig, ax, letters = build_subplots(
                                layout,
                                1.2 * default_fig_width,
                                1.2 * default_fig_height,
                                sharex=True,
                            )
                            temperature_ax = ax[letters[0]]
                            density_ax = ax[letters[1]]
                            pressure_ax = ax[letters[2]]
                        else:
                            temperature_ax = axes[0]
                            density_ax = axes[1]
                            pressure_ax = axes[2]
                    else:
                        density_ax = None
                        temperature_ax = None
                        pressure_ax = None

                    # Temperature
                    te_edges, te_slopes, te_intercepts = (
                        create_temperature_radial_plots(
                            ds_curr,
                            probe,
                            run_identifier=run_identifier,
                            see_plots=False,
                            axes=temperature_ax,
                            dataset_clor=clor[j],
                            dataset_mark=mark[j],
                            make_presentable=make_presentable,
                            sharex=True,
                            bottomx=temp_bottomx,
                            redo_grad_regions=redo_grad_regions,
                            gradient_regions=lines,
                            check_all=check_all,
                            min_points=3,
                            regions_str=p_temp_reg,
                            slopes_str=p_temp_slp,
                            intercepts_str=p_temp_int,
                            ds_save_path=pathname,
                            plot_final_fit=plot_final_fits,
                            hdf5_folder=hdf5_folder,
                            figure_folder=figure_folder,
                            save_plots=save_plots,
                            updated_nc_folder=updated_nc_folder,
                            time=t_val,
                            core_width=core_width,
                        )
                    )

                    # Density
                    ne_edges, ne_slopes, ne_intercepts = (
                        create_density_radial_plots(
                            ds_curr,
                            probe,
                            run_identifier=run_identifier,
                            see_plots=False,
                            axes=density_ax,
                            dataset_clor=clor[j],
                            dataset_mark=mark[j],
                            make_presentable=make_presentable,
                            sharex=True,
                            bottomx=dens_bottomx,
                            redo_grad_regions=redo_grad_regions,
                            gradient_regions=lines,
                            check_all=check_all,
                            min_points=3,
                            regions_str=p_dens_reg,
                            slopes_str=p_dens_slp,
                            intercepts_str=p_dens_int,
                            ds_save_path=pathname,
                            plot_final_fit=plot_final_fits,
                            hdf5_folder=hdf5_folder,
                            figure_folder=figure_folder,
                            save_plots=save_plots,
                            updated_nc_folder=updated_nc_folder,
                            time=t_val,
                        )
                    )

                    # Pressure
                    pe_edges, pe_slopes, pe_intercepts = (
                        create_pressure_radial_plots(
                            ds_curr,
                            probe,
                            run_identifier=run_identifier,
                            see_plots=False,
                            axes=pressure_ax,
                            dataset_clor=clor[j],
                            dataset_mark=mark[j],
                            make_presentable=make_presentable,
                            sharex=True,
                            bottomx=press_bottomx,
                            redo_grad_regions=redo_grad_regions,
                            gradient_regions=lines,
                            check_all=check_all,
                            min_points=3,
                            regions_str=p_press_reg,
                            slopes_str=p_press_slp,
                            intercepts_str=p_press_int,
                            ds_save_path=pathname,
                            plot_final_fit=plot_final_fits,
                            hdf5_folder=hdf5_folder,
                            figure_folder=figure_folder,
                            save_plots=save_plots,
                            updated_nc_folder=updated_nc_folder,
                            time=t_val,
                        )
                    )

                    # Mirror attributes back to primary dataset
                    if p_temp_reg in ds_curr.attrs:
                        dataset.attrs[p_temp_reg] = ds_curr.attrs[p_temp_reg]
                    if p_dens_reg in ds_curr.attrs:
                        dataset.attrs[p_dens_reg] = ds_curr.attrs[p_dens_reg]
                    if p_press_reg in ds_curr.attrs:
                        dataset.attrs[p_press_reg] = ds_curr.attrs[p_press_reg]

                    # 7. Extract Gradient Regions
                    temp_reg_val = ds_curr.attrs.get(
                        p_temp_reg, dataset.attrs.get(p_temp_reg, None)
                    )
                    dens_reg_val = ds_curr.attrs.get(
                        p_dens_reg, dataset.attrs.get(p_dens_reg, None)
                    )
                    press_reg_val = ds_curr.attrs.get(
                        p_press_reg, dataset.attrs.get(p_press_reg, None)
                    )

                    if temp_reg_val is not None or dens_reg_val is not None:
                        x_vals = dataset.coords["x"].values

                        def parse_regions(val):
                            if val is None:
                                return []
                            if isinstance(val, str):
                                try:
                                    val = json.loads(val)
                                except Exception:
                                    return []
                            return [tuple(edge) for edge in val]

                        temp_regions = parse_regions(temp_reg_val)
                        dens_regions = parse_regions(dens_reg_val)
                        press_regions = parse_regions(press_reg_val)

                        temp_slopes = ds_curr.attrs.get(
                            p_temp_slp, dataset.attrs.get(p_temp_slp, [])
                        )
                        if isinstance(temp_slopes, str):
                            temp_slopes = json.loads(temp_slopes)

                        dens_slopes = ds_curr.attrs.get(
                            p_dens_slp, dataset.attrs.get(p_dens_slp, [])
                        )
                        if isinstance(dens_slopes, str):
                            dens_slopes = json.loads(dens_slopes)

                        grad_T = np.zeros_like(x_vals, dtype=float)
                        grad_N = np.zeros_like(x_vals, dtype=float)

                        for (start, stop), slope in zip(
                            temp_regions, temp_slopes
                        ):
                            grad_T[(x_vals >= start) & (x_vals <= stop)] = slope

                        for (start, stop), slope in zip(
                            dens_regions, dens_slopes
                        ):
                            grad_N[(x_vals >= start) & (x_vals <= stop)] = slope

                        spatial_mask = (
                            (np.abs(x_vals) >= 15)
                            | (grad_T != 0)
                            | (grad_N != 0)
                        )
                        mask_T_only = (
                            (grad_T != 0) & (grad_N == 0) & spatial_mask
                        )
                        mask_N_only = (
                            (grad_T == 0) & (grad_N != 0) & spatial_mask
                        )
                        mask_same = (grad_T * grad_N > 0) & spatial_mask
                        mask_opp = (grad_T * grad_N < 0) & spatial_mask
                        mask_flat = (grad_T == 0) & (grad_N == 0) & spatial_mask

                        def extract_intervals(mask, min_width=0.5):
                            padded = np.pad(
                                mask,
                                (1, 1),
                                mode="constant",
                                constant_values=False,
                            )
                            diffs = np.diff(padded.astype(int))
                            starts = np.where(diffs == 1)[0]
                            stops = np.where(diffs == -1)[0] - 1
                            raw_intervals = [
                                (x_vals[s], x_vals[e])
                                for s, e in zip(starts, stops)
                            ]
                            return [
                                (start, stop)
                                for start, stop in raw_intervals
                                if abs(stop - start) >= min_width
                            ]

                        t_grads_dict = {
                            "temp_grads": extract_intervals(mask_T_only),
                            "density_grads": extract_intervals(mask_N_only),
                            "dt_same": extract_intervals(mask_same),
                            "dt_opposite": extract_intervals(mask_opp),
                            "flat": extract_intervals(mask_flat),
                            "full_temp_grads": temp_regions,
                            "full_density_grads": dens_regions,
                            "full_pressure_grads": press_regions,
                        }

                        if core_width:
                            if by_shot and has_shot_dim and s_val is not None:
                                if t_val not in ds_grad_dict[probe]:
                                    ds_grad_dict[probe][t_val] = {}
                                ds_grad_dict[probe][t_val][s_val] = t_grads_dict
                            else:
                                ds_grad_dict[probe][t_val] = t_grads_dict
                        else:
                            ds_grad_dict[probe] = t_grads_dict

                    # Render & Display/Save Individual Plots
                    if core_width:
                        if axes is None:
                            plt.close()

        # Store output dict cleanly per dataset index
        if len(datasets) == 1:
            gradient_regions_dict = ds_grad_dict
        else:
            gradient_regions_dict[j] = ds_grad_dict

    return gradient_regions_colors, gradient_regions_dict

def plot_slope_vs_time(
    ds,
    probe=0,
    var_name="n_e",
    slopes_base_str="probe_0_dens_grad_slopes",
    color="blue",
    mark="s",
    ax=None,
    label=None,
    normalize=False,
    show_err=True,
    add_legend=None,  # Auto-detects based on ax
    set_labels=None,  # Auto-detects based on ax
):
    """Searches ds.attrs for time-tagged slope keys, parses timestamps, and

    plots slope regions over time.
    """
    # Track whether the caller provided an existing axis
    ax_provided = ax is not None

    # Default to False if ax was passed in, True if creating a new axis
    if set_labels is None:
        set_labels = not ax_provided
    if add_legend is None:
        add_legend = not ax_provided

    time_data = {}

    # --- 1. Search ds.attrs for matching key patterns ---
    for key, value in ds.attrs.items():
        if slopes_base_str in key and "_t_" in key and "_shots" not in key:
            try:
                t_str = key.split("_t_")[-1]
                t_val = float(t_str)
            except (ValueError, TypeError):
                continue

            slopes_list = _parse_attr_list(value)
            if len(slopes_list) == 0:
                continue

            regions_key = key.replace("_slopes", "_regions")
            intercepts_key = key.replace("_slopes", "_intercepts")
            shot_slopes_key = key.replace("_slopes", "_slopes_shots")
            shot_intercepts_key = key.replace("_slopes", "_intercepts_shots")

            regions_list = _parse_attr_list(ds.attrs.get(regions_key))
            intercepts_list = _parse_attr_list(ds.attrs.get(intercepts_key))
            shot_slopes = _parse_attr_list(ds.attrs.get(shot_slopes_key))
            shot_intercepts = _parse_attr_list(
                ds.attrs.get(shot_intercepts_key)
            )

            # --- A. Process Ensemble Mean ---
            mean_vals = []
            for i, m in enumerate(slopes_list):
                if (
                    normalize
                    and i < len(regions_list)
                    and i < len(intercepts_list)
                ):
                    start_x, stop_x = regions_list[i]
                    b = intercepts_list[i]
                    x_mid = (start_x + stop_x) / 2.0
                    y_local = m * x_mid + b

                    if y_local > 0 and not np.isnan(y_local):
                        mean_vals.append(m / y_local)
                    else:
                        mean_vals.append(np.nan)
                else:
                    mean_vals.append(m)

            # --- B. Process Individual Shots ---
            processed_shots = []
            if shot_slopes and isinstance(shot_slopes, list):
                for s_idx, s_m_list in enumerate(shot_slopes):
                    s_intercepts_list = (
                        shot_intercepts[s_idx]
                        if (shot_intercepts and s_idx < len(shot_intercepts))
                        else []
                    )
                    s_vals = []
                    for i, m_s in enumerate(s_m_list):
                        if (
                            normalize
                            and i < len(regions_list)
                            and i < len(s_intercepts_list)
                        ):
                            start_x, stop_x = regions_list[i]
                            b_s = s_intercepts_list[i]
                            x_mid = (start_x + stop_x) / 2.0
                            y_local_s = m_s * x_mid + b_s

                            if y_local_s > 0 and not np.isnan(y_local_s):
                                s_vals.append(m_s / y_local_s)
                            else:
                                s_vals.append(np.nan)
                        else:
                            s_vals.append(m_s)
                    processed_shots.append(s_vals)

            time_data[t_val] = {
                "mean_vals": mean_vals,
                "shot_vals": processed_shots,
            }

    if not time_data:
        print(f"No valid slope attributes found for pattern '{slopes_base_str}'")
        return ax

    sorted_times = sorted(time_data.keys())
    max_regions = max(
        len(time_data[t]["mean_vals"])
        for t in sorted_times
        if time_data[t]["mean_vals"]
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # --- 2. Region-by-Region Aggregation & Plotting ---
    for reg_idx in range(max_regions):
        times_plot, y_plot, err_plot = [], [], []

        for t_val in sorted_times:
            entry = time_data[t_val]
            mean_vals = entry["mean_vals"]
            shot_vals = entry["shot_vals"]

            if show_err and shot_vals and len(shot_vals) > 0:
                reg_shot_values = [
                    shot[reg_idx]
                    for shot in shot_vals
                    if len(shot) > reg_idx and not np.isnan(shot[reg_idx])
                ]

                if len(reg_shot_values) > 0:
                    y_val = np.mean(reg_shot_values)
                    y_err = (
                        np.std(reg_shot_values, ddof=1)
                        / np.sqrt(len(reg_shot_values))
                        if len(reg_shot_values) > 1
                        else 0.0
                    )
                else:
                    y_val = (
                        mean_vals[reg_idx]
                        if reg_idx < len(mean_vals)
                        else np.nan
                    )
                    y_err = 0.0

                if not np.isnan(y_val):
                    times_plot.append(t_val)
                    y_plot.append(y_val)
                    err_plot.append(y_err)

            else:
                y_val = (
                    mean_vals[reg_idx] if reg_idx < len(mean_vals) else np.nan
                )
                if not np.isnan(y_val):
                    times_plot.append(t_val)
                    y_plot.append(y_val)

        reg_lbl = (
            f"{label} (Reg {reg_idx + 1})"
            if label
            else f"Region {reg_idx + 1}"
        )

        if show_err and len(err_plot) == len(y_plot) and len(err_plot) > 0:
            ax.errorbar(
                times_plot,
                y_plot,
                yerr=err_plot,
                fmt=mark,
                color=color,
                capsize=4,
                label=reg_lbl,
            )
        else:
            ax.plot(
                times_plot,
                y_plot,
                marker=mark,
                color=color,
                linestyle="-",
                label=reg_lbl,
            )

    # --- 3. Conditional Axis Formatting ---
    if set_labels:
        ax.set_xlabel("Time [ms]")
        if normalize:
            ax.set_ylabel(r"Inverse Scale Length $L^{-1}$ [cm$^{-1}$]")
        else:
            ax.set_ylabel("Gradient Slope")
        ax.grid(True, linestyle=":", alpha=0.6)

    if add_legend:
        ax.legend()

    return ax

def plot_ion_sat_curr_vs_time(datasets, figure_folder):
    figure_folder = ensure_directory(figure_folder + 'isat_figures/')
    plot_multiple_ds = ask_yes_or_no('Plot multiple datasets on the same plot? (y/n) ')

    # Color/marker mapping using determine_colors wrapper
    clors, marks = determine_colors(datasets, one_probe=True)

    if not plot_multiple_ds:
        average_shots = ask_yes_or_no('Average over shots? (y/n) ')
        if not average_shots:
            possible_shots = datasets[0]['shot'].values
            min_shot = min(possible_shots)
            max_shot = max(possible_shots)
            prompt_shot = f"Choose the shot(s) you would like to plot ({min_shot} to {max_shot}): "
            shots_to_plot = allow_only_ints(prompt_shot, min_condition=min_shot, max_condition=max_shot,
                                            accept_empty=True)
        else:
            shots_to_plot = []

        plot_all_x_together = ask_yes_or_no('Plot multiple x - values on the same plot? (y/n) ')
    else:
        average_shots = True
        shots_to_plot = []
        plot_all_x_together = False

    possible_x_vals = datasets[0]['x'].values
    min_x = min(possible_x_vals)
    max_x = max(possible_x_vals)
    prompt_x = f"Choose the x-values you would like to plot ({min_x} to {max_x}): "
    x_to_plot = allow_only_ints(prompt_x, min_condition=min_x, max_condition=max_x, accept_empty=True)

    marker_shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']

    def apply_isat_ylim(ax, label_text):
        """Restricts y-axis limits depending on species in dataset identifier(s)."""
        labels = " ".join(label_text) if isinstance(label_text, (list, tuple)) else str(label_text)
        if 'He-4+' in labels or 'He4+' in labels:
            ax.set_ylim(-0.06, 0)
        elif 'H+' in labels or 'H +' in labels:
            ax.set_ylim(-0.02, 0)

    def extract_isat_trace(ds, z_target, x_target, shot=None):
        z_array = ds['z'].values
        probe_idx = np.abs(z_array - z_target).argmin()
        z_actual = z_array[probe_idx]

        isat_da = ds['ion_isat'].isel(probe=probe_idx).sel(x=x_target, method='nearest')

        if shot is not None:
            isat_da = isat_da.sel(shot=shot)
        elif 'shot' in isat_da.dims:
            isat_da = isat_da.mean(dim='shot')

        if 'y' in isat_da.dims:
            isat_da = isat_da.mean(dim='y')

        times = isat_da['time'].values if 'time' in isat_da.coords else ds['time'].values
        return times, isat_da.values, z_actual

    target_zs = np.unique(datasets[0]['z'].values)

    for z_target in target_zs:

        # --- CASE 1: Multiple Datasets (1 x-value per figure, shot-averaged) ---
        if plot_multiple_ds:
            multiple_ds_folder = ensure_directory(figure_folder + 'multiple_ds/')
            for x in x_to_plot:
                fig, axes, letters = build_subplots([[1]])
                ax = axes[letters[0]]

                z_actual_val = z_target
                ri_dict = {}
                ds_labels_list = []
                for ds_idx, ds in enumerate(datasets):
                    color = clors[ds_idx] if ds_idx < len(clors) else None
                    marker = marks[ds_idx] if ds_idx < len(marks) else 'o'

                    ds_label = f_run_identifier(ds)
                    ds_labels_list.append(ds_label)
                    exp_date = ds_label.split(' ')[0] + '_' + ds_label.split(' ')[1]
                    run_num = ds_label.split(' ')[3]
                    if exp_date not in ri_dict:
                        ri_dict[exp_date] = f"_{run_num}"
                    else:
                        ri_dict[exp_date] = ri_dict[exp_date] + f"_{run_num}"

                    times, isat_vals, z_actual_val = extract_isat_trace(ds, z_target, x)
                    mark_step = max(1, len(times) // 12)
                    ax.plot(times, isat_vals, color=color, marker=marker, markevery=mark_step,
                            label=ds_label, linestyle='None')

                time_unit = datasets[0].attrs.get('time_units', 'ms')
                isat_unit = datasets[0]['ion_isat'].attrs.get('units', 'A')
                x_unit = datasets[0].attrs.get('x_units', 'cm')

                ax.set_xlabel(f"Time [{time_unit}]")
                ax.set_ylabel(f"Ion Saturation Current [{isat_unit}]")
                ax.set_title(f"$I_{{sat}}$ vs Time — x = {x} {x_unit}, z ≈ {z_actual_val} cm")

                apply_isat_ylim(ax, ds_labels_list)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=True)

                ri_names = ''
                for idx, key in enumerate(ri_dict.keys()):
                    ri_names += key + f"{ri_dict[key]}"
                    if idx != len(ri_dict.keys()) - 1:
                        ri_names += '_'
                plot_name = f"z_{z_actual_val}_x_{x}_{ri_names}.png"
                plt.savefig(multiple_ds_folder + plot_name, bbox_inches='tight')
                plt.close(fig)

        # --- CASE 2: Single Dataset, Multiple x-values on 1 figure ---
        elif plot_all_x_together:
            x_colors = plt.colormaps['turbo'](np.linspace(0.15, 0.85, len(x_to_plot)))

            for ds_idx, ds in enumerate(datasets):
                ds_label = f_run_identifier(ds)
                x_unit = ds.attrs.get('x_units', 'cm')
                time_unit = ds.attrs.get('time_units', 'ms')
                isat_unit = ds['ion_isat'].attrs.get('units', 'A')

                # Sub-case 2A: Shot-Averaged
                if average_shots:
                    multiple_x_folder = ensure_directory(figure_folder + 'multiple_x_one_ds/')
                    ds_folder = ensure_directory(multiple_x_folder + ds_label + '/')

                    fig, axes, letters = build_subplots([[1]])
                    ax = axes[letters[0]]

                    z_actual_val = z_target
                    x_name = ''
                    for x_idx, x in enumerate(x_to_plot):
                        x_col = x_colors[x_idx]
                        x_mrk = marker_shapes[x_idx % len(marker_shapes)]

                        times, isat_vals, z_actual_val = extract_isat_trace(ds, z_target, x)
                        mark_step = max(1, len(times) // 12)

                        ax.plot(times, isat_vals, color=x_col, marker=x_mrk, markevery=mark_step,
                                label=f"x = {x} {x_unit}", linestyle='None')
                        x_name += f"_{x}"

                    ax.set_xlabel(f"Time [{time_unit}]")
                    ax.set_ylabel(f"Ion Saturation Current [{isat_unit}]")
                    ax.set_title(f"{ds_label} — z ≈ {z_actual_val} cm")

                    apply_isat_ylim(ax, ds_label)
                    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=min(4, len(x_to_plot)),
                              frameon=True)

                    fig_name = f"z_{z_actual_val}_x{x_name}.png"
                    plt.savefig(f"{ds_folder}{fig_name}", bbox_inches='tight')
                    plt.close(fig)

                # Sub-case 2B: Single Shots (Un-averaged)
                else:
                    multiple_x_shot_folder = ensure_directory(figure_folder + 'multiple_x_one_ds/single_shot/')
                    ds_folder = ensure_directory(multiple_x_shot_folder + ds_label + '/')

                    for shot in shots_to_plot:
                        fig, axes, letters = build_subplots([[1]])
                        ax = axes[letters[0]]

                        z_actual_val = z_target
                        x_name = ''
                        for x_idx, x in enumerate(x_to_plot):
                            x_col = x_colors[x_idx]
                            x_mrk = marker_shapes[x_idx % len(marker_shapes)]

                            times, isat_vals, z_actual_val = extract_isat_trace(ds, z_target, x, shot=shot)
                            mark_step = max(1, len(times) // 12)

                            ax.plot(times, isat_vals, color=x_col, marker=x_mrk, markevery=mark_step,
                                    label=f"x = {x} {x_unit}", linestyle='None')
                            x_name += f"_{x}"

                        ax.set_xlabel(f"Time [{time_unit}]")
                        ax.set_ylabel(f"Ion Saturation Current [{isat_unit}]")
                        ax.set_title(f"{ds_label} — Shot {shot} — z ≈ {z_actual_val} cm")

                        apply_isat_ylim(ax, ds_label)
                        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=min(4, len(x_to_plot)),
                                  frameon=True)

                        fig_name = f"shot_{shot}_z_{z_actual_val}_x{x_name}.png"
                        plt.savefig(f"{ds_folder}{fig_name}", bbox_inches='tight')
                        plt.close(fig)

        # --- CASE 3: Single Dataset, 1 x-value per figure ---
        else:
            for ds_idx, ds in enumerate(datasets):
                ds_label = f_run_identifier(ds)
                x_unit = ds.attrs.get('x_units', 'cm')
                time_unit = ds.attrs.get('time_units', 'ms')
                isat_unit = ds['ion_isat'].attrs.get('units', 'A')

                # Sub-case 3A: Shot-Averaged
                if average_shots:
                    one_x_folder = ensure_directory(figure_folder + 'one_x_one_ds/')
                    ds_folder = ensure_directory(one_x_folder + ds_label + '/')

                    for x in x_to_plot:
                        fig, axes, letters = build_subplots([[1]])
                        ax = axes[letters[0]]

                        times, isat_vals, z_actual = extract_isat_trace(ds, z_target, x)
                        mark_step = max(1, len(times) // 12)

                        ax.plot(times, isat_vals, color=clors[ds_idx], marker=marks[ds_idx],
                                markevery=mark_step, label=ds_label, linestyle='None')

                        ax.set_xlabel(f"Time [{time_unit}]")
                        ax.set_ylabel(f"Ion Saturation Current [{isat_unit}]")
                        ax.set_title(f"{ds_label} — z ≈ {z_actual} cm (x = {x} {x_unit})")

                        apply_isat_ylim(ax, ds_label)
                        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=True)

                        fig_name = f"z_{z_actual}_x_{x}.png"
                        plt.savefig(f"{ds_folder}{fig_name}", bbox_inches='tight')
                        plt.close(fig)

                # Sub-case 3B: Single Shots (Un-averaged)
                else:
                    one_x_shot_folder = ensure_directory(figure_folder + 'one_x_one_ds/single_shot/')
                    ds_folder = ensure_directory(one_x_shot_folder + ds_label + '/')

                    for shot in shots_to_plot:
                        for x in x_to_plot:
                            fig, axes, letters = build_subplots([[1]])
                            ax = axes[letters[0]]

                            times, isat_vals, z_actual = extract_isat_trace(ds, z_target, x, shot=shot)
                            mark_step = max(1, len(times) // 12)

                            ax.plot(times, isat_vals, color=clors[ds_idx], marker=marks[ds_idx],
                                    markevery=mark_step, label=ds_label, linestyle='None')

                            ax.set_xlabel(f"Time [{time_unit}]")
                            ax.set_ylabel(f"Ion Saturation Current [{isat_unit}]")
                            ax.set_title(f"{ds_label} — Shot {shot} — z ≈ {z_actual} cm (x = {x} {x_unit})")

                            apply_isat_ylim(ax, ds_label)
                            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=True)

                            fig_name = f"shot_{shot}_z_{z_actual}_x_{x}.png"
                            plt.savefig(f"{ds_folder}{fig_name}", bbox_inches='tight')
                            plt.close(fig)

def center_grads_vs_experimental_params(
    datasets,
    figure_folder="",
    axes=None,
    save_plots=False,
    show_plots=False,
    make_presentable=False,
):

  if figure_folder != "":
    figure_folder = ensure_directory(
        figure_folder + "machine_params_vs_plasma_width/"
    )
  plt.rcParams.update({
      "font.size": 20,
      "axes.labelsize": 22,
      "axes.titlesize": 22,
      "xtick.labelsize": 18,
      "ytick.labelsize": 18,
      "legend.fontsize": 16,
      "axes.formatter.use_mathtext": True,
      "lines.linewidth": 2.5,
      "lines.markersize": 10,
      "errorbar.capsize": 4,
  })

  if make_presentable:
    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.transparent": True,
    })

  options_to_plot = [
      "B-field",
      "GP Voltage",
      "Cathode Current",
      "Gradient Centers",
      "Core width",
      "ion_type",
  ]

  clor, mark = determine_colors(datasets)

  print("\nChoose the axes that you would like to plot")

  while True:
    y_ax_params = int_choose_multiple_from_list(options_to_plot, "y axes")
    x_ax_params = int_choose_multiple_from_list(options_to_plot, "x axes")
    if len(y_ax_params) == len(x_ax_params):
      if all(
          y_ax_params[p] != x_ax_params[p] for p in range(len(y_ax_params))
      ):
        break
      print(
          "You cannot plot the same parameters for y and x axes. Please try"
          " again."
      )
    else:
      print(
          f"Number of y parameters ({len(y_ax_params)}) != x parameters"
          f" ({len(x_ax_params)})."
      )

  plot_color = ask_yes_or_no(
      "Plot color axes (if nothing is chosen ion_type is the color)? (y/n) "
  )

  # Added prompt for splitting temperature and density gradients
  split_diags = ask_yes_or_no(
      "Split Temperature (Te) and Density (ne) gradients into separate plots?"
      " (y/n) "
  )

  color_ax = []
  for p_idx in range(len(y_ax_params)):
    x_p, y_p = x_ax_params[p_idx], y_ax_params[p_idx]
    if plot_color:
      color_plot_options = [
          opt
          for opt in options_to_plot
          if opt not in (x_p, y_p, "Gradient Centers", "Core width")
      ]
      color_ax_list = int_choose_multiple_from_list(
          color_plot_options, "color axes", lim_length=1
      )
      color_ax.append(color_ax_list[0] if color_ax_list else "ion_type")
    else:
      if "ion_type" in (x_p, y_p):
        while_num = 0
        while True:
          color_plot_options = [
              opt for opt in options_to_plot if opt not in (x_p, y_p)
          ]
          color_ax_list = int_choose_multiple_from_list(
              color_plot_options, "color axes", lim_length=1
          )
          if color_ax_list:
            color_ax.append(color_ax_list[0])
            break
          elif while_num > 0:
            color_ax.append("Black")
            break
          else:
            print(
                "\nIon type is an x/y axis. Select another color axis or leave"
                " blank for black."
            )
            while_num += 1
      else:
        color_ax.append("ion_type")

  (
      dens_grad_regions_str,
      _,
      _,
      temp_grad_regions_str,
      _,
      _,
  ) = xarray_gradient_strings()

  def safe_mean(data):
    return np.mean(data) if len(data) > 0 else np.nan

  def extract_param_value(ds, ax_param):
    if ax_param in ["B-field", "GP Voltage", "Cathode Current"]:
      attr_raw = str(ds.attrs.get(ax_param, "0.0 a.u."))
      parts = attr_raw.split()
      val = float(parts[0])
      unit = parts[1] if len(parts) > 1 else ""
      return val, val, unit

    elif ax_param == "ion_type":
      ion = ds.attrs.get("ion_type", "Unknown")
      return ion, ion, ""

    elif ax_param in ["Core width", "Gradient Centers"]:
      has_temp = temp_grad_regions_str in ds.attrs
      has_dens = dens_grad_regions_str in ds.attrs

      temp_less, temp_big, dens_less, dens_big = [], [], [], []
      if has_temp:
        t_edges = [
            tuple(e) for e in json.loads(ds.attrs[temp_grad_regions_str])
        ]
        for start, stop in t_edges:
          mid = (start + stop) / 2.0
          (temp_less if mid < 0 else temp_big).append(mid)

      if has_dens:
        d_edges = [
            tuple(e) for e in json.loads(ds.attrs[dens_grad_regions_str])
        ]
        for start, stop in d_edges:
          mid = (start + stop) / 2.0
          (dens_less if mid < 0 else dens_big).append(mid)

      if ax_param == "Core width":
        t_lower, t_upper = safe_mean(temp_less), safe_mean(temp_big)
        d_lower, d_upper = safe_mean(dens_less), safe_mean(dens_big)

        t_width = (
            (t_upper - t_lower)
            if (not np.isnan(t_lower) and not np.isnan(t_upper))
            else np.nan
        )
        d_width = (
            (d_upper - d_lower)
            if (not np.isnan(d_lower) and not np.isnan(d_upper))
            else np.nan
        )
        return t_width, d_width, "cm"

      elif ax_param == "Gradient Centers":
        t_centers = np.array(temp_less + temp_big)
        d_centers = np.array(dens_less + dens_big)
        return t_centers, d_centers, "cm"

    return np.nan, np.nan, ""

  def broadcast_data(x_val, y_val, c_val):
    x_arr = np.atleast_1d(x_val)
    y_arr = np.atleast_1d(y_val)
    c_arr = np.atleast_1d(c_val) if c_val is not None else None

    max_len = max(len(x_arr), len(y_arr))
    if max_len == 0:
      return np.array([]), np.array([]), np.array([])

    if len(x_arr) == 1 and max_len > 1:
      x_arr = np.repeat(x_arr, max_len)
    if len(y_arr) == 1 and max_len > 1:
      y_arr = np.repeat(y_arr, max_len)
    if c_arr is not None and len(c_arr) == 1 and max_len > 1:
      c_arr = np.repeat(c_arr, max_len)

    return x_arr, y_arr, c_arr

  # Diagnostics routing setup
  diag_groups = [["Te"], ["ne"]] if split_diags else [["Te", "ne"]]

  for p_idx in range(len(y_ax_params)):
    y_param = y_ax_params[p_idx]
    x_param = x_ax_params[p_idx]
    c_param = color_ax[p_idx]

    for current_group in diag_groups:
      if axes is None:
        fig, ax_dict, letters = build_subplots([[1]])
        ax = ax_dict[letters[0]]
      else:
        ax = axes

      x_unit, y_unit, c_unit = "", "", ""

      is_b_field = c_param == "B-field"
      is_numeric_color = (
          c_param in ["GP Voltage", "Cathode Current"] or is_b_field
      )
      is_black = c_param == "Black"

      if is_numeric_color:
        all_c_vals = []
        for ds in datasets:
          c_val, _, c_u = extract_param_value(ds, c_param)
          c_unit = c_u
          if not np.isnan(c_val):
            all_c_vals.append(c_val)

        if len(all_c_vals) > 0:
          c_min, c_max = np.min(all_c_vals), np.max(all_c_vals)
          if c_min == c_max:
            c_min, c_max = (
                c_min - 0.1 * abs(c_min) if c_min != 0 else -1,
                c_max + 0.1 * abs(c_max) if c_max != 0 else 1,
            )
          norm = mcolors.Normalize(vmin=c_min, vmax=c_max)

          if is_b_field:
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "BluePink", ["royalblue", "deeppink"]
            )
          else:
            cmap = plt.cm.viridis
        else:
          is_numeric_color = False

      is_profile_plot = any(
          p in ["Core width", "Gradient Centers"] for p in [x_param, y_param]
      )

      for i, ds in enumerate(datasets):
        ds_color = clor[i]
        ds_marker = mark[i]
        ion_label = ds.attrs.get("ion_type", f"Run {i + 1}")

        x_t, x_d, x_u = extract_param_value(ds, x_param)
        y_t, y_d, y_u = extract_param_value(ds, y_param)
        c_t, c_d, _ = extract_param_value(ds, c_param)

        x_unit, y_unit = x_u, y_u

        for diag_name, (x_val, y_val, c_val) in [
            ("Te", (x_t, y_t, c_t)),
            ("ne", (x_d, y_d, c_d)),
        ]:
          # Filter diagnostic based on choice
          if diag_name not in current_group:
            continue

          x_b, y_b, c_b = broadcast_data(x_val, y_val, c_val)

          valid_mask = ~np.isnan(np.asarray(x_b, dtype=float)) & ~np.isnan(
              np.asarray(y_b, dtype=float)
          )

          x_plot = x_b[valid_mask]
          y_plot = y_b[valid_mask]
          c_plot = c_b[valid_mask]

          if len(x_plot) == 0:
            continue

          if is_numeric_color:
            base_color = cmap(norm(c_plot.astype(float)))
          elif is_black:
            base_color = "black"
          else:
            base_color = ds_color

          if is_profile_plot:
            if diag_name == "Te":
              face_col = "none"
              edge_col = base_color
              hatch_pattern = None
              lw = 2.0
            else:  # Density ('ne')
              face_col = "none"
              edge_col = base_color
              hatch_pattern = r"\\\\\\"
              lw = 1.2

            legend_label = ion_label if not is_numeric_color else None
          else:
            face_col = base_color
            edge_col = "k"
            hatch_pattern = None
            lw = 1.0

            legend_label = (
                f"{ion_label} ({diag_name})"
                if not is_numeric_color
                else f"{diag_name}"
            )

          ax.scatter(
              x_plot,
              y_plot,
              facecolors=face_col,
              edgecolors=edge_col,
              hatch=hatch_pattern,
              linewidths=lw,
              marker=ds_marker,
              s=130,
              label=legend_label,
          )

      if is_profile_plot:
        if "Te" in current_group:
          ax.scatter(
              [],
              [],
              facecolors="none",
              edgecolors="black",
              marker="o",
              s=130,
              linewidths=2.0,
              label=r"$T_e$ (Hollow)",
          )
        if "ne" in current_group:
          ax.scatter(
              [],
              [],
              facecolors="none",
              edgecolors="black",
              hatch=r"\\\\\\",
              marker="o",
              s=130,
              linewidths=1.2,
              label=r"$n_e$ (Hatched)",
          )

      xlabel_str = f"{x_param} [{x_unit}]" if x_unit else x_param
      ylabel_str = f"{y_param} [{y_unit}]" if y_unit else y_param
      ax.set_xlabel(xlabel_str)
      ax.set_ylabel(ylabel_str)

      handles, labels = ax.get_legend_handles_labels()
      by_label = dict(zip(labels, handles))
      if by_label:
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(by_label),
            frameon=True,
        )

      if is_numeric_color:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar_label = f"{c_param} [{c_unit}]" if c_unit else c_param
        cbar.set_label(cbar_label)

      if save_plots and figure_folder:
        suffix = f"_{current_group[0]}" if split_diags else ""
        plot_title = f"{figure_folder}{y_param}_vs_{x_param}{suffix}"
        add_to_title = ask_yes_or_no(
            f"To be saved as {plot_title}.png. Add anything? (y/n) "
        )
        if add_to_title:
          addition = input("What do you want to add to the save title?")
          plot_title = plot_title + addition
        plt.savefig(f"{plot_title}.png", bbox_inches="tight")
        print(f"Saved plot to: {plot_title}.png")

      if show_plots:
        plt.show()







    # for plot_name in plot_list:
    #     if axes is None:
    #         fig, ax = plt.subplots(figsize=(14,11))
    #     else:
    #         ax = axes
    #
    #     # Tracker for presentation mode to prevent duplicate labels
    #     seen_labels = set()
    #
    #     if len(datasets) < 10:
    #         list_plot_names = []
    #     else:
    #         list_plot_names = None
    #
    #     for i, dataset in enumerate(datasets):
    #         run_identifier = f_run_identifier(dataset)
    #         if list_plot_names is not None:
    #             exp_name = run_identifier.split(',')[0].strip()
    #             run_num = run_identifier.split(',')[1].strip()
    #             list_plot_names.append(f'{exp_name}_{run_num}')
    #         gpv = dataset.attrs['GP Voltage']
    #         cath_curr = dataset.attrs['Cathode Current']
    #         b_field = dataset.attrs['B-field']
    #         y_full_str = dataset.attrs[plot_name]
    #         y_str = y_full_str.split(' ')[0]
    #         y_units = y_full_str.split(' ')[1]
    #         if i == 0:
    #             y_units = y_full_str.split(' ')[1]
    #         x = float(y_str)
    #
    #         if make_presentable:
    #             # --- PRESENTATION MODE: Simplified groups (Max 4) ---
    #             parts = run_identifier.split(',')
    #             if len(parts) >= 3:
    #                 group_label = f"{parts[0].strip()}, {parts[-1].strip()}"
    #             else:
    #                 group_label = run_identifier
    #
    #             if group_label not in seen_labels:
    #                 ax.plot([], [], color=clor[i], marker=mark[i], linestyle='None', label=group_label)
    #                 seen_labels.add(group_label)
    #         else:
    #             # --- ANALYSIS MODE: Plot every individual run ---
    #             ax.plot([], [], color=clor[i], marker=mark[i], linestyle='None',
    #                     label=run_identifier + f'_{gpv}_{cath_curr}_{b_field}')
    #
    #         if (temp_grad_regions_str in dataset.attrs) and (dens_grad_regions_str in dataset.attrs):
    #             temp_load_regions = json.loads(dataset.attrs[temp_grad_regions_str])
    #             temp_regions = [tuple(edge) for edge in temp_load_regions]
    #
    #             dens_load_regions = json.loads(dataset.attrs[dens_grad_regions_str])
    #             dens_regions = [tuple(edge) for edge in dens_load_regions]
    #
    #             for start, stop in temp_regions:
    #                 y = (start + stop) / 2
    #                 print(f'{run_identifier} temperature {y}')
    #                 ax.plot(x, y, color=clor[i], marker=mark[i], fillstyle='none')
    #             for start, stop in dens_regions:
    #                 y = (start + stop) / 2
    #                 print(f'{run_identifier} density {y}')
    #                 ax.plot(x, y, color=clor[i], marker=mark[i])
    #
    #     ax.set_xlabel(f'{plot_name} [{y_units}]')
    #     ax.set_ylabel('Gradient center x [cm]')
    #
    #     if make_presentable:
    #         # --- PRESENTATION LEGEND: 4 Groups + Shape Key ---
    #         temp_handle = mlines.Line2D([], [], color='k', marker='^', linestyle='None',
    #                                     fillstyle='none', label='Temp Gradient', markersize=12)
    #         dens_handle = mlines.Line2D([], [], color='k', marker='^', linestyle='None',
    #                                     label='Density Gradient', markersize=12)
    #
    #         handles, labels = ax.get_legend_handles_labels()
    #         handles.extend([temp_handle, dens_handle])
    #         labels.extend(['Temp Gradient', 'Density Gradient'])
    #
    #         # 1. Baseline the subplots first
    #         plt.tight_layout()
    #
    #         # 2. Aggressively raise the bottom margin of the plots (from 0.28 to 0.35)
    #         # This clears out a massive, clean gap below the x-axis label
    #         if axes is None:
    #             fig.subplots_adjust(bottom=0.28, left=0.15, top=0.92, right=0.95)
    #
    #         # 3. Pin the legend to the absolute bottom margin edge (0.01)
    #         # Dropping the font size to 24 keeps it from expanding upwards into your labels
    #         fig.legend(
    #             handles=handles,
    #             labels=labels,
    #             loc='lower center',
    #             bbox_to_anchor=(0.5, 0.01),
    #             ncol=2,  # Keeping it to 2 columns stacks it cleanly without overflowing horizontally
    #             framealpha=0.8,
    #             fontsize=24
    #         )
    #
    #     else:
    #         # --- ANALYSIS LEGEND ---
    #         plt.tight_layout()
    #         if axes is None:
    #             fig.subplots_adjust(bottom=0.30, left=0.15, top=0.92, right=0.95)
    #
    #         fig.legend(
    #             loc='lower center',
    #             bbox_to_anchor=(0.5, 0.01),
    #             ncol=3,
    #             framealpha=0.8,
    #             fontsize=14
    #         )
    #
    #     if save_plots:
    #         save_str = ''
    #         if list_plot_names is not None:
    #             for run_name in list_plot_names:
    #                 save_str += run_name + '_'
    #         save_folder = ensure_directory(figure_folder + 'gradient_center_vs_x/')
    #         plot_str = ''
    #         if plot_name == 'B-field':
    #             save_folder = ensure_directory(save_folder + 'gradient_center_vs_b/')
    #             plot_str = 'Bfield' + save_str
    #
    #         elif plot_name == 'GP Voltage':
    #             save_folder = ensure_directory(save_folder + 'gradient_center_vs_gpv/')
    #             plot_str = 'GPVoltage' + save_str
    #         elif plot_name == 'Cathode Current':
    #             save_folder = ensure_directory(save_folder + 'gradient_center_cc/')
    #             plot_str = 'CathCurr' + save_str
    #
    #         plt.savefig(save_folder + plot_str + '.png', bbox_inches='tight')
    #         print('Saved figure to: ', save_folder + plot_str + '.png')
    #
    # if show_plots:
    #     plt.show()
    #     plt.close()
    # else:
    #     plt.close()


def overlapping_radial_plots(datasets, pathnames='', figure_folder='',
                             axes=None, see_plots=True, one_probe=True, save_plots=False):
    '''
    Parameters
    ----------
    datasets - A list of xarray datasets corresponding to different runs of LAPD
    '''
    default_fig_height, default_fig_width = default_fig_params()
    clrs, mark = determine_colors(datasets)
    (dens_regions_str, dens_slopes_str, dens_intercepts_str,
     temp_regions_str, temp_slopes_str, temp_intercepts_str) = xarray_gradient_strings()

    if one_probe:
        probes = [0]
    else:
        probes = [probe for probe in datasets[0].coords['probe'].values]

    temp_axes = []
    dens_axes = []

    # ------------------------------------------------------------------
    # NEW: Pre-calculate all base run identifiers to detect matching runs
    # ------------------------------------------------------------------
    all_base_ids = [f_run_identifier(ds=d) for d in datasets]

    for probe in probes:
        if axes is None:
            layout = [[1],
                      [1]]
            fig, ax, letters = build_subplots(layout, 1.2 * default_fig_width, 1.2 * default_fig_height,
                                              sharex=True)
            temperature_ax = ax[letters[0]]
            density_ax = ax[letters[1]]
            temp_axes.append(temperature_ax)
            dens_axes.append(density_ax)
        else:
            whole_ax = axes[probe]
            temperature_ax = whole_ax[0]
            density_ax = whole_ax[1]
            temp_axes.append(temperature_ax)
            dens_axes.append(density_ax)

        i = 0
        for ds in datasets:
            pathname = pathnames[i]
            color = clrs[i]
            mk = mark[i]

            # Get the base identifier
            base_run_identifier = f_run_identifier(ds=ds)

            # ------------------------------------------------------------------
            # DYNAMIC NORMALIZATION CHECK
            # If the base identifier appears more than once, we are plotting
            # a raw vs updated file for the same run. Don't normalize!
            # ------------------------------------------------------------------
            if all_base_ids.count(base_run_identifier) > 1:
                should_normalize = False
            else:
                should_normalize = True

            run_identifier = base_run_identifier

            if 'updated' in pathname.lower():
                color = 'black'
                run_identifier = f"{run_identifier}_updated"

            plot_axes = True if i == 0 else False

            temp_ax = temp_axes[-1]
            dens_ax = dens_axes[-1]

            # Pass our dynamic 'should_normalize' flag to both plot functions
            create_temperature_radial_plots(ds, probe, run_identifier=run_identifier,
                                            see_plots=True,
                                            axes=temp_ax, dataset_clor=color, dataset_mark=mk,
                                            make_presentable=False, sharex=True,
                                            redo_grad_regions=False, gradient_regions=False,
                                            regions_str=temp_regions_str, slopes_str=temp_slopes_str,
                                            intercepts_str=temp_intercepts_str, ds_save_path=pathname,
                                            plot_final_fit=False, plot_axes=plot_axes, plot_title=False,
                                            normalize=should_normalize)

            create_density_radial_plots(ds, probe, run_identifier=run_identifier,
                                        see_plots=True,
                                        axes=dens_ax, dataset_clor=color, dataset_mark=mk,
                                        make_presentable=False, sharex=True,
                                        redo_grad_regions=False, gradient_regions=False,
                                        regions_str=dens_regions_str, slopes_str=dens_slopes_str,
                                        intercepts_str=dens_intercepts_str, ds_save_path=pathname,
                                        plot_final_fit=False, plot_axes=plot_axes, plot_title=False,
                                        normalize=should_normalize)

            dens_ax.plot([], [], color=color, marker=mk, linestyle='None', label=run_identifier)
            i += 1

        dens_ax = dens_axes[-1]
        dens_ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            framealpha=0.8,
            fontsize=20
        )

        plt.tight_layout()

        if save_plots:
            save_folder = ensure_directory(figure_folder + 'overlapping_radial_plots/')
            save_path = save_folder + f'overlapping_radial_plot_{len(datasets)}_datasets_probe_{probe}.svg'
            plt.savefig(save_path, bbox_inches='tight')
            print('Saved figure to: ', save_path)

        if see_plots:
            plt.show()

        plt.close()


def generate_nu_eff_values(ds, probe_idx, color, mark, L,
                           axes = None, run_identifier = None):
    min_time = ds.attrs[f'steady state start probe {probe_idx}']
    max_time = ds.attrs[f'steady state end probe {probe_idx}']

    mean_data = ds['t_e'].sel(probe=probe_idx).mean('shot')
    std_data = ds['t_e'].sel(probe=probe_idx).std('shot')

    t_e_filtered_data = filter_data(mean_data, std_data)

    t_e_mask = (t_e_filtered_data['time'] >= min_time) & (t_e_filtered_data['time'] <= max_time)

    t_e_to_plot = t_e_filtered_data.sel(sweep=ds['sweep'][t_e_mask]).mean('sweep')
    t_e_std_to_plot = t_e_filtered_data.sel(sweep=ds['sweep'][t_e_mask]).std('sweep')

    # Format everything for matplot.lib plotting
    x_vals = t_e_filtered_data['x'].values
    t_e_vals = t_e_to_plot.squeeze().values
    t_e_std = t_e_std_to_plot.squeeze().values
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111)

    if run_identifier is None:
        run_identifier = f_run_identifier(ds)

    particle_str = ds.attrs['ion_type']
    ion_particle = Particle(particle_str)

    e_charge = c.e.si
    e_charge = e_charge.to(u.C)

    z_eff = ion_particle.charge_number
    ion_mass = ion_particle.mass
    ion_mass = ion_mass.to(u.kg)


    b_field_str = ds.attrs["B-field"]
    b_field = Quantity(b_field_str)
    b_field = b_field.to('T')

    t_e_units = ds['t_e'].units

    t_e_w_units = t_e_vals * u.Unit(t_e_units)
    t_e_std_w_units = t_e_std * u.Unit(t_e_units)

    min_time = ds.attrs[f'steady state start probe {probe_idx}']
    max_time = ds.attrs[f'steady state end probe {probe_idx}']

    mean_data = ds['t_e'].sel(probe=probe_idx).mean('shot')
    std_data = ds['t_e'].sel(probe=probe_idx).std('shot')
    #
    t_e_filtered_data = filter_data(mean_data, std_data)
    where_nans = t_e_filtered_data.isnull()
    nu_ei_filtered_data = ds['nu_ei'].sel(probe=probe_idx).mean('shot').where(~where_nans)

    nu_ei_mask = (nu_ei_filtered_data['time'] >= min_time) & (nu_ei_filtered_data['time'] <= max_time)

    nu_ei_to_plot = nu_ei_filtered_data.sel(sweep=ds['sweep'][nu_ei_mask]).mean('sweep')
    nu_ei_std_to_plot = nu_ei_filtered_data.sel(sweep=ds['sweep'][nu_ei_mask]).std('sweep')


    # Format everything for matplot.lib plotting
    nu_ei_to_plot_vals = nu_ei_to_plot.squeeze().values
    nu_ei_std_to_plot_vals = nu_ei_std_to_plot.squeeze().values

    nu_units = ds['nu_ei'].units

    nu_ei_w_units = nu_ei_to_plot_vals * u.Unit(nu_units)
    nu_ei_std_units = nu_ei_std_to_plot_vals * u.Unit(nu_units)

    if t_e_w_units.unit == Unit('eV'):
        t_e_joules = t_e_w_units.to(u.J, equivalencies=u.temperature_energy())
        t_e_std_joules = t_e_std_w_units.to(u.J, equivalencies=u.temperature_energy())
    elif t_e_w_units.unit == Unit('K'):
        t_e_joules = t_e_w_units.to(u.J, equivalencies=u.temperature_energy())
        t_e_std_joules = t_e_std_w_units.to(u.J, equivalencies=u.temperature_energy())
    else:
        t_e_joules = t_e_w_units
        t_e_std_joules = t_e_std_w_units
    core_idxs = np.where((x_vals >= -10) & (x_vals <= 10))[0]
    nu_eff = (2 * np.pi * np.sqrt(t_e_joules / ion_mass) / (nu_ei_w_units * L)).to(u.dimensionless_unscaled)
    avg_core_nu_eff = np.mean(nu_eff[core_idxs])
    print(f'Average core nu_eff: {avg_core_nu_eff}')
    rel_err_te = t_e_std_joules / t_e_joules
    rel_err_nu_ei = nu_ei_std_units / nu_ei_w_units
    nu_eff_rel_err = np.sqrt((0.5 * rel_err_te) ** 2 + (rel_err_nu_ei) ** 2)
    nu_eff_total_err = nu_eff * nu_eff_rel_err

    axes.errorbar(x_vals, nu_eff.value, yerr = nu_eff_total_err.value, color=color, fmt = mark, capsize = 3,
                  label=run_identifier, linestyle='None')

def generate_rhostar_values(ds, probe_idx, color, mark, a,
                            axes = None, run_identifier = None):
    min_time = ds.attrs[f'steady state start probe {probe_idx}']
    max_time = ds.attrs[f'steady state end probe {probe_idx}']

    mean_data = ds['t_e'].sel(probe=probe_idx).mean('shot')
    std_data = ds['t_e'].sel(probe=probe_idx).std('shot')

    t_e_filtered_data = filter_data(mean_data, std_data)

    t_e_mask = (t_e_filtered_data['time'] >= min_time) & (t_e_filtered_data['time'] <= max_time)

    t_e_to_plot = t_e_filtered_data.sel(sweep=ds['sweep'][t_e_mask]).mean('sweep')
    t_e_std_to_plot = t_e_filtered_data.sel(sweep=ds['sweep'][t_e_mask]).std('sweep')

    # Format everything for matplot.lib plotting
    x_vals = t_e_filtered_data['x'].values
    t_e_vals = t_e_to_plot.squeeze().values
    t_e_std = t_e_std_to_plot.squeeze().values
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111)

    if run_identifier is None:
        run_identifier = f_run_identifier(ds)
    particle_str = ds.attrs['ion_type']
    ion_particle = Particle(particle_str)

    e_charge = c.e.si
    e_charge = e_charge.to(u.C)

    z_eff = ion_particle.charge_number
    ion_mass = ion_particle.mass
    ion_mass = ion_mass.to(u.kg)


    b_field_str = ds.attrs["B-field"]
    b_field = Quantity(b_field_str)
    b_field = b_field.to('T')

    t_e_units = ds['t_e'].units

    t_e_w_units = t_e_vals * u.Unit(t_e_units)
    t_e_std_units = t_e_std * u.Unit(t_e_units)

    if t_e_w_units.unit == Unit('eV'):
        t_e_joules = t_e_w_units.to(u.J, equivalencies=u.temperature_energy())
        t_e_std_joules = t_e_std_units.to(u.J, equivalencies=u.temperature_energy())

    elif t_e_w_units.unit == Unit('K'):
        t_e_joules = t_e_w_units.to(u.J, equivalencies=u.temperature_energy())
        t_e_std_joules = t_e_std_units.to(u.J, equivalencies=u.temperature_energy())
    else:
        t_e_joules = t_e_w_units
        t_e_std_joules = t_e_std_units
    core_idxs = np.where((x_vals >= -10) & (x_vals <= 10))[0]
    rhostar = ((ion_mass * t_e_joules) ** 0.5 / (e_charge * b_field * a)).to(u.dimensionless_unscaled)
    avg_core_rhostar = np.mean(rhostar[core_idxs])
    print(f'Average core rhostar: {avg_core_rhostar}')
    rel_err_te = t_e_std_joules / t_e_joules
    rhostar_rel_err = 0.5 * rel_err_te
    rhostar_total_err = rhostar * rhostar_rel_err

    # print('mark ', mark)
    axes.errorbar(x_vals,
                        rhostar.value,
                        yerr=rhostar_total_err.value,
                        color=color,
                        fmt=mark,
                        capsize=3,
                        label=run_identifier,
                        linestyle='None')


