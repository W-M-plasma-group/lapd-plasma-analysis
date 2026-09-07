import json
import numpy as np
import ast
import os
import matplotlib

from lapd_plasma_analysis.obtain_plots.xarray_plots import build_subplots

matplotlib.use('TkAgg')
from scipy.ndimage import binary_closing

from lapd_plasma_analysis.obtain_plots.Auxillary_functions import *
from lapd_plasma_analysis.obtain_plots.plot_from_netcdf_helpers.process_temp_dens_data import *
from lapd_plasma_analysis.file_access import *
from lapd_plasma_analysis.Read_hdf5.read_metadata import *
from lapd_plasma_analysis.Read_hdf5.primary_functions import *

def create_radial_plot(
    ds,
    probe,
    var_name='n_e',
    run_identifier='',
    see_plots=True,
    axes=None,
    dataset_clor=None,
    dataset_mark=None,
    make_presentable=False,
    sharex=False,
    bottomx=True,
    gradient_regions=False,
    redo_grad_regions=False,
    regions_str=None,
    slopes_str=None,
    intercepts_str=None,
    ds_save_path=None,
    plot_final_fit=False,
    plot_axes=True,
    plot_title=True,
    normalize=False,
    hdf5_folder=None,
    save_plots=False,
    figure_folder=None,
    updated_nc_folder=None
):
    """
    Parameters
    ----------
    ds : xarray.Dataset
        xarray dataset of diagnostics indexed by probe, x, y, shot, sweep
    probe : int
        Index of the probe to plot
    var_name : str, optional
        Diagnostic variable key inside ds to plot (e.g., 'n_e', 't_e')
    run_identifier : str, optional
        Formatted string identifying the experimental run
    see_plots : bool, optional
        Whether to generate and display the matplotlib figures
    axes : matplotlib.axes.Axes, optional
        Existing axis object to plot into
    dataset_clor : str, optional
        Color for plot points and error bars
    dataset_mark : str, optional
        Marker style for plot points
    make_presentable : bool, optional
        Apply clean presentation/publication style formatting
    sharex : bool, optional
        Whether the x-axis is shared across subplots
    bottomx : bool, optional
        Whether to draw the bottom x-axis labels
    gradient_regions : bool, optional
        Whether to calculate and display gradient region bounds
    redo_grad_regions : bool, optional
        Prompt user to interactively re-select gradient regions
    regions_str : str, optional
        Attribute key in ds storing gradient region start and end boundaries
    slopes_str : str, optional
        Attribute key in ds storing fitted slopes for each region
    intercepts_str : str, optional
        Attribute key in ds storing fitted intercepts for each region
    ds_save_path : str, optional
        File path location for saving updated netCDF datasets
    plot_final_fit : bool, optional
        Whether to plot the linear fit line over gradient regions
    plot_axes : bool, optional
        Whether to draw labels and ticks on axes
    plot_title : bool, optional
        Whether to display diagnostic titles on plots
    normalize : bool, optional
        Normalize profile values relative to core plasma average (-5 to 5 cm)
    hdf5_folder : str, optional
        Folder containing raw HDF5 files for reprocessing
    save_plots : bool, optional
        Whether to save figure images to disk
    figure_folder : str, optional
        Output folder path for saving figures
    updated_nc_folder : str, optional
        Output folder path for saving updated netCDF files

    Returns
    -------
    ds : xarray.Dataset
        The original or updated xarray dataset
    """

    # Map variable names to readable titles and LaTeX representations
    var_title_map = {'n_e': 'Density', 't_e': 'Temperature'}
    var_symbol_map = {'n_e': r'n_e', 't_e': r'T_e'}

    var_label_name = var_title_map.get(var_name, var_name)
    var_symbol = var_symbol_map.get(var_name, var_name)

    print(f'Building {var_label_name.lower()} radial plots...')

    if dataset_clor is None:
        dataset_clor = 'royalblue'
        dataset_mark = 'o'

    # Extract plot values, standard deviation error, and radial positions
    vals, std_vals, x_vals = process_variable_data(ds, probe, var_name=var_name)

    # Double-check temperature/density data if NaNs are present and HDF5 directory exists
    if np.any(np.isnan(vals)) and (hdf5_folder is not None) and (var_name == 'n_e'):
        updated_ds = double_check_temp_data(
            hdf5_folder, ds, probe, x_vals=x_vals, y_vals=vals,
            save_plots=save_plots, figure_folder=figure_folder
        )

        if updated_ds is not None:
            # Recompute every t_e-dependent variable (n_e, n_i, nu_ei, p_e, p_ei)
            # now that t_e has changed. v_f/v_p/ion_isat/electron_isat are untouched.
            updated_ds = recalculate_derived_variables(updated_ds)

            base_name = os.path.splitext(os.path.basename(ds_save_path))[0]
            if base_name.endswith('_updated'):
                # Keep the exact same name, no extra tags
                updated_ds_save_path = os.path.join(updated_nc_folder, f"{base_name}.nc")
                print("Existing updated file detected. Overwriting with new modifications...")
            else:
                # It's a raw file being updated for the first time
                updated_ds_save_path = os.path.join(updated_nc_folder, f"{base_name}_updated.nc")
                print("Raw file detected. Creating new _updated.nc file...")

            updated_ds.to_netcdf(updated_ds_save_path, mode='w', engine='netcdf4')
            print(f'Successfully saved the dataset at: \n {updated_ds_save_path}.')

            # Re-assign ds and update plot values so the rest of the function uses the newly fixed data
            ds = updated_ds
            vals, std_vals, x_vals = process_variable_data(ds, probe, var_name=var_name)

    # Get metadata attributes for labels
    long_var_name = ds[var_name].attrs.get("long_name", var_name)
    units_str = ds[var_name].attrs.get('units', var_name)
    ylabel = fr"${var_symbol}$ [{units_str}]"
    xlabel = f'x [{ds.attrs.get("x_units")}]'

    # Handle gradient region detection and linear polyfit calculations
    if redo_grad_regions:
        print(f'redo {var_name} str: ', redo_grad_regions)
        edges = obtain_gradient_regions(
            ds, x_vals, vals, std_vals,
            color=dataset_clor, marker=dataset_mark,
            ylabel=ylabel, xlabel=xlabel, regions_str=regions_str
        )

        slopes = []
        intercepts = []
        for start_x, stop_x in edges:
            region_mask = (x_vals >= start_x) & (x_vals <= stop_x)
            x_region = x_vals[region_mask]
            var_region = vals[region_mask]
            valid_mask = ~np.isnan(x_region) & ~np.isnan(var_region)
            valid_x = x_region[valid_mask]
            valid_var = var_region[valid_mask]

            slope, intercept = np.polyfit(valid_x, valid_var, deg=1)
            slopes.append(slope)
            intercepts.append(intercept)

        if ds_save_path is not None and regions_str:
            ds.attrs[regions_str] = json.dumps(edges)
            ds.attrs[slopes_str] = slopes
            ds.attrs[intercepts_str] = intercepts
            ds.to_netcdf(ds_save_path, mode='a', engine='netcdf4')
            print(f'Updated the dataset at \n {ds_save_path} \n to include {var_label_name} gradient locations.')

    elif not redo_grad_regions and (regions_str and regions_str in ds.attrs.keys()) and gradient_regions:
        loaded_edges = json.loads(ds.attrs[regions_str])
        edges = [tuple(edge) for edge in loaded_edges]
        slopes = ds.attrs[slopes_str]
        intercepts = ds.attrs[intercepts_str]
    else:
        edges = []
        slopes = []
        intercepts = []

    # Construct plots
    if see_plots:
        if normalize:
            core_x_idxs = np.where((x_vals <= 5) & (x_vals >= -5))[0]
            core_vals = vals[core_x_idxs]
            core_avg = np.mean(core_vals)
            vals = vals / core_avg
            std_vals = std_vals / core_avg

        if axes is None:
            fig, ax, letters = build_subplots(layout=[[1]])
            axes = ax[letters[0]]

        axes.errorbar(x_vals, vals, yerr=std_vals, fmt=dataset_mark, capsize=3, color=dataset_clor)
        axes.set_ylabel(ylabel, rotation=0, labelpad=60)

        if gradient_regions:
            for i, (start_x, stop_x) in enumerate(edges):
                # Draw the start line
                axes.axvline(x=start_x, color='black', linestyle='--', linewidth=2)
                # Draw the stop line
                axes.axvline(x=stop_x, color='black', linestyle='--', linewidth=2)
                if plot_final_fit:
                    region_mask = (x_vals >= start_x) & (x_vals <= stop_x)
                    axes.plot(
                        x_vals[region_mask],
                        slopes[i] * x_vals[region_mask] + intercepts[i],
                        color='gold', linestyle='--'
                    )

        if not make_presentable:
            if plot_axes:
                if plot_title:
                    axes.set_title(f"{long_var_name}")
                probe_z = ds['z'].isel(probe=probe).item()
                ss_start = ds.attrs.get(f'steady state start probe {probe}', 0)
                ss_end = ds.attrs.get(f'steady state end probe {probe}', 0)

                axes.set_title(
                    f"{run_identifier} \n "
                    f"{long_var_name} \n "
                    f"z: {probe_z:.2f} \n "
                    f"Between times ({ss_start:.2f},{ss_end:.2f}) ms"
                )
                if not sharex:
                    axes.set_xlabel(xlabel)
        else:
            if plot_axes:
                axes.ticklabel_format(style='sci', axis='y', scilimits=(-1, 2))
                if bottomx:
                    axes.set_xlabel(xlabel)
                    axes.tick_params(axis='x', labelbottom=True)
                else:
                    axes.set_xlabel("")
                    axes.tick_params(axis='x', labelbottom=False)

                # 2. FORCE THE MATH ENGINE (Bypass canvas.draw completely)
                ax_formatter = axes.yaxis.get_major_formatter()
                ax_formatter.set_locs(axes.yaxis.get_majorticklocs())
                offset = ax_formatter.get_offset()

                # 3. Hide the default floating text
                axes.yaxis.get_offset_text().set_visible(False)

                # 4. Build the dynamic label strings using f-strings
                if not sharex:
                    if offset:
                        # Because we turned on 'use_mathtext' globally, offset is
                        # automatically formatted as beautiful LaTeX (e.g., $\times10^{4}$)
                        label_str = fr'{long_var_name} ({offset})'
                    else:
                        label_str = rf'{long_var_name}'
                else:
                    label_str = fr'{offset}' if offset else ''

                axes.text(
                    0.5, 0.05, label_str,
                    transform=axes.transAxes,
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    color='k'
                )

    return ds







def create_density_radial_plots(ds, probe, run_identifier = '', see_plots = True,
                                axes = None, dataset_clor = None,
                                dataset_mark = None,make_presentable = False,
                                sharex = False, bottomx = True, gradient_regions = False, redo_grad_regions = False,
                                regions_str = None, slopes_str = None, intercepts_str = None, ds_save_path = None,
                                plot_final_fit = False, plot_axes = True, plot_title = True, normalize = False,
                                hdf5_folder = None, save_plots = False, figure_folder = None, updated_nc_folder = None):

    print('Building density radial plots...')
    if dataset_clor is None:
        dataset_clor = 'royalblue'
        dataset_mark = 'o'
    n_e_to_plot_vals, n_e_std_to_plot_vals, x_vals = process_variable_data(ds, probe, var_name = 'n_e')
    if (np.any(np.isnan(n_e_to_plot_vals))) and (hdf5_folder is not None):
        updated_ds = double_check_temp_data(hdf5_folder, ds, probe, x_vals=x_vals, y_vals=n_e_to_plot_vals,
                                            save_plots=save_plots, figure_folder=figure_folder)

        if updated_ds is not None:
            # Recompute every t_e-dependent variable (n_e, n_i, nu_ei, p_e, p_ei)
            # now that t_e has changed.  v_f/v_p/ion_isat/electron_isat are untouched.
            updated_ds = recalculate_derived_variables(updated_ds)

            base_name = os.path.splitext(os.path.basename(ds_save_path))[0]
            if base_name.endswith('_updated'):
                # Keep the exact same name, no extra tags
                updated_ds_save_path = os.path.join(updated_nc_folder, f"{base_name}.nc")
                print("Existing updated file detected. Overwriting with new modifications...")
            else:
                # It's a raw file being updated for the first time
                updated_ds_save_path = os.path.join(updated_nc_folder, f"{base_name}_updated.nc")
                print("Raw file detected. Creating new _updated.nc file...")

            updated_ds.to_netcdf(updated_ds_save_path, mode='w', engine='netcdf4')

            print(f'Successfully saved the dataset at: \n {updated_ds_save_path}.')
    n_e_name = ds['n_e'].attrs.get("long_name", 'n_e')
    ylabel = fr"$n_e$ [$\mathregular{{{ds['n_e'].attrs.get('units', 'n_e')}}}$]"
    xlabel = f'x [{ds.attrs.get("x_units")}]'
    if redo_grad_regions:
        print('redo n str: ', redo_grad_regions)
        ne_edges = obtain_gradient_regions(ds,x_vals,n_e_to_plot_vals,n_e_std_to_plot_vals,color = dataset_clor,
                                               marker=dataset_mark, ylabel = ylabel, xlabel = xlabel, regions_str = regions_str)

        ne_slopes = []
        ne_intercepts = []
        for start_x, stop_x in ne_edges:
            region_mask = (x_vals >= start_x) & (x_vals <= stop_x)
            x_region = x_vals[region_mask]
            ne_region = n_e_to_plot_vals[region_mask]
            valid_mask = ~np.isnan(x_region) & ~np.isnan(ne_region)
            valid_x = x_region[valid_mask]
            valid_ne = ne_region[valid_mask]
            ne_slope, ne_intercept = np.polyfit(valid_x, valid_ne, deg=1)
            ne_slopes.append(ne_slope)
            ne_intercepts.append(ne_intercept)
        if ds_save_path is not None:
            ds.attrs[regions_str] = json.dumps(ne_edges)
            ds.attrs[slopes_str] = ne_slopes
            ds.attrs[intercepts_str] = ne_intercepts
            ds.to_netcdf(ds_save_path, mode='a', engine='netcdf4')
            print(f'Updated the dataset at \n {ds_save_path} \n to include Density gradient locations.')
    elif not redo_grad_regions and (regions_str in ds.attrs.keys()) and gradient_regions:
        loaded_edges = json.loads(ds.attrs[regions_str])
        ne_edges = [tuple(edge) for edge in loaded_edges]
        ne_slopes = ds.attrs[slopes_str]
        ne_intercepts = ds.attrs[intercepts_str]
    else:
        ne_edges = []
        ne_slopes = []
        ne_intercepts = []

    if see_plots:
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        if normalize:
            core_x_idxs = np.where((x_vals <= 5) & (x_vals >= -5))[0]
            core_denses = n_e_to_plot_vals[core_x_idxs]
            core_dens = np.mean(core_denses)
            n_e_to_plot_vals = n_e_to_plot_vals/core_dens
            n_e_std_to_plot_vals = n_e_std_to_plot_vals/core_dens

        axes.errorbar(x_vals, n_e_to_plot_vals, yerr=n_e_std_to_plot_vals, fmt=dataset_mark, capsize=3,
                      color = dataset_clor)

        axes.set_ylabel(ylabel, rotation=0, labelpad=60)

        if gradient_regions:
            for i, (start_x, stop_x) in enumerate(ne_edges):
                # Draw the start line
                axes.axvline(x=start_x, color='black', linestyle='--', linewidth=2)
                # Draw the stop line
                axes.axvline(x=stop_x, color='black', linestyle='--', linewidth=2)
                if plot_final_fit:
                    region_mask = (x_vals >= start_x) & (x_vals <= stop_x)
                    axes.plot(x_vals[region_mask], ne_slopes[i] * x_vals[region_mask] + ne_intercepts[i],
                              color='gold', linestyle = '--')

        if not make_presentable and not sharex:
            if plot_axes:
                if plot_title:
                    axes.set_title(f"{n_e_name}")
                axes.set_xlabel(f'x [{ds.attrs.get("x_units")}]')
                probe_z = ds['z'].isel(probe=probe).item()
                ss_start = ds.attrs[f'steady state start probe {probe}']
                ss_end = ds.attrs[f'steady state end probe {probe}']
                axes.set_title(f"{run_identifier} \n "
                               f"{n_e_name} \n "
                               f"z: {probe_z:.2f} \n "
                               f"Between times ({ss_start:.2f},{ss_end:.2f}) ms")
        else:
            if plot_axes:
                if bottomx:
                    axes.ticklabel_format(style='sci', axis='y', scilimits=(-1, 2))
                    axes.set_xlabel(f'x [{ds.attrs.get("x_units")}]')
                    axes.tick_params(axis='x', labelbottom=True)
                else:
                    axes.set_xlabel("")
                    axes.tick_params(axis='x', labelbottom=False)

                # 2. FORCE THE MATH ENGINE (Bypass canvas.draw completely)
                ax_formatter = axes.yaxis.get_major_formatter()
                ax_formatter.set_locs(axes.yaxis.get_majorticklocs())
                n_e_offset = ax_formatter.get_offset()

                # 3. Hide the default floating text
                axes.yaxis.get_offset_text().set_visible(False)

                # 4. Build the dynamic label strings using f-strings
                if not sharex:
                    if n_e_offset:
                        # Because we turned on 'use_mathtext' globally, rho_offset is
                        # automatically formatted as beautiful LaTeX (e.g., $\times10^{4}$)
                        n_e_label_str = fr'{n_e_name} ({n_e_offset})'
                    else:
                        n_e_label_str = rf'{n_e_name}'

                else: n_e_label_str = fr'{n_e_offset}'


                axes.text(0.5, 0.05, n_e_label_str,
                                transform=axes.transAxes,
                                horizontalalignment='center',
                                verticalalignment='bottom',
                                color='k')
        # axes.legend(loc='best')


def create_temperature_radial_plots(ds, probe, run_identifier = '', see_plots = True, axes = None,
                                    gradient_regions = True, dataset_clor = None,
                                    dataset_mark = None, make_presentable = False,
                                    sharex = False, bottomx = True, redo_grad_regions = False,
                                    regions_str = None, slopes_str = None, intercepts_str = None, ds_save_path = None,
                                    plot_final_fit = False,plot_axes = True, plot_title = True, normalize = False,
                                    hdf5_folder = None, save_plots = False, figure_folder = None, updated_nc_folder = None):

    grad_region_str = 'te_grad_regions'
    if dataset_clor is None:
        dataset_clor = 'royalblue'
        dataset_mark = 'o'

    t_e_to_plot_vals, t_e_std_to_plot_vals, x_vals = process_variable_data(ds, probe, var_name='t_e')

    ylabel = fr"$T_e$ [{ds['t_e'].attrs.get('units', 't_e')}]"
    xlabel = f'x [{ds.attrs.get("x_units")}]'
    t_e_name = ds['t_e'].attrs.get("long_name", 't_e')
    if redo_grad_regions:
        print('redo t str: ', redo_grad_regions)
        te_edges = obtain_gradient_regions(ds, x_vals, t_e_to_plot_vals,t_e_std_to_plot_vals,color = dataset_clor,
                                           marker=dataset_mark, ylabel = ylabel, xlabel = xlabel, regions_str = regions_str)
        te_slopes = []
        te_intercepts = []
        for start_x, stop_x in te_edges:
            region_mask = (x_vals >= start_x) & (x_vals <= stop_x)
            x_region = x_vals[region_mask]
            te_region = t_e_to_plot_vals[region_mask]
            valid_mask = ~np.isnan(x_region) & ~np.isnan(te_region)
            valid_x = x_region[valid_mask]
            valid_te = te_region[valid_mask]
            te_slope, te_intercept = np.polyfit(valid_x, valid_te, deg=1)
            te_slopes.append(te_slope)
            te_intercepts.append(te_intercept)
        if ds_save_path is not None:
            ds.attrs[regions_str] = json.dumps(te_edges)
            ds.attrs[slopes_str] = te_slopes
            ds.attrs[intercepts_str] = te_intercepts
            ds.to_netcdf(ds_save_path, mode='a', engine='netcdf4')
            print(f'Updated the dataset at {ds_save_path} to include Temperature gradient locations.')
    elif not redo_grad_regions and (regions_str in ds.attrs.keys()) and gradient_regions:
        loaded_edges = json.loads(ds.attrs[regions_str])
        te_edges = [tuple(edge) for edge in loaded_edges]
        te_slopes = ds.attrs[slopes_str]
        te_intercepts = ds.attrs[intercepts_str]
    else:
        te_edges = []
        te_slopes = []
        te_intercepts = []

    if see_plots:
        if normalize:
            core_x_idxs = np.where((x_vals <= 5) & (x_vals >= -5))[0]
            core_temps = t_e_to_plot_vals[core_x_idxs]
            core_temp = np.mean(core_temps)
            t_e_to_plot_vals = t_e_to_plot_vals/core_temp
            t_e_std_to_plot_vals = t_e_std_to_plot_vals/core_temp
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        axes.errorbar(x_vals, t_e_to_plot_vals, yerr=t_e_std_to_plot_vals, fmt=dataset_mark, capsize=3,
                      color = dataset_clor)

        axes.set_ylabel(ylabel, rotation=0, labelpad=60)
        if gradient_regions:
            for i, (start_x, stop_x) in enumerate(te_edges):
                # Draw the start line
                axes.axvline(x=start_x, color='black', linestyle='--', linewidth=2)
                # Draw the stop line
                axes.axvline(x=stop_x, color='black', linestyle='--', linewidth=2)
                if plot_final_fit:
                    region_mask = (x_vals >= start_x) & (x_vals <= stop_x)
                    axes.plot(x_vals[region_mask], te_slopes[i] * x_vals[region_mask] + te_intercepts[i],
                              color='gold', linestyle = '--')

        if not make_presentable:
            if plot_axes:
                if plot_title:
                    axes.set_title(f"{t_e_name}")
                probe_z = ds['z'].isel(probe=probe).item()
                ss_start = ds.attrs[f'steady state start probe {probe}']
                ss_end = ds.attrs[f'steady state end probe {probe}']
                axes.set_title(f"{run_identifier} \n "
                               f"z: {probe_z:.2f} \n "
                               f"Between times ({ss_start:.2f},{ss_end:.2f}) ms")
                if not sharex:
                    axes.set_xlabel(xlabel)
        else:
            if plot_axes:
                axes.ticklabel_format(style='sci', axis='y', scilimits=(-1, 2))
                axes.tick_params(axis='x', labelbottom=False)

                # 2. FORCE THE MATH ENGINE (Bypass canvas.draw completely)
                ax_formatter = axes.yaxis.get_major_formatter()
                ax_formatter.set_locs(axes.yaxis.get_majorticklocs())
                t_e_offset = ax_formatter.get_offset()

                # 3. Hide the default floating text
                axes.yaxis.get_offset_text().set_visible(False)

                # 4. Build the dynamic label strings using f-strings
                if t_e_offset:
                    # Because we turned on 'use_mathtext' globally, rho_offset is
                    # automatically formatted as beautiful LaTeX (e.g., $\times10^{4}$)
                    t_e_label_str =  f'{t_e_offset}'
                else:
                    t_e_label_str = ''

                axes.text(0.5, 0.05, t_e_label_str,
                                transform=axes.transAxes,
                                horizontalalignment='center',
                                verticalalignment='bottom',
                                color='k')


def obtain_gradient_regions(ds, x,y, error, color = 'royalblue', marker = 'o', ylabel ='', xlabel = '', regions_str = ''):
    if regions_str not in ds.attrs.keys():
        edges = find_pedestals_strict_thresh(x, y)
    else:
        loaded_edges = json.loads(ds.attrs[regions_str])
        data_edges = [tuple(edge) for edge in loaded_edges]
        edges = {'left': [], 'right': []}
        for edge in data_edges:
            if edge[0]<0:
                edges['left'].append(edge)
            else:
                edges['right'].append(edge)

    fig_post_grad, axes, letters = build_subplots([[1]])
    ax_post_grad = axes[letters[0]]
    ax_post_grad.errorbar(x, y, yerr=error, fmt=marker, capsize=3,
                          color=color)
    ax_post_grad.set_ylabel(ylabel, rotation=90)
    ax_post_grad.set_xlabel(xlabel)


    for side in ['left', 'right']:
        # ne_edges[side] looks like [(-21.0, -16.0)] or []
        for start_x, stop_x in edges[side]:
            # Draw the start line
            ax_post_grad.axvline(x=start_x, color='red', linestyle='--', linewidth=2,
                                 label='Edge Boundary' if side == 'left' else "")
            # Draw the stop line
            ax_post_grad.axvline(x=stop_x, color='red', linestyle='--', linewidth=2)

    plt.tight_layout()
    fig_post_grad.show()
    plt.close(fig_post_grad)
    while ask_yes_or_no('Adjust x-positions of gradients? (y/n) '):
        print(f'Current Gradient edges: {edges["left"]}, {edges["right"]}')
        while True:
            chosen_edges_str = input(
                'Choose where you would like the gradient edges. \n '
                'Use the format (1,2),(3,4),(5,6)... ')
            # Allow the user to skip by pressing Enter
            if not chosen_edges_str.strip():
                print("No edges entered. Skipping.")
                chosen_edges = []
                break

            try:
                # Wrap the raw string in brackets.
                # "(1,2)" becomes "[(1,2)]"
                # "(1,2),(3,4)" becomes "[(1,2),(3,4)]"
                formatted_str = f"[{chosen_edges_str}]"

                # Safely evaluate the string into a literal Python list
                chosen_edges = ast.literal_eval(formatted_str)

                # Verify the structure is what we expect
                if all(isinstance(i, tuple) for i in chosen_edges):
                    print(f"Successfully parsed {len(chosen_edges)} edges:", chosen_edges)
                    break
                else:
                    print("Input parsed, but it didn't look like tuples. Try again.")

            except (ValueError, SyntaxError):
                print("Invalid format. Please make sure to use parentheses and commas, like (1,2),(3,4).")

        # chosen_edges is now a guaranteed list of tuples: [(1, 2), (3, 4)]
        if len(chosen_edges) >= 0:
            left_list = []
            right_list = []
            for chosen_edge in chosen_edges:
                if chosen_edge[0] < 0:
                    left_list.append(chosen_edge)
                else:
                    right_list.append(chosen_edge)
            edges['left'] = left_list
            edges['right'] = right_list

        fig, axes, letters = build_subplots([[1]])
        ax_subfigure = axes[letters[0]]
        ax_subfigure.errorbar(x, y, yerr=error, fmt=marker, capsize=3,
                              color=color)
        ax_subfigure.set_ylabel(ylabel, rotation=90)
        ax_subfigure.set_xlabel(xlabel)

        for side in ['left', 'right']:
            # ne_edges[side] looks like [(-21.0, -16.0)] or []
            for start_x, stop_x in edges[side]:
                # Draw the start line
                ax_subfigure.axvline(x=start_x, color='red', linestyle='--', linewidth=2,
                                     label='Edge Boundary' if side == 'left' else "")
                # Draw the stop line
                ax_subfigure.axvline(x=stop_x, color='red', linestyle='--', linewidth=2)

        plt.tight_layout()
        fig.show()
        plt.close(fig)
    edges_full = edges['left'] + edges['right']
    return edges_full

def find_pedestals_strict_thresh(x, y, slope_ratio=0.25, stop_ratio=0.05, max_slope_error=0.20,
                                 window=7, poly_order=3,
                                 edge_limit=15, max_center_bleed=0.98, gap_tolerance=2,
                                 min_width=0.0, min_height_ratio=0.15):

    '''
        Finds steep gradients, calculates opposing slopes independently to prevent merging,
        but returns them in a flat dictionary: {'left': [(x1, x2)...], 'right': [(x1, x2)...]}
        '''

    y_smooth = savgol_filter(y, window_length=window, polyorder=poly_order)
    dy_dx = np.gradient(y_smooth, x)

    results = {'left': [], 'right': []}

    for side in ['left', 'right']:
        mask = (x <= 0) if side == 'left' else (x > 0)
        if not np.any(mask):
            continue

        x_half = x[mask]
        dy_dx_half = dy_dx[mask]
        y_smooth_half = y_smooth[mask]

        # --- THE STRICT EDGE MASK ---
        strict_edge_mask = (x_half <= -edge_limit) if side == 'left' else (x_half >= edge_limit)

        if not np.any(strict_edge_mask):
            continue

        # --- DIRECTIONAL THRESHOLD ENFORCEMENT ---
        expected_sign = 1 if side == 'left' else -1
        directional_edge_slopes = dy_dx_half[strict_edge_mask] * expected_sign
        valid_edge_slopes = directional_edge_slopes[directional_edge_slopes > 0]

        if len(valid_edge_slopes) == 0:
            continue

        max_slope_edge = np.nanmax(valid_edge_slopes)
        dynamic_thresh = max_slope_edge * slope_ratio

        # The global stop threshold used exclusively for the edge-facing expansion
        global_edge_stop = max_slope_edge * stop_ratio

        # --- THE EVALUATION (SEED) ---
        if side == 'left':
            is_steep = (dy_dx_half > dynamic_thresh)
            direction = 'up'
        else:
            is_steep = (dy_dx_half < -dynamic_thresh)
            direction = 'down'

        # --- GAP CLOSING ---
        if gap_tolerance > 0:
            structure = np.ones(gap_tolerance + 1)
            is_steep = binary_closing(is_steep, structure=structure)

        # --- HYBRID EXPANSION ---
        def get_expanded_cliffs(steep_mask, dir_flag):
            padded = np.pad(steep_mask, (1, 1), mode='constant', constant_values=False)
            diffs = np.diff(padded.astype(int))
            starts = np.where(diffs == 1)[0]
            stops = np.where(diffs == -1)[0] - 1

            expanded = []
            for start, stop in zip(starts, stops):
                # Calculate the strict local threshold for the core-facing expansion
                chunk_slopes = dy_dx_half[start:stop + 1]
                if dir_flag == 'up':
                    local_peak = np.max(chunk_slopes)
                else:
                    local_peak = np.min(chunk_slopes)

                local_core_stop = abs(local_peak) * (1.0 - max_slope_error)

                # Assign thresholds asymmetrically based on the hemisphere
                if side == 'left':
                    thresh_start = global_edge_stop  # expanding left = towards edge
                    thresh_stop = local_core_stop  # expanding right = towards core
                else:
                    thresh_start = local_core_stop  # expanding left = towards core
                    thresh_stop = global_edge_stop  # expanding right = towards edge

                # Expand backwards (using thresh_start)
                while start > 0:
                    slope = dy_dx_half[start - 1]
                    if (dir_flag == 'up' and slope > thresh_start) or \
                            (dir_flag == 'down' and slope < -thresh_start):
                        start -= 1
                    else:
                        break

                # Expand forwards (using thresh_stop)
                while stop < len(dy_dx_half) - 1:
                    slope = dy_dx_half[stop + 1]
                    if (dir_flag == 'up' and slope > thresh_stop) or \
                            (dir_flag == 'down' and slope < -thresh_stop):
                        stop += 1
                    else:
                        break

                expanded.append({
                    'start_idx': start,
                    'stop_idx': stop,
                    'coords': (x_half[start], x_half[stop])
                })
            return expanded

        all_cliffs = get_expanded_cliffs(is_steep, direction)

        # --- MERGE OVERLAPPING INTERVALS ---
        if all_cliffs:
            all_cliffs.sort(key=lambda item: item['start_idx'])
            merged_cliffs = [all_cliffs[0]]
            for current in all_cliffs[1:]:
                prev = merged_cliffs[-1]
                if current['start_idx'] <= prev['stop_idx']:
                    new_stop = max(prev['stop_idx'], current['stop_idx'])
                    merged_cliffs[-1] = {
                        'start_idx': prev['start_idx'],
                        'stop_idx': new_stop,
                        'coords': (x_half[prev['start_idx']], x_half[new_stop])
                    }
                else:
                    merged_cliffs.append(current)
            all_cliffs = merged_cliffs

        # --- MACROSCOPIC FILTERING ---
        hemisphere_y_range = np.nanmax(y_smooth_half) - np.nanmin(y_smooth_half)

        valid_regions = []
        for cliff in all_cliffs:
            start_x, stop_x = cliff['coords']
            if start_x == stop_x:
                continue

            total_length = stop_x - start_x
            if total_length < min_width:
                continue

            center_start = max(start_x, -edge_limit)
            center_stop = min(stop_x, edge_limit)
            center_length = max(0.0, center_stop - center_start)
            bleed_ratio = center_length / total_length

            if bleed_ratio >= max_center_bleed:
                continue

            cliff_height = abs(y_smooth_half[cliff['stop_idx']] - y_smooth_half[cliff['start_idx']])
            if cliff_height < (hemisphere_y_range * min_height_ratio):
                continue

            valid_regions.append({
                'coords': (start_x, stop_x),
                'height': cliff_height
            })

        valid_regions.sort(key=lambda item: item['height'], reverse=True)
        results[side] = [region['coords'] for region in valid_regions]

    return results


def double_check_temp_data(hdf5_folder, ds, probe, x_vals_to_check = None, x_vals = None, y_vals = None,
                           save_plots = False, figure_folder = None):
    min_time = ds.attrs[f'steady state start probe {probe}']
    max_time = ds.attrs[f'steady state end probe {probe}']
    run_check = False
    # ---------Get x's to search---------
    if x_vals is not None and y_vals is not None:
        x_vals_with_nans = x_vals[np.isnan(y_vals)]
        print('\nFound NaNs in x_vals at ', x_vals_with_nans)
        run_check = ask_yes_or_no('\nDouble check temperature region manually? (y/n) ')
        if run_check:
            use_nan_xs = ask_yes_or_no("\n Use ONLY and ALL x's defined as NaN above to check? (y/n) ")
            if use_nan_xs:
                x_vals_to_check = x_vals_with_nans
            else:
                print(f"\n Minimum x value: {np.min(x_vals)} \n Maximum x value: {np.max(x_vals)}")
                x_prompt = "\nEnter integer x-values to check"
                x_vals_to_check = allow_only_ints(x_prompt, min_condition=np.min(x_vals), max_condition=np.max(x_vals))
    elif x_vals_to_check is not None:
        run_check = True

    if not run_check:
        return None

    # ---------Get times to search---------
    print(f'\nSteady State start time is {min_time} \nSteady State end time is {max_time}')
    ss_time_check = ask_yes_or_no('Use steady state times as the upper and lower check bounds? (y/n) ')
    if ss_time_check:
        start_time_int = round(min_time)
        end_time_int = round(max_time)
        time_to_check = list(range(start_time_int, end_time_int + 1))
        print('time_to_check = ', time_to_check)
    else:
        min_time = ds.time.min().item()
        max_time = ds.time.max().item()
        min_time_int = round(min_time)
        max_time_int = round(max_time)
        print(f'Minimum time value: {min_time_int} \n Maximum time value: {max_time_int} ')
        time_prompt = "\nEnter integer time values to check"
        time_to_check = allow_only_ints(time_prompt, min_time_int, max_time_int)

    # ---------Find Corresponding HDF5 file---------
    exp_name = ds.attrs.get('Exp name')
    run_num = ds.attrs.get('Run number')
    hdf5_files = [f for f in os.listdir(hdf5_folder) if f.endswith('.hdf5')]
    matched_file = get_hdf5_filename(exp_name, run_num, hdf5_files)
    print('Matched_file: ', matched_file)
    hdf5_pathname = os.path.join(hdf5_folder, matched_file)

    # -----------------------------------------------------------------------------
    # Which physical port does the requested `probe` index map to in the dataset?
    # We only need to load the one probe we are checking.
    # -----------------------------------------------------------------------------
    target_port = int(ds['port'].sel(probe=probe).item())

    # ---------Load raw bias/current for the matched file (this one probe)---------
    with lapd.File(hdf5_pathname) as hdf5_file:

        params_dict = metadata_dict(hdf5_file.info['run description'],
                                    hdf5_file.info['exp name'],
                                    hdf5_file.info['file'])

        # Same probe-set convention as your conversion pipeline
        if ('jan' in params_dict['Exp name'].lower()
                and '24' in params_dict['Exp name'].lower()):
            valid_probes = [0, 2]
        else:
            valid_probes = [0, 1]

        (exp_params_dict, vsweep_bc, langmuir_configs, config_id,
         voltage_gain, orientation, current_bc) = n_IV_parameters(hdf5_file, hdf5_pathname)

        # Bias sweep is shared across probes
        bias, dt = n_get_sweep_voltage(hdf5_file, vsweep_bc, voltage_gain)
        ramp_bounds = isolate_ramps(bias)  # (num_ramps, 2): [start, end] frame indices
        ramp_times = ramp_bounds[:, 1] * dt.to(u.ms)  # time of each ramp (ms), same order as sweeps

        # Pick the config index whose port matches the requested probe
        probe_num = None
        for k in valid_probes:
            if langmuir_configs[k]['port'] == target_port:
                probe_num = k
                break
        if probe_num is None:
            raise ValueError(f"No langmuir config with port {target_port} "
                             f"(probe {probe}) in {matched_file}")

        probe_bias = bias.copy()
        probe_current, motor_data = n_get_sweep_current(hdf5_file,
                                                        langmuir_configs[probe_num],
                                                        orientation)

    # ---------Reshape to (position, shot, frame), exactly as the conversion step---------
    (probe_position_array, num_positions,
     shots_per_position, selected_shots) = get_shot_positions(motor_data)

    if probe_bias.ndim == 2:  # certain shots already selected in bias
        probe_bias = probe_bias[selected_shots, ...]
    probe_current = probe_current[selected_shots, ...]

    probe_bias = probe_bias.reshape(num_positions, shots_per_position, -1)
    probe_current = probe_current.reshape(num_positions, shots_per_position, -1)
    # probe_bias / probe_current dims:  (position, shot, frame)

    # Ramp count should line up with the sweep dimension of ds
    n_ramps = ramp_bounds.shape[0]
    if n_ramps != ds.sizes['sweep']:
        print(f"Warning: {n_ramps} ramps found in HDF5 but ds has "
              f"{ds.sizes['sweep']} sweeps; mapping time -> nearest ramp by time.")

    # Coordinate value arrays (labels, not positions) for clean .loc write-back
    x_coord_vals = ds['x'].values
    sweep_coord_vals = ds['sweep'].values
    ds_time_vals = ds['time'].values
    y_val = float(ds['y'].values[0])  # single y row for this probe

    # ---------------------------------------------------------------------------------
    #  Figure output directories (built once).  All manual-fit plots live under one
    #  run directory so the per-sweep plots sit inside it:
    #      <figure_folder>/manual_temp_fits/<expname>_<runnumber>/                 <- 8-pack overviews
    #      <figure_folder>/manual_temp_fits/<expname>_<runnumber>/individual_sweeps/  <- per-shot fits
    #  ensure_directory requires an ABSOLUTE path ending in a separator.
    # ---------------------------------------------------------------------------------
    if save_plots:
        if not figure_folder:
            raise ValueError("save_plots=True but figure_folder is empty/None; "
                             "pass a figure_folder so plots have somewhere to go.")
        run_dir = ensure_directory(
            os.path.join(figure_folder, "manual_temp_fits", f"{exp_name}_{run_num}") + os.sep)
        sweep_dir = ensure_directory(os.path.join(run_dir, "individual_sweeps") + os.sep)
        print(f"[save_plots=True] overviews -> {run_dir}")
        print(f"[save_plots=True] sweeps    -> {sweep_dir}")
    else:
        print("[save_plots=False] no plots will be written to disk this run.")

    # ---------------------------------------------------------------------------------
    #  Attribute logs.  Stored in ds.attrs as "; "-joined STRINGS so the dataset can be
    #  written to NetCDF (NetCDF cannot serialize a Python list of strings). On reload,
    #  split back into a list so we can keep appending.
    # ---------------------------------------------------------------------------------
    def _attr_to_list(val):
        if not val:
            return []
        if isinstance(val, str):
            return [s for s in val.split("; ") if s]
        return list(val)

    fit_log = _attr_to_list(ds.attrs.get('manually_fit_t_e'))
    bad_log = _attr_to_list(ds.attrs.get('potential_bad_sweeps'))

    # ---------------------------------------------------------------------------------
    #  Small local input helper: read a float (used for voltage edges)
    # ---------------------------------------------------------------------------------
    def _ask_float(prompt):
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("Invalid input. Please enter a number.")

    # ---------------------------------------------------------------------------------
    #  Safe figure teardown.  The TkAgg backend on macOS raises TclError if a figure
    #  window was already destroyed (e.g. closed by hand, or a prior close's delayed
    #  destroy already fired).  Swallow that so the interactive loop never crashes.
    # ---------------------------------------------------------------------------------
    def _safe_close():
        try:
            plt.close('all')
        except Exception:
            pass

    # ---------------------------------------------------------------------------------
    #  Electron current for a single ramp.  get_ion_current / get_floating_potential
    #  expect the bias & current SORTED by ascending bias; our ramp arrays are in frame
    #  order, so we sort first.  get_ion_current returns the ion-saturation current
    #  evaluated at every sorted-bias point, so I_electron = I_total - I_ion elementwise.
    #  (Assumes both functions accept plain numpy arrays; strip astropy units first.)
    # ---------------------------------------------------------------------------------
    def _electron_current(V_raw, I_raw):
        # Keep astropy units ON: get_floating_potential / get_ion_current index the
        # arrays as Quantities (they call .value internally).  Sort by ascending bias
        # first, then strip units only on the final numeric arrays we plot/fit.
        order = np.argsort(np.asarray(getattr(V_raw, 'value', V_raw)))
        V_sorted = V_raw[order]  # Quantity (or ndarray) preserved
        I_sorted = I_raw[order]

        # get_floating_potential returns (v_f_bias, current_at_v_f, index).
        # We only need the bias.  It can be None (empty / no clean zero-crossing).
        v_f_bias, _, _ = get_floating_potential(V_sorted, I_sorted)

        V_num = np.asarray(getattr(V_sorted, 'value', V_sorted), dtype=float)
        I_num = np.asarray(getattr(I_sorted, 'value', I_sorted), dtype=float)

        # ---- Hard reject: junk electron branch ----------------------------------
        # If a significant fraction of the HIGH-bias points still have non-positive
        # current, the electron branch never developed -> throw the sweep out no matter
        # what V_f was reported.  (Above ~30 V the electron current should be strongly
        # positive; lots of I<=0 there means the sweep is unusable.)
        HIGH_V = 30.0  # bias threshold (V) beyond which current should be positive
        FRAC_BAD = 0.3  # fraction of high-bias points allowed to be non-positive
        high_mask = V_num > HIGH_V
        if np.count_nonzero(high_mask) >= 3:
            frac_neg = float(np.mean(I_num[high_mask] <= 0))
            if frac_neg >= FRAC_BAD:
                print(f"    Sweep rejected: {100 * frac_neg:.0f}% of points above "
                      f"{HIGH_V:.0f} V have I<=0 (electron branch never develops).")
                return V_num, None, float('nan')

                # ---- Choose V_f purely as a noise cutoff --------------------------------
                # V_f here serves ONLY to trim the noisy low-bias part of the sweep.
                # We require any candidate to be STRICTLY POSITIVE (> 0) and BELOW HIGH_V.
                candidates = []

                # (a) reported V_f from get_floating_potential
                if v_f_bias is not None:
                    v_f_rep = float(getattr(v_f_bias, 'value', v_f_bias))
                    if 0 < v_f_rep < HIGH_V:
                        candidates.append(v_f_rep)

                # (b) independent ion->electron zero-crossing of the smoothed total current
                if I_num.size >= 5:
                    I_s = np.convolve(I_num, np.ones(5) / 5.0, mode='same')
                else:
                    I_s = I_num
                crossings = np.where(np.diff(np.sign(I_s)) > 0)[0]  # negative -> positive
                if crossings.size > 0:
                    i0 = int(crossings[0])
                    v_cross = 0.5 * (V_num[i0] + V_num[i0 + 1])
                    if 0 < v_cross < HIGH_V:
                        candidates.append(v_cross)

                if not candidates:
                    print(f"    No valid positive floating potential below {HIGH_V:.0f} V; "
                          f"flagging sweep bad.")
                    return V_num, None, float('nan')

                # Lowest valid positive estimate -> cuts the least data while still removing the noise.
                v_f_low = min(candidates)
                print(f"    V_f (noise cutoff) = {v_f_low:.2f} V  (from {len(candidates)} "
                      f"valid positive candidate(s): {', '.join(f'{c:.2f}' for c in candidates)})")

                unit = getattr(V_sorted, 'unit', None)
                v_f_bias = (v_f_low * unit) if unit is not None else v_f_low
        if v_f_bias is None:
            # No floating potential -> sweep is not worth fitting; signal caller to
            # flag it bad (current returned as None).
            return V_num, None, float('nan')

        ion = get_ion_current(V_sorted, I_sorted, v_f_bias)
        ion_num = np.asarray(getattr(ion, 'value', ion), dtype=float)
        v_f_num = float(getattr(v_f_bias, 'value', v_f_bias))
        return V_num, I_num - ion_num, v_f_num

    # ---------------------------------------------------------------------------------
    #  Overlay line for an already-stored T_e.  We know the slope (1/te) but not where
    #  the original fit sat, so anchor the intercept by least squares over the electron
    #  branch (points above V_f with I>0).  Returns (Vline, Iline) or None.
    # ---------------------------------------------------------------------------------
    def _existing_fit_line(V, I, te, v_f, vlo, vhi):
        if I is None or not (np.isfinite(te) and te > 0):
            return None
        slope = 1.0 / te
        # Anchor the intercept over the electron-RETARDING region only (roughly V_f up
        # to a few Te above it).  Past that the branch saturates / rolls over, and
        # including those points drags the offset so the line sits below the data.
        # Use the median offset (robust to the couple of outliers near the knee).
        mask = I > 0
        if np.isfinite(v_f):
            mask = mask & (V >= v_f) & (V <= v_f + 4.0 * te)
        if np.count_nonzero(mask) < 2:  # fall back: any pts above V_f
            mask = I > 0
            if np.isfinite(v_f):
                mask = mask & (V >= v_f)
        if np.count_nonzero(mask) < 2:
            return None
        intercept = np.median(np.log(I[mask]) - slope * V[mask])
        # Only draw the reference line across the region it was anchored to.
        vhi_line = min(vhi, v_f + 4.0 * te) if np.isfinite(v_f) else vhi
        Vline = np.linspace(vlo, vhi_line, 100)
        return Vline, np.exp(slope * Vline + intercept)

    # ---------------------------------------------------------------------------------
    #  Keep every pop-up window pinned to the same upper-screen spot so they don't
    #  march down over the terminal.  Backend-dependent; silently ignore if unsupported.
    # ---------------------------------------------------------------------------------
    def _place(fig):
        try:
            win = fig.canvas.manager.window
            if hasattr(win, 'wm_geometry'):  # TkAgg
                win.wm_geometry("+60+40")
            elif hasattr(win, 'move'):  # Qt5/6
                win.move(60, 40)
        except Exception:
            pass

    # =================================================================================
    #  MAIN LOOP:  for every requested (x, time) show all shots, let user investigate
    # =================================================================================
    for x in x_vals_to_check:
        # x value -> position row index in probe_bias
        x = float(x)
        pos_matches = np.nonzero(np.isclose(probe_position_array[:, 0], x, atol=0.5))[0]
        if len(pos_matches) == 0:
            print(f"\nx = {x} not found in probe positions; skipping.")
            continue
        pos_idx = int(pos_matches[0])

        for time in time_to_check:
            # requested integer time -> nearest sweep index in ds (and same index into ramp_bounds)
            sweep_idx = int(np.argmin(np.abs(ds_time_vals - time)))
            if n_ramps != ds.sizes['sweep']:
                # fall back: match against actual ramp times if counts disagree
                ramp_idx = int(np.argmin(np.abs(ramp_times.value - time)))
            else:
                ramp_idx = sweep_idx
            ramp_slice = slice(int(ramp_bounds[ramp_idx, 0]), int(ramp_bounds[ramp_idx, 1]))

            x_label = x_coord_vals[np.argmin(np.abs(x_coord_vals - x))]
            sweep_label = int(sweep_coord_vals[sweep_idx])
            actual_time = float(ds_time_vals[sweep_idx])

            # -----------------------------------------------------------------------
            #  STAGE 1: overview grid of ln(I) vs V for ALL shots at this (x, time)
            # -----------------------------------------------------------------------
            ncols = min(4, shots_per_position)
            nrows = int(np.ceil(shots_per_position / ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                    squeeze=False, dpi=100)
            axs = axs.ravel()
            for shot in range(shots_per_position):
                Vs, Ie, vf_o = _electron_current(probe_bias[pos_idx, shot, ramp_slice],
                                                 probe_current[pos_idx, shot, ramp_slice])
                ax = axs[shot]
                if Ie is None:  # no floating potential -> unusable
                    ax.text(0.5, 0.5, "no V_f\n(bad sweep)", ha='center', va='center',
                            transform=ax.transAxes, fontsize=11, color='red')
                    ax.set_title(f"shot {shot}", fontsize=11)
                    ax.set_xticks([]);
                    ax.set_yticks([])
                    continue
                pos_mask = Ie > 0
                ax.semilogy(Vs[pos_mask], Ie[pos_mask], '.', ms=3)
                # Overlay the existing stored T_e fit (if present) on this shot.
                shot_label_o = int(ds['shot'].values[shot])
                sel_o = dict(probe=probe, x=float(x_label), y=y_val,
                             shot=shot_label_o, sweep=sweep_label)
                te_o = float(ds['t_e'].loc[sel_o].item())
                vlo_o = vf_o if np.isfinite(vf_o) else float(Vs.min())
                el_o = _existing_fit_line(Vs, Ie, te_o, vf_o, vlo_o, float(Vs.max()))
                if el_o is not None:
                    ax.semilogy(el_o[0], el_o[1], 'm--', lw=1.5,
                                label=f'T_e={te_o:.1f} eV')
                    ax.legend(loc='best', fontsize=8)
                ax.set_title(f"shot {shot}", fontsize=11)
                ax.set_xlabel("bias V");
                ax.set_ylabel("ln(I)")
            for j in range(shots_per_position, len(axs)):
                axs[j].axis('off')
            fig.suptitle(f"probe {probe} (port {target_port})  |  x = {x_label}  |  "
                         f"time = {actual_time:.2f} ms (sweep {sweep_label})", fontsize=13)
            fig.tight_layout()

            # ----- Optionally save the 8-pack overview figure (probe first for sorting) -----
            # <figure_folder>/manual_temp_fits/<expname>_<runnumber>/
            #     probe_{}_x_{}_shot_all_time_{}.png   ('all' -> the 8-pack holds every shot)
            if save_plots:
                overview_name = f"probe_{probe}_x_{x_label}_shot_all_time_{time}.png"
                overview_path = os.path.join(run_dir, overview_name)
                fig.savefig(overview_path, bbox_inches="tight", dpi=150)
                print('Plot saved to ' + overview_path)
            else:
                print("    [save_plots=False] overview figure not saved.")

            _place(fig)
            try:
                plt.show(block=False)
                plt.pause(0.5)
            except Exception:
                pass

            if not ask_yes_or_no(f"\nInvestigate any shots at x={x_label}, "
                                 f"time={actual_time:.2f} ms? (y/n) "):
                _safe_close()
                continue

            shot_prompt = (f"\nEnter shot indices to investigate "
                           f"(0 to {shots_per_position - 1})")
            shots_to_check = allow_only_ints(shot_prompt, 0, shots_per_position - 1)
            _safe_close()

            # -----------------------------------------------------------------------
            #  STAGE 2: per selected shot -> left edge, zoom, right edge, fit, confirm
            # -----------------------------------------------------------------------
            for shot in shots_to_check:
                shot = int(shot)
                V_full, I_full, v_f_val = _electron_current(
                    probe_bias[pos_idx, shot, ramp_slice],
                    probe_current[pos_idx, shot, ramp_slice])
                shot_label = int(ds['shot'].values[shot])

                coord_str = (f"probe={probe}, x={x_label}, y={y_val}, "
                             f"shot={shot_label}, sweep={sweep_label}")

                # Save helper for this shot's plots (probe first so files sort nicely).
                # kind is a suffix: 'fit' | 'passed' | 'compare' | 'noVf'.
                def _save_sweep(fig, kind):
                    if not save_plots:
                        print(f"    [save_plots=False] NOT saving {kind} plot for "
                              f"shot {shot_label}.")
                        return
                    name = (f"probe_{probe}_x_{x_label}_shot_{shot_label}"
                            f"_time_{time}_{kind}.png")
                    out_path = os.path.join(sweep_dir, name)
                    fig.savefig(out_path, bbox_inches="tight", dpi=150)
                    print('Plot saved to ' + out_path)

                # No floating potential -> not worth fitting; flag bad and move on.
                # Still save the raw ln(I) vs V so the flagged sweep is documented.
                if I_full is None:
                    print(f"No floating potential for shot {shot}; "
                          f"auto-flagging {coord_str} as a bad sweep.")
                    if coord_str not in bad_log:
                        bad_log.append(coord_str)
                    ds.attrs['potential_bad_sweeps'] = "; ".join(bad_log)
                    if save_plots:
                        Vr = probe_bias[pos_idx, shot, ramp_slice]
                        Ir = probe_current[pos_idx, shot, ramp_slice]
                        Vr = np.asarray(getattr(Vr, 'value', Vr), dtype=float)
                        Ir = np.asarray(getattr(Ir, 'value', Ir), dtype=float)
                        o = np.argsort(Vr);
                        Vr, Ir = Vr[o], Ir[o]
                        fig0, ax0 = plt.subplots(figsize=(14, 6), dpi=100)
                        mm = Ir > 0
                        ax0.semilogy(Vr[mm], Ir[mm], '.', ms=4)
                        ax0.locator_params(axis='x', nbins=10)
                        ax0.set_xlabel("bias V");
                        ax0.set_ylabel("ln(I) [raw total]")
                        ax0.set_title(f"{coord_str}\n(t={actual_time:.2f} ms) "
                                      f"- no V_f, flagged bad", fontsize=11)
                        fig0.tight_layout()
                        _save_sweep(fig0, 'noVf')
                        _safe_close()
                    continue

                print(f"\nshot {shot}: floating potential ~ {v_f_val:.2f} V "
                      f"(pick LEFT edge above this)")

                # Existing stored T_e (if any) so we can overlay it on the sweep.
                sel = dict(probe=probe, x=float(x_label), y=y_val,
                           shot=shot_label, sweep=sweep_label)
                existing_te = float(ds['t_e'].loc[sel].item())
                existing_line = None
                has_existing = np.isfinite(existing_te) and existing_te > 0
                skip_fit = False  # user is happy with the existing fit as-is
                if has_existing:
                    vlo0 = v_f_val if np.isfinite(v_f_val) else float(V_full.min())
                    el = _existing_fit_line(V_full, I_full, existing_te, v_f_val,
                                            vlo0, float(V_full.max()))
                    if el is not None:
                        existing_line = (el[0], el[1], existing_te)
                        print(f"  (existing T_e = {existing_te:.3f} eV shown dashed)")

                def _draw(xlim=None, left=None, right=None, fit=None, te=None,
                          existing=None, v_f_line=None, show=True, ylim=None):
                    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
                    m = I_full > 0
                    ax.semilogy(V_full[m], I_full[m], '.', ms=4, label='ln(I) data')
                    if v_f_line is not None and np.isfinite(v_f_line):
                        ax.axvline(v_f_line, color='k', ls=':', lw=1.5, label='V_f')
                    if left is not None:
                        ax.axvline(left, color='r', lw=2, label='left edge')
                    if right is not None:
                        ax.axvline(right, color='g', lw=2, label='right edge')
                    if existing is not None:
                        Ve, Ie_line, te_old = existing
                        ax.semilogy(Ve, Ie_line, 'm--', lw=2,
                                    label=f'existing T_e = {te_old:.2f} eV')
                    if fit is not None:
                        Vfit, Ifit = fit
                        lbl = 'linear fit' if te is None else f'new fit: T_e = {te:.2f} eV'
                        ax.semilogy(Vfit, Ifit, 'k-', lw=2, label=lbl)
                    if xlim is not None:
                        ax.set_xlim(xlim)
                    # Y-limits: an explicit `ylim` wins (used by the final-fit view so
                    # the sloped line fills the plot); otherwise derive from the positive
                    # data actually in view so a few low outliers don't squash the
                    # electron branch into the top decade.
                    if ylim is not None:
                        ax.set_ylim(ylim)
                    else:
                        if xlim is not None:
                            vis = m & (V_full >= xlim[0]) & (V_full <= xlim[1])
                        else:
                            vis = m
                        if np.any(vis):
                            lo = np.percentile(I_full[vis], 2)
                            hi = I_full[vis].max()
                            if lo > 0 and hi > lo:
                                ax.set_ylim(lo / 3.0, hi * 3.0)
                    # Denser x tick labels make edge selection easier to read
                    ax.locator_params(axis='x', nbins=10)
                    ax.set_xlabel("bias V");
                    ax.set_ylabel("ln(I)")
                    ax.set_title(f"{coord_str}\n(t={actual_time:.2f} ms)", fontsize=11)
                    ax.legend(loc='best', fontsize=11)
                    fig.tight_layout()
                    if show:
                        _place(fig)
                        try:
                            plt.show(block=False)
                            plt.pause(0.4)
                        except Exception:
                            pass
                    return fig, ax

                passed = False  # user gave up on this sweep -> flag as bad
                t_e_value = None

                # If a T_e is already stored, show the current fit and ask whether it
                # is good enough to keep (skip re-fitting) before touching the edges.
                if has_existing:
                    _draw(xlim=((v_f_val - 3.0, float(V_full.max()))
                                if np.isfinite(v_f_val) else None),
                          existing=existing_line,
                          v_f_line=(v_f_val if np.isfinite(v_f_val) else None))
                    keep = ask_yes_or_no(
                        f"An existing T_e = {existing_te:.3f} eV is stored here. "
                        f"Happy with the current fit (y = keep, n = re-fit)? ")
                    _safe_close()
                    if keep:
                        skip_fit = True

                # View for LEFT-edge selection: start ~3 V BELOW the floating potential
                # (so a little of the transition is visible) up to the sweep max, with a
                # dotted black line marking V_f itself.
                left_view = ((v_f_val - 3.0, float(V_full.max()))
                             if np.isfinite(v_f_val) else None)
                vf_marker = v_f_val if np.isfinite(v_f_val) else None

                # ---- LEFT EDGE (with confirmation) ----
                left_edge = None
                while not skip_fit:
                    _draw(xlim=left_view, left=left_edge, existing=existing_line,
                          v_f_line=vf_marker)
                    ui = input("\nEnter LEFT edge (bias V), or 'p' to pass this sweep: ").strip().lower()
                    _safe_close()
                    if ui == 'p':
                        passed = True
                        break
                    try:
                        left_edge = float(ui)
                    except ValueError:
                        print("Invalid input.");
                        continue
                    _draw(xlim=left_view, left=left_edge, existing=existing_line,
                          v_f_line=vf_marker)
                    happy = ask_yes_or_no("Happy with this left edge (y/n)? ")
                    _safe_close()
                    if happy:
                        break

                if not passed and not skip_fit:
                    # ---- RIGHT EDGE (narrow window past the left edge) ----
                    right_edge = None
                    # Show only ~15 V past the left edge (clipped to the sweep max) so the
                    # exponential region is spread out and the right edge is easy to pick.
                    RIGHT_VIEW_SPAN = 15.0
                    zoom = (left_edge - 2.0,
                            min(float(V_full.max()), left_edge + RIGHT_VIEW_SPAN))
                    while True:
                        _draw(xlim=zoom, left=left_edge, right=right_edge,
                              existing=existing_line, v_f_line=vf_marker)
                        ui = input(f"\nEnter RIGHT edge (bias V, > {left_edge}, "
                                   f"up to {float(V_full.max()):.1f}), or 'p' to pass: ").strip().lower()
                        _safe_close()
                        if ui == 'p':
                            passed = True
                            break
                        try:
                            right_edge = float(ui)
                        except ValueError:
                            print("Invalid input.");
                            continue
                        if right_edge <= left_edge:
                            print("Right edge must be greater than left edge.")
                            right_edge = None
                            continue
                        _draw(xlim=zoom, left=left_edge, right=right_edge,
                              existing=existing_line, v_f_line=vf_marker)
                        if ask_yes_or_no("Happy with this right edge (y/n)? "):
                            _safe_close()
                            break
                        _safe_close()

                if not passed and not skip_fit:
                    # ---- FIT ln(I) vs V between the edges;  T_e[eV] = 1/slope ----
                    while True:
                        window = (V_full >= left_edge) & (V_full <= right_edge) & (I_full > 0)
                        n_pts = int(np.count_nonzero(window))
                        if n_pts < 2:
                            print(f"Only {n_pts} positive-current points in window; "
                                  "cannot fit. Passing this sweep.")
                            passed = True
                            break
                        Vw = V_full[window]
                        lnI = np.log(I_full[window])
                        slope, intercept = np.polyfit(Vw, lnI, 1)
                        t_e_fit = 1.0 / slope if slope > 0 else np.nan

                        Vf = np.linspace(left_edge, right_edge, 100)
                        Ifit = np.exp(slope * Vf + intercept)
                        # Zoom tight on the fit window so you can judge how well the
                        # line tracks the data (small margin on each side for context).
                        win_w = right_edge - left_edge
                        fit_margin = max(0.5, 0.15 * win_w)
                        fit_view = (left_edge - fit_margin, right_edge + fit_margin)
                        # Y-limits set from the fitted line's own span so it fills the
                        # plot (>=50% of the axis height) instead of looking nearly flat.
                        fit_lo, fit_hi = float(Ifit.min()), float(Ifit.max())
                        if fit_hi > fit_lo > 0:
                            fit_ylim = (fit_lo / 3.0, fit_hi * 3.0)
                        else:
                            fit_ylim = None
                        fit_fig, _ = _draw(xlim=fit_view, left=left_edge, right=right_edge,
                                           fit=(Vf, Ifit), te=t_e_fit,
                                           existing=existing_line, v_f_line=vf_marker,
                                           ylim=fit_ylim)
                        print(f"\nslope = {slope:.4f} 1/V   ->   T_e = {t_e_fit:.3f} eV   "
                              f"(fit on {n_pts} points)")
                        ans = input("Happy with this fit? (y/n), or 'p' to pass this sweep: ").strip().lower()
                        if ans == 'p':
                            _safe_close()
                            passed = True
                            break
                        if ans.startswith('y'):
                            if not np.isfinite(t_e_fit):
                                print("Fit slope <= 0, T_e not physical. Re-selecting edges.")
                                _safe_close()
                            else:
                                # ----- Save the accepted per-sweep fit -----
                                # Save the FULL (unzoomed) view -- essentially the 8-pack
                                # tile with both fits overlaid -- not the zoomed display.
                                save_fig, _ = _draw(left=left_edge, right=right_edge,
                                                    fit=(Vf, Ifit), te=t_e_fit,
                                                    existing=existing_line,
                                                    v_f_line=vf_marker, show=False)
                                _save_sweep(save_fig, 'fit')
                                _safe_close()

                                # ----- Secondary check: a T_e already exists here -----
                                # Show old vs new fit overlaid and confirm the replacement.
                                if existing_line is not None:
                                    # Interactive compare (zoomed) so old vs new is clear.
                                    _draw(xlim=fit_view, left=left_edge,
                                          right=right_edge, fit=(Vf, Ifit),
                                          te=t_e_fit, existing=existing_line,
                                          v_f_line=vf_marker)
                                    print(f"  existing T_e = {existing_te:.3f} eV  vs  "
                                          f"new T_e = {t_e_fit:.3f} eV")
                                    replace = ask_yes_or_no(
                                        "Replace the existing fit with this new one (y/n)? ")
                                    _safe_close()
                                    # Save the FULL (unzoomed) compare view with both fits.
                                    cmp_fig, _ = _draw(left=left_edge, right=right_edge,
                                                       fit=(Vf, Ifit), te=t_e_fit,
                                                       existing=existing_line,
                                                       v_f_line=vf_marker, show=False)
                                    _save_sweep(cmp_fig, 'compare')
                                    _safe_close()
                                    if replace:
                                        t_e_value = float(t_e_fit)
                                    else:
                                        print("  Keeping existing T_e; new fit discarded.")
                                        t_e_value = None
                                    break
                                else:
                                    t_e_value = float(t_e_fit)
                                    break
                        else:
                            _safe_close()
                        # 'n' -> re-pick edges for this shot
                        left_edge = _ask_float("Re-enter LEFT edge (bias V): ")
                        right_edge = _ask_float("Re-enter RIGHT edge (bias V): ")

                # ---- WRITE-BACK / LOGGING ----
                # (sel was defined above, right after the no-V_f guard.)
                if skip_fit:
                    # User was happy with the already-stored fit -> leave it untouched.
                    print(f"Kept existing T_e = {existing_te:.3f} eV at {coord_str}.")
                elif passed:
                    # User gave up on this sweep -> flag bad and save the ln(I) vs V it saw.
                    # If a (bad) value was stored here, overwrite it with NaN so the
                    # unsatisfactory fit does not linger in the dataset.
                    print(f"Passed -> flagging {coord_str} as a potential bad sweep.")
                    if coord_str not in bad_log:
                        bad_log.append(coord_str)
                    if has_existing:
                        prev = float(ds['t_e'].loc[sel].item())
                        ds['t_e'].loc[sel] = np.nan
                        print(f"  Replaced stored T_e ({prev:.3f} eV) with NaN.")
                    passed_fig, _ = _draw(existing=existing_line, v_f_line=vf_marker,
                                          show=False)
                    _save_sweep(passed_fig, 'passed')
                    _safe_close()
                elif t_e_value is None:
                    # Fit was made but the user chose to keep the existing value.
                    print(f"No change written at {coord_str} (kept existing T_e).")
                else:
                    # A new value was accepted -> write it.  (Guard against None just in
                    # case an upstream branch was edited; never format None.)
                    if t_e_value is None:
                        print(f"No T_e value to write at {coord_str}; skipping.")
                    else:
                        prev = float(ds['t_e'].loc[sel].item())
                        ds['t_e'].loc[sel] = t_e_value
                        print(f"Wrote T_e = {t_e_value:.3f} eV at {coord_str} "
                              f"(was {prev}).")
                        fit_log.append(f"{coord_str} : t_e={t_e_value:.4f} eV "
                                       f"(manual ln(I)-V fit)")

                # Persist logs after each shot (as NetCDF-safe strings) so nothing is
                # lost if you stop early
                ds.attrs['manually_fit_t_e'] = "; ".join(fit_log)
                ds.attrs['potential_bad_sweeps'] = "; ".join(bad_log)

    print(f"\nDone. {len(fit_log)} manual fits written, "
          f"{len(bad_log)} sweeps flagged as potentially bad.")
    return ds


def recalculate_derived_variables(ds):
    """
    Recompute every t_e-dependent diagnostic (n_i, n_e, nu_ei, p_e, p_ei) from the
    stored electron temperature and the stored ion-saturation current, for EVERY
    (probe, x, y, shot, sweep) cell.  Run this after double_check_temp_data has
    changed t_e values (manual re-fits and bad-sweep NaNs).

    NOT recomputed (they do not depend on t_e): v_f, v_p, ion_isat, electron_isat.
    ion_isat is t_e-independent and already in ds, so no HDF5 reload is needed.

    The per-cell math mirrors your original diagnostic assembly exactly, so the
    numbers are identical to a from-scratch run for the cells whose t_e is unchanged.
    """
    import numpy as np
    import astropy.units as u
    # --- Import your diagnostic helpers (adjust the module path to yours) ---
    from lapd_plasma_analysis.obtain_plots.Auxillary_functions import (
        get_ion_density,
        get_electron_ion_collision_frequency,
        l_get_pressure,
    )
    from lapd_plasma_analysis.net_cdf_infrastructure.Build_netcdf import safe_value

    # ------------------------------------------------------------------ #
    #  Hard-coded probe area (one entry per probe index in ds['probe']).
    #  Fill these in; if every probe shares an area, set them equal.
    # ------------------------------------------------------------------ #
    A_p_by_probe = {
        0: 2.0 * u.mm ** 2,   # <-- FILL IN probe 0 area
        1: 4.0 * u.mm ** 2,   # <-- FILL IN probe 1 area
        2: 4.0 * u.mm ** 2,   # <-- FILL IN probe 2 area (jan24 uses [0, 2]); remove if unused
    }

    # ------------------------------------------------------------------ #
    #  Constants
    # ------------------------------------------------------------------ #
    ion_type = ds.attrs['ion_type']    # e.g. "He-4+", read from the dataset
    t_i = 1.0 * u.eV                    # ion temperature: always 1 eV

    # Pull raw numpy arrays out once (dims: probe, x, y, shot, sweep)
    t_e_arr  = ds['t_e'].values
    isat_arr = ds['ion_isat'].values

    n_e_arr   = np.full_like(t_e_arr, np.nan, dtype=float)
    n_i_arr   = np.full_like(t_e_arr, np.nan, dtype=float)
    nu_ei_arr = np.full_like(t_e_arr, np.nan, dtype=float)
    p_e_arr   = np.full_like(t_e_arr, np.nan, dtype=float)
    p_ei_arr  = np.full_like(t_e_arr, np.nan, dtype=float)

    probe_vals = ds['probe'].values
    shape = t_e_arr.shape
    n_cells = int(np.prod(shape))
    print(f"Recalculating derived variables for {n_cells} cells "
          f"(probe x x x y x shot x sweep = {shape})...")

    done = 0
    for ip in range(shape[0]):
        probe_id = int(probe_vals[ip])
        if probe_id not in A_p_by_probe:
            raise KeyError(f"No probe area defined for probe index {probe_id}; "
                           f"add it to A_p_by_probe.")
        A_p = A_p_by_probe[probe_id]
        for ix in range(shape[1]):
            for iy in range(shape[2]):
                for ish in range(shape[3]):
                    for isw in range(shape[4]):
                        t_e_value = t_e_arr[ip, ix, iy, ish, isw]
                        i_ion_sat_value = isat_arr[ip, ix, iy, ish, isw]

                        # No valid temperature -> the whole cascade is NaN
                        # (matches the original t_e-is-NaN branch).
                        if not np.isfinite(t_e_value):
                            done += 1
                            continue

                        # Rebuild the astropy Quantities the helpers expect.
                        t_e_q = t_e_value * u.eV
                        i_ion_sat_q = (i_ion_sat_value * u.A
                                       if np.isfinite(i_ion_sat_value) else np.nan)

                        try:
                            n_i_value = get_ion_density(ion_type, i_ion_sat_q, A_p, t_e_q)
                            n_e_value = n_i_value
                        except Exception:
                            n_i_value = np.nan
                            n_e_value = np.nan

                        try:
                            nu_ei_value = get_electron_ion_collision_frequency(
                                n_e_value, ion_type, t_e_q)
                        except Exception:
                            nu_ei_value = np.nan

                        try:
                            p_e_value = l_get_pressure(t_e_q, n_e_value)
                        except Exception:
                            p_e_value = np.nan

                        try:
                            p_ei_value = l_get_pressure(t_e_q + t_i, n_e_value)
                        except Exception:
                            p_ei_value = np.nan

                        n_i_arr[ip, ix, iy, ish, isw]   = safe_value(n_i_value)
                        n_e_arr[ip, ix, iy, ish, isw]   = safe_value(n_e_value)
                        nu_ei_arr[ip, ix, iy, ish, isw] = safe_value(nu_ei_value)
                        p_e_arr[ip, ix, iy, ish, isw]   = safe_value(p_e_value)
                        p_ei_arr[ip, ix, iy, ish, isw]  = safe_value(p_ei_value)
                        done += 1
        print(f"  probe {probe_id} done ({done}/{n_cells} cells).")

    # Write the recomputed arrays back into the dataset in place.
    ds['n_e'][:]   = n_e_arr
    ds['n_i'][:]   = n_i_arr
    ds['nu_ei'][:] = nu_ei_arr
    ds['p_e'][:]   = p_e_arr
    ds['p_ei'][:]  = p_ei_arr

    print("Done recalculating derived variables "
          "(n_e, n_i, nu_ei, p_e, p_ei updated; v_f/v_p/ion_isat/electron_isat unchanged).")
    return ds