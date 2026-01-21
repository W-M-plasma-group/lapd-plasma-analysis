import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import math

from lapd_plasma_analysis.file_access import ask_yes_or_no
from lapd_plasma_analysis.obtain_plots.Auxillary_functions import filter_data, find_steady_state, filter_ne_data
from scipy.optimize import curve_fit


def polynomial_function(x, *coeffs):
    return np.polyval(coeffs, x)

def contour_plot(ds, diagnostic_to_plot, probe, run_identifier):
    """

    Parameters
    ----------
    ds - xarray of diagnostics and their corresponding values indexed by probe, x, y, shot, sweep
    diagnostic_to_plot
    probe - Index of the probe to plot

    Returns
    -------

    """

    # print('diagnostic_to_plot: ', repr(diagnostic_to_plot))
    if ask_yes_or_no("Also plot standard deviation? (y/n) "):
        filt_data = ask_yes_or_no("Filter data? (y/n) ")
        if filt_data:
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes = axes.flatten()

        mean_data = ds[diagnostic_to_plot].sel(probe=probe).mean('shot')
        std_data = ds[diagnostic_to_plot].sel(probe=probe).std('shot')
        if 't_e' in diagnostic_to_plot:
            # Make a uniform color bar from 0 eV to 20 eV for Temperature plots
            v_max = 15
            v_min = 0
            mean_plot = mean_data.plot(
                ax = axes[0],
                x='time',
                y='x',
                vmin=v_min,
                vmax=v_max,
                cmap = 'turbo',
                add_colorbar = True
            )
        elif 'n_e' in diagnostic_to_plot or 'n_i' in diagnostic_to_plot:
            v_max = 3e18
            v_min = 0
            mean_plot = mean_data.plot(
                ax=axes[0],
                x='time',
                y='x',
                vmin=v_min,
                vmax = v_max,
                cmap='turbo',
                add_colorbar=True
            )

        else:
            # TODO Other limits will be determined shortly
            v_max = None
            v_min = None
            mean_plot = mean_data.plot(
                ax = axes[0],
                x='time',
                y='x',
                add_colorbar = True,
                cmap = 'turbo'
            )

        axes[0].set_title('Mean')
        mean_plot.colorbar.set_label(
            f"{diagnostic_to_plot} ({ds[diagnostic_to_plot].attrs.get('units', diagnostic_to_plot)})"
        )

        std_plot = std_data.plot(ax = axes[1], x = 'time', y = 'x',
                                 vmin = 5,
                                 vmax = 0,
                                 add_colorbar = True,
                                 cmap = 'turbo')
        axes[1].set_title('Standard Deviation')
        std_plot.colorbar.set_label(f"({ds[diagnostic_to_plot].attrs.get('units', diagnostic_to_plot)})")

        if filt_data:
            filtered_data1 = filter_data(mean_data, std_data, first_filter = True)
            filt_plot1 = filtered_data1.plot(
                ax = axes[2],
                x='time',
                y='x',
                vmin=v_min,
                vmax=v_max,
                cmap='turbo',
                add_colorbar=True
            )
            axes[2].set_title('First filter')
            filt_plot1.colorbar.set_label(f"{diagnostic_to_plot} "
                                         f"({ds[diagnostic_to_plot].attrs.get('units', diagnostic_to_plot)})")

            filtered_data2 = filter_data(mean_data, std_data, first_filter=False)
            filt_plot2 = filtered_data2.plot(
                ax=axes[3],
                x='time',
                y='x',
                vmin=v_min,
                vmax=v_max,
                cmap='turbo',
                add_colorbar=True
            )
            axes[3].set_title('Second filter')
            filt_plot2.colorbar.set_label(f"{diagnostic_to_plot} "
                                         f"({ds[diagnostic_to_plot].attrs.get('units', diagnostic_to_plot)})")


        for ax in axes:
            ax.set_xlabel(f'Time  ({ds.attrs.get("time_units")})')
            ax.set_ylabel(f'x ({ds.attrs.get("x_units")})')


        diagnostic_name = ds[diagnostic_to_plot].attrs.get("long_name", diagnostic_to_plot)
        probe_port = ds['port'].isel(probe=probe).item()

        # print('diagnostic_name: ', diagnostic_name)
        # print('probe_port: ', probe_port)
        plt.tight_layout()

        fig.suptitle(f"{run_identifier} \n {diagnostic_name}\n  Port: {probe_port}")
        # plt.tight_layout()



    else:
        if 't_e' in diagnostic_to_plot:
            # Make a uniform color bar from 0 eV to 20 eV for Temperature plots
            cp = ds[diagnostic_to_plot].sel(probe=probe).mean('shot').plot(
                x = 'time',
                y = 'x',
                vmin = 0,
                vmax = 15,
                cmap = 'turbo'
                )
        elif 'n_e' in diagnostic_to_plot or 'n_i' in diagnostic_to_plot:
            cp = ds[diagnostic_to_plot].sel(probe=probe).mean('shot').plot(
                x='time',
                y='x',
                vmin=0,
                cmap='turbo',
                add_colorbar=True
            )

        else:
            # TODO Other limits will be determined shortly
            cp = ds[diagnostic_to_plot].sel(probe=probe).mean('shot').plot(
                x='time',
                y='x',
                cmap = 'turbo'
            )

        cbar = cp.colorbar
        ax = plt.gca()
        ax.set_xlabel(f'Time  ({ds.attrs.get("time_units")})')
        ax.set_ylabel(f'x ({ds.attrs.get("x_units")})')
        cbar.set_label(f'{diagnostic_to_plot} ({ds[diagnostic_to_plot].attrs.get("units",diagnostic_to_plot)})')

        diagnostic_name = ds[diagnostic_to_plot].attrs.get("long_name", diagnostic_to_plot)
        probe_port = ds['port'].isel(probe=probe).item()

        # print('diagnostic_name: ', diagnostic_name)
        # print('probe_port: ', probe_port)
        ax.set_title(f"{run_identifier} \n {diagnostic_name}\n  Port: {probe_port}")

    plt.subplots_adjust(top=0.85)
    plt.show()

def contour_subplots(datasets,diagnostic_to_plot,nc_list, nc_choice):
    """

    Parameters
    ----------
    datasets
    diagnostics_to_plot_list
    nc_list
    nc_choice

    Returns
    -------

    """
    num_runs = len(datasets)


    n_cols = math.ceil(math.sqrt(num_runs))
    n_rows = math.ceil(num_runs / n_cols)
    if ask_yes_or_no("Divide first two chosen datasets? (y/n) "):
        n_rows += 1
        divide = True
    else:
        divide = False

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    if ask_yes_or_no('Adjust any of the plasma py data (y/n) '):
        while True:
            try:
                adj_idx_str = input('Insert the indices (from the selected .nc list) to adjust (multiple indices must be '
                                    'comma separated): ').strip()
                adj_idx = [int(idx.strip()) for idx in adj_idx_str.split(',')]
            except ValueError:
                print("Invalid input. Please enter numeric indices")

            adjustments = []
            for index in adj_idx:
                try:
                    if index < 0 or index >= num_runs:
                        print(f"Index {index} is out of range.")
                        adjustments.append(float(1))
                        continue
                    adj_str = input(f"Adjust the index {index} by this specified amount: ").strip()
                    adjustments.append(float(adj_str))
                except ValueError:
                    print("Invalid input. Please enter float value")

            break
    else:
        adj_idx = []
        adjustments = []

    full_adj_array = []
    for j in range(len(datasets)):
        if j in adj_idx:
            idx = adj_idx.index(j)
            full_adj_array.append(adjustments[idx])
        else:
            full_adj_array.append(float(1))

    for i, ds in enumerate(datasets):
        ax = axes[i]

        if diagnostic_to_plot == 't_e':
            Min = 0
            Max = 15
        elif diagnostic_to_plot == 'electron_isat':
            Min = 0
            Max = 0.6
        else:
            Min = None
            Max = None

        cp = (ds[diagnostic_to_plot].sel(probe=0).mean('shot') * full_adj_array[i]).plot(
        x='time',
        y='x',
        ax=ax,
        vmin=Min,
        vmax=Max,
        cmap = 'turbo'
        )
        cbar = cp.colorbar
        ax.set_xlabel(f'Time  ({ds.attrs.get("time_units")})')
        ax.set_ylabel(f'x ({ds.attrs.get("x_units")})')
        cbar.set_label(f'{diagnostic_to_plot} ({ds[diagnostic_to_plot].attrs.get("units", diagnostic_to_plot)})')

        filename = nc_list[nc_choice[i]]

        if "Mar" in filename:
            run_identifier = "Mar 22 run " + filename.split("_")[1]
        elif filename.split("_")[0].isdigit():
            run_identifier = "Jan 24 run " + filename.split("_")[0]
        else:
            run_identifier = "filename not yet supported"

        if "pp" in filename:
            run_identifier = "Plasma Py " + run_identifier

        if "adj" in filename:
            run_identifier = "Adj v_p " + run_identifier

        if full_adj_array[i] != 1.0:
            run_identifier = run_identifier + f"( * {full_adj_array[i]})"

        diagnostic_name = ds[diagnostic_to_plot].attrs.get("long_name", diagnostic_to_plot)
        probe_port = ds['port'].isel(probe=0).item()

        ax.set_title(f"{run_identifier} \n {diagnostic_name}\n  Port: {probe_port}")

    if divide:
        ax = axes[len(axes) - 1]
        ds1 = datasets[0]
        ds2 = datasets[1]
        data_1 = ds1[diagnostic_to_plot].sel(probe = 0).mean('shot') * full_adj_array[0]
        data_2 = ds2[diagnostic_to_plot].sel(probe = 0).mean('shot') * full_adj_array[1]
        cp = (data_1/
              data_2).plot(
            x='time',
            y='x',
            ax=ax,
            vmin=0,
            vmax=2,
            cmap = 'turbo'
        )
        cbar = cp.colorbar
        ax.set_xlabel(f'Time  ({ds.attrs.get("time_units")})')
        ax.set_ylabel(f'x ({ds.attrs.get("x_units")})')
        filename_a = filename = nc_list[nc_choice[0]]
        filename_b = filename = nc_list[nc_choice[1]]
        if "Mar" in filename_a:
            run_identifier_a = "Mar 22 run " + filename_a.split("_")[1]
        elif filename_a.split("_")[0].isdigit():
            run_identifier_a = "Jan 24 run " + filename_a.split("_")[0]
        else:
            run_identifier_a = "filename not yet supported"
        if "Mar" in filename_b:
            run_identifier_b = "Mar 22 run " + filename_b.split("_")[1]
        elif filename_b.split("_")[0].isdigit():
            run_identifier_b = "Jan 24 run " + filename_b.split("_")[0]
        else:
            run_identifier_b = "filename not yet supported"
        if "pp" in filename_a:
            run_identifier_a = "Plasma Py " + run_identifier_a
        if"pp" in filename_b:
            run_identifier_b = "Plasma Py " + run_identifier_b
        if full_adj_array[0] != 1.0:
            run_identifier_a = run_identifier_a + f"( * {full_adj_array[0]})"
        if full_adj_array[1] != 1.0:
            run_identifier_b = run_identifier_b + f"( * {full_adj_array[1]})"

        diagnostic_name = ds[diagnostic_to_plot].attrs.get("long_name", diagnostic_to_plot)
        ax.set_title(f"{diagnostic_name} \n {run_identifier_a}/{run_identifier_b}")
        print(f"Average divided value: {(data_1/data_2).mean('x').mean('sweep')}")



    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    plt.show()

def michael_density_plots(ds, diagnostic_to_plot, probe, run_identifier):

    ds_diag = ds[diagnostic_to_plot].sel(probe = probe).mean('shot')
    mask = (ds['time'] >= 7.0) & (ds['time'] <= 15.0)
    diag_to_plot = ds_diag.sel(sweep = ds['sweep'][mask]).mean('sweep')
    std_to_plot = ds_diag.sel(sweep = ds['sweep'][mask]).std('sweep')

    # Format everything for matplot.lib plotting
    x_vals = diag_to_plot['x'].values
    diag_to_plot_vals = diag_to_plot.squeeze().values
    std_to_plot_vals = std_to_plot.squeeze().values

    fig, ax = plt.subplots()
    ax.errorbar(x_vals, diag_to_plot_vals, yerr=std_to_plot_vals, fmt='o', capsize=3)
    ax.set_xlabel(f'x ({ds.attrs.get("x_units")})')

    diagnostic_name = ds[diagnostic_to_plot].attrs.get("long_name", diagnostic_to_plot)
    ax.set_ylabel(f'{diagnostic_name} ({ds[diagnostic_to_plot].attrs.get("units", diagnostic_to_plot)})')
    probe_port = ds['port'].isel(probe=probe).item()

    ax.set_title(f"{run_identifier} \n {diagnostic_name} Gradient plot \n Port: {probe_port}")
    plt.subplots_adjust(top=0.85)
    plt.show()

def show_steady_state(ds, probe, run_identifier):
    """

    Parameters
    ----------
    ds - xarray.DataArray
    probe - Probe index so we can get the proper data out of the xarray.DataArray
    run_identifier - Run identifier indicating which experiment day and run we are looking at

    Returns
    -------

    """
    mean_data = ds['t_e'].sel(probe=probe).mean('shot')
    std_data = ds['t_e'].sel(probe=probe).std('shot')

    filtered_data = filter_data(mean_data, std_data)

    fig1, axes1, range_tot = plot_time_series(ds, probe, run_identifier, return_range = True)
    plt.show()

    # Ask to find the steady state period
    if ask_yes_or_no('Search for steady state?" (y/n) '):
        zero_index = range_tot.index(0)
        if len(range_tot) >= 5:
            search_range = range_tot[(zero_index - 2) : (zero_index + 3)]
        elif len(range_tot) >= 3:
            search_range = range_tot[(zero_index - 1) : (zero_index + 2)]
        else:
            search_range = range_tot[zero_index]

        while True:
            user_input = input("Guess a center time for the steady state: ")
            try:
                int_user_input = int(user_input)
                break

            except ValueError:
                print("Please enter an integer.")

        t_e_data_arrays = [filtered_data.sel(x = x_val, y = 0) for x_val in search_range]
        n_e_data_arrays = [ds['n_e'].sel(probe = probe, x = x_val, y = 0).mean('shot') for x_val in search_range]

        min_time, max_time = find_steady_state(t_e_data_arrays, n_e_data_arrays, int_user_input)
        fig2, axes2 = plot_time_series(ds, probe, run_identifier, return_range = False)

        for ax in axes2:
            ax.axvline(min_time, color='k', linestyle='--')
            ax.axvline(max_time, color='k', linestyle='--')
        plt.show()

def plot_time_series(ds, probe, run_identifier, return_range = False):
    mean_data = ds['t_e'].sel(probe=probe).mean('shot')
    std_data = ds['t_e'].sel(probe=probe).std('shot')

    filtered_data = filter_data(mean_data, std_data)

    min_x = int(min(filtered_data['x'].values))
    max_x = int(max(filtered_data['x'].values))

    range_up = list(range(0, max_x + 1, 5))
    if min_x <= - 5:
        range_down = list(range(-5, min_x + 1, -5))
    else:
        range_down = []
    range_tot = range_down + range_up
    range_tot = sorted(range_tot)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes = axes.flatten()
    for x_test in range_tot:
        test_data_t_e = filtered_data.sel(x=x_test, y=0)
        test_data_n_e = ds['n_e'].sel(probe=probe, x=x_test, y=0).mean('shot')
        num_nan = test_data_t_e.isnull().sum().item()
        if num_nan < len(test_data_t_e) / 2:
            test_data_t_e.plot(ax=axes[0],
                               x='time',
                               marker='o',
                               label=f'x = {x_test}')
            test_data_n_e.plot(ax=axes[1],
                               x='time',
                               marker='o',
                               label=f'x = {x_test}')

    axes[0].set_xlabel(f'Time ({ds.attrs.get("time_units")})',fontsize=28)
    axes[0].set_ylabel(rf'$T_{{e}}$ ({ds["t_e"].attrs.get("units", "t_e")})',fontsize=28)
    axes[0].set_title(f'{ds["t_e"].attrs.get("long_name", "t_e")}',fontsize=28)
    axes[0].legend(loc = 'upper left',fontsize=14)
    axes[0].tick_params(labelsize=20)

    axes[1].set_xlabel(f'Time ({ds.attrs.get("time_units")})',fontsize=28)
    axes[1].set_ylabel(rf'$n_{{e}}$ (${ds["n_e"].attrs.get("units", "n_e")}$)',fontsize=28)
    axes[1].set_title(f'{ds["n_e"].attrs.get("long_name", "n_e")}',fontsize = 28)
    axes[1].legend(loc = 'best',fontsize=14)
    axes[1].tick_params(labelsize=24)
    axes[1].ticklabel_format(style='sci', axis = 'y', scilimits = (0, 0))
    axes[1].yaxis.get_offset_text().set_fontsize(24)


    probe_port = ds['port'].isel(probe=probe).item()

    plt.tight_layout(rect=[0, 0, 1, .9])
    fig.suptitle(f"Time Series: {run_identifier},  Probe: {probe}", fontsize=28)
    if return_range:
        return fig, axes, range_tot

    return fig, axes

def create_gradient_plot(ds, run_identifier, gradient_times_dict):
    """
    Parameters
    ----------
    ds
    probe
    run_identifier

    Returns
    -------

    """
    fig1, axes, range_tot = plot_time_series(ds, 0, run_identifier, return_range = True)
    plt.show()
    while True:
        user_input = input("Guess a center time for the steady state: ")
        try:
            int_user_input = int(user_input)
            break

        except ValueError:
            print("Please enter an integer.")
    # plt.close('all')
    d_probe_gradients = {}
    for probe in range(ds.sizes['probe']):
        mean_data = ds['t_e'].sel(probe=probe).mean('shot')
        std_data = ds['t_e'].sel(probe=probe).std('shot')

        t_e_filtered_data = filter_data(mean_data, std_data)
        where_nans = t_e_filtered_data.isnull()
        n_e_filtered_data = ds['n_e'].sel(probe=probe).mean('shot').where(~where_nans)
        zero_index = range_tot.index(0)
        if len(range_tot) >= 5:
            search_range = range_tot[(zero_index - 2): (zero_index + 3)]
        elif len(range_tot) >= 3:
            search_range = range_tot[(zero_index - 1): (zero_index + 2)]
        else:
            search_range = range_tot[zero_index]

        t_e_data_arrays = [t_e_filtered_data.sel(x=x_val, y=0) for x_val in search_range]
        n_e_data_arrays = [n_e_filtered_data.sel(x=x_val, y=0) for x_val in search_range]
        min_time, max_time = find_steady_state(t_e_data_arrays, n_e_data_arrays, int_user_input)
        filtered_ne_mean_profile, filtered_ne_std_profile = filter_ne_data(n_e_filtered_data, min_time, max_time)

        t_e_mask = (t_e_filtered_data['time'] >= min_time) & (t_e_filtered_data['time'] <= max_time)
        n_e_mask = (n_e_filtered_data['time'] >= min_time) & (n_e_filtered_data['time'] <= max_time)

        t_e_to_plot = t_e_filtered_data.sel(sweep=ds['sweep'][t_e_mask]).mean('sweep')
        t_e_std_to_plot = t_e_filtered_data.sel(sweep=ds['sweep'][t_e_mask]).std('sweep')

        # n_e_to_plot = n_e_filtered_data.sel(sweep=ds['sweep'][n_e_mask]).mean('sweep')
        # n_e_std_to_plot = n_e_filtered_data.sel(sweep=ds['sweep'][n_e_mask]).std('sweep')
        n_e_to_plot = filtered_ne_mean_profile
        n_e_std_to_plot = filtered_ne_std_profile
        # Format everything for matplot.lib plotting

        x_vals = t_e_filtered_data['x'].values
        t_e_to_plot_vals = t_e_to_plot.squeeze().values
        t_e_std_to_plot_vals = t_e_std_to_plot.squeeze().values
        n_e_to_plot_vals = n_e_to_plot.squeeze().values
        n_e_std_to_plot_vals = n_e_std_to_plot.squeeze().values

        threshold = 25
        outlier_mask = t_e_to_plot_vals < threshold
        x_vals = x_vals[outlier_mask]
        t_e_to_plot_vals = t_e_to_plot_vals[outlier_mask]
        t_e_std_to_plot_vals = t_e_std_to_plot_vals[outlier_mask]
        n_e_to_plot_vals = n_e_to_plot_vals[outlier_mask]
        n_e_std_to_plot_vals = n_e_std_to_plot_vals[outlier_mask]

        # Test to see if the last edge of the temperature regime increases rapidly - We don't think that's possible
        step = 3
        # How far from the end do you want to check
        max_value = 20
        end_index = 1
        beginning_index = 0
        beg_threshold_slope = -.3
        end_threshold_slope = .5
        it_vars = list(reversed(list(range(max_value))))
        for i in it_vars:

            if end_index == 1 and -(i + 1) + step <= -1:
                end_numerator = (-t_e_to_plot_vals[-(i+1)] + t_e_to_plot_vals[-(i+1)+ step])
                end_denom = (-x_vals[-(i+1)] + x_vals[-(i+1) + step])
                try:
                    end_slope = end_numerator / end_denom
                except ZeroDivisionError:
                    end_slope = 0
                if end_slope > end_threshold_slope:
                    end_index = i + 1
                # print('end_slope: ', end_slope)

            if beginning_index == 0 and i - step >= 0:
                beginning_numerator = t_e_to_plot_vals[i] - t_e_to_plot_vals[i - step]
                beginning_denom = x_vals[i] - x_vals[i - step]
                try:
                    beginning_slope = beginning_numerator / beginning_denom
                except ZeroDivisionError:
                    beginning_slope = 0
                if beginning_slope < beg_threshold_slope:
                    beginning_index = i
                # print('beginning_slope: ', beginning_slope)
        adj_t_e_to_plot_vals = t_e_to_plot_vals[beginning_index: len(t_e_to_plot_vals) - end_index]
        adj_t_e_std_to_plot_vals = t_e_std_to_plot_vals[beginning_index: len(t_e_to_plot_vals) - end_index]
        adj_x_vals = x_vals[beginning_index: len(x_vals) - end_index]
        # print(f'Beginning index: {beginning_index}, Ending index: {end_index}')
        num_t_vals = len(adj_x_vals)
        num_n_vals = len(x_vals)


        if 'Apr 18' in run_identifier:
            key_num = 0
        elif 'Mar 22' in run_identifier:
            key_num = 1
        elif 'Nov 22' in run_identifier:
            key_num = 2
        elif 'Jan 24' in run_identifier:
            key_num = 3
        else:
            key_num = 4

        if key_num not in gradient_times_dict:

            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
            axes2 = axes2.flatten()

            axes2[0].errorbar(x_vals, t_e_to_plot_vals, yerr=t_e_std_to_plot_vals, fmt='o', capsize=3)
            axes2[0].set_xlabel(f'x ({ds.attrs.get("x_units")})',fontsize=24)

            axes2[1].errorbar(x_vals, n_e_to_plot_vals, yerr=n_e_std_to_plot_vals, fmt='o', capsize=3)
            axes2[1].set_xlabel(f'x ({ds.attrs.get("x_units")})',fontsize=24)

            t_e_name = ds['t_e'].attrs.get("long_name", 't_e')
            n_e_name = ds['n_e'].attrs.get("long_name", 'n_e')

            axes2[0].set_ylabel(rf"$T_{{e}}$ ({ds['t_e'].attrs.get('units', 't_e')})",fontsize=24)
            axes2[1].set_ylabel(rf'$n_{{e}}$ (${ds["n_e"].attrs.get("units", "n_e")}$)',fontsize=24)

            axes2[0].set_title(f"{t_e_name}",fontsize=24)
            axes2[1].set_title(f"{n_e_name}",fontsize=24)

            axes2[0].tick_params(labelsize=20)
            axes2[1].tick_params(labelsize=20)

            axes2[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axes2[1].yaxis.get_offset_text().set_fontsize(24)

            probe_port = ds['port'].isel(probe=probe).item()
            plt.tight_layout(rect=[0, 0, 1, .9])
            fig2.suptitle(f"Radial Profile: {run_identifier}, Probe: {probe}",fontsize=24)

            plt.show()
            while True:
                try:
                    user_input = input("Enter intervals to use for the gradients (e.g., 10 15, 20 25, 30 35): ")

                    # Split the input into interval strings using commas
                    interval_strings = user_input.split(',')

                    # Parse each interval string into a list of integers
                    intervals = [list(map(int, interval.strip().split())) for interval in interval_strings]
                    gradient_times_dict[key_num] = intervals
                    break
                except ValueError:
                    print("Invalid interval input. Make sure all values are integers.")
            # plt.close('all')

        fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12))
        fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
        axes3 = axes3.flatten()
        axes4 = axes4.flatten()

        axes3[0].errorbar(x_vals, t_e_to_plot_vals, yerr=t_e_std_to_plot_vals, fmt='o', capsize=3)
        axes3[0].set_xlabel(f'x ({ds.attrs.get("x_units")})')

        axes4[0].set_xlabel(f'x ({ds.attrs.get("x_units")})')

        axes3[2].errorbar(adj_x_vals, adj_t_e_to_plot_vals, yerr=adj_t_e_std_to_plot_vals, fmt='o', capsize=3)
        axes3[2].set_xlabel(f'x ({ds.attrs.get("x_units")})')

        axes4[0].errorbar(adj_x_vals, adj_t_e_to_plot_vals, yerr=adj_t_e_std_to_plot_vals, fmt='o', capsize=3)

        axes3[1].errorbar(x_vals, n_e_to_plot_vals, yerr=n_e_std_to_plot_vals, fmt='o', capsize=3)
        axes3[1].set_xlabel(f'x ({ds.attrs.get("x_units")})')

        axes4[1].errorbar(x_vals, n_e_to_plot_vals, yerr=n_e_std_to_plot_vals, fmt='o', capsize=3)
        axes4[1].set_xlabel(f'x ({ds.attrs.get("x_units")})')

        axes3[3].errorbar(x_vals, n_e_to_plot_vals, yerr=n_e_std_to_plot_vals, fmt='o', capsize=3)
        axes3[3].set_xlabel(f'x ({ds.attrs.get("x_units")})')

        t_e_name = ds['t_e'].attrs.get("long_name", 't_e')
        n_e_name = ds['n_e'].attrs.get("long_name", 'n_e')

        axes3[0].set_ylabel(f"t_e ({ds['t_e'].attrs.get('units', 't_e')})")
        axes3[2].set_ylabel(f"t_e ({ds['t_e'].attrs.get('units', 't_e')})")
        axes4[0].set_ylabel(f"t_e ({ds['t_e'].attrs.get('units', 't_e')})")

        axes3[1].set_ylabel(f"n_e ({ds['n_e'].attrs.get('units', 'n_e')})")
        axes3[3].set_ylabel(f"n_e ({ds['n_e'].attrs.get('units', 'n_e')})")
        axes4[1].set_ylabel(f"n_e ({ds['n_e'].attrs.get('units', 'n_e')})")

        axes3[0].set_title(f"{t_e_name}")
        axes4[0].set_title(f"{t_e_name}")
        axes3[2].set_title(f"Cleaned {t_e_name} data")

        axes3[1].set_title(f"{n_e_name}")
        axes4[1].set_title(f"{n_e_name}")
        axes3[3].set_title(f"{n_e_name}")

        intervals = gradient_times_dict[key_num]
        d_gradients = {}
        d_gradients['intervals'] = len(intervals)
        interval_num = 0
        for interval in intervals:
            for time in interval:
                # Plot a vertical line at each of the chosen time intervals
                for ax in axes3:
                    ax.axvline(x=time, color='k', linestyle='--')
                for ax in axes4:
                    ax.axvline(x=time, color='k', linestyle='--')
            start, end = interval
            x_mask = (x_vals >= start) & (x_vals <= end)
            adj_x_mask = (adj_x_vals >= start) & (adj_x_vals <= end)

            t_vals = t_e_to_plot_vals[x_mask]
            d_gradients[f'{interval_num}_t_vals'] = t_vals
            adj_t_vals = adj_t_e_to_plot_vals[adj_x_mask]
            d_gradients[f'{interval_num}_adj_t_vals'] = adj_t_vals

            x_vals_masked = x_vals[x_mask]
            num_n_vals = num_n_vals - len(x_vals_masked)
            d_gradients[f'{interval_num}_x_vals'] = x_vals_masked
            adj_x_vals_masked = adj_x_vals[adj_x_mask]
            num_t_vals = num_t_vals - len(adj_x_vals_masked)
            n_vals = n_e_to_plot_vals[x_mask]
            d_gradients[f'{interval_num}_n_vals'] = n_vals
        #     print('x vals: ', len(x_vals))
        #     print('x_vals_masked', len(x_vals_masked))
        #     t_coeffs = np.polyfit(x_vals_masked, t_vals, deg=1)
        #     t_m, t_b = t_coeffs
        #
        #     try:
        #         adj_t_coeffs = np.polyfit(adj_x_vals_masked, adj_t_vals, deg=1)
        #     except TypeError:
        #         adj_t_coeffs = np.polyfit(x_vals_masked, t_vals, deg=1)
        #         adj_x_vals_masked = x_vals_masked
        #     adj_t_m, adj_t_b = adj_t_coeffs
        #     d_gradients[f'{interval_num}_adj_x_vals_masked'] = adj_x_vals_masked
        #
        #     if np.where(x_vals >= start)[0][0] <= beginning_index + 1 and t_m <= beg_threshold_slope:
        #         t_m = 0
        #         t_b = 0
        #     if np.where(x_vals >= start)[0][0] <= beginning_index + 1 and adj_t_m <= beg_threshold_slope:
        #         adj_t_m = 0
        #         adj_t_b = 0
        #
        #     if np.where(x_vals <= end)[0][-1] >= len(x_vals) - end_index and t_m >= end_threshold_slope:
        #         t_m = 0
        #         t_b = 0
        #     if np.where(x_vals <= end)[0][-1] >=len(x_vals) - end_index and adj_t_m >= end_threshold_slope:
        #         adj_t_m = 0
        #         adj_t_b = 0
        #
        #     if t_m == 0 and t_b == 0:
        #         continue
        #     else:
        #         d_gradients[f'{interval_num}_t_m'] = t_m
        #         d_gradients[f'{interval_num}_t_b'] = t_b
        #         d_gradients[f'{interval_num}_adj_t_m'] = adj_t_m
        #         d_gradients[f'{interval_num}_adj_t_b'] = adj_t_b
        #
        #     n_coeffs = np.polyfit(x_vals_masked, n_vals, deg=1)
        #     n_m, n_b = n_coeffs
        #     d_gradients[f'{interval_num}_n_m'] = n_m
        #     axes3[0].plot(x_vals_masked, t_m * x_vals_masked + t_b, linestyle='--',
        #                   label = f'Fit: y = {t_m:.2f}x + {t_b:.2f}')
        #     axes3[2].plot(adj_x_vals_masked, adj_t_m * adj_x_vals_masked + adj_t_b, linestyle='--',
        #                   label=f'Fit: y = {adj_t_m:.2f}x + {adj_t_b:.2f}')
        #     axes4[0].plot(adj_x_vals_masked, adj_t_m * adj_x_vals_masked + adj_t_b, linestyle='--',
        #                   label=f'Fit: y = {adj_t_m:.2f}x + {adj_t_b:.2f}')
        #
        #     axes3[1].plot(x_vals_masked, n_m * x_vals_masked + n_b, linestyle='--',
        #                   label = f'Fit: y = {n_m:.2e}x + {n_b:.2e}')
        #     axes4[1].plot(x_vals_masked, n_m * x_vals_masked + n_b, linestyle='--',
        #                   label=f'Fit: y = {n_m:.2e}x + {n_b:.2e}')
        #     axes3[3].plot(x_vals_masked, n_m * x_vals_masked + n_b, linestyle='--',
        #                   label=f'Fit: y = {n_m:.2e}x + {n_b:.2e}')
        #
        #     axes3[0].legend(loc = 'best')
        #     axes3[1].legend(loc = 'best')
        #     axes3[2].legend(loc = 'best')
        #     axes3[3].legend(loc = 'best')
        #     axes4[0].legend(loc='best')
        #     axes4[1].legend(loc='best')
        #     interval_num += 1

        probe_port = ds['port'].isel(probe=probe).item()
        # plt.tight_layout(rect=[0, 0, 1, .9])
        # fig3.suptitle(f"{run_identifier} \n Gradient plot \n Port: {probe_port}")
        # fig4.suptitle(f"{run_identifier} \n Gradient plot \n Port: {probe_port}")
        # # plt.show()
        # # plt.show()
        plt.close('all')

        # Do a fit of the entire profile of order number of total points - number of points in interval

        # Get rid of edge case
        num_t_vals = num_t_vals - 4
        num_n_vals = num_n_vals - 4

        test_adj_x_vals = np.linspace(adj_x_vals[0], adj_x_vals[-1], 100)
        test_x_vals = np.linspace(x_vals[0], x_vals[-1], 100)

        # np.polyfit
        t_poly_fit = np.polyfit(adj_x_vals,adj_t_e_to_plot_vals, deg = num_t_vals)
        n_poly_fit = np.polyfit(x_vals,n_e_to_plot_vals, deg = num_n_vals)
        t_func = np.poly1d(t_poly_fit)
        n_func = np.poly1d(n_poly_fit)
        poly_t_vals = t_func(test_adj_x_vals)
        poly_n_vals = n_func(test_x_vals)

        # curve fit
        t_curvefit, t_curve_cov = curve_fit(polynomial_function, adj_x_vals, adj_t_e_to_plot_vals, p0 = t_poly_fit,
                                           sigma = adj_t_e_std_to_plot_vals, absolute_sigma = True)
        n_curvefit, n_curve_cov = curve_fit(polynomial_function, x_vals, n_e_to_plot_vals, p0=n_poly_fit,
                                           sigma=n_e_std_to_plot_vals, absolute_sigma=True)
        t_curve_func = np.poly1d(t_curvefit)
        n_curve_func = np.poly1d(n_curvefit)
        curve_t_vals = polynomial_function(test_adj_x_vals, *t_curvefit)
        curve_n_vals = polynomial_function(test_x_vals, *n_curvefit)

        # Remove edge wackiness from polynomials
        points_on_edge = 2
        t_fit_x_vals_mask = (test_adj_x_vals >= adj_x_vals[points_on_edge -1]) & (test_adj_x_vals <= adj_x_vals[-points_on_edge])
        n_fit_x_vals_mask = (test_x_vals >= x_vals[points_on_edge -1]) & (test_x_vals <= x_vals[-points_on_edge])
        curve_t_vals_masked = curve_t_vals[t_fit_x_vals_mask]
        curve_n_vals_masked = curve_n_vals[n_fit_x_vals_mask]
        test_adj_x_vals_masked = test_adj_x_vals[t_fit_x_vals_mask]
        test_x_vals_masked = test_x_vals[n_fit_x_vals_mask]


        n_adj_x_vals = adj_x_vals[points_on_edge - 1:len(adj_x_vals) - points_on_edge]
        n_x_vals = x_vals[points_on_edge - 1:len(x_vals) - points_on_edge]
        n_e_edge_mask = n_e_to_plot_vals[points_on_edge - 1:len(n_e_to_plot_vals) - points_on_edge]
        t_e_edge_mask = adj_t_e_to_plot_vals[points_on_edge - 1:len(adj_t_e_to_plot_vals) - points_on_edge]
        t_grad = t_curve_func.deriv()(n_adj_x_vals)
        n_grad = n_curve_func.deriv()(n_x_vals)
        n_x_vals_t_mask = (n_x_vals >= n_adj_x_vals[0]) & (n_x_vals <= n_adj_x_vals[-1])
        n_grad_masked = n_grad[n_x_vals_t_mask]
        n_e_masked = n_e_edge_mask[n_x_vals_t_mask]
        normalized_grad_n = n_grad/n_e_edge_mask
        adj_normalized_grad_n = n_grad_masked/n_e_masked
        normalized_grad_t = t_grad/t_e_edge_mask
        eta_e = normalized_grad_t/adj_normalized_grad_n



        fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
        axes5 = axes5.flatten()

        axes5[0].errorbar(adj_x_vals, adj_t_e_to_plot_vals, yerr=adj_t_e_std_to_plot_vals, fmt='o', capsize=3)
        # axes5[0].plot(test_adj_x_vals, poly_t_vals, label=f'Polynomial fit order: {num_t_vals}')
        axes5[0].plot(test_adj_x_vals_masked, curve_t_vals_masked, label=f'Curve fit order: {num_t_vals}')
        axes5[0].set_ylabel(f"$t_e$ ({ds['t_e'].attrs.get('units', 't_e')})")
        axes5[0].set_xlabel(f'x ({ds.attrs.get("x_units")})')
        axes5[0].set_title(f"{t_e_name}")


        axes5[1].errorbar(x_vals, n_e_to_plot_vals, yerr=n_e_std_to_plot_vals, fmt='o', capsize=3)
        # axes5[1].plot(test_x_vals, poly_n_vals, label=f'Polynomial fit order: {num_n_vals}')
        axes5[1].plot(test_x_vals_masked, curve_n_vals_masked, label=f'Curve fit order: {num_n_vals}')
        axes5[1].set_ylabel(f"$n_e$ (${ds['n_e'].attrs.get('units', 'n_e')}$)")
        axes5[1].set_xlabel(f'x ({ds.attrs.get("x_units")})')
        axes5[1].set_title(f"{n_e_name}")

        axes5[0].legend(loc='best')
        axes5[1].legend(loc='best')

        plt.tight_layout(rect=[0, 0, 1, .9])
        fig5.suptitle(f"Radial Profile: {run_identifier} \n Probe: {probe}")
        plt.show()
        d_gradients['x_vals'] = n_x_vals
        d_gradients['adj_x_vals'] = n_adj_x_vals
        d_gradients['normalized grad t'] = normalized_grad_t
        d_gradients['normalized grad n'] = normalized_grad_n
        d_gradients['eta e'] = eta_e

        d_probe_gradients[f'{probe}'] = d_gradients





    return gradient_times_dict, d_probe_gradients




def f_run_identifier(filename):
    """

    Parameters
    ----------
    filename - String identifying what run the data is from

    Returns
    -------
    run_identifier - String identifying what run the data is from to be used in a plot title
    """

    if "Mar" in filename:
        run_identifier = "Mar 22 run " + filename.split("_")[1]
    elif 'kG' in filename:
        try:
            addition = int(filename.split("_")[1])
        except ValueError:
            addition = filename.split("_")[0]

        run_identifier = "Jan 24 run " + str(addition)

        if "H2" in filename:
            run_identifier = run_identifier + " H2"
        else:
            run_identifier = run_identifier + " He"

    else:
        run_identifier = "filename not yet supported"
    # print(run_identifier)
    if "pp" in filename:
        run_identifier = "Plasma Py " + run_identifier
    if "adj" in filename:
        run_identifier = 'Adj V_P ' + run_identifier

    return run_identifier


