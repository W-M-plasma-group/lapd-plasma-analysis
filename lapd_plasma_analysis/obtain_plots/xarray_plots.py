import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xarray as xr
import math
import colorsys
import ast
import json

from astropy.units.quantity_helper.function_helpers import concatenate

from lapd_plasma_analysis.file_access import *
from lapd_plasma_analysis.obtain_plots.Auxillary_functions import filter_data
from lapd_plasma_analysis.calculation_helpers import sound_speed_calculation
from lapd_plasma_analysis.obtain_plots.plot_from_netcdf_helpers.process_temp_dens_data import process_variable_data
from scipy.optimize import curve_fit
from scipy.ndimage import binary_closing
from plasmapy.particles import *
from astropy import constants as c
from astropy import units as u
from astropy.units import Unit, Quantity

from scipy.signal import savgol_filter, find_peaks


def polynomial_function(x, *coeffs):
    return np.polyval(coeffs, x)


def contour_plot(ds, diagnostic_to_plot, probe=0, run_identifier=None, figure_folder=None,
                 plot_std = False, filt_data = False, save_plots = True, show_plot = False, check_shots = False,
                 shots_to_plot = None
                 ):
    """
    Generate 2D contour plot(s) for a single xarray dataset across time and position.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing diagnostic measurements.
    diagnostic_to_plot : str
        Variable key inside ds to plot (e.g., 't_e', 'n_e').
    probe : int, default=0
        Probe index to plot.
    run_identifier : str, optional
        Custom identifier string. If None, generated via f_run_identifier.
    figure_folder : str, optional
        Folder path for saving plots.
    plot_std : bool, default=False
        Whether to plot standard deviation for the plot diagnostic in a separate subplot
    filt_data : bool, default=False
        Whether to include filtered data in a separate subplot.
    save_plots : bool, default=True
        Whether to save plots in an associated figure folder
    show_plot : bool, default=False
        Whether to show plots directly in this window
    check_shots : bool, default=False
        Whether to check shots individually
    shots_to_plot : list, optional
        List of shot numbers to plot, if check_shots is true this will be populated
    """

    # Sort spatial and temporal coordinates monotonically for 2D plotting
    coords_to_sort = [c for c in ['time', 'x'] if c in ds.coords]
    if coords_to_sort:
        ds = ds.sortby(coords_to_sort)

    # PyCharm warning mitigation
    if run_identifier is None:
        run_identifier = f_run_identifier(ds=ds)

    if save_plots:
        figure_folder = ensure_directory(figure_folder + f'Contour/{diagnostic_to_plot}/{run_identifier}/')
        if check_shots:
            figure_folder = ensure_directory(figure_folder + f'Individual_shots/probe_{probe}/')
        else:
            figure_folder = ensure_directory(figure_folder + f'Averaged_over_shots/')

    # PyCharm warning mitigation
    if run_identifier is None:
        run_identifier = f_run_identifier(ds=ds)

    # Determine layout structure for build_subplots
    if plot_std and filt_data:
        layout = [[2], [2]]  # 2x2 grid (4 plots total)
    elif plot_std:
        layout = [[2]]  # 1x2 grid (2 plots total)
    else:
        layout = [[1]]  # 1x1 grid (1 plot total)



    if not check_shots:
        fig, ax, letters = build_subplots(layout=layout, fig_width=6.0, fig_height=4.5)
        # Compute mean and standard deviation
        mean_data = ds[diagnostic_to_plot].sel(probe=probe).mean('shot')
        std_data = ds[diagnostic_to_plot].sel(probe=probe).std('shot')


        # Dynamic colorbar bounds
        vals = mean_data.values
        v_min, v_max = np.nanpercentile(vals[~np.isnan(vals)], [2, 98]) if np.any(~np.isnan(vals)) else (0, 1)
        print('Min and Max values', v_min, v_max)

        # Force lower bound to 0 for non-negative physical diagnostics
        negative_diagnostics = ['ion_isat', 'v_f']

        # Force floor to 0.0 for positive physical quantities (t_e, n_e, n_i, etc.)
        if not any(tag in diagnostic_to_plot for tag in negative_diagnostics):
            v_min = max(0.0, v_min)

        # Plot Mean values across shots
        mean_ax = ax[letters[0]]
        cp_mean = mean_data.plot(
            ax=mean_ax,
            x='time',
            y='x',
            vmin=v_min,
            vmax=v_max,
            cmap='turbo',
            add_colorbar=True
        )
        mean_ax.set_title("Mean", fontsize=16)

        # Plot Standard Deviation across shots (if requested)
        if plot_std:
            std_ax = ax[letters[1]]
            std_vals = std_data.values
            s_min, s_max = np.nanpercentile(std_vals[~np.isnan(std_vals)], [2, 98]) if np.any(
                ~np.isnan(std_vals)) else (0,
                                           1)

            cp_std = std_data.plot(
                ax=std_ax,
                x='time',
                y='x',
                vmin=max(0.0, s_min),
                vmax=s_max,
                cmap='turbo',
                add_colorbar=True
            )
            std_ax.set_title("Standard Deviation", fontsize=16)

        # Plot Filtered Data (if requested)
        if filt_data:
            filtered_data1 = filter_data(mean_data, std_data, first_filter=True)
            filt_ax1 = ax[letters[2]]
            filtered_data1.plot(
                ax=filt_ax1,
                x='time',
                y='x',
                vmin=v_min,
                vmax=v_max,
                cmap='turbo',
                add_colorbar=True
            )
            filt_ax1.set_title("First Filter", fontsize=16)

            filtered_data2 = filter_data(mean_data, std_data, first_filter=False)
            filt_ax2 = ax[letters[3]]
            filtered_data2.plot(
                ax=filt_ax2,
                x='time',
                y='x',
                vmin=v_min,
                vmax=v_max,
                cmap='turbo',
                add_colorbar=True
            )
            filt_ax2.set_title("Second Filter", fontsize=16)

        # Format labels across all active subplots
        unit_str = ds[diagnostic_to_plot].attrs.get('units', diagnostic_to_plot)
        for ltr in letters[:len(ax)]:
            curr_ax = ax[ltr]
            curr_ax.set_xlabel(f'Time ({ds.attrs.get("time_units", "s")})', fontsize=14)
            curr_ax.set_ylabel(f'x ({ds.attrs.get("x_units", "cm")})', fontsize=14)

        diagnostic_name = ds[diagnostic_to_plot].attrs.get("long_name", diagnostic_to_plot)
        probe_port = ds['port'].isel(probe=probe).item() if 'port' in ds else "Unknown"
        probe_z = ds['z'].isel(probe=probe).item() if 'z' in ds else "Unknown"

        fig.suptitle(f"{run_identifier}\n{diagnostic_name} (z: {probe_z} m)", fontsize=20)
        plt.tight_layout()
        if save_plots:
            plot_name = run_identifier + f"probe_{probe}_{diagnostic_to_plot}.png"
            if plot_std:
                plot_name += f"_std"
            if filt_data:
                plot_name += f"_filt"
            plt.savefig(figure_folder + plot_name, bbox_inches='tight')
            print('Figure saved to', figure_folder + plot_name)
        if show_plot:
            plt.show()
        plt.close()


    else:
        for shot in shots_to_plot:

            fig, ax, letters = build_subplots(layout=layout, fig_width=6.0, fig_height=4.5)
            data = ds[diagnostic_to_plot].sel(shot=shot, probe = probe)
            # Dynamic colorbar bounds
            vals = data.values
            v_min, v_max = np.nanpercentile(vals[~np.isnan(vals)], [2, 98]) if np.any(~np.isnan(vals)) else (0, 1)

            # Force lower bound to 0 for non-negative physical diagnostics
            negative_diagnostics = ['isat', 'v_f', 'vf']

            # Check if current diagnostic belongs to negative quantities
            is_neg = any(tag in diagnostic_to_plot for tag in negative_diagnostics)

            if np.any(~np.isnan(vals)):
                if not is_neg:
                    # Positive diagnostics (T_e, n_e): isolate edge (0 to 75th percentile)
                    v_min, v_max = np.nanpercentile(vals[~np.isnan(vals)], [0, 75])
                    v_min = max(0.0, v_min)
                else:
                    # Negative diagnostics (I_sat, V_f): isolate edge (50th to 98th percentile)
                    v_min, v_max = np.nanpercentile(vals[~np.isnan(vals)], [30, 98])
                    v_max = min(0.0, v_max)
            else:
                # Safe fallbacks if array contains only NaNs
                v_min, v_max = (-0.03, 0.0) if is_neg else (0.0, 1.0)

            # Guard against v_min >= v_max (e.g., flat signals or identical percentiles)
            if v_min >= v_max:
                v_max = v_min + 1e-5

            # Plot Mean values across shots
            data_ax = ax[letters[0]]
            cp_mean = data.plot(
                ax=data_ax,
                x='time',
                y='x',
                vmin=v_min,
                vmax=v_max,
                cmap='turbo',
                add_colorbar=True
            )
            data_ax.set_title(f"Shot = {shot}", fontsize=16)
            if save_plots:
                plot_name = run_identifier + f'probe_{probe}_shot{shot}_{diagnostic_to_plot}.png'
                plt.savefig(figure_folder + plot_name, bbox_inches='tight')
                print('Figure saved to', figure_folder + plot_name)
            if show_plot:
                plt.show()
            plt.close()


def contour_subplots(datasets, diagnostic_to_plot):
    """
    Generate a grid of contour subplots comparing multiple xarray datasets.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        List of dataset objects to evaluate and plot.
    diagnostic_to_plot : str
        Variable key inside datasets to plot.
    """
    datasets = list(datasets)
    num_runs = len(datasets)

    divide = ask_yes_or_no("Divide first two chosen datasets? (y/n) ")
    total_plots = num_runs + (1 if divide else 0)

    # Determine layout rows and columns for build_subplots
    n_cols = math.ceil(math.sqrt(total_plots))
    n_rows = math.ceil(total_plots / n_cols)

    # Construct inner list representation for build_subplots (e.g. [[2], [1]])
    layout = []
    remaining_plots = total_plots
    for _ in range(n_rows):
        row_count = min(n_cols, remaining_plots)
        layout.append([row_count])
        remaining_plots -= row_count

    fig, ax, letters = build_subplots(layout=layout, fig_width=5.5, fig_height=4.0)

    # Global dynamic colorbar bounds (Across ALL compared datasets)
    all_means = [ds[diagnostic_to_plot].sel(probe=0).mean('shot') for ds in datasets]
    all_vals = np.concatenate([m.values.ravel() for m in all_means])
    valid_vals = all_vals[~np.isnan(all_vals)]

    if len(valid_vals) > 0:
        v_min, v_max = np.nanpercentile(valid_vals, [2, 98])
    else:
        v_min, v_max = 0, 1

    # Force lower bound to 0 for non-negative diagnostics
    if any(tag in diagnostic_to_plot for tag in ['t_e', 'n_e', 'n_i', 'isat']) or v_min > 0:
        v_min = 0.0

    # Plot individual datasets
    for i, ds in enumerate(datasets):
        curr_ax = ax[letters[i]]
        mean_data = all_means[i]

        cp = mean_data.plot(
            ax=curr_ax,
            x='time',
            y='x',
            vmin=v_min,
            vmax=v_max,
            cmap='turbo',
            add_colorbar=True
        )

        run_id = f_run_identifier(ds=ds)
        diagnostic_name = ds[diagnostic_to_plot].attrs.get("long_name", diagnostic_to_plot)
        probe_port = ds['port'].isel(probe=0).item() if 'port' in ds else "Unknown"

        curr_ax.set_title(f"{run_id}\n{diagnostic_name} (Port: {probe_port})", fontsize=12)
        curr_ax.set_xlabel(f'Time ({ds.attrs.get("time_units", "s")})', fontsize=10)
        curr_ax.set_ylabel(f'x ({ds.attrs.get("x_units", "cm")})', fontsize=10)

    # Optional Division Plot (Ratio of Dataset 1 / Dataset 2)
    if divide and num_runs >= 2:
        div_ax = ax[letters[num_runs]]
        data_1 = all_means[0]
        data_2 = all_means[1]
        ratio = data_1 / data_2

        # Dynamic bounds for ratio plot (clipping extreme division artifacts)
        r_vals = ratio.values.ravel()
        valid_r = r_vals[~np.isnan(r_vals) & ~np.isinf(r_vals)]
        r_min, r_max = np.nanpercentile(valid_r, [5, 95]) if len(valid_r) > 0 else (0.5, 2.0)

        cp_div = ratio.plot(
            ax=div_ax,
            x='time',
            y='x',
            vmin=r_min,
            vmax=r_max,
            cmap='turbo',
            add_colorbar=True
        )

        run_id_a = f_run_identifier(ds=datasets[0])
        run_id_b = f_run_identifier(ds=datasets[1])

        div_ax.set_title(f"Ratio: {run_id_a} / {run_id_b}", fontsize=12)
        div_ax.set_xlabel(f'Time ({datasets[0].attrs.get("time_units", "s")})', fontsize=10)
        div_ax.set_ylabel(f'x ({datasets[0].attrs.get("x_units", "cm")})', fontsize=10)

    plt.tight_layout()
    plt.show()

def show_steady_state(ds, probe, run_identifier):
    """
    Purely visual function to display the currently saved steady state bounds.

    Parameters
    ----------
    ds: xarray.Dataset
        Data containing the time series the user wants to plot.
    probe: int
        Probe number in ds where the time series was saved.
    run_identifier: str
        Unique run identifier for the selected dataset

    """

    # Safely extract the saved bounds
    start = ds.attrs.get(f'steady state start probe {probe}')
    end = ds.attrs.get(f'steady state end probe {probe}')

    if start is None or end is None:
        print(f"No steady state bounds saved for probe {probe} yet.")
        return

    # Generate the plot
    fig, axes, _ = plot_time_series(ds, probe, run_identifier, return_range = False)

    temp_ax, dens_ax = axes

    # Draw the saved lines
    temp_ax.axvline(x=start, color='k', linestyle='--', linewidth=2, label='SS Start')
    dens_ax.axvline(x=start, color='k', linestyle='--', linewidth=2, label='SS Start')

    temp_ax.axvline(x=end, color='k', linestyle=':', linewidth=2, label='SS End')
    dens_ax.axvline(x=end, color='k', linestyle=':', linewidth=2, label='SS End')

    # Update legend
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=14)

    plt.show()


def plot_time_series(ds, probe, run_identifier, return_range = False):
    """

    Parameters
    ----------
    ds: xarray.Dataset
        Data containing the time series the user wants to plot.
    probe: int
        Probe number in ds where the time series was saved.
    run_identifier: str
        Unique run identifier for the selected dataset
    return_range: bool
        If true, the function returns 3 arguments with the last argument being a list of all the x-positions plotted

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure object containing the time series plot.
    axes: list
        List of axes objects containing the time series plot. axes[0] yields the temperature axis and
        axes[1] yields the density axis and axes[2].
    range_tot: list
        Contains all the x values used to plot the temperature and the density
    """

    # Average, get the standard deviation, and filter the temperature across shots
    mean_data = ds['t_e'].sel(probe=probe).mean('shot')
    std_data = ds['t_e'].sel(probe=probe).std('shot')
    filtered_data = filter_data(mean_data, std_data)

    min_x = int(min(filtered_data['x'].values))
    max_x = int(max(filtered_data['x'].values))


    # Starting from x = 0, list all values up to the maximum x value in increments of 5
    range_up = list(range(0, max_x + 1, 5))

    # Starting from x = -5 list all values from x = -5 to the minimum x value in increments of 5
    if min_x <= - 5:
        range_down = list(range(-5, min_x + 1, -5))
    else:
        range_down = []

    # Combine the two up and down lists
    range_tot = range_down + range_up
    range_tot = sorted(range_tot)

    # Build the figure and axis objects
    layout = [[2]]
    fig, ax, letters = build_subplots(layout = layout)
    temp_letter, dens_letter = letters
    temp_axis = ax[temp_letter]
    dens_axis = ax[dens_letter]

    # For each x in the selected x's, plot the temperature and density on their respective axes
    for x_test in range_tot:
        test_data_t_e = filtered_data.sel(x=x_test, y=0)
        test_data_n_e = ds['n_e'].sel(probe=probe, x=x_test, y=0).mean('shot')
        num_nan = test_data_t_e.isnull().sum().item()
        if num_nan < len(test_data_t_e) / 2:
            test_data_t_e.plot(ax=temp_axis,
                               x='time',
                               marker='o',
                               label=f'x = {x_test}',
                               linestyle='None')
            test_data_n_e.plot(ax=dens_axis,
                               x='time',
                               marker='o',
                               label=f'x = {x_test}',
                               linestyle='None')

    # Temperature axis formatting
    temp_axis.set_xlabel(f'Time ({ds.attrs.get("time_units")})', fontsize=28)
    temp_axis.set_ylabel(rf'$T_{{e}}$ ({ds["t_e"].attrs.get("units", "t_e")})', fontsize=28)
    temp_axis.set_title(f'{ds["t_e"].attrs.get("long_name", "t_e")}', fontsize=28)
    temp_axis.tick_params(labelsize=20)

    # Density axis formatting
    dens_axis.set_xlabel(f'Time ({ds.attrs.get("time_units")})', fontsize=28)
    dens_axis.set_ylabel(rf'$n_{{e}}$ (${ds["n_e"].attrs.get("units", "n_e")}$)', fontsize=28)
    dens_axis.set_title(f'{ds["n_e"].attrs.get("long_name", "n_e")}', fontsize=28)
    dens_axis.tick_params(labelsize=24)
    dens_axis.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    dens_axis.yaxis.get_offset_text().set_fontsize(24)

    # Extract legend items from temp_axis (since both axes plot the same x labels)
    handles, labels = temp_axis.get_legend_handles_labels()

    # Determine columns so legend entries span horizontally across the bottom
    num_cols = min(len(labels), 3) if labels else 1

    # Place one single legend centered underneath the figure
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),
        ncol=num_cols,
        fontsize=16,
        frameon=True
    )
    plt.tight_layout()

    # Get the z position of the probe
    probe_z = ds['z'].isel(probe=probe).item()

    # Set super title
    fig.suptitle(f"Time Series: {run_identifier},  z: {probe_z} m", fontsize=28)
    fig.subplots_adjust(top=0.91, bottom=0.14)


    axes = [temp_axis, dens_axis]

    if return_range:
        return fig, axes, range_tot

    return fig, axes, None

def dim_num_params(filename, ds):
    """
    An easy place to gather values useful in dimensionless number calculations.

    Parameters
    ----------
    filename: string
        Save name of the .nc file in the langmuir folder. Used to get the magnetic field if it is not in the dataset
        attributes

    ds: xarray.Dataset
        Of which the user wants to look at the dimensionless numbers for
    Returns
    -------
    ion_mass: Quantity
        Mass of the ion of interest in kg.
    z_eff: int
        Ionization of the neutral particle -- unitless.
    e_charge: Quantity
        Charge of an electron in C
    b_field: Quantity
        Magnetic field for the experiment in T
    """

    particle_name = ds.attrs['ion_type']
    part = Particle(particle_name)

    e_charge = c.e.si

    ion_mass = part.mass

    z_eff = part.charge_number


    if 'B-field' in ds.attrs:
        b_field_str = ds.attrs['B-field']
        b_field = u.Quantity(b_field_str)

    else:
        split_filename = filename.split('_')
        b_field = 0 * u.kG
        for snippet in split_filename:
            if 'kG' in snippet:
                b_field = float(snippet.split('kG')[0]) * u.kG

    b_field = b_field.to(u.T)

    return ion_mass.to(u.kg), z_eff, e_charge.to(u.C), b_field

def compute_dimesionless_plots(ds, ion_mass, z_eff, e_charge, b_field, a, L, run_identifier):
    """
    Parameters
    ----------
    ds: xarray.Dataset
        Full dataset corresponing to the experiment you want to get the dimensionless numbers for
    ion_mass: astropy.units.Quantity
        Mass of the ion of interest in kg
    z_eff: int
        Ionization value of the ion particle
    e_charge: astropy.units.Quantity
        Charge of the electron
    b_field: astropy.units.Quantity
        Magnetic field of LAPD in T
    a: astropy.units.Quantity
        LAPD core radius in m
    L: astropy.units.Quantity
        LAPD length in m
    run_identifier: str
        Identifies the experiment by experiment name and run number

    Returns
    -------
    probe_dict: dict
        Dictionary with the values needed to plot the dimensionless numbers sorted by probe

    """

    probe_dict = {}
    for probe in range(ds.sizes['probe']):

        # Find the steady state values for each probe
        min_time = ds.attrs[f'steady state start probe {probe}']
        max_time = ds.attrs[f'steady state end probe {probe}']

        probe_dict[probe] = {}

        # Process data
        t_e_to_plot_vals, _, t_x_vals = process_variable_data(ds, probe, var_name='t_e')
        nu_ei_to_plot_vals, _, nu_ei_x_vals = process_variable_data(ds, probe, var_name='nu_ei')

        # Safely extract attributes/units
        t_e_units = getattr(ds['t_e'], 'units', ds['t_e'].attrs.get('units', 'eV'))
        nu_units = getattr(ds['nu_ei'], 'units', ds['nu_ei'].attrs.get('units', '1/s'))

        # Attach units to the values
        t_e_w_units = t_e_to_plot_vals * u.Unit(t_e_units)
        nu_ei_w_units = nu_ei_to_plot_vals * u.Unit(nu_units)

        # Sound speed calculation for rho* -> sqrt(T_e/m_i) or other variations as outlined in the docstring of the
        # function
        c_s = sound_speed_calculation(t_e=t_e_w_units, ion_mass=ion_mass, z = z_eff)

        # Dimensionless Parameters Calculation
        rhostar = (ion_mass * c_s / (e_charge * b_field * a)).to(u.dimensionless_unscaled)
        nu_eff = (2 * np.pi * c_s / (nu_ei_w_units * L)).to(u.dimensionless_unscaled)

        # Assign keys to the values we want to save in the dictionary
        probe_dict[probe]['rhostar'] = rhostar
        probe_dict[probe]['nu_eff'] = nu_eff
        probe_dict[probe]['x_vals'] = t_x_vals
        probe_dict[probe]['run identifier'] = run_identifier

    return probe_dict



def shortened_exp_name(long_exp_name):
    """
    
    Parameters
    ----------
    long_exp_name: str 
        Full length name of the experiment from exp_params_dict such as January_2024

    Returns
    -------
    exp_name: str
        Shortened experiment name with 3 letters indicating the month followed by the year

    """""

    exp_name_split = long_exp_name.split('_')
    month = exp_name_split[0][: 3]
    year = exp_name_split[1]
    exp_name = month + year
    return exp_name





def f_run_identifier(ds = None, filename = ''):
    """
    Parameters
    ----------
    filename: string
        Raw filename coming from how the .nc file is saved
    ds: xarray.Dataset
        Data and, most importantly for this function, attributes corresponding to the user's selected experiment run

    Returns
    -------
    run_identifier: String
        In the format expname_runnumber_iontype
    """
    run_identifier = ""

    if ds is not None:
        # Create the run identifier from the dataset attributes
        if 'Exp name' in ds.attrs:
            exp_name = shortened_exp_name(ds.attrs['Exp name'])
            run_identifier = exp_name + ', ' +  ds.attrs['Run number'] + ', ' + ds.attrs['ion_type']

    else:
        # Create run_identifier from the filename given
        if "Mar" in filename:
            run_identifier = "Mar 22 run " + filename.split("_")[1]
        elif 'kG' in filename:
            try:
                addition = int(filename.split("_")[1])
            except ValueError:
                addition = filename.split("_")[0]

            run_identifier = "Jan 24 run " + str(addition)

            if "H2" in filename:
                run_identifier = run_identifier + " H+"
            else:
                run_identifier = run_identifier + " He+"

        else:
            run_identifier = "filename not yet supported"

    return run_identifier


def generic_run_identifiers(lang_datasets):
    """
    Parameters
    ----------
    lang_datasets: list
        List of xarray.Dataset objects corresponding to what the user would like to plot

    Returns
    -------
    run_identifiers: list
        List of strings containing generic run identifiers corresponding to the order in which the original
        lang_datasets list was passed in. (He 1, H 1, He 2, etc.)
    """

    # Deal with PyCharm's annoying error handling
    lang_datasets = list(lang_datasets)
    he_num = 1
    h_num = 1
    run_identifiers = []

    # This dictionary will store the mapping: {'raw_run_id': 'generic_run_id'}
    id_mapping = {}

    for ds in lang_datasets:
        raw_identifier = f_run_identifier(ds)

        # Check to see if the dataset has already been assigned an identifier and if not create one
        if raw_identifier not in id_mapping:
            if 'He' in raw_identifier:
                gen_identifier = f'He {he_num}'
                he_num += 1
            elif 'H' in raw_identifier:
                gen_identifier = f'H {h_num}'
                h_num += 1
            else:
                gen_identifier = 'Unknown'
            id_mapping[raw_identifier] = gen_identifier
        else:
            gen_identifier = id_mapping[raw_identifier]

        run_identifiers.append(gen_identifier)

    return run_identifiers


def determine_colors(datasets, one_probe =True):

    """
    Parameters
    ----------
    datasets: list
        list of user selected xarray.Datasets to assign colors and markers to
    one_probe: boolean
        Indicates whether to plot one probe from each dataset or all probes from each dataset

    Returns
    -------
    clor: list
        List of the colors corresponding to each dataset. Ordered by the order in the original dataset list
    mark: list
        List of the marks corresponding to each dataset. Ordered by the order in the original dataset list
    """

    # Supress PyCharm annoying warnings and ensure we don't have a generator object
    datasets = list(datasets)

    list_run_identifiers = []
    probe_num_list = []

    # Create run identifiers and look for the number of probes in each
    for dataset in datasets:
        run_identifier = f_run_identifier(ds = dataset)
        list_run_identifiers.append(run_identifier)
        probe_num_list.append(len(dataset.probe))

    if one_probe:
        # Create a list of 1s in the probe list equal in length to the number of datasets to evaluate
        probes_in_file = [1] * len(datasets)
        clor, mark = generate_file_colors(probes_in_file, list_run_identifiers)
    else:
        # Repeat each dataset's run identifier by the exact number of probes in that dataset
        n_l_run_identifiers = []
        for run_id, num_probes in zip(list_run_identifiers, probe_num_list):
            n_l_run_identifiers.extend([run_id] * num_probes)

        clor, mark = generate_file_colors(probe_num_list, n_l_run_identifiers)

    return clor, mark

def generate_file_colors(probes_per_file, l_run_identifiers):
    """
    Generate colors for multiple files.
    - Hydrogen (H): Red spectrum
    - Helium (He): Green spectrum
    - Mar22: Blue spectrum
    - Nov22: Yellow spectrum
    Probes within files vary dramatically in brightness and saturation.

    Parameters
    ----------
    probes_per_file: list
        List where the length corresponds to the number of datasets to evaluate and each cell corresponds to how many
        probes are in each dataset
    l_run_identifiers: list
        List of strings corresponding to which run is being looked at. Run identifiers are repeated for multiple probes
        in the dataset

    Returns
    -------
    colors: list
        List of the colors corresponding to each dataset. Ordered by the order in the original dataset list and
        repeated based on how many probes are in each dataset
    markers: list
        List of the marks corresponding to each dataset. Ordered by the order in the original dataset list and
        repeated based on how many probes are in each dataset
    """

    # To deal with PyCharm's annoying warnings and ensure we are dealing with lists
    probes_per_file = list(probes_per_file)
    l_run_identifiers = list(l_run_identifiers)

    # Master pool of distinct Matplotlib markers
    master_markers = ['^', 'v', '<', '>', 'D', 'd', 'o', 's', 'P', '*', 'X', 'h']

    # Split the master list in half so He and H have completely disjoint marker sets.
    # This guarantees a Helium marker will NEVER be used for a Hydrogen plasma.
    half = len(master_markers) // 2
    he_markers = master_markers[:half]  # First half assigned to Helium  : ['^', 'v', '<', '>', 'D', 'd']
    h_markers = master_markers[half:]  # Second half assigned to Hydrogen: ['o', 's', 'P', '*', 'X', 'h']

    # Check to make sure we have the correct number of probes corresponding to run identifiers
    if sum(probes_per_file) != len(l_run_identifiers):
        raise ValueError("Sum of probes_per_file must match length of l_run_identifiers")

    colors = []
    markers = []

    # Categorize and count files. Datasets from Jan 2024 and later will be categorized by Hydrogen and Helium
    # and earlier experiments that were entirely Helium runs will be categorized with their run
    file_categories = []
    file_species = []
    start_idx = 0
    for n in probes_per_file:
        run_id = l_run_identifiers[start_idx]
        # Check identifiers in order of priority
        if "Mar2022" in run_id:
            file_categories.append("Mar22")
        elif "Nov2022" in run_id:
            file_categories.append("Nov22")
        elif "He+" in run_id or "He-4+" in run_id:
            file_categories.append("He")
        else:
            file_categories.append("H")

        if "He+" in run_id or "He-4+" in run_id or "Mar2022" in run_id or "Nov2022" in run_id:
            file_species.append("He")
        else:
            file_species.append("H")

        start_idx += n

    # Tally up the total number of files in each category
    counts = {
        "H": file_categories.count("H"),
        "He": file_categories.count("He"),
        "Mar22": file_categories.count("Mar22"),
        "Nov22": file_categories.count("Nov22")
    }

    # Keep track of which file we are on for each category
    indices = {"H": 0, "He": 0, "Mar22": 0, "Nov22": 0}

    # Start assigning colors and markers. Start index refers to the index in the run_identifiers list where this
    # probe set started
    start_idx = 0
    for file_idx, num_probes in enumerate(probes_per_file):
        category = file_categories[file_idx]
        species = file_species[file_idx]
        total_in_cat = counts[category]
        current_idx = indices[category]

        # Select target marker pool based on species
        target_markers = he_markers if species == "He" else h_markers
        pool_size = len(target_markers)

        # Print message if probes exceed available unique markers for this gas species
        if num_probes > pool_size:
            run_name = l_run_identifiers[start_idx]

            # Calculate how many of each marker will be plotted
            marker_counts = {}
            for i, m in enumerate(target_markers):
                count = (num_probes // pool_size) + (1 if i < (num_probes % pool_size) else 0)
                marker_counts[m] = count

            counts_str = ", ".join(f"'{m}': {count}" for m, count in marker_counts.items())
            print(
                f"Notice: File '{run_name}' has {num_probes} probes, exceeding the {pool_size} "
                f"unique markers available. \n Marker plot counts -> {counts_str}"
            )

        # Assign base hue depending on the category
        if category == "Mar22":
            # Cyan/Teal range: 0.45 to 0.55
            if total_in_cat > 1:
                base_hue = 0.45 + 0.10 * (current_idx / (total_in_cat - 1))
            else:
                base_hue = 0.50

        elif category == "Nov22":
            # Yellow range: 0.10 (Golden-Orange) to 0.18 (Bright Lemon)
            if total_in_cat > 1:
                base_hue = 0.10 + 0.08 * (current_idx / (total_in_cat - 1))
            else:
                base_hue = 0.15  # Standard Yellow


        elif category == "He":
            # Deep Blue range: 0.60 (True Blue) to 0.70 (Deep Indigo)
            if total_in_cat > 1:
                base_hue = 0.60 + 0.10 * (current_idx / (total_in_cat - 1))
            else:
                base_hue = 0.65


        else:  # "H"
            # Pink range: 0.82 (Magenta) to 0.92 (Hot Pink)
            if total_in_cat > 1:
                base_hue = 0.82 + 0.10 * (current_idx / (total_in_cat - 1))
            else:
                base_hue = 0.87

        # Increment the index for whichever category we just processed
        indices[category] += 1

        # Calculate Saturation and value and determine the marker for each probe
        for p in range(num_probes):
            if num_probes > 1:
                fraction = p / (num_probes - 1)
                # Saturation goes from 1.0 (intense) down to 0.45 (washed out)
                saturation = 1.0 - 0.55 * fraction
                # Value (Brightness) goes from 0.45 (dark) up to 0.95 (bright)
                value = 0.45 + 0.50 * fraction
            else:
                saturation = 0.8
                value = 0.8

            # Convert hue saturation and value numbers into normalized RGB float
            rgb = colorsys.hsv_to_rgb(base_hue, saturation, value)

            # Scale floats to 8-bit integers and format into standard Hex format
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )

            colors.append(hex_color)

            # Assign marker (cycling through the species pool)
            marker = target_markers[p % pool_size]
            markers.append(marker)

        start_idx += num_probes

    return colors, markers

def build_subplots(layout, fig_width = 6.4, fig_height = 4.8, **kwargs):
    """

    Parameters
    ----------
    layout: list
        List of lists with each list containing an integer indicating the number of subplots in each row.
    fig_width: float
        Width of ONE subplot figure in inches.
    fig_height: float
        Height of ONE subplot figure in inches.
    **kwargs: additional keyword arguments
        Passed directly to plt.subplot_mosaic (e.g., sharex=True, sharey=True).

    Returns
    -------
    fig: figure
        Figure object with the associated subplots
    ax: dict
        Dictionary of axes mapped to letters
    letters: list
        List of strings with all the letters needed to place figures within the subplot
    """

    # Deal with PyCharm's annoying warnings
    layout = list(layout)

    # Find the maximum number of plots in a row to determine the width of the subplot diagram
    max_plots = max(row[0] for row in layout)
    total_cols = max_plots * 2

    # Find the size of the total figure
    total_fig_width = fig_width * 1.25 * max_plots
    total_fig_height = fig_height * 1.25 * len(layout)

    mosaic_layout = []
    letters = []
    current_ascii = ord('A')
    for row in layout:
        num_plots = row[0]
        row_mosaic = []

        # How much space needs to be padded with the '.'s
        cols_used = num_plots * 2
        padding = (total_cols - cols_used) // 2

        # Add padding to the left side (One '.' on each side for each set of columns not used)
        row_mosaic.extend(['.'] * padding)

        # Add in the letters
        for _ in range(num_plots):
            # Convert the ascii number back into a letter
            letter = chr(current_ascii)
            letters.append(letter)
            # Add those letters to the row
            row_mosaic.extend([letter,letter])
            # Move on to the next letter
            current_ascii += 1


        # Add padding to the right side
        row_mosaic.extend(['.'] * padding)

        # Add the completed row to the mosaic
        mosaic_layout.append(row_mosaic)

    # Create the figure and axis objects
    fig, ax = plt.subplot_mosaic(mosaic_layout, figsize=(total_fig_width, total_fig_height),dpi = 100, **kwargs)

    return fig, ax, letters







