from colorsys import rgb_to_hsv
import json
from lapd_plasma_analysis.obtain_plots.plot_from_netcdf import *
from lapd_plasma_analysis.obtain_plots.plot_from_netcdf_helpers.radial_plots import *
from numpy import *
import matplotlib.ticker as ticker
from lapd_plasma_analysis.file_access import *
from astropy import units as u
import matplotlib.colors as mcolors
from lapd_plasma_analysis.obtain_plots.xarray_plots import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lapd_plasma_analysis.fluctuations.fourier import *
from lapd_plasma_analysis.obtain_plots.plot_from_netcdf_helpers.process_temp_dens_data import *
from lapd_plasma_analysis.mach.luke_main_mach_helpers import *



def temp_dens_spectrogram(lang_ds, fluct_ds, quantity, pathname, figure_folder, dataset_color=None, dataset_mark=None,
                          make_presentable=True, fluc_probez=None, lang_z_idx=None, save_plots = False, see_plots = False,
                          run_identifier = ""):

    q_shortname = get_quantity_short_name(quantity)

    plt.rcParams.update({
        'font.size': 30,  # Global font size
        'axes.labelsize': 40,  # x and y labels
        'axes.titlesize': 30,  # Title size
        'xtick.labelsize': 40,  # x-axis tick labels
        'ytick.labelsize': 40,  # y-axis tick labels
        'legend.fontsize': 30,  # Legend text
        'legend.title_fontsize': 30,  # Legend title (if any)
        'axes.formatter.use_mathtext': True, # Use LaTeX style math font
        'lines.linewidth': 3,  # Global line width (default is 1.5)
        'lines.markersize': 8,  # Global marker size (default is 6.0)
        'errorbar.capsize': 5  # Global error bar cap width (default is 0.0)
    })

    if make_presentable:
        plt.rcParams.update({
                'figure.facecolor': 'none',
                'axes.facecolor': 'none',
                'savefig.transparent': True
            })

    default_fig_height, default_fig_width = default_fig_params()
    dens_region_str, _, _, temp_region_str, _, _ = xarray_gradient_strings()



    layout = [[1],
              [1],
              [1]]

    fig, ax, letters = build_subplots(layout, 2.25 * default_fig_width, 2 * default_fig_height,
                                      sharex=True)

    # Apply layout adjustments BEFORE carving out colorbar spaces
    fig.tight_layout()

    temp_axis = ax[letters[0]]
    dens_axis = ax[letters[1]]
    spec_axis = ax[letters[2]]

    # Carve out invisible colorbar space for Temperature
    div_temp = make_axes_locatable(temp_axis)
    cax_temp = div_temp.append_axes("right", size="5%", pad=0.1)
    cax_temp.set_visible(False)

    # Carve out invisible colorbar space for Density
    div_dens = make_axes_locatable(dens_axis)
    cax_dens = div_dens.append_axes("right", size="5%", pad=0.1)
    cax_dens.set_visible(False)

    # Pre-allocate the exact same space for the Spectrogram
    div_spec = make_axes_locatable(spec_axis)
    cax_spec = div_spec.append_axes("right", size="5%", pad=0.1)

    # Build inputs for the spectrogram function
    lang_probe_ss_start = round(lang_ds.attrs[f'steady state start probe {lang_z_idx}'])
    lang_probe_ss_end = round(lang_ds.attrs[f'steady state end probe {lang_z_idx}'])
    bin_vals = ast.literal_eval(f'({lang_probe_ss_start}, {lang_probe_ss_end})')

    x_chosen = None
    shot = None
    file_str = pathname.split('/')[-1]
    filename = file_str.split('.')[0]

    if dens_region_str in lang_ds.attrs.keys() and temp_region_str in lang_ds.attrs.keys():
        radial_axes = [temp_axis, dens_axis]
        gradient_regions_colors, gradient_regions_dict = (
            build_radial_plot([lang_ds], pathnames=[pathname],
                              figure_folder=figure_folder,
                              make_presentable=make_presentable, see_temp_and_dens_plots=False, lines=False, shaded=True,
                              plot_final_fits=False, save_plots=False, clor=[dataset_color], mark=[dataset_mark],
                              from_main=False, one_probe=True, axes=radial_axes, dens_bottomx=False, temp_bottomx=False,
                              run_identifiers = run_identifier))
    else:
        gradient_regions_colors = {}
        gradient_regions_dict = {}

    grads_per_probe = gradient_regions_dict.get(lang_z_idx, None)
    if grads_per_probe:
        temp_x = np.round(np.mean(grads_per_probe['full_temp_grads'][0]))
        dens_x = np.round(np.mean(grads_per_probe['full_density_grads'][1]))

    else:
        temp_x = np.nan
        dens_x = np.nan

    get_radial_spectrogram(fluct_ds[quantity].sel(z=fluc_probez), x=x_chosen,
                           bin=bin_vals, shot=shot, z=fluc_probez, plot=True,
                           axis=spec_axis, filename=filename,
                           gradient_regions=gradient_regions_dict.get(lang_z_idx, None), # Added .get() for safety
                           return_x=False,
                           cax=cax_spec,
                           make_presentable=make_presentable,
                           gradient_region_colors=gradient_regions_colors,
                           q_shortname=q_shortname)

    temp_axis.set_ylabel(r'$T_e$ [eV]', fontsize = 40, rotation= 90, labelpad=15)
    dens_axis.set_ylabel(r'$n_e$ [$\text{m}^{-3}$]', fontsize = 40, rotation = 90, labelpad=15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8, bottom=0.15, left=0.25, right=0.88, hspace=0.3)

    if save_plots:
        ensure_directory(figure_folder + 'radial_spectrograms/')
        figure_name = quantity + filename.split('_')[0] + filename.split('_')[1]
        fig.savefig(figure_folder + 'radial_spectrograms/' + figure_name + '.svg', bbox_inches='tight')
        print('Plot saved to ' + figure_folder + f'radial_spectrograms/' + figure_name + '.svg')

    if see_plots:
        fig.show()
        plt.close()
    else:
        plt.close()

    return temp_x, dens_x

def dimless_plots(fluc_datasets, filenames, figure_folder, langmuir_nc_folder, dataset_colors, dataset_marks, a, L,
                  make_presentable = False, save_plots = False, see_plots = False, run_identifiers = ""):
    plt.rcParams.update({
        'font.size': 30,  # Global font size
        'axes.labelsize': 30,  # x and y labels
        'axes.titlesize': 30,  # Title size
        'xtick.labelsize': 30,  # x-axis tick labels
        'ytick.labelsize': 30,  # y-axis tick labels
        'legend.fontsize': 30,  # Legend text
        'legend.title_fontsize': 30,  # Legend title (if any)
        'axes.formatter.use_mathtext': True,  # Use LaTeX style math font
        'lines.linewidth': 3,  # Global line width (default is 1.5)
        'lines.markersize': 8,  # Global marker size (default is 6.0)
        'errorbar.capsize': 5  # Global error bar cap width (default is 0.0)
    })

    if make_presentable:
        plt.rcParams.update({
            'figure.facecolor': 'none',
            'axes.facecolor': 'none',
            'savefig.transparent': True
        })
    layout = [[2]]
    default_fig_height, default_fig_width = default_fig_params()

    fig, ax, letters = build_subplots(layout, 1.5 * default_fig_width, 1.5 * default_fig_height)
    rhos_axis = ax[letters[0]]
    nu_e_axis = ax[letters[1]]
    i = 0
    j = 0
    run_nums = ''
    for fluct_ds in fluc_datasets:
        fluct_probes = [fluct_ds.coords["z"].values[np.argmax(fluct_ds.coords["z"].values)]]

        if filenames[i] + '_tanh.nc' in os.listdir(langmuir_nc_folder):
            lang_ds = xr.open_dataset(langmuir_nc_folder + filenames[i] + '_tanh.nc')
            lang_ds_zs = lang_ds['z'].values
            lang_z_idx = np.abs(lang_ds_zs - fluct_probes).argmin()
            lang_run_identifier = run_identifiers[j]
            dataset_clor = dataset_colors[j]
            dataset_mark = dataset_marks[j]



            generate_rhostar_values(lang_ds, lang_z_idx, dataset_clor, dataset_mark, a,
                                    axes = rhos_axis, run_identifier = lang_run_identifier)

            generate_nu_eff_values(lang_ds, lang_z_idx, dataset_clor,dataset_mark,  L,
                                    axes=nu_e_axis, run_identifier=lang_run_identifier)

            run_nums = run_nums + f'{filenames[i].split("_")[0]}{filenames[i].split("_")[1]}'
            j += 1
        i+=1


    rhos_axis.set_ylabel(r'$\rho^*$', rotation=0, labelpad=35)
    rhos_axis.set_xlabel(r'x [cm]')
    nu_e_axis.set_ylabel(r'$\nu_{eff}$', rotation=0, labelpad=45)
    nu_e_axis.set_xlabel(r'x [cm]')

    handles, labels = rhos_axis.get_legend_handles_labels()

    if make_presentable:
        formatter_rhos = ticker.ScalarFormatter(useMathText=True)
        formatter_rhos.set_powerlimits((0, 0))
        rhos_axis.yaxis.set_major_formatter(formatter_rhos)
        rhos_axis.yaxis.get_offset_text().set_visible(False)

        # Manually calculate exponent based on maximum y-data limits
        y_max_rho = max(abs(tick) for tick in rhos_axis.get_yticks() if tick != 0)
        exp_rhos = int(np.floor(np.log10(y_max_rho))) if y_max_rho > 0 else 0

        if exp_rhos != 0:
            rhos_axis.text(0.5, 0.08, rf'$\times 10^{{{exp_rhos}}}$',
                           transform=rhos_axis.transAxes,
                           fontsize=30,  # Match global tick size
                           horizontalalignment='center',
                           verticalalignment='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='none', edgecolor='none', alpha=0.8),
                           zorder=5)  # Forces text to sit on top of transparent layers

        # 2. Handle nu_eff (Upper Center)
        formatter_nue = ticker.ScalarFormatter(useMathText=True)
        formatter_nue.set_powerlimits((0, 0))
        nu_e_axis.yaxis.set_major_formatter(formatter_nue)
        nu_e_axis.yaxis.get_offset_text().set_visible(False)

        # Manually calculate exponent based on maximum y-data limits
        y_max_nue = max(abs(tick) for tick in nu_e_axis.get_yticks() if tick != 0)
        exp_nue = int(np.floor(np.log10(y_max_nue))) if y_max_nue > 0 else 0

        if exp_nue != 0:
            nu_e_axis.text(0.5, 0.92, rf'$\times 10^{{{exp_nue}}}$',
                           transform=nu_e_axis.transAxes,
                           fontsize=30,  # Match global tick size
                           horizontalalignment='center',
                           verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='none', edgecolor='none', alpha=0.8),
                           zorder=5)
    fig.legend(handles, labels,
               loc='lower center',
               bbox_to_anchor=(0.5, .85),  # 0.5 is the exact center of the figure, 1.02 is just above the top
               ncol=len(labels))
    plt.tight_layout()
    fig.subplots_adjust(top=0.85, wspace = 0.5)

    if save_plots:
        ensure_directory(figure_folder + 'dimless_plots/')
        fig.savefig(figure_folder + 'dimless_plots/' + run_nums + '.svg')
        print('plot saved to ' + figure_folder + 'dimless_plots/' + run_nums + '.svg')
    if see_plots:
        plt.show()
        plt.close()


def psd_plot(datasets, filenames, quantity, temp_xs, dens_xs, figure_folder, make_presentable = False,
             langmuir_nc_folder = None,  save_plots = False, see_plots = False,
             dataset_colors = None, run_identifiers = ""):
    plt.rcParams.update({
        'font.size': 30,  # Global font size
        'axes.labelsize': 30,  # x and y labels
        'axes.titlesize': 30,  # Title size
        'xtick.labelsize': 30,  # x-axis tick labels
        'ytick.labelsize': 30,  # y-axis tick labels
        'legend.fontsize': 30,  # Legend text
        'legend.title_fontsize': 30,  # Legend title (if any)
        'axes.formatter.use_mathtext': True,  # Use LaTeX style math font
        'lines.linewidth': 3,  # Global line width (default is 1.5)
        'lines.markersize': 8,  # Global marker size (default is 6.0)
        'errorbar.capsize': 5  # Global error bar cap width (default is 0.0)
    })

    if make_presentable:
        plt.rcParams.update({
            'figure.facecolor': 'none',
            'axes.facecolor': 'none',
            'savefig.transparent': True
        })
    q_shortname = get_quantity_short_name(quantity)
    default_fig_height, default_fig_width = default_fig_params()

    fig_temp = plt.figure(figsize=(1.5 * default_fig_width,1.5 * default_fig_height))
    ax_temp = plt.subplot(111)
    fig_dens = plt.figure(figsize=(1.5 * default_fig_width,1.5 * default_fig_height))
    ax_dens = plt.subplot(111)

    i = 0
    j = 0
    run_nums =''
    for fluct_ds in datasets:
        lang_pathname = langmuir_nc_folder + filenames[i] + '_tanh.nc'
        fluct_probes = [fluct_ds.coords["z"].values[np.argmax(fluct_ds.coords["z"].values)]]

        if filenames[i] + '_tanh.nc' in os.listdir(langmuir_nc_folder):
            lang_ds = xr.open_dataset(lang_pathname)
            ri = run_identifiers[j]
            dataset_clor = dataset_colors[j]
            lang_ds_zs = lang_ds['z'].values
            lang_z_idx = np.abs(lang_ds_zs - fluct_probes).argmin()
            lang_probe_ss_start = round(lang_ds.attrs[f'steady state start probe {lang_z_idx}'])
            lang_probe_ss_end = round(lang_ds.attrs[f'steady state end probe {lang_z_idx}'])
            bin_vals = ast.literal_eval(f'({lang_probe_ss_start}, {lang_probe_ss_end})')
            dens_x = dens_xs[j]
            temp_x = temp_xs[j]
            q_units = fluct_ds[quantity].attrs.get('units')
            get_spectrum_from_data(fluct_ds[quantity].sel(z=fluct_probes[0]),
                                  x=dens_x,
                                  bin=bin_vals,
                                  shot=None,
                                  z=fluct_probes[0],
                                  plot=True,
                                  scaling="psd",
                                  axis= ax_dens,
                                  dataset_color = dataset_clor,
                                  make_presentable= make_presentable,
                                  run_identifier = ri +f' x={dens_x}',
                                  q_shortname = q_shortname)

            get_spectrum_from_data(fluct_ds[quantity].sel(z=fluct_probes[0]),
                                  x=temp_x,
                                  bin=bin_vals,
                                  shot=None,
                                  z=fluct_probes[0],
                                  plot=True,
                                  scaling="psd",
                                  axis= ax_temp,
                                  dataset_color=dataset_clor,
                                  make_presentable= make_presentable,
                                  run_identifier = ri +f' x={temp_x}',
                                  q_shortname = q_shortname)
            run_nums = run_nums + f'{filenames[i].split("_")[0]}{filenames[i].split("_")[1]}'
            j += 1
        i += 1

    fig_temp.tight_layout()
    fig_dens.tight_layout()

    # For the temperature plot
    ax_temp.legend(loc='lower center', bbox_to_anchor=(0.5, .98), ncol=2)

    # For the density plot
    ax_dens.legend(loc='lower center', bbox_to_anchor=(0.5, .98), ncol=2)

    fig_temp.tight_layout()
    fig_dens.tight_layout()
    fig_temp.subplots_adjust(top=0.8)
    fig_dens.subplots_adjust(top=0.8)
    run_nums = quantity + '_' + run_nums
    if save_plots:
        ensure_directory(figure_folder + 'psd_plots/')
        fig_temp.savefig(figure_folder + 'psd_plots/' + run_nums + '_temp_grads.svg', bbox_inches='tight')
        fig_dens.savefig(figure_folder + 'psd_plots/' + run_nums + '_dens_grads.svg', bbox_inches='tight')

    if see_plots:
        plt.show()
        plt.close()
    else:
        plt.close()


def get_deltan_over_n(run_dict, fluct_z, quantity, x_ax_arg='x',
                      b_dens_grads=False,
                      b_temp_grads=False,
                      split_x=False,
                      axis_list=None,
                      letters=None,
                      v_var_name='v_para',
                      cmap_name='viridis',
                      target_x_lhs=None,
                      target_x_rhs=None):

    # 0. Initialize return variables
    temp_color = '#C04000'
    density_color = '#000080'
    extracted_velocities = []
    plotted_x_vals = []

    fluct_ds = run_dict.get('fluct_ds', None)
    lang_ds = run_dict.get('lang_ds', None)
    mach_ds = run_dict.get('mach_ds', None)

    if fluct_ds is None or lang_ds is None:
        return temp_color, density_color, extracted_velocities, plotted_x_vals

    ri = run_dict.get('run_id', '')
    clor = run_dict.get('color', 'black')
    mark = run_dict.get('marker', 'o')

    # 1. Spatial & Temporal Probe Alignment
    try:
        lang_z_idx = np.abs(lang_ds['z'].values - fluct_z).argmin()
        lang_probe_ss_start = round(lang_ds.attrs[f'steady state start probe {lang_z_idx}'])
        lang_probe_ss_end = round(lang_ds.attrs[f'steady state end probe {lang_z_idx}'])
        bin_vals = ast.literal_eval(f'({lang_probe_ss_start}, {lang_probe_ss_end})')
    except (KeyError, ValueError, AttributeError):
        return temp_color, density_color, extracted_velocities, plotted_x_vals

    # 2. Extract Background Profiles
    n_vals, n_std, x_vals = process_variable_data(lang_ds, lang_z_idx, var_name='n_e')
    n_units = u.Unit(lang_ds['n_e'].attrs['units'])
    n_vals = n_vals * n_units
    n_std = n_std * n_units

    if mach_ds is not None:
        mach_z_idx = np.abs(mach_ds['z'].values - fluct_z).argmin()
        v_vals, v_x_vals = process_mach_data(
            mach_ds, mach_z_idx, v_var_name,
            lang_probe_ss_start, lang_probe_ss_end
        )
        sort_idx = np.argsort(v_x_vals)
        v_x_vals = v_x_vals[sort_idx]
        v_vals = v_vals[sort_idx]
    else:
        v_vals, v_x_vals = None, None

    # 3. Subplot & Target Axes Mapping
    if axis_list is None:
        new_layout = [[1]]
        new_fig, new_ax, new_letters = build_subplots(new_layout)
        axes = new_ax[new_letters[0]]
        lhs_axis, rhs_axis = axes, axes
    elif letters is not None and not split_x:
        lhs_axis = axis_list[letters[0]]
        rhs_axis = None
    elif letters is not None and split_x:
        lhs_axis = axis_list[letters[0]]
        rhs_axis = axis_list[letters[1]]
    else:
        lhs_axis, rhs_axis = None, None

    d_lhs, d_rhs = lhs_axis, rhs_axis
    t_lhs, t_rhs = lhs_axis, rhs_axis

    # 4. Parse Gradient Regions
    valid_dens_regions, valid_dens_slopes = [], []
    if b_dens_grads and 'dens_grad_regions' in lang_ds.attrs:
        dens_grad_regions = json.loads(lang_ds.attrs['dens_grad_regions'])
        d_slopes_str = lang_ds.attrs['dens_grad_slopes']
        dens_slopes = json.loads(d_slopes_str.replace('array(', '').replace(')', '')) if isinstance(d_slopes_str, str) else d_slopes_str

        for idx, region in enumerate(dens_grad_regions):
            if int(region[0]) < 0 and dens_slopes[idx] > 0:
                valid_dens_slopes.append(-1 * dens_slopes[idx])
                valid_dens_regions.append(region)
            if int(region[0]) > 0 and dens_slopes[idx] < 0:
                valid_dens_slopes.append(dens_slopes[idx])
                valid_dens_regions.append(region)

    valid_temp_regions = []
    if b_temp_grads and 'temp_grad_regions' in lang_ds.attrs:
        valid_temp_regions = json.loads(lang_ds.attrs['temp_grad_regions'])

    def plot_shaded_regions(valid_regions, ax_left, ax_right, is_split, color):
        for region in valid_regions:
            x_start, x_end = region[0], region[1]
            if is_split:
                if x_end <= 0 and ax_left is not None:
                    abs_start, abs_end = min(abs(x_start), abs(x_end)), max(abs(x_start), abs(x_end))
                    ax_left.axvline(abs_start, color=color, linestyle='--', alpha=0.7, zorder=1)
                    ax_left.axvline(abs_end, color=color, linestyle='--', alpha=0.7, zorder=1)
                    ax_left.axvspan(abs_start, abs_end, color=color, alpha=0.2, zorder=0)
                elif x_start >= 0 and ax_right is not None:
                    ax_right.axvspan(x_start, x_end, color=color, alpha=0.2, zorder=0)
                    ax_right.axvline(x_start, color=color, linestyle='--', alpha=0.7, zorder=1)
                    ax_right.axvline(x_end, color=color, linestyle='--', alpha=0.7, zorder=1)
            else:
                if ax_left is not None:
                    ax_left.axvspan(x_start, x_end, color=color, alpha=0.2, zorder=0)
                    ax_left.axvline(x_start, color=color, linestyle='--', alpha=0.7, zorder=1)
                    ax_left.axvline(x_end, color=color, linestyle='--', alpha=0.7, zorder=1)

    # 5. Plotting Logic: L_n vs delta_n / n
    if x_ax_arg != 'x':
        if lhs_axis is not None:
            ln_fluct_samples = []
            fluct_x_vals = fluct_ds['x'].values

            for region_idx, density_region in enumerate(valid_dens_regions):
                region_density_slope = valid_dens_slopes[region_idx]

                region_x_mask = (x_vals >= density_region[0]) & (x_vals <= density_region[1])
                region_x_positions = x_vals[region_x_mask]
                region_density_vals = n_vals[region_x_mask]
                region_density_stds = n_std[region_x_mask]

                for point_idx, x_val in enumerate(region_x_positions):
                    nearest_fluct_x = fluct_x_vals[np.abs(fluct_x_vals - x_val).argmin()]

                    try:
                        _, _, _, delta, delta_err = get_spectrum_from_data(
                            fluct_ds[quantity].sel(z=fluct_z),
                            x=nearest_fluct_x, bin=bin_vals, shot=None, z=fluct_z,
                            plot=False, scaling="psd", run_identifier=ri,
                            integrate_psd=True
                        )
                    except ValueError:
                        delta, delta_err = np.nan, np.nan

                    if np.isnan(delta):
                        continue

                    n_m3 = region_density_vals[point_idx].to(1 / (u.m ** 3)).value
                    n_std_m3 = region_density_stds[point_idx].to(1 / (u.m ** 3)).value

                    l_n = -n_m3 / region_density_slope
                    l_n_err = n_std_m3 / abs(region_density_slope)

                    if l_n <= 0:
                        continue

                    n_cm3 = region_density_vals[point_idx].to(1 / (u.cm ** 3)).value
                    n_std_cm3 = region_density_stds[point_idx].to(1 / (u.cm ** 3)).value

                    fluct_ratio = delta / n_cm3
                    rel_err_delta = delta_err / delta
                    rel_err_n = n_std_cm3 / n_cm3
                    fluct_ratio_err = abs(fluct_ratio) * np.sqrt(rel_err_delta ** 2 + rel_err_n ** 2)

                    if v_vals is not None and len(v_vals) > 0:
                        v_val = np.interp(x_val, v_x_vals, v_vals)
                    else:
                        v_val = np.nan

                    if not (np.isnan(l_n) or np.isnan(fluct_ratio)):
                        ln_fluct_samples.append({
                            'x_val': x_val,
                            'l_n': l_n,
                            'l_n_err': l_n_err,
                            'fluct_ratio': fluct_ratio,
                            'fluct_ratio_err': fluct_ratio_err,
                            'velocity': v_val
                        })

            def select_target_samples(sample_list, user_target_xs=None):
                if not sample_list:
                    return []

                # Target x-positions specified by user
                if user_target_xs:
                    selected_samples = []
                    for target_x in user_target_xs:
                        closest = min(sample_list, key=lambda s: abs(s['x_val'] - target_x))
                        if closest not in selected_samples:
                            selected_samples.append(closest)
                    return selected_samples

                # Fallback: Strip bottom 15% of L_n values (keep values >= 15th percentile)
                if len(sample_list) <= 2:
                    target_median = np.median([s['l_n'] for s in sample_list])
                    return [min(sample_list, key=lambda s: abs(s['l_n'] - target_median))]

                ln_vals = [s['l_n'] for s in sample_list]
                p15 = np.percentile(ln_vals, 15)

                trimmed_samples = [s for s in sample_list if s['l_n'] >= p15]
                if not trimmed_samples:
                    trimmed_samples = sample_list

                target_median = np.median([s['l_n'] for s in trimmed_samples])
                return [min(trimmed_samples, key=lambda s: abs(s['l_n'] - target_median))]

            plot_axes_lhs = set([ax for ax in [d_lhs if b_dens_grads else None, t_lhs if b_temp_grads else None] if ax])
            plot_axes_rhs = set([ax for ax in [d_rhs if b_dens_grads else None, t_rhs if b_temp_grads else None] if ax])

            if not b_dens_grads and not b_temp_grads:
                plot_axes_lhs = {lhs_axis}
                plot_axes_rhs = {rhs_axis}

            extracted_velocities = [s['velocity'] for s in ln_fluct_samples if not np.isnan(s['velocity'])]

            if ln_fluct_samples and extracted_velocities:
                v_min, v_max = min(extracted_velocities), max(extracted_velocities)
                if v_min == v_max: v_max += 1e-5
                norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
                cmap = plt.get_cmap(cmap_name)
            else:
                norm, cmap = None, None

            def plot_sample_with_cbar(sample, target_axes):
                if not sample:
                    return
                sample_color = cmap(norm(sample['velocity'])) if (norm is not None and cmap is not None and not np.isnan(sample['velocity'])) else clor

                for ax in target_axes:
                    ax.errorbar(
                        sample['l_n'], sample['fluct_ratio'],
                        xerr=sample['l_n_err'], yerr=sample['fluct_ratio_err'],
                        fmt=mark,
                        markersize=8,
                        markerfacecolor=sample_color,
                        markeredgecolor=clor,
                        markeredgewidth=1.0,
                        ecolor=clor,
                        capsize=3
                    )

            if split_x:
                lhs_samples = [s for s in ln_fluct_samples if s['x_val'] < 0]
                rhs_samples = [s for s in ln_fluct_samples if s['x_val'] > 0]

                selected_lhs = select_target_samples(lhs_samples, target_x_lhs)
                selected_rhs = select_target_samples(rhs_samples, target_x_rhs)

                for sample in selected_lhs:
                    plot_sample_with_cbar(sample, plot_axes_lhs)
                    plotted_x_vals.append(sample['x_val'])

                for sample in selected_rhs:
                    plot_sample_with_cbar(sample, plot_axes_rhs)
                    plotted_x_vals.append(sample['x_val'])
            else:
                selected_samples = select_target_samples(ln_fluct_samples, target_x_lhs)
                for sample in selected_samples:
                    plot_sample_with_cbar(sample, plot_axes_lhs)
                    plotted_x_vals.append(sample['x_val'])

    # 6. Plotting Logic: x vs delta_n / n
    if x_ax_arg == 'x':
        if b_dens_grads:
            plot_shaded_regions(valid_dens_regions, d_lhs, d_rhs, split_x, density_color)
        if b_temp_grads:
            plot_shaded_regions(valid_temp_regions, t_lhs, t_rhs, split_x, temp_color)

        if lhs_axis is not None:
            fluct_x_vals = fluct_ds['x'].values
            for idx, x_val in enumerate(x_vals):
                nearest_fluct_x = fluct_x_vals[np.abs(fluct_x_vals - x_val).argmin()]
                try:
                    _, _, _, delta, delta_err = get_spectrum_from_data(
                        fluct_ds[quantity].sel(z=fluct_z),
                        x=nearest_fluct_x, bin=bin_vals, shot=None, z=fluct_z,
                        plot=False, scaling="psd", run_identifier=ri,
                        integrate_psd=True
                    )
                except ValueError:
                    delta, delta_err = np.nan, np.nan

                if not np.isnan(delta):
                    plotted_x_vals.append(x_val)
                    n = n_vals[idx].to(1 / (u.cm ** 3))
                    n_err = n_std[idx].to(1 / (u.cm ** 3))

                    fluct_ratio = delta / n.value
                    total_rel_error = np.sqrt((delta_err / delta) ** 2 + (n_err.value / n.value) ** 2)
                    total_err = abs(fluct_ratio) * total_rel_error

                    axes_to_plot = set()
                    if split_x:
                        radius = abs(x_val)
                        if x_val < 0:
                            if b_dens_grads: axes_to_plot.add(d_lhs)
                            if b_temp_grads: axes_to_plot.add(t_lhs)
                            if not b_dens_grads and not b_temp_grads: axes_to_plot.add(lhs_axis)
                        elif x_val > 0:
                            if b_dens_grads: axes_to_plot.add(d_rhs)
                            if b_temp_grads: axes_to_plot.add(t_rhs)
                            if not b_dens_grads and not b_temp_grads: axes_to_plot.add(rhs_axis)

                        for ax in axes_to_plot:
                            if ax is not None:
                                ax.errorbar(radius, fluct_ratio, yerr=total_err, color=clor, fmt=mark, capsize=3)
                    else:
                        if b_dens_grads: axes_to_plot.add(d_lhs)
                        if b_temp_grads: axes_to_plot.add(t_lhs)
                        if not b_dens_grads and not b_temp_grads: axes_to_plot.add(lhs_axis)

                        for ax in axes_to_plot:
                            if ax is not None:
                                ax.errorbar(x_val, fluct_ratio, yerr=total_err, color=clor, fmt=mark, capsize=3)

    return temp_color, density_color, extracted_velocities, plotted_x_vals