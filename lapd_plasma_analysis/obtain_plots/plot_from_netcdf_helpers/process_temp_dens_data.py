from lapd_plasma_analysis.obtain_plots.Auxillary_functions import *

def process_variable_data(ds, probe, var_name, mask_var='t_e'):
    """
    Standardizes the shot-averaging, NaN-filtering, and steady-state sweep
    averaging for ANY variable in a Langmuir probe xarray Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Processed xarray Dataset from Langmuir probe analysis.
    probe : int
        Integer corresponding to the probe number in the dataset.
    var_name : str
        Data variable to process (e.g., 't_e', 'n_e', 'nu_ei', 'v_f').
    mask_var : str or None, default 't_e'
        Variable used to filter out bad fits/NaNs across shots.
        Secondary variables (n_e) inherit NaNs from bad t_e fits.
        Set to None if no quality mask is needed.

    Returns
    -------
    vals : np.ndarray
        Array of steady-state averaged parameter values matched to x_vals.
    std_vals : np.ndarray
        Array of standard deviations across the steady-state period matched to x_vals.
    x_vals : np.ndarray
        Array of spatial x-positions corresponding to the output arrays.
    """

    # Extract steady-state time boundaries
    min_time = ds.attrs[f'steady state start probe {probe}']
    max_time = ds.attrs[f'steady state end probe {probe}']

    # Quality filter based on mask_var (defaults to t_e)
    if mask_var is not None and mask_var in ds:
        mask_mean = ds[mask_var].sel(probe=probe).mean('shot')
        mask_std = ds[mask_var].sel(probe=probe).std('shot')
        filtered_mask_data = filter_data(mask_mean, mask_std)
        where_nans = filtered_mask_data.isnull()
    else:
        where_nans = None

    # Process target variable across shots
    target_data = ds[var_name].sel(probe=probe).mean('shot')
    if where_nans is not None:
        target_data = target_data.where(~where_nans)

    # Steady-state time window mask
    time_mask = (target_data['time'] >= min_time) & (target_data['time'] <= max_time)
    steady_state_sweeps = ds['sweep'][time_mask]

    # Average across steady-state sweeps
    var_to_plot = target_data.sel(sweep=steady_state_sweeps).mean('sweep')
    var_std_to_plot = target_data.sel(sweep=steady_state_sweeps).std('sweep')

    # Extract numpy arrays for plotting
    x_vals = target_data['x'].values
    vals = var_to_plot.squeeze().values
    std_vals = var_std_to_plot.squeeze().values

    return vals, std_vals, x_vals




# def process_dens_data(ds, probe):
#     """
#     Parameters
#     ----------
#     ds: Processed xarray Dataset obtained from analysis of a swept Langmuir probe
#     probe: Integer corresponding to the probe number in the xarray Dataset
#
#     Returns
#     -------
#     n_e_to_plot_vals: Array of density values in m^{-3} matched with the x_vals array
#     n_e_std_to_plot_vals: Array of standard deviations of the density values obtained from taking the standard deviation
#                           of the averaged shot and steady state density values, matched with the
#                           x_vals array
#     x_vals: Array of x values that match the density and standard deviation arrays
#     """
#
#     min_time = ds.attrs[f'steady state start probe {probe}']
#     max_time = ds.attrs[f'steady state end probe {probe}']
#
#     mean_data = ds['t_e'].sel(probe=probe).mean('shot')
#     std_data = ds['t_e'].sel(probe=probe).std('shot')
#
#     t_e_filtered_data = filter_data(mean_data, std_data)
#     where_nans = t_e_filtered_data.isnull()
#     n_e_filtered_data = ds['n_e'].sel(probe=probe).mean('shot').where(~where_nans)
#     filtered_ne_mean_profile, filtered_ne_std_profile = filter_ne_data(n_e_filtered_data, min_time, max_time)
#
#     n_e_to_plot = filtered_ne_mean_profile
#     n_e_std_to_plot = filtered_ne_std_profile
#
#     # Format everything for matplot.lib plotting
#     x_vals = t_e_filtered_data['x'].values
#     n_e_to_plot_vals = n_e_to_plot.squeeze().values
#     n_e_std_to_plot_vals = n_e_std_to_plot.squeeze().values
#
#     return n_e_to_plot_vals, n_e_std_to_plot_vals, x_vals

# def process_temp_data(ds, probe):
#     """
#     Function to standardize the averaging of temperature values across shot and steady state period
#     Parameters
#     ----------
#     ds: Processed xarray Dataset obtained from analysis of a swept Langmuir probe
#     probe: Integer corresponding to the probe number in the xarray Dataset
#
#     Returns
#     -------
#     t_e_to_plot_vals: Array of temperature values in eV matched with the x_vals array
#     t_e_std_to_plot_vals: Array of standard deviations of the temperature values obtained from taking the standard
#                           deviation of the averaged shot and steady state temperature values, matched with the
#                           x_vals array
#     x_vals: Array of x values that match the temperature and standard deviation arrays
#     """
#
#     min_time = ds.attrs[f'steady state start probe {probe}']
#     max_time = ds.attrs[f'steady state end probe {probe}']
#
#     mean_data = ds['t_e'].sel(probe=probe).mean('shot')
#     std_data = ds['t_e'].sel(probe=probe).std('shot')
#
#     t_e_filtered_data = filter_data(mean_data, std_data)
#
#     t_e_mask = (t_e_filtered_data['time'] >= min_time) & (t_e_filtered_data['time'] <= max_time)
#
#     t_e_to_plot = t_e_filtered_data.sel(sweep=ds['sweep'][t_e_mask]).mean('sweep')
#     t_e_std_to_plot = t_e_filtered_data.sel(sweep=ds['sweep'][t_e_mask]).std('sweep')
#
#     # Format everything for matplot.lib plotting
#     x_vals = t_e_filtered_data['x'].values
#     t_e_to_plot_vals = t_e_to_plot.squeeze().values
#     t_e_std_to_plot_vals = t_e_std_to_plot.squeeze().values
#
#     return t_e_to_plot_vals, t_e_std_to_plot_vals, x_vals
