import numpy as np
import xarray as xr
import pandas as pd
import re

from scipy.signal import find_peaks, hilbert
import astropy.units as u
from astropy import constants as const
from bapsflib import lapd

from langmuir.file_access import search_folder
from langmuir.configurations import get_config_id


# "interferometry" is abbreviated as "itfm" throughout
def interferometry_calibration(density_da: xr.DataArray,
                               exp_attrs: dict,
                               itfm_folder: str,              # path to either HDF5 or text folder
                               steady_state: tuple,
                               core_radius: u.Quantity = 26. * u.cm,
                               ) -> xr.DataArray:

    itfm_id = get_config_id(exp_attrs['Exp name'])

    # All calculations are in cm, as coordinates are labeled in cm
    density_da *= (1 / u.m ** 3).to(1 / u.cm ** 3).value  # density in 1/cm^3

    # Detect spatial dimensions (1D or 2D)
    spatial_dims = [dim for dim in ('x', 'y') if density_da.sizes[dim] > 1]
    # Assume all spatial dimensions are labeled in centimeters

    run_str = exp_attrs['Run name'][:2]

    # Use only probe listed first for generating scale factors
    isweep_coord = density_da['isweep']
    core_density = density_da.isel(isweep=0)

    # Select core region in x and y and interpolate nan values to allow integration; 10 cm (arbitrary) max gap
    # TODO describe averaging across shots! Preserve structure in that direction, since only time varies in itfm data?
    core_density = core_density.mean(dim="shot")
    for dim in spatial_dims:
        core_density = core_density.where(np.abs(density_da.coords[dim]) < core_radius.value, drop=True)
        core_density = core_density.interpolate_na(dim=dim, use_coordinate=True, max_gap=10.)

    if itfm_id == 0:  # April 2018
        itfm_file = lapd.File(itfm_file_search_hdf5(run_str, itfm_folder))
        itfm = itfm_file.read_msi("Interferometer array")
        density_scale_factor = itfm_calib_56ghz(core_density, itfm, spatial_dims).expand_dims({"isweep": isweep_coord})
        itfm_file.close()

    elif itfm_id == 1:  # March 2022
        itfm_file = lapd.File(itfm_file_search_hdf5(run_str, itfm_folder))
        num_fringes = find_fringes_96ghz(itfm_file)
        density_scale_factor = itfm_calib_96ghz(core_density, num_fringes, spatial_dims, core_radius)
        itfm_file.close()

    elif itfm_id == 2:  # November 2022
        itfm_paths = itfm_file_search_288ghz(run_str, itfm_folder)
        itfm_density_da = itfm_density_288ghz(*itfm_paths)
        density_scale_factor = itfm_calib_288ghz(core_density, itfm_density_da, spatial_dims, core_radius)

    elif itfm_id == 3:  # January 2024
        itfm_file = lapd.File(itfm_file_search_hdf5(run_str, itfm_folder))
        itfm = itfm_file.read_msi("Interferometer array")
        density_scale_factor = itfm_calib_jan_2024(core_density, itfm, spatial_dims
                                                   ).expand_dims({"isweep": isweep_coord})
        itfm_file.close()

    else:
        raise NotImplementedError("Unsupported interferometry id " + repr(itfm_id))

    # Select steady state region only (given by plateau indices)
    steady_state_density_scale_factor = density_scale_factor.where(np.logical_and(
        density_da.plateau >= steady_state[0], density_da.plateau <= steady_state[1]))
    print(f"Run {exp_attrs['Run name'][:2]}: \t"
          f"{np.nanmean(np.asarray(steady_state_density_scale_factor)):.2f}")

    density_da *= density_scale_factor.expand_dims({dim: density_da.sizes[dim] for dim in spatial_dims + ["shot"]})
    density_da *= (1 / u.cm ** 3).to(1 / u.m ** 3).value  # density in 1/m^3

    return density_da


def itfm_calib_56ghz(langmuir_da, itfm, spatial_dimensions) -> xr.DataArray:
    r"""
    Calibrates density data from sweep probe to measurements from interferometry probe.

    Parameters
    ----------
    :param langmuir_da: DataArray of n_e in cm^-3, core region only
    :param itfm: structured array of interferometry data and metadata from HDF5 file
    :param spatial_dimensions: list of spatial dimensions of density data, e.g. ('x', 'y')

    :return: DataArray of dimensionless scale factors for density data
    """

    # INTERFEROMETRY DATA #
    # ___________________ #

    itfm_nd = np.array(itfm['signal'][:, 0, :])
    itfm_raw_to_line_integrated_density = itfm.info['n_bar_L'][0] / u.cm ** 2
    itfm_dt = itfm.info['dt'][0] * u.s
    # itfm_t0 = itfm.info['t0'][0] * u.s  # unused because we align times manually here

    itfm_nd = np.mean(itfm_nd, axis=0)  # average out unnecessary length-2 dimension

    itfm_nd = itfm_nd * itfm_raw_to_line_integrated_density  # previously: itfm_nd *= real_density_scale
    itfm_times = np.arange(len(itfm_nd)) * itfm_dt.to(u.ms)

    # Create interferometry DataArray with interferometry time as coordinates
    itfm_da = xr.DataArray(itfm_nd, coords=(('time', itfm_times, {'units': str(u.ms)}),))

    # TODO correct interferometry issues for bimaxwellian diagnostics

    # INTERFEROMETRY - DENSITY TIME ALIGNMENT #
    # _______________________________________ #

    # Find index of interferometry collapse; at point with most negative slope
    itfm_collapse_time = itfm_da.differentiate('time').idxmin('time')

    # Find average step in density (n_e) time coordinate by dividing total time elapsed by number of time measurements
    density_time_coord = langmuir_da.coords['time']
    dt = (density_time_coord[-1] - density_time_coord[0]) / len(density_time_coord)  # time step for density time coord.

    # DENSITY LINE INTEGRALS #
    # ______________________ #

    density_scales = dict()  # create empty dictionary to hold x and y scaling factors where necessary

    for dim in spatial_dimensions:
        # Line integral along dimension
        integral = langmuir_da.integrate(dim)

        # Find time of density collapse in dimension
        # Take average to get one single average dimension density collapse time (scalar)
        density_collapse_time = integral.idxmax('time').mean()

        # Align times so that interferometry data collapse matches x density data max
        aligned_time = itfm_da.coords['time'] - itfm_collapse_time + density_collapse_time
        itfm_da = itfm_da.assign_coords({dim + '_time': ('time', aligned_time.data)})

        # Average all interferometry measurements into data point with the closest corresponding density time coordinate
        itfm_time_avg = crunch_data(itfm_da, dim + '_time', langmuir_da.coords['time'], dt)

        density_scales[dim] = itfm_time_avg / integral

    # Average together x and y scale factors, if they exist, to get scale factor array in right number of dimensions
    return np.sum(*[density_scales[dim] for dim in spatial_dimensions]) / len(spatial_dimensions)  # 1D xarray?


def find_fringes_96ghz(inter_file):
    uwave_data = inter_file.read_data(board=4, channel=1, silent=True)  # hardcoded

    plasma_shutoff = 15 * u.ms  # TODO hardcoded
    uwave_dt = uwave_data.dt
    shutoff_frame = int(plasma_shutoff / uwave_dt)

    mean_peaks = np.mean([len(find_peaks(signal[shutoff_frame:], **{"prominence": 1.})[0])
                          for signal in uwave_data['signal']])

    return 2 * mean_peaks


def itfm_calib_96ghz(langmuir_density: xr.DataArray, num_fringes, spatial_dimensions, core_radius) -> u.Quantity:
    # Peak interferometry density
    uwave_avg_density_at_peak = ((num_fringes * 1.88e13 / u.cm ** 2) / (2 * core_radius)).to(u.cm ** -3).value
    # Formula from HDF5 file run description

    # Peak Langmuir density
    for dim in spatial_dimensions:
        langmuir_density = (langmuir_density.integrate(dim) / (2 * core_radius.to(u.cm).value)
                            ).assign_attrs(langmuir_density.attrs)

    time_size = langmuir_density.sizes['time']
    langmuir_density_at_peak = langmuir_density.max('time', skipna=True)

    return uwave_avg_density_at_peak / langmuir_density_at_peak.expand_dims(
        {"time": time_size}, axis=-1)  # [x,] y, ..., time


def itfm_file_search_hdf5(run_str: str, folder_path: str):
    hdf5_paths = search_folder(folder_path, "hdf5")

    # find hdf5 path corresponding to hdf5 file
    return [path for path in hdf5_paths if os.path.split(path)[1].startswith(run_str)][0]


def itfm_density_288ghz(reference_filename: str, signal_filename: str) -> xr.DataArray:
    """
    Created on Wed Sep 14 15:55:57 2022
    @author: Stephen Vincena
    """

    uwave_freq = 288 * u.GHz
    e = const.e.to(u.C)         # elementary_charge
    m_e = const.m_e             # electron_mass
    eps0 = const.eps0
    c = const.c

    num_passes = 2.
    diameter = 0.35 * u.m

    calibration = 1. / ((num_passes / 4. / np.pi / uwave_freq) * (e ** 2 / m_e / c / eps0))

    # read data from csv files via pandas dataframes
    ref_df = pd.read_csv(reference_filename, names=["time", "reference"], dtype=np.float64, skiprows=5)
    sig_df = pd.read_csv(signal_filename,    names=["time", "signal"],    dtype=np.float64, skiprows=5)

    # the traces are saved together, so the time is redundant
    # convert a named series in dataframes to numpy arrays
    time = ref_df["time"].to_numpy() * u.s
    ref_np = ref_df["reference"].to_numpy()
    sig_np = sig_df["signal"].to_numpy()
    time_ms = time.to(u.ms)

    # Construct analytic function versions of the reference and the plasma signal
    # Note: scipy's hilbert function actually creates an analytic function using the Hilbert transform,
    #    which is what we want in the end anyway
    # So, given real X(t): analytic function = X(t) + i * HX(t), where H is the actual Hilbert transform
    # https://en.wikipedia.org/wiki/Hilbert_transform

    aref = hilbert(ref_np)
    asig = hilbert(sig_np)
    aref -= np.mean(aref)
    asig -= np.mean(asig)

    pref = np.unwrap(np.angle(aref))
    psig = np.unwrap(np.angle(asig))

    dphi = (pref - psig)
    dphi -= dphi[0]

    itfm_density = (dphi * calibration / diameter).to(u.cm ** -3).value  # returns array of *average* density over core
    itfm_density_da = xr.DataArray(itfm_density,
                                   dims=["time"],
                                   coords={"time": time_ms.value})

    return itfm_density_da


def itfm_calib_288ghz(density_da: xr.DataArray,
                      itfm_da:    xr.DataArray, spatial_dimensions, core_radius=26*u.cm):

    dt = (itfm_da.time[-1] - itfm_da.time[0]) / len(itfm_da.time)   # time step for density time coord. in ms

    mean_density_da = density_da.copy()
    for dim in spatial_dimensions:
        mean_density_da = (mean_density_da.integrate(dim) / (2 * core_radius.to(u.cm).value)
                           ).assign_attrs(mean_density_da.attrs)

    # Average all interferometry measurements into data point with the closest corresponding density time coordinate
    itfm_da_crunched = crunch_data(itfm_da, 'time', density_da.coords['time'], dt)

    return itfm_da_crunched / mean_density_da  # density scale factor


def itfm_file_search_288ghz(run_str: str, folder_path: str):
    text_paths = search_folder(folder_path, "txt")

    # find channel 1, channel 2 text file paths corresponding to hdf5 file
    c1_match = [path for path in text_paths
                if "c1" in path.lower()
                and re.search("(?<=[Rr]un).?[0-9]{2}", path).group(0)[-2:] == run_str][0]
    c2_match = [path for path in text_paths
                if "c2" in path.lower()
                and re.search("(?<=[Rr]un).?[0-9]{2}", path).group(0)[-2:] == run_str][0]
    return c1_match, c2_match


def itfm_calib_jan_2024(lang_da, itfm, spatial_dimensions) -> xr.DataArray:
    r"""
    Calibrates density data from sweep probe to measurements from interferometry probe.

    Parameters
    ----------
    :param lang_da: DataArray of n_e in cm^-3, core region only
    :param itfm: structured array of interferometry data and metadata from HDF5 file
    :param spatial_dimensions: list of spatial dimensions of density data, e.g. ('x', 'y')

    :return: DataArray of dimensionless scale factors for density data
    """

    # INTERFEROMETRY DATA #
    # ___________________ #

    itfm_signal = np.array(itfm['signal'][:, 1, :])  # Port 20 interferometer
    itfm_signal = itfm_signal.mean(axis=0)  # average out unnecessary length-2 dimension "shot"; check similarity?
    itfm_signal = np.clip(itfm_signal, a_min=0, a_max=None)  # TODO document this clipping

    # Convert interferometry signal from arbitrary dimensionless values to real units
    itfm_raw_to_line_integrated_density = itfm.info['n_bar_L'][1] / u.cm ** 2
    itfm_signal = itfm_signal * itfm_raw_to_line_integrated_density

    # Calculate time coordinates for interferometry data
    itfm_t0 = itfm.info['t0'][0] * u.s
    itfm_dt = itfm.info['dt'][0] * u.s
    itfm_time_ms = itfm_t0.to(u.ms) + itfm_dt.to(u.ms) * np.arange(len(itfm_signal))

    # Create interferometry DataArray with interferometry time as coordinates
    itfm_da = xr.DataArray(itfm_signal, coords=(('time', itfm_time_ms, {'units': str(u.ms)}),))

    # TODO correct interferometry issues for bimaxwellian diagnostics (?)

    # Find average step in density (n_e) time coordinate by dividing total time elapsed by number of time measurements
    lang_time_ms = lang_da.coords['time']
    lang_dt = (lang_time_ms[-1] - lang_time_ms[0]) / len(lang_time_ms)  # time step for density time coord.

    # DENSITY LINE INTEGRALS #
    # ______________________ #

    # Average all interferometry measurements into data point with the closest corresponding density time coordinate
    itfm_crunched = crunch_data(source_da=itfm_da, source_coord_name='time',
                                destination_coord_da=lang_da.coords['time'], step=lang_dt)

    """
    Note: Maybe just extend each dimension that was integrated out back to its original size?
    
    density_scales = dict()  # create empty dictionary to hold x and y scaling factors where necessary
    for dim in spatial_dimensions:
        # Line integral along dimension
        integral = lang_da.integrate(dim)
        density_scales[dim] = itfm_time_avg / integral
    
    # Average together x and y scale factors, if they exist, to get scale factor array in right number of dimensions
    return np.sum(*[density_scales[dim] for dim in spatial_dimensions]) / len(spatial_dimensions)  # 1D xarray?
    """

    if len(spatial_dimensions) > 1:
        raise NotImplementedError("2D areal data detected, which is not supported for January 2024.")
    integral = lang_da.fillna(0).integrate(spatial_dimensions[0])
    density_scales = itfm_crunched / integral

    return density_scales


def crunch_data(source_da: xr.DataArray,
                source_coord_name: str,
                destination_coord_da: xr.DataArray,
                step):
    # "Crunch" interferometry data into the density data timescale by averaging all interferometry measurements
    #     into a "bucket" around the closest matching density time coordinate (within half a time step)
    #     [inter. ]   (*   *) (*) (*   *) (*) (*   *)   <-- average together all (grouped together) measurements
    #     [density]   |__o__|__o__|__o__|__o__|__o__|   <-- measurements grouped by closest density measurement "o"
    # Take the mean of all interferometry measurements in the same "bucket" to match timescales
    r"""
    Group data along a specified dimension into bins determined by a destination coordinate and a step size,
    then return the mean of each bin with the dimensions and coordinates of the destination coordinate.

    Parameters
    ----------
    :param source_da: DataArray containing data to bin and average
    :param source_coord_name: string, dimension in data_array; used to bin data
    :param destination_coord_da: xarray DataArray, used as coordinate
    :param step:
    :return:
    """

    # Group input data "source_da" along the dimension specified by "source_coord_name"
    #    by the coordinate in the xarray "destination_coord_da", assumed to have regular spacing "step", and take means
    grouped_mean = source_da.groupby_bins(source_coord_name,
                                          np.linspace(destination_coord_da[0] - step / 2,
                                                      destination_coord_da[-1] + step / 2,
                                                      len(destination_coord_da) + 1
                                                      ), labels=destination_coord_da.data
                                          ).mean()

    # This result has only one dimension, the input data "dimension" + "_bins", labeled with the destination coordinate.
    #    We want to return a DataArray with all the dimensions and coordinates (in this case: time dimension,
    #    time dimension coordinate, plateau non-dimension coordinate) of the destination data.
    #    This involves renaming the "_bins" dimension to match the destination coordinate,
    #    creating a new coordinate identical to the destination coordinate's dimension coordinate,
    #    and swapping the two new coordinates to give the xarray the same dimension coordinate as the destination.

    destination_dimension = destination_coord_da.dims[0]  # The name of the dimension of the 1D destination coordinate
    destination_coordinate_name = destination_coord_da.name  # The name of the destination coordinate

    # Rename position-time-"_bins" dimension name to match destination coordinate, for example "x_time_bins" to "time"
    named_mean = grouped_mean.rename({source_coord_name + "_bins": destination_coordinate_name})
    # Add the destination dimension coordinate to the output xarray as a new coordinate
    named_mean = named_mean.assign_coords({destination_dimension: (destination_coordinate_name,
                                                                   destination_coord_da[destination_dimension].data)})
    # Make the new destination dimension coordinate the main (dimension) coordinate of the output as well
    named_mean = named_mean.swap_dims({destination_coordinate_name: destination_dimension})

    return named_mean
