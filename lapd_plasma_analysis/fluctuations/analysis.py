import numpy as np
import astropy.units as u
from bapsflib import lapd
from warnings import warn
from numpy.random import normal
from lapd_plasma_analysis.langmuir.configurations import get_config_id, get_langmuir_config
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.fft import fft, fftfreq, ifft
from lapd_plasma_analysis.langmuir.analysis import get_langmuir_datasets
from lapd_plasma_analysis.experimental import get_exp_params
from lapd_plasma_analysis.langmuir.configurations import get_ion
import xarray as xr
from tqdm import tqdm
from matplotlib.colors import LogNorm
from scipy.sparse import lil_matrix, csr_matrix
from scipy.optimize import curve_fit
from lapd_plasma_analysis.fluctuations.fourier import *
import os

def get_isat_vf(filename, hdf5_path, flux_nc_folder):
    """
    Obtains saturation current, floating potential, floating potential difference, sound speed,
    and density data from the HDF5 file specified by `filename`. Deposits a NetCDF file containing this data (as
    an `xarray.Dataset` in `flux_nc_folder`. The NetCDF file will have the same name as the HDF5 file.

    Parameters
    ----------
    filename : `str`
        String containing path to specific HDF5 file

    hdf5_path : `str`
        String containing path to all HDF5 files

    flux_nc_folder : `str`
        String giving path to the folder containing the NetCDF files for fluctuation data

    Returns
    -------
    dataset : `xarray.Dataset`
        Dataset containing floating voltage, saturation current, sound speed and density data

    """
    print("File name: ", filename)
    assert os.path.exists(filename)
    assert get_config_id(lapd.File(filename).info['exp name']) in [1, 2]

    file = lapd.File(filename)

    config_id = get_config_id(file.info['exp name'])

    if config_id == 1:
        isat_receptacle = 1
        isat_board = 2
        isat_channel = 1
        vf_receptacle = 2
        vf_board = 2
        vftop_channel = 2
        vfbot_channel = 3

    if config_id == 2:
        isat_receptacle = 1
        isat_board = 2
        isat_channel = 3
        # vf_receptacle = 2
        # vf_board = 2
        # vftop_channel = 2
        # vfbot_channel = 3


    isat_data = file.read_data(isat_board, isat_channel, silent=True, add_controls=[('6K Compumotor', isat_receptacle)])
    dt = isat_data.dt.value
    times = np.linspace(0, len(isat_data[0][1])*dt, len(isat_data[0][1]))

    isat_vs_x = []
    isat_vs_x.append([])
    x_index = 0
    x_array = []
    x_array.append(isat_data[0][2][0])
    for discharge in isat_data:
        if abs(discharge[2][0] - x_array[x_index]) <= 0.01:
            isat_vs_x[x_index].append(discharge[1]/15) # 15 ohm resistance

        else:
            x_index += 1
            isat_vs_x.append([])
            x_array.append(discharge[2][0])
            isat_vs_x[x_index].append(discharge[1]/15)


    isat_vs_x = np.array(isat_vs_x)
    _, numshot, __ = isat_vs_x.shape
    shot_array = range(numshot)
    isat_data_array = xr.DataArray(isat_vs_x*u.ampere, dims = ['x', 'shot', 'time'],
                                                       coords = {'x': np.round(x_array)*u.cm,
                                                                 'shot': shot_array,
                                                                 'time': times*u.s.to(u.ms)},
                                                       name = 'isat',
                                                       attrs={'units': 'A'})
    isat_data_array.coords['x'].attrs['units'] = 'cm'
    isat_data_array.coords['time'].attrs['units'] = 'ms'

    if config_id == 1:
        vf_top_data = file.read_data(vf_board, vftop_channel, silent=True, add_controls=[('6K Compumotor', vf_receptacle)])
        vf_bottom_data = file.read_data(vf_board, vfbot_channel, silent=True, add_controls=[('6K Compumotor', vf_receptacle)])
        dt = vf_bottom_data.dt.value
        times = np.linspace(0, len(vf_bottom_data[0][1])*dt, len(vf_bottom_data[0][1]))

        dvf_vs_x = []
        vf_vs_x = []
        dvf_vs_x.append([])
        vf_vs_x.append([])
        x_index = 0
        x_array = []
        x_array.append(vf_bottom_data[0][2][0])
        for index in range(len(vf_bottom_data)):
            if abs(vf_bottom_data[index][2][0] - x_array[x_index]) <= 0.01:
                dvf_vs_x[x_index].append((vf_top_data[index][1] - vf_bottom_data[index][1])*10)  # /10 in exp
                vf_vs_x[x_index].append(vf_bottom_data[index][1]*10)

            else:
                x_index += 1
                dvf_vs_x.append([])
                vf_vs_x.append([])
                x_array.append(vf_bottom_data[index][2][0])
                dvf_vs_x[x_index].append((vf_top_data[index][1] - vf_bottom_data[index][1]) * 10)
                vf_vs_x[x_index].append(vf_bottom_data[index][1]*10)

        dvf_vs_x = np.array(dvf_vs_x)
        vf_vs_x = np.array(vf_vs_x)
        dvf_data_array = xr.DataArray(dvf_vs_x*u.volt, dims=['x', 'shot', 'time'],
                                                       coords = {'x': np.round(x_array)*u.cm,
                                                                 'shot': shot_array,
                                                                 'time': times*u.s.to(u.ms)},
                                                       name = 'dvf',
                                                       attrs = {'units': 'V'})
        dvf_data_array.coords['x'].attrs['units'] = 'cm'
        dvf_data_array.coords['time'].attrs['units'] = 'ms'

        vf_data_array = xr.DataArray(vf_vs_x*u.volt, dims = ['x', 'shot', 'time'],
                                                     coords = {'x': np.round(x_array)*u.cm,
                                                               'shot': shot_array,
                                                               'time': times*u.s.to(u.ms)},
                                                     name = 'vf',
                                                     attrs={'units': 'V'})
        vf_data_array.coords['x'].attrs['units'] = 'cm'
        vf_data_array.coords['time'].attrs['units'] = 'ms'

    sound_speed_data_array, density_data_array = get_density_data(filename, hdf5_path, isat_data_array, shot_array)

    if config_id == 1:
        isat_vf_dataset = xr.merge([isat_data_array, dvf_data_array,
                                    vf_data_array, sound_speed_data_array, density_data_array])

    else:
        isat_vf_dataset = xr.merge([isat_data_array, sound_speed_data_array, density_data_array])

    isat_vf_dataset.to_netcdf(flux_nc_folder +
                              filename.replace(hdf5_path, '').replace('.hdf5', '') + '.nc')
    return isat_vf_dataset


def get_density_data(filename, hdf5_folder, isat_data_array, shot_array):
    """
    Auxiliary function to `get_isat_vf`. Retrieves electron temperature data from langmuir
    analysis, uses it to compute the sound speed and density data using the saturation current data.
    Upsampling via linear interpolation of the electron temperature data is done to align the coordinates
    of the sound speed dataset with those of the saturation current data array.

    Right now, it requires the user to go through the prompting as if they are about to plot the Langmuir
    analysis plots.

    Parameters
    ----------
    filename : str
        Currently unused.

    isat_data_array : xarray.DataArray or xarray.Dataset
        The saturation current data array calculated in `get_isat_vf`.

    shot_array : numpy.array
        Currently unused.

    Returns
    -------
    `tuple`
        A tuple of `xarray.DataArray` or `xarray.Dataset`,`(sound_speed_data_array, density_data_array)`.

    """
    #todo hardcoded

    # note this only supports the use of 1 hdf5 file at a time, so as to be compatible
    # with the rest of this module

    interferometry_folder = hdf5_folder
    langmuir_nc_folder = hdf5_folder + 'lang_nc/'
    interferometry_mode = 'skip'
    isweep_choices = [[[1, 0], [0, 0]]]
    bimaxwellian = False
    core_radius = 21. * u.cm
    plot_save_folder = " "
    datasets, steady_state_times_runs, hdf5_paths = get_langmuir_datasets(
        langmuir_nc_folder, hdf5_folder, interferometry_folder, interferometry_mode,
        core_radius, bimaxwellian, plot_save_folder)

    c = 299792458*u.m/u.s
    M = 4.002603254 * (931.49410372*u.MeV/(c**2)).to(u.eV*u.s*u.s/u.m/u.m) #helium
    print("mass: ", M.to(u.kg))
    T_e_data = datasets[0]['T_e'].mean(dim = ['probe', 'y', 'face'])
    print("T_e_data:", T_e_data)
    T_e_interp = T_e_data.interp(time=isat_data_array.time, method = 'linear')
    sound_speed_data_array = np.sqrt(T_e_interp/M)

    sound_speed_data_array.attrs['units'] = 'm/s'
    sound_speed_data_array.coords['time'].attrs['units'] = 'ms'
    sound_speed_data_array.coords['x'].attrs['units'] = 'cm'

    area = 5 * u.mm ** 2  # area of probe, I think -- Leo said 1, Phil suggested off by x5 (circle vs cylinder)
    area = area.to(u.m ** 2) # but i feel like there was an extra factor of 2
    e_plus = 1.60217663e-19*u.C
    e = np.exp(1)

    density_data_array = (1e-6*isat_data_array*np.sqrt(e)/(sound_speed_data_array*area*e_plus)).rename('density')
    # density_data_array = (isat_data_array / (sound_speed_data_array * area * e_plus)).rename('density')
    density_data_array.attrs['units'] = '$cm^{-3}$'
    return sound_speed_data_array, density_data_array

def nan_filter(data, time):
    #todo I don't think this is ever used-- double check and then delete

    # finds the number of nans up front
    nan_front_count = 0
    found_first_not_nan = False
    first_not_nan_index = 0
    while found_first_not_nan == False:
        if not np.isnan(data[first_not_nan_index]):
            found_first_not_nan = True
        elif first_not_nan_index >= len(data):
            raise ValueError("Can't find not nan values")
        else:
            first_not_nan_index +=1

    found_last_not_nan = False
    last_not_nan_index = len(data)-1
    while found_last_not_nan == False:
        if not np.isnan(data[last_not_nan_index]):
            found_last_not_nan = True
        elif last_not_nan_index == 0:
            raise ValueError("Can't find not nan values")
        else:
            last_not_nan_index += -1

    data = data[first_not_nan_index:last_not_nan_index]
    time = time[first_not_nan_index:last_not_nan_index]

    intermediate_nan_count = 0
    for i in range(len(time)):
        if not np.isnan(data[i]):
            continue
        else:
            data[i] = 0.5*(data[i-1] + data[i+1])
            intermediate_nan_count += 1
            if np.isnan(data[i]):
                raise ValueError("Multiple nan values in a row")

    #print(intermediate_nan_count)
    return data, time


def derivative(y_values, dx):
    #todo delete if this is used nowhere

    n=len(y_values)
    derivative_matrix = lil_matrix((n, n))
    derivative_matrix.setdiag(0.5/dx, 1)
    derivative_matrix.setdiag(-0.5/dx, -1)
    derivative_matrix[0, 0] = -1/dx
    derivative_matrix[0, 1] = 1/dx
    derivative_matrix[n-1, n-2] = -1/dx
    derivative_matrix[n-1, n-1] = 1/dx
    derivative_matrix = derivative_matrix.tocsr()
    return derivative_matrix*y_values

