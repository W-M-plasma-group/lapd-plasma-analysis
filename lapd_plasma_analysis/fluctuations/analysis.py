
"""
Takes in HDF5 file

gets vsat data, averages over shot (one array per x position)

gets isat data, removes outliers, averages over shot (one array per x position)

gets temperature data, averages over shot (one array per x position)

combines isat and temperature data to get density data

visualizes spectra and cross-spectra, cross-phase


"""
import numpy as np
import astropy.units as u
from bapsflib import lapd
from warnings import warn

from numpy.random import normal

from langmuir.configurations import get_config_id, get_langmuir_config
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.fft import fft, fftfreq, ifft
from langmuir.analysis import get_langmuir_datasets
from experimental import get_exp_params
from langmuir.configurations import get_ion
import xarray as xr
from tqdm import tqdm
from matplotlib.colors import LogNorm
from scipy.sparse import lil_matrix, csr_matrix
from scipy.optimize import curve_fit
from lapd_plasma_analysis.fluctuations.fourier import *

C1 = ['#115740', '#B9975B']
C2 = ['#F0B323', '#D0D3D4']
C3 = ['#00B388', '#CAB64B', '#84344E', '#64CCC9', '#E56A54', '#789D4A',
      '#789F90', '#5B6770', '#183028', '#00313C']


def get_isat_vf(filename, hdf5_path, flux_nc_folder):
    """
    Obtains saturation current, floating potential, floating potential difference, sound speed,
    and density data

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
    assert get_config_id(lapd.File(filename).info['exp name']) in [1]

    file = lapd.File(filename)
    receptacle = 1  #todo: hardcoded from config

    isat_data = file.read_data(2, 1, silent=True, add_controls=[('6K Compumotor', receptacle)])
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

    receptacle = 2 # hard coded from configurations

    vf_top_data = file.read_data(2, 2, silent=True, add_controls=[('6K Compumotor', receptacle)])
    vf_bottom_data = file.read_data(2, 3, silent=True, add_controls=[('6K Compumotor', receptacle)])
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
            dvf_vs_x[x_index].append((vf_top_data[index][1] - vf_bottom_data[index][1]) * 10)  # /10 in exp
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

    sound_speed_data_array, density_data_array = get_density_data(filename, isat_data_array, shot_array)
    isat_vf_dataset = xr.merge([isat_data_array, dvf_data_array,
                                vf_data_array, sound_speed_data_array, density_data_array])

    isat_vf_dataset.to_netcdf(flux_nc_folder +
                              filename.replace(hdf5_path, '').replace('.hdf5', '') + '.nc')
    return isat_vf_dataset


def get_density_data(filename, isat_data_array, shot_array):
    #todo hardcoded

    # note this only supports the use of 1 hdf5 file at a time, so as to be compatible
    # with the rest of this module

    hdf5_folder = '/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/HDF5 Files/March_2022_HDF5 and NetCDF/'
    interferometry_folder = hdf5_folder
    langmuir_nc_folder = hdf5_folder + 'lang_nc/'
    interferometry_mode = 'skip'
    isweep_choices = [[[1, 0], [0, 0]]]
    bimaxwellian = False
    core_radius = 21. * u.cm
    plot_save_folder = " "
    datasets, steady_state_times_runs, hdf5_paths = get_langmuir_datasets(
        langmuir_nc_folder, hdf5_folder, interferometry_folder, interferometry_mode,
        isweep_choices, core_radius, bimaxwellian, plot_save_folder)

    c = 299792458*u.m/u.s
    M = 4.002603254 * (931.49410372*u.MeV/(c**2)).to(u.eV*u.s*u.s/u.m/u.m)
    T_e_data = datasets[0]['T_e'].mean(dim = ['probe', 'y', 'face'])
    T_e_interp = T_e_data.interp(time=isat_data_array.time, method = 'linear')
    sound_speed_data_array = np.sqrt(T_e_interp/(M*2*np.pi))

    sound_speed_data_array.attrs['units'] = 'm/s'
    sound_speed_data_array.coords['time'].attrs['units'] = 'ms'
    sound_speed_data_array.coords['x'].attrs['units'] = 'cm'

    area = 5 * u.mm ** 2  # area of probe, I think -- Leo said 1, Phil suggested off by x5 (circle vs cylinder)
    area = area.to(u.m ** 2) # but i feel like there was an extra factor of 2
    e_plus = 1.60217663e-19*u.C
    e = np.exp(1)

    density_data_array = (1e6*isat_data_array*np.sqrt(e)/(sound_speed_data_array*area*e_plus)).rename('density')
    # density_data_array = (isat_data_array / (sound_speed_data_array * area * e_plus)).rename('density')
    density_data_array.attrs['units'] = '$cm^{-3}$'
    return sound_speed_data_array, density_data_array

def nan_filter(data, time):

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


#make_cross_spectrogram_averaged(data['dvf'], data['isat'], bins = (5,17), plot = True)
# note Conor took the mean over shot and radial position of the conj(F_1)*F_2 and the cross phase was the
# argument of the result
# I think the binning may be the issue, although I don't understand why
# print('ran')





def derivative(y_values, dx):
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

if __name__ == '__main__':
    import os

    files = os.listdir('/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/'
                           'HDF5 Files/March_2022_HDF5 and NetCDF/')

    filenames = []
    for file in files:
        if file.endswith('.hdf5'):
            filenames.append(file)

    filename_0 = ('/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/'
                'HDF5 Files/March_2022_HDF5 and NetCDF/18_line_valves105V_7000A.hdf5')

    hdf5_path = ('/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/'
                 'HDF5 Files/March_2022_HDF5 and NetCDF/')

    flux_nc_path = ('/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/'
                    'HDF5 Files/March_2022_HDF5 and NetCDF/flux_nc/')

    plot_save_folder = '/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/plot_save_folder_nov14/'

    # print(filenames)
    #
    # Lns = []
    # psdsums = []
    # for filename in filenames:
    #     print('get_isat_vf')
    #     print('Current file:', filename_0)
    #     try:
    #         dat = get_isat_vf(filename_0, hdf5_path, flux_nc_path)
    #     except:
    #         print("Skipped this file due to an error")
    #         continue
    #     print('done get_isat_vf')
    #     data = xr.open_dataset((filename_0).replace('.hdf5', '.nc'))
    #     Ln, psdsum = plot_total_flux_vs_Ln(data['density'], x=(-26, -19))
    #     print(Ln, psdsum)
    #     inp = input('Accept?')
    #     if inp == 'y':
    #         Lns.append(Ln)
    #         psdsums.append(psdsum)
    #
    # print(Lns)
    # print(psdsums)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # ax.plot(Lns, psdsums, marker = 'o', color = 'black')
    # ax.set_xlabel(r'$L_n = \frac{n_e}{\nabla n_e}$ ($cm$)')
    # ax.set_ylabel(r'$\frac{1}{n_e}\int PSD$ ($cm^{-3}$)')
    # ax.set_title(r'$\frac{1}{n_e}\int PSD$ vs $L_n$')
    #
    #
    # fig.show()
    # fig.savefig(plot_save_folder+'mutli_experiment_Ln.png')



    print('get_isat_vf')
    dat = get_isat_vf(filename_0, hdf5_path, flux_nc_path)
    print('done get_isat_vf')

    data = xr.open_dataset('/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/HDF5 Files/March_2022_HDF5 '
                           'and NetCDF/flux_nc/18_line_valves105V_7000A.nc')

    plot_save_folder = '/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/plot_save_folder_nov14/'

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # get_profile(data['density'], axis=ax)
    # linear_fit_profile(data['density'], x=(-27, -19), plot=True, axis=ax)
    # plot_total_flux_vs_Ln(data['density'], axis=ax, x=(-26, -19), plot=True)
    # f=get_psd_from_data(data['density'], axis=ax, plot=True, bin=(7, 14), x=-26)[0]
    # s=get_time_series(data['density'], time=(7, 14), x=-26)[0]
    # print("print", np.sum(f), np.sum(s*s))


    # for i in range(8):
    #     get_profile(data['density'], plot=True, shot=i, time=(2,6), axis=ax)

    fig.show()
    fig.savefig(plot_save_folder+'norm_int_psd_vs_Ln.png')

    #Diagnostics selected: [np.str_('n_e_cal'), np.str_('n_i_cal'), np.str_('n_i_OML_cal'), np.str_('n_i'), np.str_('n_i_OML')]
    # hdf5_folder = "/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/HDF5 Files/March_2022_HDF5 and NetCDF/"
    # interferometry_folder = ("/Users/leomurphy/lapd-data/November_2022/uwave_288_GHz_waveforms/"
    #                          if "November_2022" in hdf5_folder else hdf5_folder)
    # interferometry_mode = "skip"
    # isweep_choices = [[[1, 0], [0, 0]],     # . 1st combination to plot: 1 * (first face on first probe)
    #                   [[0, 0], [1, 0]]]     # . 2nd combination to plot: 1 * (first face on second probe)
    # bimaxwellian = False
    # core_radius = 21. * u.cm
    # langmuir_nc_folder = hdf5_folder + "lang_nc/"
    # mach_nc_folder = hdf5_folder + "mach_nc/"
    # from lapd_plasma_analysis.langmuir.plots import *
    #
    # datasets, steady_state_times_runs, hdf5_paths = get_langmuir_datasets(
    #         langmuir_nc_folder, hdf5_folder, interferometry_folder, interferometry_mode, isweep_choices,
    #         core_radius, bimaxwellian, plot_save_directory=' ')
    #
    # plot_diagnostic = np.str_('n_i_OML')
    # multiplot_linear_diagnostic(datasets, plot_diagnostic, isweep_choices, 'time',
    #                                         steady_state_by_runs=steady_state_times_runs, core_rad=core_radius,
    #                                         save_directory=plot_save_folder)