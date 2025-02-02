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

def get_fft(time_series, sample_rate = 1, bin = None, plot = False):
    """
    Assumes a real time series- returns 1 sided transform

    Parameters
    ----------
    time_series
    sample_rate

    Returns
    -------

    """
    if bin is not None:
        try:
            assert type(time_series) is xr.DataArray or time_series is xr.Dataset
        except AssertionError:
            print('Your data type: ', type(time_series))
            raise ValueError('Only xarray DataArray or Dataset are currently supported for binning')
        if not isinstance(bin, tuple):
            time_series = time_series.sel(time=slice(bin+0.02, bin+1-0.02)).values
        else:
            fts = []
            #len_fts = []
            for b in range(bin[0], bin[1]):
                ft, freq = get_fft(time_series, sample_rate=sample_rate, bin=b, plot=False)
                fts.append(ft)
                #len_fts.append(len(ft))
            ft=np.mean(fts, axis=0)

            if plot:
                fig = plt.figure()
                plt.plot(freq, np.abs(ft))
                plt.yscale('log')
                plt.xscale('log')
                fig.show()
            return ft, freq

    if type(time_series) is xr.DataArray or time_series is xr.Dataset:
        times=time_series.coords['time'].values
        sample_rate=1000/(times[1]-times[0])
        time_series = time_series.values

    ft = fft(time_series)
    freq = fftfreq(len(time_series), 1/sample_rate)
    ft_index = int(len(ft) / 2)
    if len(time_series) % 2 == 0:
        ft_index += -1
    ft = ft[1:ft_index]
    freq = freq[1:ft_index] / (2*np.pi)

    if plot:
        fig = plt.figure()
        plt.plot(freq, np.abs(ft))
        plt.yscale('log')
        plt.xscale('log')
        fig.show()

    return ft, freq

def get_psd(time_series, sample_rate = 1, bin = None, plot = False):

    ft, freq = get_fft(time_series, sample_rate = sample_rate, bin = bin)
    ps = 2*abs(ft)**2
    psd = ps/(len(time_series)*sample_rate)

    if plot:
        fig = plt.figure()
        plt.plot(freq, np.abs(psd))
        plt.yscale('log')
        plt.xscale('log')
        fig.show()

    return psd, freq

def get_cross_spectrum(time_series1, time_series2, sample_rate = 1, bin = None, plot = False):

    ft1, freq = get_fft(time_series1, sample_rate = sample_rate, bin = bin)
    ft2 = get_fft(time_series2, sample_rate = sample_rate, bin = bin)[0]

    cross_spectrum = np.conjugate(ft1)*ft2

    return cross_spectrum, freq

def get_cross_phase(time_series1, time_series2, sample_rate = 1, bin = None, plot = False):

    cross_spec, freq = get_cross_spectrum(time_series1, time_series2, sample_rate = sample_rate, bin = bin)

    cross_phase = np.cos(np.angle(cross_spec))

    return cross_phase, freq

def get_cross_phase_spectrogram(data1, data2, bin, plot=False, axis=None):
    """

    Parameters
    ----------
    data1
    data2

    Returns
    -------

    """
    if type(bin) is type((1, 2)):
        assert bin[1]>bin[0], 'select valid bin range'
        cross_phase = []
        for i in range(bin[0], bin[1]):
            cross_phases, x_positions, freq = get_cross_phase_spectrogram(data1, data2, bin = i, plot = False)
            cross_phase.append(cross_phases)
        cross_phases = np.mean(np.array(cross_phase), axis=0)

    else:
        sample_rate = 1000*(data1.coords['time'].values[1]-data1.coords['time'].values[0])**(-1)
        x_positions = data1.coords['x'].values
        shot = data1.coords['shot'].values

        cross_phases = []
        for x_pos in x_positions:
            cross_phases_for_this_x = []
            for shot in shot:
                phase, freq = get_cross_phase(data1.sel(x=x_pos, shot = shot), data2.sel(x=x_pos, shot = shot),
                                              sample_rate = sample_rate, bin = bin)
                cross_phases_for_this_x.append(phase)
            cross_phase_for_this_x = np.mean(cross_phases_for_this_x, axis = 0)
            cross_phases.append(cross_phase_for_this_x)

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis
        plt.contourf(x_positions, freq, np.transpose(cross_phases), cmap='plasma')
        plt.colorbar()
        ax.set_yscale('log')
        ax.set_ylabel('frequency (Hz)')
        ax.set_xlabel('x position (cm)')
        ax.set_title('cross phase (' + data1.name + ', ' + data2.name + '), bins ' + str(bin))
        if axis is None:
            fig.show()
    return cross_phases, x_positions, freq

def get_cross_spectrogram(data1, data2, bin, plot=False, axis=None):
    if type(bin) is type((1, 2)):
        assert bin[1] > bin[0], 'select valid bin range'
        cross_spectrogram = []
        for i in range(bin[0], bin[1]):
            cross_spectra, x_positions, freq = get_cross_spectrogram(data1, data2, bin = i)
            cross_spectrogram.append(cross_spectra)
        cross_spectra = np.mean(np.array(cross_spectrogram), axis = 0)

    else:
        sample_rate = 1000 * (data1.coords['time'].values[1] - data1.coords['time'].values[0]) ** (-1)
        x_positions = data1.coords['x'].values
        shot = data1.coords['shot'].values

        cross_spectra = []
        for x_pos in x_positions:
            cross_spectra_for_this_x = []
            for shot in shot:
                spectrum, freq = get_cross_spectrum(data1.sel(x=x_pos, shot=shot), data2.sel(x=x_pos, shot=shot),
                                              sample_rate=sample_rate, bin=bin)
                spectrum = 2*abs(spectrum)**2 / (len(data1.coords['time'].values)**2)
                cross_spectra_for_this_x.append(spectrum)
            cross_spectrum_for_this_x = np.mean(cross_spectra_for_this_x, axis=0)
            cross_spectra.append(cross_spectrum_for_this_x)

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis
        plt.contourf(x_positions, freq, np.transpose(cross_spectra), locator = ticker.LogLocator(), cmap='plasma')
        plt.colorbar()
        ax.set_yscale('log')
        ax.set_ylabel('frequency (Hz)')
        ax.set_xlabel('x position (cm)')
        ax.set_title('cross spectrogram (' + data1.name + ', ' + data2.name + ')')
        if axis is None:
            fig.show()
    return cross_spectra, x_positions, freq

def get_fft_from_data(data, time=None, shot=None, x=None, bin=None, plot=False, axis=None):
    time_series, params_desc = get_time_series(data, time=time, shot=shot, x=x)
    times = data.coords['time'].values
    sample_rate = 1000/(times[1]-times[0]) #hardcoded unit conversion
    ft, freq = get_fft(time_series, sample_rate=sample_rate, bin=bin)

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(freq, abs(ft), color='black')
        ax.set_xlabel('frequency (Hz)')
        ax.set_ylabel('amplitude ('+data.attrs['units']+'/Hz)')
        ax.set_title(data.name+' fft\n time series '+params_desc+'\n bin: '+str(bin), fontsize=8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        if axis is None:
            fig.show()

def get_psd_from_data(data, time=None, shot=None, x=None, bin=None, plot=False, plot_power_law=None,
                      power_law_freq=None, axis=None):
    times=data.coords['time'].values
    sample_rate=1000/(times[1]-times[0])
    time_series, params_desc, std = get_time_series(data, time=time, shot=shot, x=x)
    psd, freq = get_psd(time_series, sample_rate=sample_rate, bin=bin)

    if plot_power_law is not None:
        assert power_law_freq is not None, 'give a frequency in the range the power law should apply'
        print('fi', np.where(abs(freq-power_law_freq)<100)[0][0])
        psd_start = psd[np.where(abs(freq-power_law_freq)<100)[0][0]]
        line = psd_start * (freq/power_law_freq)**(plot_power_law)

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis
        ax.plot(freq, psd, color='black')
        ax.set_xlabel('frequency (Hz)')
        ax.set_ylabel('psd ('+data.attrs['units']+'^2/Hz)')
        if plot_power_law is not None:
            ax.plot(freq, line, 'darkgreen', label=str(plot_power_law)+' power scaling')
            ax.legend()
        ax.set_title(data.name+' psd\n time series '+params_desc+'\n bin: '+str(bin), fontsize=8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(np.min(psd)/5, np.max(psd)*5)
        if axis is None:
            fig.show()

    return psd, freq

def get_profile(data, time=None, shot=None, x=None, plot=False, axis=None):
    #if time and shot are none, average over both
    #if time or shot are some number, then it will be a specific time or shot
    #if time or shot are tuples, then it will averaged over the range specified by the tuple
    if axis is not None and not plot: plot = True
    if time is None:
        profile = data.mean(dim = ['time'])
        params_desc = 'averaged over all time'
    elif isinstance(time, tuple):
        profile = data.sel(time=slice(time[0], time[1])).mean(dim=['time'])
        params_desc = 'averaged over times: '+str(time)+'ms'
    else:
        profile = data.sel(time=time, method='nearest')
        params_desc = 'at time: '+str(time)+'ms'
    if shot is None:
        #calculates error bar
        std = profile.std(dim = ['shot'], ddof=1)
        profile = profile.mean(dim = ['shot'])
        params_desc = params_desc+'\naveraged over all shot'
    elif isinstance(shot, tuple):
        std = profile.sel(shot=slice(shot[0], shot[1])).std(dim=['shot'], ddof=1)
        profile = profile.sel(shot=slice(shot[0], shot[1])).mean(dim=['shot'])
        params_desc = params_desc + '\naveraged over shot: '+str(shot)
    else:
        std = None
        profile = profile.sel(shot=shot, method='nearest')
        params_desc = params_desc + '\nat shot: '+str(shot)

    if x is None:
        x_array = data.coords['x']
    else:
        x_array = data.coords['x'].sel(x=slice(x[0], x[1]))
        std = std.sel(x=slice(x[0], x[1]))
        profile = profile.sel(x=slice(x[0], x[1]))

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis
        ax.set_xlabel('x position (cm)')
        ax.set_ylabel(data.name+" ("+data.attrs['units']+")")
        ax.set_title('radial '+data.name+' profile\n'+params_desc)
        ax.errorbar(x_array, profile, yerr=std, color='black', linestyle='', marker='o', capsize=0, markersize=3)
        if axis is None:
            fig.show()

    return profile, std, params_desc

def get_time_series(data, time=None, shot=None, x=None, plot=False, axis=None):
    if x is None:
        time_series = data.mean(dim = ['x'])
        params_desc = 'averaged over all x'
    elif isinstance(x, tuple):
        time_series = data.sel(x=slice(x[0], x[1])).mean(dim=['x'])
        params_desc = 'averaged over x: '+str(x)+'cm'
    else:
        time_series = data.sel(x=x, method='nearest')
        params_desc = 'at x: '+str(x)+'cm'
    if shot is None:
        std = time_series.std(dim = ['shot'], ddof=1)
        time_series = time_series.mean(dim = ['shot'])
        params_desc = params_desc+', averaged over all shot'
    elif isinstance(shot, tuple):
        std = time_series.sel(shot=slice(shot[0], shot[1])).std(dim=['shot'], ddof=1)
        time_series = time_series.sel(shot=slice(shot[0], shot[1])).mean(dim=['shot'])
        params_desc = params_desc + ', averaged over shot: '+str(shot)
    else:
        std = None
        time_series = time_series.sel(shot=shot, method='nearest')
        params_desc = params_desc + ' at shot: '+str(shot)

    if time is None:
        time_array = data.coords['time']
    else:
        time_array = data.coords['time'].sel(time=slice(time[0], time[1]))
        time_series = time_series.sel(time=slice(time[0], time[1]))

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis
        ax.set_xlabel('time (ms)')
        ax.set_ylabel(data.name+" ("+data.attrs['units']+")")
        ax.set_title(data.name+' time series\n'+params_desc)
        ax.errorbar(time_array, time_series, yerr=std, color='fuchsia', linestyle='', capsize=1, alpha = 0.5)
        ax.plot(time_array, time_series, color='black', linestyle='', marker='o', markersize=1)
        if axis is None:
            fig.show()

    return time_series, params_desc, std

def get_contour(data, time=None, shot=None, x=None, plot=True, axis=None):

    data_unit = data.attrs['units']
    if shot is None:
        data = data.mean(dim = ['shot'])
        params_desc = 'averaged over all shot'
    elif isinstance(shot, tuple):
        data = data.sel(shot=slice(shot[0], shot[1])).mean(dim=['shot'])
        params_desc = 'averaged over shot: '+str(shot)
    else:
        data = data.sel(shot=shot, method='nearest')
        params_desc = 'at shot: '+str(shot)

    if time is None:
        time_array = data.coords['time']
    else:
        data = data.sel(time=slice(time[0], time[1]))
        time_array = data.coords['time'].sel(time=slice(time[0], time[1]))

    if x is None:
        x_array = data.coords['x']
    else:
        data = data.sel(x=slice(x[0], x[1]))
        x_array = data.coords['x'].sel(x=slice(x[0], x[1]))

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis
        c=ax.contourf(time_array, x_array, data.values)
        plt.colorbar(c, ax=ax, label=data.name+" ("+data_unit+")")
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('x position (cm)')
        ax.set_title(data.name+' '+params_desc)
        if axis is None:
            fig.show()

def get_avg_flux_amplitude(data, time=None, shot=None, x=None, bin=None, freq_slice=None):
    psd, freq = get_psd_from_data(data, time=time, shot=shot, x=x, bin=bin)

    if freq_slice is not None:
        assert isinstance(freq_slice, tuple), 'freq_slice must be a tuple'
        indx1 = np.where(freq>freq_slice[0])[0][0]
        indx2 = np.where(freq<freq_slice[1])[0][-1]
        psd = psd[indx1:indx2]

    df = freq[1]-freq[0]
    return 2*np.sqrt(np.sum(psd)*df)

def linear_fit_profile(data, x=(-30, -20), time=None, shot=None, plot=False, axis=None):
    if data.name == 'density' and data.units == '$m^{-3}$':
        print('adjusted units in fit profile')
        data = 1e-6 * data
        data.attrs['units'] = '$cm^{-3}$'
    assert data.units == '$cm^{-3}$' or data.name != 'density'
    profile,  std, params_desc = get_profile(data, time=time, shot=shot, plot=True)
    yerr = std.sel(x=slice(x[0], x[1]))
    ydata = profile.sel(x=slice(x[0], x[1]))
    x_array = ydata.coords['x'].values
    y = ydata.values
    y_err = yerr.values
    linear_model = lambda x, slope, intercept: slope * x + intercept


    x_scale = np.mean(np.abs(x_array))
    y_scale = np.mean(np.abs(y))
    x_scaled = x_array / x_scale
    y_scaled = y / y_scale
    yerr_scaled = yerr / y_scale
    slope_guess = 0.5 * np.mean((y_scaled[1:] - y_scaled[:-1]) / (x_scaled[1] - x_scaled[0]))

    fit_params, covariance_matrix = curve_fit(linear_model, x_scaled, y_scaled, p0=[slope_guess, 0.0],
                                              sigma=yerr_scaled, absolute_sigma=True)
    slope, intercept = fit_params
    cov_slope_intercept = covariance_matrix[1, 0]
    slope_err, intercept_err = np.sqrt(np.diag(covariance_matrix))

    slope *= y_scale/x_scale
    intercept *= y_scale
    slope_err *= y_scale/x_scale
    intercept_err *= y_scale
    cov_slope_intercept *= y_scale*y_scale/x_scale
    print(cov_slope_intercept)
    # cov_slope_intercept = 0

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis
        y_fit = linear_model(x_array, slope, intercept)
        ax.set_xlabel('x position (cm)')
        ax.set_ylabel(data.name + " (" + data.attrs['units'] + ")")
        ax.set_title('radial ' + data.name + ' profile\n' + params_desc)
        ax.errorbar(profile.coords['x'].values, profile.values, yerr=std.values, color='black', linestyle='',
                     marker='o', capsize=1, markersize=2 )
        ax.plot(x_array, y_fit, color='fuchsia', linestyle = 'dotted', label='linear fit')
        ax.legend()
        if axis is None:
            fig.show()

    return slope, intercept, slope_err, intercept_err, cov_slope_intercept

def get_data_over_grad_data(data, x=(-27, -19), time=None, shot=None, plot=False, axis=None):
    if data.name == 'density' and data.units == '$m^{-3}$':
        print('adjusted units in get_dogd')
        data = 1e-6*data
        data.attrs['units'] = '$cm^{-3}$'
    assert data.units == '$cm^{-3}$' or data.name != 'density'
    profile, profile_err, params_desc = get_profile(data, time=time, shot=shot, x=x, plot=False)
    grad, _, grad_err, ___, cov = linear_fit_profile(data, x=x, time=time, shot=shot, plot=False)
    d_grad_d = profile/grad
    d_grad_d_err = np.sqrt( (profile_err/grad)**2 + (profile*grad_err/(grad**2))**2
                            -2*profile*cov/(grad**3))

    print((profile_err/grad)**2)
    print((profile*grad_err/(grad**2))**2)
    print(-2*profile*cov/(grad**3))

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis
        ax.errorbar(profile.coords['x'].values, d_grad_d, yerr=d_grad_d_err, capsize=1, markersize=2,
                    color='black', linestyle='', marker = 'o')
        ax.set_xlabel('x position (cm)')
        ax.set_ylabel(data.name + r"/$\nabla$"+data.name+" (" + data.attrs['units'] + "/cm)")
        ax.set_title(data.name+r"/$\nabla$"+data.name+'\n' + params_desc)
        if axis is None:
            fig.show()

    return d_grad_d, d_grad_d_err

def plot_total_flux_vs_Ln(data, x=(-27, -19), time=None, shot=None, bin=(7,14), plot=False, axis=None):
    if data.name == 'density' and data.units == '$m^{-3}$':
        print('adjusted units in plot')
        data = 1e6*data
        data.attrs['units'] = '$cm^{-3}$'
    assert data.units == '$cm^{-3}$' or data.name != 'density'
    d_grad_d, d_grad_d_err = get_data_over_grad_data(data, x=x, time=time, shot=shot, plot=False)
    profile, _, __ = get_profile(data, time=time, shot=shot, x=x, plot=False)
    total_flux_list = []
    for x_value in d_grad_d.coords['x'].values:
        psd, freq = get_psd_from_data(data, time=time, shot=shot, x=x_value, plot=True, bin=bin)
        dfreq = freq[1]-freq[0]
        total_flux = np.sum(psd)*dfreq
        total_flux_list.append(total_flux)
    normalized_total_flux = np.array(total_flux_list)/profile

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis

        ax.errorbar(d_grad_d, normalized_total_flux, xerr=d_grad_d_err, markersize=3, linestyle='', capsize=1,
                    marker = 'o', color='black')
        #todo hardcoded plot axes
        ax.set_xlabel(r'$L_n = \frac{n_e}{\nabla n_e}$ ($cm$)')
        ax.set_ylabel(r'$\frac{1}{n_e}\int PSD$ ($cm^{-3}$)')
        ax.set_title(r'$\frac{1}{n_e}\int PSD$ vs $L_n$')

        if axis is None:
            fig.show()

    return np.mean(d_grad_d), np.mean(normalized_total_flux)
