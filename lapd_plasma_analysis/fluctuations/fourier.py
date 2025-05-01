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
from scipy.signal import welch
from datetime import datetime
from time import sleep

try:
    from nmmn.plots import parulacmap
    parula = parulacmap()
    cmap = parula
except:
    cmap = "plasma"




SMALL_SIZE = 12 # font size code from Pedro Duarte on stack exchange
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def get_time():
    """

    Useful for attaching unique numerical string to the end of the name of plots.

    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_fft(time_series, scaling="spectrum", dt=1, bin=None, plot=False):
    """
    Performs a 1D, one-sided FFT of given time series data.

    Parameters
    ----------
    time_series : `xarray.DataArray` or `xarray.Dataset` or `numpy.array`
        Time series data.

    dt : `float`
        Time between successive data points in the time series.

    bin : `tuple` or `int` or `None`
        If not given, the FFT of the entire data will be returned.
        If it is an integer, the FFT of the data in the time window (bin+0.02, bin+1-0.02)
        will be returned.
        If it is a tuple of integers, the returned spectra will be the averaged FFT result of
        the data in each of the time windows corresponding to the range of integers specified by
        the tuple.

    for_power_spec: `bool`
        If true, it will modulus square each of the spectra as they are returned-- if
        bin is a tuple, the averaged spectra will then be the average of the squared
        spectra

    plot: `bool`
        If true, it will plot the FFT of the data in the time series as it is computed.


    Returns
    _______
    `tuple`
        `ft, freq, dt` where `ft` is the spectra, `freq` is the frequency domain, and `dt` is the
        time separation between data points in the time series.

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
                ft, freq= get_fft(time_series, dt=dt, bin=b, plot=False, scaling=scaling)
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

    if isinstance(time_series, xr.DataArray) or isinstance(time_series, xr.Dataset):
        times=time_series.coords['time'].values
        dt = (times[1]-times[0])/1000 #converts to seconds
        time_series = time_series.values

    ft = fft(time_series)
    freq = fftfreq(len(time_series), dt)
    ft_index = int(len(ft) / 2)
    if len(time_series) % 2 == 0:
        ft_index += -1
    spec = ft[0:ft_index]
    freq = freq[0:ft_index]

    if scaling=="amplitude":
        spec = (abs(spec) / len(time_series))
        dc, nyquist = spec[0], spec[-1]
        spec = 2*spec
        spec[0], spec[-1] = dc, nyquist

    if scaling=="power spectrum":
        spec = (abs(spec) / len(time_series)) ** 2
        dc, nyquist = spec[0], spec[-1]
        spec = 2*spec
        spec[0], spec[-1] = dc, nyquist

    if scaling=="psd":
        spec = (abs(spec)**2 / (len(time_series)*(1/dt)))
        dc, nyquist = spec[0], spec[-1]
        spec = 2*spec
        spec[0], spec[-1] = dc, nyquist

    # print("other dt: ", dt)
    # print("length of time: ", len(time_series)*dt)
    # freq, spec = welch(time_series, 1/dt, scaling=scaling, nperseg=len(time_series)//32, detrend="linear")
    # var = np.var(time_series)
    # total_power = np.trapezoid(spec, freq)
    # print("variance: ", np.var(time_series))
    # print(var, total_power, total_power / var)

    if plot:
        fig = plt.figure()
        plt.plot(freq, spec)
        plt.yscale('log')
        plt.xscale('log')
        fig.show()

    return spec, freq

def get_psd(time_series, dt=1, bin=None, plot=False):
    """
        Performs a 1D, one-sided PSD of given time series data.

        Parameters
        ----------
        time_series : `xarray.DataArray` or `xarray.Dataset` or `numpy.array`
            Time series data.

        dt : `float`
            Time between successive data points in the time series.

        bin : `tuple` or `int` or `None`
            If not given, the PSD of the entire data will be returned.
            If it is an integer, the PSD of the data in the time window (bin+0.02, bin+1-0.02)
            will be returned.
            If it is a tuple of integers, the returned spectra will be the averaged PSD result of
            the data in each of the time windows corresponding to the range of integers specified by
            the tuple.

        plot: `bool`
            If true, it will plot the PSD of the data in the time series as it is computed.


        Returns
        -------
        `tuple`
            `psd, freq where `psd` is the power spectral density and `freq` is the frequency domain.

        """

    ft, freq, dt = get_fft(time_series, dt=dt, for_power_spec=True, bin=bin)
    psd = ft/(len(freq)*dt)
    print("time spacing in s: ", dt)
    print("time over which psd computed in ms: ", dt*len(freq)*2*1000)
    print("integrated psd: ", np.sum(psd)*(freq[1]-freq[0]))

    if plot:
        fig = plt.figure()
        plt.plot(freq, np.abs(psd))
        plt.yscale('log')
        plt.xscale('log')
        fig.show()

    return psd, freq

def get_cross_spectrum(time_series1, time_series2, sample_rate = 1, bin = None, plot = False):

    ft1, freq, dt = get_fft(time_series1, bin = bin)
    ft2, freq, dt = get_fft(time_series2, bin = bin)

    cross_spectrum = np.conjugate(ft1)*ft2

    return cross_spectrum, freq

def get_cross_phase(time_series1, time_series2, sample_rate = 1, bin = None, plot = False):

    cross_spec, freq = get_cross_spectrum(time_series1, time_series2, bin = bin)

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
        plt.contourf(x_positions, freq, np.transpose(cross_phases), cmap=cmap)
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
        plt.contourf(x_positions, freq, np.transpose(cross_spectra), locator = ticker.LogLocator(), cmap=cmap)
        plt.colorbar()
        ax.set_yscale('log')
        ax.set_ylabel('frequency (Hz)')
        ax.set_xlabel('x position (cm)')
        ax.set_title('cross spectrogram (' + data1.name + ', ' + data2.name + ')')
        if axis is None:
            fig.show()
    return cross_spectra, x_positions, freq

def get_fft_from_data(data, time=None, shot=None, x=None, bin=None, plot=False, axis=None, plot_save_folder=None):
    """
        Performs a 1D, one-sided FFT of given time series data, given just the `xarray.DataArray` from the
        NetCDF file.

        Parameters
        ----------
        data : `xarray.DataArray`
            Time series data array.

        time : `tuple` or `None`
            It's probably best to leave this as none, and specify the bins to compute the FFT over.
            If `bin` is left as none, this is useful to compute the FFT over a given time range, and
            it should be given as a tuple ex. `(7.1, 7.5)` (values are given in ms)

        shot : `int` or `tuple` or `None`
            Determines which shots to use in averaging. If `None`, all 8 are used. If an integer `n`, then
            the `n-1`th shot is used. If a tuple, uses all shots in between the bounds specified by the tuple.
            ex. `(1, 4)` will use shots 2, 3, 4, 5. (Indexing starts at 0). This averaging happens to the time
            series data, not to the FFT-- it's a good idea to choose an integer.

        x : `float` or `tuple` or `None`
            If `None`, the FFT is computed from a time series which is averaged over all radial positions.
            If `tuple`, say (x1, x2), the FFT is computed from a time series which is averaged over all x
            in between x1 and x2.
            It is recommended to provide a `float`, since time series averaging will remove data from the
            spectrum. This provides the FFT using only data obtained at the given x position.

        bin : `tuple` or `int` or `None`
            If not given, the FFT of the entire data will be returned.
            If it is an integer, the FFT of the data in the time window (bin+0.02, bin+1-0.02)
            will be returned.
            If it is a tuple of integers, the returned spectra will be the averaged FFT result of
            the data in each of the time windows corresponding to the range of integers specified by
            the tuple.


        plot: `bool`
            If true, it will plot the FFT of the data in the time series as it is computed.

    """
    time_series, params_desc, std = get_time_series(data, time=time, shot=shot, x=x)
    dt = (data.coords['time'].values[1] - data.coords['time'].values[0]) / 1000
    ft, freq, dt = get_fft(time_series, dt=dt, bin=bin)

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
        ax.set_title(data.name+' fft\n time series '+params_desc+'\n bin: '+str(bin))
        ax.set_xscale('log')
        ax.set_yscale('log')
        if axis is None:
            fig.show()

def get_spectrum_from_data(data, time=None, shot=None, x=None, bin=None, plot=False, plot_power_law=None,
                      scaling="power spectrum", power_law_freq=None, axis=None, plot_save_folder=None):
    """
        Performs a 1D, one-sided PS of given time series data, given just the `xarray.DataArray` from the
        NetCDF file.

        Parameters
        ----------
        data : `xarray.DataArray`
            Time series data array.

        time : `tuple` or `None`
            It's probably best to leave this as none, and specify the bins to compute the PS over.
            If `bin` is left as none, this is useful to compute the PS over a given time range, and
            it should be given as a tuple ex. `(7.1, 7.5)` (values are given in ms)

        shot : `int` or `tuple` or `None`
            Determines which shots to use in averaging. If `None`, all 8 are used. If an integer `n`, then
            the `n-1`th shot is used. If a tuple, uses all shots in between the bounds specified by the tuple.
            ex. `(1, 4)` will use shots 2, 3, 4, 5. (Indexing starts at 0). This averaging happens to the time
            series data, not to the PS-- it's a good idea to choose an integer.

        x : `float` or `tuple` or `None`
            If `None`, the PS is computed from a time series which is averaged over all radial positions.
            If `tuple`, say (x1, x2), the PS is computed from a time series which is averaged over all x
            in between x1 and x2.
            It is recommended to provide a `float`, since time series averaging will remove data from the
            spectrum. This provides the PSD using only data obtained at the given x position.

        bin : `tuple` or `int` or `None`
            If not given, the PS of the entire data will be returned.
            If it is an integer, the PS of the data in the time window (bin+0.02, bin+1-0.02)
            will be returned.
            If it is a tuple of integers, the returned spectra will be the averaged PSD result of
            the data in each of the time windows corresponding to the range of integers specified by
            the tuple.

        plot: `bool`
            If true, it will plot the PS of the data in the time series as it is computed.

        plot_power_law: `float` or `None`
            Plots a line representing the power law specified by `float`, say 5/3 or 7/3.

        power_law_freq: `float` or `None`
            Give any frequency in the frequency range where the power law provided in plot_power_law
            is expected to apply.

        axis : `matplotlib.axes.Axes` or `None`
            If `None`, this function will create its own figure. Supply axis if the output will be a smaller part of
            an existing figure.

        plot_save_folder: `str` or `None`
            If `None`, the plot will not be saved. The `string`, if provided, should be the file path to the
            directory where the plot will be saved. The path should end with a `/`.

        Returns
        -------
        `tuple`
            `(ps, freq)` where `ps` is the PS of the data in the time series and `freq` is the frequency
            domain

    """
    dt = (data.coords['time'].values[1] - data.coords['time'].values[0])/1000 #todo hardcoded to change to seconds

    tseries, params_desc, _ = get_time_series(data, time=time, shot=shot, x=x)
    err = None

    if x is not None and not isinstance(x, tuple):
        if shot is not None and not isinstance(shot, tuple):
            spec, freq = get_fft(tseries, dt=dt, bin=bin, scaling=scaling)
        if shot is None:
            shot = (0, 7) #todo potentially hardcoded- must be changed if the number of shots at each position is not 8
        if isinstance(shot, tuple):
            spectra = []
            for s in range(shot[0], shot[1]):
                tseries, _, _ = get_time_series(data, time=time, shot=s, x=x)
                spec, freq = get_fft(tseries, dt=dt, bin=bin, scaling=scaling)
                spectra.append(spec)
            spec = np.mean(spectra, axis=0)
            err = np.std(spectra, axis=0)

    if isinstance(x, tuple):
        spectra = []
        errors = []
        for xval in tqdm(range(x[0], x[1]+1), desc="Averaging..."):
            spec, freq, err = get_spectrum_from_data(data, time=time, shot=shot, x=xval, bin=bin, scaling=scaling)
            spectra.append(spec)
            errors.append(err)
        spec = np.mean(spectra, axis=0)
        err = np.sqrt(np.mean(errors, axis=0)**2 + np.var(spectra, axis=0))

    if plot_power_law is not None:
        assert power_law_freq is not None, 'give a frequency in the range the power law should apply'
        print('fi', np.where(abs(freq-power_law_freq)<100)[0][0])
        spec_start = spec[np.where(abs(freq-power_law_freq)<100)[0][0]]
        line = spec_start * (freq/power_law_freq)**(plot_power_law)

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis
        if err is not None:
            # ax.plot(freq, spec+err/2, linestyle='dashed', color='black')
            # ax.plot(freq, spec-err/2, linestyle='dashed', color='black')
            ax.fill_between(freq, spec+err, spec-err, edgecolor='olivedrab', facecolor='olive', alpha=0.6)
        ax.plot(freq, spec, color='black')
        ax.set_xlabel('frequency (Hz)')
        ylabel1 = scaling+' ('+data.attrs['units']
        if scaling != "amplitude":
            ylabel2 = '^2)'
        else:
            ylabel2 = ')'
        ax.set_ylabel(ylabel1+ylabel2)
        if plot_power_law is not None:
            ax.plot(freq, line, 'darkgreen', label=str(plot_power_law)+' power scaling')
            ax.legend()
        ax.set_title(data.name+' '+ scaling + '\n time series '+params_desc+'\n bin: '+str(bin))
        ax.set_xscale('log')
        ax.set_yscale('log')
        if axis is None:
            fig.show()
            if plot_save_folder is not None:
                fig.savefig(plot_save_folder+data.name+'_'+scaling+'_'+get_time()+'.png', dpi=150)
                sleep(1)

    return spec, freq, err

def get_radial_spectrogram(data, x=None, shot=None, bin=None, scaling="amplitude", plot=False, axis=None,
                           plot_save_folder=None):
    spectra = []
    assert isinstance(x, tuple) or x is None, "x should be a tuple"
    if x is None:
        x1, x2 = data.coords['x'].values.min(), data.coords['x'].values.max()
        x = (int(x1+0.3), int(x2+0.1))
    x_positions = np.array(range(x[0], x[1]+1))
    for x_pos in x_positions:
        spectrum, freq, err = get_spectrum_from_data(data, x=x_pos, shot=shot, bin=bin, scaling=scaling)
        spectrum = spectrum[1:]
        freq = freq[1:]
        spectra.append(spectrum)

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis
        im = ax.imshow(
            np.transpose(spectra),
            aspect='auto',
            origin='lower',
            extent=(
                x_positions.min(), x_positions.max(),
                freq.min(), freq.max()
            ),
            cmap=cmap,
            norm=LogNorm()
        )
        fig.colorbar(im, ax=ax, label='amplitude spectrum ('+data.attrs['units']+')')
        ax.set_yscale('log')
        ax.set_ylabel('frequency (Hz)')
        ax.set_xlabel('x position (cm)')
        ax.set_title('radial '+scaling+' spectrogram (' + data.name +')')
        if axis is None:
            fig.show()
            if plot_save_folder is not None:
                fig.savefig(plot_save_folder+data.name+'_'+scaling+'_spectrogram_'+get_time()+'.png', dpi=150)
    return spectra, x_positions, freq

def get_profile(data, time=None, shot=None, x=None, plot=False, axis=None, plot_save_folder=None):
    """
        Given the data from an `xarray.DataArray` object (from the NetCDF file), obtains and plots
        the radial profile of the data.

        Parameters
        ----------
        data : `xarray.DataArray`
            Time series data array.

        time : `tuple` or `float` or `None`
            Determines the range of the time series data over which to average. Give a tuple to
            provide a range, or a float to obtain the profile at a given time. The `None` option will
            automatically average over the entire time series.

        shot : `int` or `tuple` or `None`
            Determines which shots to use in averaging. If `None`, all 8 are used. If an integer `n`, then
            the `n-1`th shot is used. If a tuple, uses all shots in between the bounds specified by the tuple.
            ex. `(1, 4)` will use shots 2, 3, 4, 5. (Indexing starts at 0).

        x : `tuple` or `None`
            The range of x values over which to plot the profile. `None` will adjust the domain to be as wide as
            possible, but a smaller range may be specified with a tuple.

        plot: `bool`
            If true, it will plot the profile as it is generated.

        axis : `matplotlib.axes.Axes` or `None`
            If `None`, this function will create its own figure. Supply axis if the output will be a smaller part of
            an existing figure.

        plot_save_folder: `str` or `None`
            If `None`, the plot will not be saved. The `string`, if provided, should be the file path to the
            directory where the plot will be saved. The path should end with a `/`.

        Returns
        -------
        `tuple`
            `(profile, std, params_desc)` where `profile` is the profile of the time series, `std` is the standard
            deviation (square root of the variance of the data with respect to shot), and `params_desc` is a string
            used in the plot to describe how the profile was obtained (over what values were averaged in each
            coordinate).

    """
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
        if not isinstance(shot, tuple) or shot is not None:
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
            if plot_save_folder is not None:
                fig.savefig(plot_save_folder+data.name+'_profile'+get_time()+'.png', dpi=150)
                sleep(1)

    return profile, std, params_desc

def get_time_series(data, time=None, shot=None, x=None, plot=False, axis=None, plot_save_folder=None):
    """
        Given the data from an `xarray.DataArray` object (from the NetCDF file), obtains and plots
        the data versus time.

        Parameters
        ----------
        data : `xarray.DataArray`
            Time series data array.

        time : `tuple` or `None`
            The range of time values over which to plot the data. `None` will adjust the domain to be as long as
            possible, but a smaller range may be specified with a tuple.

        shot : `int` or `tuple` or `None`
            Determines which shots to use in averaging. If `None`, all 8 are used. If an integer `n`, then
            the `n-1`th shot is used. If a tuple, uses all shots in between the bounds specified by the tuple.
            ex. `(1, 4)` will use shots 2, 3, 4, 5. (Indexing starts at 0).

        x : `tuple` or `float` or `None`
            Determines the span of the position data over which to average. Give a tuple to
            provide a range, or a float to obtain the profile at a given location. The `None` option will
            automatically average over the entire range of x values.

        plot: `bool`
            If true, it will plot the time series as it is obtained.

        axis : `matplotlib.axes.Axes` or `None`
            If `None`, this function will create its own figure. Supply axis if the output will be a smaller part of
            an existing figure.

        plot_save_folder: `str` or `None`
            If `None`, the plot will not be saved. The `string`, if provided, should be the file path to the
            directory where the plot will be saved. The path should end with a `/`.

        Returns
        -------
        `tuple`
            `(time_series, params_desc, std)` where `time_series` is the (averaged) time series as an `xarray.DataArray`
            , `std` is the standard deviation (square root of the variance of the data with respect to shot), also an
            `xarray.DataArray`, and `params_desc` is a string used in the plot to describe how the time series was obtained
            (over what values were averaged in each coordinate).

    """
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
        std = std.sel(time=slice(time[0], time[1]))
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
            if plot_save_folder is not None:
                fig.savefig(plot_save_folder+data.name+'_time-series'+get_time()+'.png', dpi=150)
                sleep(1)

    return time_series, params_desc, std

def get_contour(data, time=None, shot=None, x=None, plot=True, axis=None, plot_save_folder=None):
    """
        Given the data from an `xarray.DataArray` object (from the NetCDF file), obtains and plots
        how the profile changes over time on a 2D color plot.

        Parameters
        ----------
        data : `xarray.DataArray`
            Time series data array.

        time : `tuple` or `None`
            The range of time values over which to plot. `None` will adjust the domain to be as long as
            possible, but a smaller range may be specified with a tuple.

        shot : `int` or `tuple` or `None`
            Determines which shots to use in averaging. If `None`, all 8 are used. If an integer `n`, then
            the `n-1`th shot is used. If a tuple, uses all shots in between the bounds specified by the tuple.
            ex. `(1, 4)` will use shots 2, 3, 4, 5. (Indexing starts at 0).

        x : `tuple` or `None`
            The range of x values over which to plot the profile. `None` will adjust the domain to be as wide as
            possible, but a smaller range may be specified with a tuple.

        plot: `bool`
            If true, the plot will appear as it is created.

        axis : `matplotlib.axes.Axes` or `None`
            If `None`, this function will create its own figure. Supply axis if the output will be a smaller part of
            an existing figure.

        plot_save_folder: `str` or `None`
            If `None`, the plot will not be saved. The `string`, if provided, should be the file path to the
            directory where the plot will be saved. The path should end with a `/`.

    """

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
            if plot_save_folder is not None:
                fig.savefig(plot_save_folder+data.name+'_contour'+get_time()+'.png', dpi=150)
                sleep(1)

def get_avg_flux_amplitude(data, time=None, shot=None, x=None, bin=None, freq_slice=None):
    psd, freq = get_spectrum_from_data(data, scaling="psd", time=time, shot=shot, x=x, bin=bin)

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
        psd, freq = get_ps_from_data(data, time=time, shot=shot, x=x_value, plot=False, bin=bin)
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
        ax.set_ylabel(r'$\frac{1}{n_e}\int PS$ ($cm^{-3}$)')
        ax.set_title(r'$\frac{1}{n_e}\int PS$ vs $L_n$')

        if axis is None:
            fig.show()

    return np.mean(d_grad_d), np.mean(normalized_total_flux)

if __name__ == "__main__":
    x = np.linspace(0, 8*np.pi, 10000)
    y = np.sin(x) + 0.5*np.cos(2*x) + 0.5*np.sin(3*x)
    get_fft(y, dt = x[1]-x[0], scaling="spectrum", plot = True)