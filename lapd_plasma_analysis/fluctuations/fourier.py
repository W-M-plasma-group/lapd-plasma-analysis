import numpy as np
import astropy.units as u
from bapsflib import lapd
from warnings import warn

from numpy.random import normal

from lapd_plasma_analysis.langmuir.configurations import get_config_id, get_langmuir_config
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.fft import fft, fftfreq, ifft
from lapd_plasma_analysis.langmuir.analysis import get_langmuir_datasets, print_user_file_choices
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

def get_fft(time_series, scaling="spectrum", dt=1, bin=None, plot=False, returnError=False):
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
                if not np.isnan(ft).all():
                    fts.append(ft)
                #len_fts.append(len(ft))
            if len(fts) <= 1:
                raise ValueError("Bad spectrum")
            ft=np.mean(fts, axis=0)

            ft_err = np.std(fts, axis=0)

            if plot:
                fig = plt.figure()
                plt.plot(freq, np.abs(ft))
                plt.yscale('log')
                plt.xscale('log')
                fig.show()
            if returnError:
                return ft, freq, ft_err
            return ft, freq

    if isinstance(time_series, xr.DataArray) or isinstance(time_series, xr.Dataset):
        times=time_series.coords['time'].values
        dt = (times[1]-times[0])/1000 #converts to seconds
        time_series = time_series.values

    if np.isnan(time_series).any():
        print(time_series)
        print("All nan: ", np.isnan(time_series).all())
        raise ValueError("Bad time series")
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

        # print("---validation---")
        # mom2 = np.mean(time_series**2)
        # int_psd = np.sum(spec)*(freq[1]-freq[0])
        # print("2nd moment", mom2)
        # print("integrated psd", int_psd)
        # print("2nd moment/int psd (should be 1)", mom2/int_psd)
        # print("----------------")

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

def get_fft_from_data(data, time=None, shot=None, x=None, bin=None, scaling="amplitude", plot=False, axis=None, plot_save_folder=None):
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
    ft, freq, ft_err = get_fft(time_series, dt=dt, scaling=scaling, bin=bin, returnError=True)

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

    return ft, freq, ft_err

def lowpass(ft, freq, cutoff_freq=250000):
    """Takes in two arrays of numbers, ft and freq, and returns the same
    arrays but with all values after a certain freq value removed."""
    filter_array = freq > cutoff_freq
    cutoff_index = np.where(filter_array)[0][0]
    return ft[:cutoff_index], freq[:cutoff_index]


def get_spectrum_from_data(data, time=None, shot=None, x=None, bin=None, z=None, plot=False, plot_power_law=None,
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

    tseries, params_desc, _ = get_time_series(data, time=time, shot=shot, x=x, z=z)
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

def get_radial_spectrogram(data, x=None, shot=None, bin=None, z=None, scaling="amplitude", plot=False, axis=None,
                           plot_save_folder=None):
    spectra = []
    valid_x = []
    assert isinstance(x, tuple) or x is None, "x should be a tuple"
    if x is None:
        x1, x2 = data.coords['x'].values.min(), data.coords['x'].values.max()
        x = (int(x1+0.3), int(x2+0.1))
    x_positions = np.array(range(x[0], x[1]+1))
    for x_pos in x_positions:
        try:
            spectrum, freq, err = get_spectrum_from_data(data, x=x_pos, shot=shot, bin=bin, scaling=scaling)
            spectrum, freq = lowpass(spectrum, freq)
            spectrum = spectrum[1:]
            freq = freq[1:]
            spectra.append(spectrum)
            valid_x.append(x_pos)
        except ValueError:
            print(f"Probably bad time series at x={x_pos}")

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis
        # im = ax.imshow(
        #     np.transpose(spectra),
        #     aspect='auto',
        #     origin='lower',
        #     extent=(
        #         x_positions.min(), x_positions.max(),
        #         freq.min(), freq.max()
        #     ),
        #     cmap=cmap,
        #     norm=LogNorm()
        # )
        # fig.colorbar(im, ax=ax, label='amplitude spectrum ('+data.attrs['units']+')')
        X, Y = np.meshgrid(valid_x, freq)
        vmin = np.min(spectra)
        vmax = np.max(spectra)
        levels = np.logspace(np.log10(vmin), np.log10(vmax), 100)
        cs = ax.contourf(
            X, Y, np.transpose(spectra),
            levels=levels,
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax)
        )
        # ax.colorbar(cs, ax=ax, label='amplitude spectrum ('+data.attrs['units']+')')
        ax.set_yscale('log')
        ax.set_ylabel('frequency (Hz)')
        ax.set_xlabel('x position (cm)')
        ax.set_title('radial '+scaling+' spectrogram (' + data.name +f') (z={z})')

        # to superimpose vertical lines for cutoff radii
        # r1, r2, r3, r4, r5 = -25, 17, 22, 28, 31
        # ax.plot([r1, r1], [min(freq), max(freq)], color='black')
        # ax.plot([r2, r2], [min(freq), max(freq)], color='black')
        # ax.plot([r3, r3], [min(freq), max(freq)], color='black')
        # ax.plot([r4, r4], [min(freq), max(freq)], color='black')
        # ax.plot([r5, r5], [min(freq), max(freq)], color='black')

        if axis is None:
            fig.show()
            if plot_save_folder is not None:
                fig.savefig(plot_save_folder+data.name+'_'+scaling+'_spectrogram_'+get_time()+'.png', dpi=150)
    return spectra, x_positions, freq


def timesplitter(time, correction=0):
    t1, t2 = time
    t1 += -1.0*correction
    t2 += -1.0*correction
    assert t2 > t1
    if t2 - t1 < 0.02:
        assert t1 - int(t1) > 0.02 and t2 - int(t2) > 0.02
    new_times = []
    if t1 - int(t1) > 0.02:
        if int(t1) != int(t2):
            new_times.append((t1 + correction, int(t1) + 1 - 0.02 + correction))

    for bin in range(int(t1) + 1, int(t2)):
        new_times.append((bin + 0.02 + correction, bin + 1 - 0.02 + correction))

    if t2 - int(t2) > 0.02:
        if int(t1) == int(t2):
            new_times.append((max(t1, int(t1) + 0.02) + correction, t2 + correction))
        else:
            new_times.append((int(t2) + 0.02 + correction, t2 + correction))

    weights = []
    for bin in new_times:
        weights.append(bin[1] - bin[0])

    weights = np.array(weights)
    weights = weights / np.sum(weights)

    return new_times, weights

def get_profile(data, time=None, shot=None, x=None, z=None, plot=False, axis=None, plot_save_folder=None):
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

    # print(data.mean(dim=["time", "shot"]).values)
    params_desc = f'at z={z}'
    if axis is not None and not plot: plot = True
    if time is None:
        profile = data.mean(dim = ['time'])
        time_var_avg = data.std(dim = ['time'])
        params_desc = params_desc + ' averaged over all time'
    elif isinstance(time, tuple):
        new_times, weights = timesplitter(time)

        profiles_mean = []
        profiles_std = []
        weights_used = []

        for (t1, t2), w in zip(new_times, weights):

            sel = data.sel(time=slice(t1, t2))

            p_mean = sel.mean(dim="time", skipna=True)
            p_std = sel.std(dim="time", ddof=1)  # <-- this is the time std you want

            if np.all(np.isnan(p_mean)):
                continue

            profiles_mean.append(p_mean)
            profiles_std.append(p_std)

            # weight bins by # of valid time samples
            n_valid = sel.count(dim="time")
            if n_valid.size > 0:
                weights_used.append(float(n_valid.max().item()))
            else:
                weights_used.append(0.0)

        weights_used = np.array(weights_used, dtype=float)
        weights_used /= weights_used.sum()
        weights_da = xr.DataArray(weights_used, dims=["bin"])

        mean_stack = xr.concat(profiles_mean, dim="bin")
        std_stack = xr.concat(profiles_std, dim="bin")

        profile = (mean_stack * weights_da).sum(dim="bin")
        mean_diff_sq = (mean_stack - profile) ** 2
        time_var = (weights_da * (std_stack ** 2 + mean_diff_sq)).sum(dim="bin")

        time_std = np.sqrt(time_var)

        time_var_avg = (time_std ** 2).mean(dim="shot")

        params_desc += f' averaged over times: {time} ms'
    else:
        profile = data.sel(time=time, method='nearest')
        params_desc = params_desc + ' at time: '+str(time)+'ms'
    if shot is None:
        #calculates error bar
        std = np.sqrt(profile.std(dim = ['shot'], ddof=1)**2 + time_var_avg)
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
        print("PLOTTING PROFILE")
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis
        ax.set_xlabel('x position (cm)')
        ax.set_ylabel(data.name+" ("+data.attrs['units']+")")
        ax.set_title('radial '+data.name+' profile\n'+params_desc)
        print(x_array)
        print(profile)
        print(std)
        ax.errorbar(x_array, profile, yerr=std, color='black', linestyle='', marker='o', capsize=0, markersize=3)
        if axis is None:
            fig.show()
            if plot_save_folder is not None:
                fig.savefig(plot_save_folder+data.name+'_profile'+get_time()+'.png', dpi=150)
                sleep(1)

    # print("PROFILE")
    # print(profile)
    # print(std)
    # print(params_desc)
    return profile, std, params_desc

def get_time_series(data, time=None, shot=None, x=None, z=None, plot=False, axis=None, plot_save_folder=None):
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
    params_desc = f"at z={z}, "
    # if "z" in data.coords:
    #     print(data.coords)
    #     data = data.sel(z=z, method="nearest")
    if x is None:
        time_series = data.mean(dim = ['x'])
        params_desc = params_desc + 'averaged over all x'
    elif isinstance(x, tuple):
        time_series = data.sel(x=slice(x[0], x[1])).mean(dim=['x'])
        params_desc = params_desc + 'averaged over x: '+str(x)+'cm'
    else:
        time_series = data.sel(x=x, method='nearest')
        params_desc = params_desc + 'at x: '+str(x)+'cm'
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

def get_contour(data, time=None, shot=None, x=None, z=None, plot=True, axis=None, plot_save_folder=None):
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
    params_desc = f"at z={z}, "
    data_unit = data.attrs['units']
    if shot is None:
        data = data.mean(dim = ['shot'])
        params_desc = params_desc + 'averaged over all shot'
    elif isinstance(shot, tuple):
        data = data.sel(shot=slice(shot[0], shot[1])).mean(dim=['shot'])
        params_desc = params_desc + 'averaged over shot: '+str(shot)
    else:
        data = data.sel(shot=shot, method='nearest')
        params_desc = params_desc + 'at shot: '+str(shot)

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
    profile,  std, params_desc = get_profile(data, time=time, shot=shot, plot=False)
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
    y_err_scaled = y_err / y_scale
    slope_guess = 0.5 * np.mean((y_scaled[1:] - y_scaled[:-1]) / (x_scaled[1] - x_scaled[0]))

    # print("y", y)
    # print("yscale", y_scale)
    # print("y_scaled", y_scaled)

    fit_params, covariance_matrix = curve_fit(linear_model, x_scaled, y_scaled, p0=[slope_guess, 0.0],
                                              sigma=y_err_scaled, absolute_sigma=True)

    if np.isnan(covariance_matrix).any() or np.isinf(covariance_matrix).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan

    slope, intercept = fit_params
    cov_slope_intercept = covariance_matrix[1, 0]
    slope_err, intercept_err = np.sqrt(np.diag(covariance_matrix))

    slope *= y_scale/x_scale
    intercept *= y_scale
    slope_err *= y_scale/x_scale
    intercept_err *= y_scale
    cov_slope_intercept *= y_scale*y_scale/x_scale
    # print(covariance_matrix)
    # print(cov_slope_intercept)
    # cov_slope_intercept = 0

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis
        y_fit = linear_model(x_array, slope, intercept)
        ax.set_xlabel('x position (cm)')
        try:
            ax.set_ylabel(data.name + " (" + data.attrs['units'] + ")")
        except KeyError:
            ax.set_ylabel(data.name + " (unit fail) ")
        ax.set_title('radial ' + data.name + ' profile\n' + params_desc)
        ax.errorbar(profile.coords['x'].values, profile.values, yerr=std.values, color='black', linestyle='',
                     marker='o', capsize=1, markersize=2, alpha=0.5)
        ax.plot(x_array, y_fit, color='fuchsia', label='linear fit')
        ax.legend()
        if axis is None:
            fig.show()

    return slope, intercept, slope_err, intercept_err, cov_slope_intercept

def get_data_over_grad_n(data, x=(-27, -19), time=None, shot=None, plot=False, axis=None):
    assert data.units == '$cm^{-3}$' or data.name != 'density'
    profile, profile_err, params_desc = get_profile(data, time=time, shot=shot, x=x, plot=False)
    grad, _, grad_err, ___, cov = linear_fit_profile(data, x=x, time=time, shot=shot, plot=False)
    d_grad_d = profile/grad
    d_grad_d_err = np.sqrt( (profile_err/grad)**2 + (profile*grad_err/(grad**2))**2
                            -2*profile*cov/(grad**3)*0) # don't know how to handle error yet

    # print("d_grad_d_err", d_grad_d_err)

    # print((profile_err/grad)**2)
    # print((profile*grad_err/(grad**2))**2)
    # print(-2*profile*cov/(grad**3))

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


def plot_total_flux_vs_Ln(data, quantity, x=(-28, -19), Ln_range=(-28, -23), time=None, shot=None, bin=(7,14), plot=False, axis=None):
    """
    flux here is short for fluctuations! not magnetic flux

    Parameters
    ----------
    data
    quantity
    x
    time
    shot
    bin
    plot
    axis

    Returns
    -------

    """
    assert data["density"].units == '$cm^{-3}$'
    data_density = data["density"]
    data = data[quantity]
    slope, intercept, slope_err, intercept_err, cov_slope_intercept =\
        linear_fit_profile(data_density, x=x, time=time, shot=shot, plot=False)
    profile, _, __ = get_profile(data, time=time, shot=shot, x=Ln_range, plot=False)

    Ln = intercept/slope + 0.5*(Ln_range[1] + Ln_range[0])
    Ln_err = np.sqrt((intercept_err/slope)**2 + (intercept*slope_err/(slope**2))**2 - 2*cov_slope_intercept/(slope)**2)
    # check this is right
    # average_quantity = profile.mean(dim=["x"], skipna=True).to_numpy()
    # print("av quant", average_quantity)
    total_flux_list = []
    total_flux_err_list = []
    good_x = []
    for x_value in profile.coords['x'].values:
        try:
            psd, freq, psd_err = get_fft_from_data(data, time=time, shot=shot, x=x_value, plot=False, bin=bin, scaling="psd")
            psd, freq = lowpass(psd, freq)
            dfreq = freq[2]-freq[1]
            total_flux = np.sqrt(np.sum(psd[1:])*dfreq)
            total_flux_err = 0.5*np.sum(psd_err[1:])*dfreq/total_flux
            total_flux_list.append(total_flux)
            total_flux_err_list.append(total_flux_err)
            good_x.append(x_value)
        except ValueError:
            pass

    middle_range_density = intercept + 0.5*slope*(np.max(good_x) + np.min(good_x))
    if quantity == "density":
        normalized_total_flux = np.array(total_flux_list)/middle_range_density
        normalized_total_flux_err = np.array(total_flux_err_list)/middle_range_density
    else:
        normalized_total_flux = np.array(total_flux_list)/np.mean(profile.to_numpy())
        normalized_total_flux_err = np.array(total_flux_err_list)/np.mean(profile.to_numpy())

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis

        ax.errorbar(abs(Ln), normalized_total_flux, yerr=normalized_total_flux_err, xerr=Ln_err, markersize=3, linestyle='', capsize=1,
                    marker = 'o', color='black')
        #todo hardcoded plot axes
        ax.set_xlabel(r'$L_n = \frac{n_e}{\nabla n_e}$ ($cm$)')
        ax.set_ylabel(r'$\frac{1}{n_e}\int PS$ ($cm^{-3}$)')
        ax.set_title(r'$\frac{1}{n_e}\int PS$ vs $L_n$')

        if axis is None:
            fig.show()

    # print(Ln, normalized_total_flux)
    print("IN plot total_Flux_vs_ln")
    print(abs(Ln), abs(np.mean(normalized_total_flux)), Ln_err, abs(np.mean(normalized_total_flux_err)))
    return abs(Ln), abs(np.mean(normalized_total_flux)), Ln_err, abs(np.mean(normalized_total_flux_err))


def get_langmuir_profiles(data, name, z, x=None, time=None, shot=None, plot=False, axis=None):
    """
    Plot a Langmuir probe profile as a function of x, averaging over time and shot,
    with error bars from the standard deviation.

    Parameters
    ----------
    data : xr.Dataset
        The xarray dataset containing the data.
    name : str
        The name of the variable to plot (e.g. "T_e").
    z : float
        The z position to select (nearest available).
    x : None, tuple (a, b), or None
        The x range to plot. None = all x.
    time : None, tuple (a, b), or float
        Time range or specific time. None = average over all time.
    shot : None, tuple (a, b), or int
        Shot range or specific shot. None = average over all shots.
    """

    da = data[name].sel(z=z, method="nearest")

    if x is not None:
        if isinstance(x, tuple):
            x_min = da.x.sel(x=x[0], method="nearest").item()
            x_max = da.x.sel(x=x[1], method="nearest").item()
            if x_min == x_max:
                da = da.sel(x=x_min)
            else:
                da = da.sel(x=slice(x_min, x_max))
        else:
            raise ValueError("x must be None or a tuple (a, b)")

    if time is not None:
        if isinstance(time, tuple):
            t_min = da.time.sel(time=time[0], method="nearest").item()
            t_max = da.time.sel(time=time[1], method="nearest").item()
            if t_min == t_max:
                da = da.sel(time=t_min)
            else:
                da = da.sel(time=slice(t_min, t_max))
        else:
            t_sel = da.time.sel(time=time, method="nearest").item()
            da = da.sel(time=t_sel)
    else:
        pass

    if shot is not None:
        if isinstance(shot, tuple):
            s_min = da.shot.sel(shot=shot[0], method="nearest").item()
            s_max = da.shot.sel(shot=shot[1], method="nearest").item()
            if s_min == s_max:
                da = da.sel(shot=s_min)
            else:
                da = da.sel(shot=slice(s_min, s_max))
        else:
            s_sel = da.shot.sel(shot=shot, method="nearest").item()
            da = da.sel(shot=s_sel)
    else:
        pass

    avg_dims = []
    if "time" in da.dims and (time is None or isinstance(time, tuple)):
        avg_dims.append("time")
    if "shot" in da.dims and (shot is None or isinstance(shot, tuple)):
        avg_dims.append("shot")

    mean_da = da.mean(dim=avg_dims)
    std_da = da.std(dim=avg_dims)

    if plot:
        if axis is not None:
            ax = axis
        else:
            fig=plt.figure(figsize=(6, 4))
            ax = fig.add_subplot()
        if "x" in mean_da.dims:
            ax.errorbar(mean_da.x, mean_da, yerr=std_da, fmt="o-", capsize=3, label=f"z={z:.2f}")
            ax.set_xlabel("x")
        else:
            ax.errorbar([0], [mean_da.item()], yerr=[std_da.item()], fmt="o")
            ax.set_xlabel("(no x dimension)")

        ax.set_ylabel(name)
        ax.set_title(f"{name} profile at z={z:.2f}")
        ax.legend()
        if axis is None:
            fig.tight_layout()
            fig.show()

    return mean_da, std_da

def get_linear_fit_langmuir(data, name, z, x=None, time=None, shot=None, plot=False, axis=None):
    mean, std = get_langmuir_profiles(data, name, z, x=x, time=time, shot=shot, plot=False)
    full_mean, full_std = get_langmuir_profiles(data, name, z, x=None, time=time, shot=shot, plot=False)
    x_array = mean.coords['x'].values

    y = mean.values
    y_err = std.values
    linear_model = lambda x, slope, intercept: slope * x + intercept

    x_scale = np.mean(np.abs(x_array))
    y_scale = np.mean(np.abs(y))
    x_scaled = x_array / x_scale
    y_scaled = y / y_scale
    y_err_scaled = y_err / y_scale
    slope_guess = 0.5 * np.mean((y_scaled[1:] - y_scaled[:-1]) / (x_scaled[1] - x_scaled[0]))

    # print("y", y)
    # print("yscale", y_scale)
    # print("y_scaled", y_scaled)

    fit_params, covariance_matrix = curve_fit(linear_model, x_scaled, y_scaled, p0=[slope_guess, 0.0],
                                              sigma=y_err_scaled, absolute_sigma=True)

    if np.isnan(covariance_matrix).any() or np.isinf(covariance_matrix).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan

    slope, intercept = fit_params
    cov_slope_intercept = covariance_matrix[1, 0]
    slope_err, intercept_err = np.sqrt(np.diag(covariance_matrix))

    slope *= y_scale / x_scale
    intercept *= y_scale
    slope_err *= y_scale / x_scale
    intercept_err *= y_scale
    cov_slope_intercept *= y_scale * y_scale / x_scale

    if plot:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axis
        y_fit = linear_model(x_array, slope, intercept)
        ax.set_xlabel('x position (cm)')
        ax.set_ylabel(f"{name} (unspecified units)")
        ax.set_title('radial ' + name + ' profile\n' + "unspecified params_desc")
        ax.errorbar(full_mean.coords['x'].values, full_mean.values, yerr=full_std.values, color='black', linestyle='',
                     marker='o', capsize=1, markersize=2, alpha=0.5)
        ax.plot(x_array, y_fit, color='fuchsia', label='linear fit')
        ax.legend()
        if axis is None:
            fig.show()

    return slope, intercept, slope_err, intercept_err, cov_slope_intercept




if __name__ == "__main__":
    x = np.linspace(0, 8*np.pi, 10000)
    y = np.sin(x) + 0.5*np.cos(2*x) + 0.5*np.sin(3*x)
    get_fft(y, dt = x[1]-x[0], scaling="spectrum", plot = True)