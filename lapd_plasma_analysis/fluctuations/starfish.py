from lapd_plasma_analysis.fluctuations.ranges import *


def get_whole_time_series(data, time_window, x, z, shot):
    t1, t2 = time_window
    data = data.sel(time=slice(t1, t2)).sel(z=z, x=x, shot=shot, method="nearest")
    return data, data.coords["time"].values/1000 #seconds

def s_fft(data, time_window, x, z, shot, normalize=False):
    time_series, times = get_whole_time_series(data, time_window, x, z, shot)
    dt = (times[1] - times[0])

    ft = fft(time_series)
    freq = fftfreq(len(time_series), dt)
    ft_index = int(len(ft) / 2)
    if len(time_series) % 2 == 0:
        ft_index += -1
    spec = ft[0:ft_index]
    freq = freq[0:ft_index]
    phase = np.angle(spec)

    spec = (abs(spec)**2 / (len(time_series)*(1/dt)))
    dc, nyquist = spec[0], spec[-1]
    spec = 2*spec
    spec[0], spec[-1] = dc, nyquist

    if normalize:
        spec = spec / dc

    return spec, phase, freq

def make_spectrum_plots(filepath, z, time_window):
    datasets = [xr.open_dataset(filepath + netcdf) for netcdf in os.listdir(filepath)]
    __ = datasets[0]
    xs = __.coords["x"].values
    shots = __.coords["shot"].values
    times = __.coords["time"].values
    for dataset in datasets:
        data = dataset["isat"]
        fig = plt.figure()
        ax = plt.subplot(111)
        for x in xs:
            for shot in shots:
                ft, phase, freq = s_fft(data, time_window, x, z, shot)
                s_freq = freq[4]
                print("s_freq", freq[4])
                ft_min, ft_max = np.min(ft), np.max(ft)
                ax.plot(freq[1:], ft[1:])
                ax.plot([s_freq, s_freq], [ft_min, ft_max])
                ax.set_xscale("log")
                ax.set_yscale('log')
                fig.show()

def make_starfish_plot(data, z, time_window):
    xs = data.coords["x"].values
    shots = data.coords["shot"].values
    times = data.coords["time"].values
    isat_data = data["isat"]
    fig = plt.figure()
    ax = plt.subplot(111)
    ax2 = ax.twinx()
    # phase_colors = ["#4C72B0","#55A868","#C44E52","#EFC94C",
    #                 "#8172B3","#EE854A","#8C8C8C","#64B5CD"]
    for x in xs:
        amps = []
        for shot in shots:
            ft, phase, freq = s_fft(isat_data, time_window, x, z, shot)
            df = freq[1]-freq[0]
            normalization = np.sum(ft)
            amps.append(ft[4]/normalization)
            ax2.plot([x], [phase[4]], color="green", linestyle="None", marker="o", alpha=0.3)
            # ax.plot([x], [ft[4]/normalization], color="black", linestyle="None", marker="o", alpha=0.3)
        mean = np.mean(amps)
        std = np.std(amps)
        if not std > abs(mean):
            ax.errorbar([x], [mean], yerr=[std], linestyle="", marker="o", color="black")

    ax.set_xlabel("$x$ [cm]")
    ax.set_title(f"{data.attrs['Run name']}, z={z}")
    # ax.set_yscale('log')

    fig.show()

def get_Er(data, z, time_window):
    t1, t2 = time_window
    lang_data = get_langmuir_dataset(data)
    V_P = lang_data["V_P"].sel(z=z, method="nearest").sel(time=slice(t1, t2))
    V_P_mean = V_P.mean(dim=["shot", "time"])
    V_P_std = V_P.std(dim=["shot", "time"])
    Er = (V_P_mean[:-1].values - V_P_mean[1:].values)/0.02
    Er_err = (V_P_std[:-1].values + V_P_std[1:].values)/0.02
    Er = np.concatenate(([(V_P_mean[0] - V_P_mean[1])/0.02], Er))
    Er = np.concatenate((Er, [V_P_mean[-2]/0.02 - V_P_mean[-1]/0.02]))
    Er_err = np.concatenate(([(V_P_std[0] + V_P_std[1])/0.01], Er_err))
    Er_err = np.concatenate((Er_err, [V_P_std[-2] / 0.01 + V_P_std[-1] / 0.01]))
    # print("Er", Er)
    # print("Er_err", Er_err)
    return Er, Er_err

def get_grad_isat(data, z, time_window):
    t1, t2 = time_window
    data = data["isat"].sel(z=z, method="nearest").sel(time=slice(t1, t2))
    isat_mean = data.mean(dim=["shot", "time"])
    isat_std = data.std(dim=["shot", "time"])
    grad_isat = (isat_mean[:-1].values - isat_mean[1:].values) / 0.02
    grad_isat_err = (isat_std[:-1].values + isat_std[1:].values) / 0.02
    grad_isat = np.concatenate(([(isat_mean[0] - isat_mean[1]) / 0.02], grad_isat))
    grad_isat = np.concatenate((grad_isat, [isat_mean[-2] / 0.02 - isat_mean[-1] / 0.02]))
    grad_isat_err = np.concatenate(([(isat_std[0] + isat_std[1]) / 0.01], grad_isat_err))
    grad_isat_err = np.concatenate((grad_isat_err, [isat_std[-2] / 0.01 + isat_std[-1] / 0.01]))
    return grad_isat, grad_isat_err

def make_starfish_plot_vs_Er_grad_isat(data, z, time_window):
    Er, Er_err = get_Er(data, z, time_window)
    grad_isat, grad_isat_err = get_grad_isat(data, z, time_window)
    grad_isat_Er = Er*grad_isat
    grad_isat_Er_err = Er_err*grad_isat_err
    xs = data.coords["x"].values
    shots = data.coords["shot"].values
    times = data.coords["time"].values
    data = data["isat"]
    fig = plt.figure()
    ax = plt.subplot(111)
    ax2 = ax.twinx()
    # phase_colors = ["#4C72B0","#55A868","#C44E52","#EFC94C",
    #                 "#8172B3","#EE854A","#8C8C8C","#64B5CD"]

    for i in range(len(xs)):
        amps = []
        for shot in shots:
            ft, phase, freq = s_fft(data, time_window, xs[i], z, shot)
            df = freq[1]-freq[0]
            normalization = np.sum(ft)*df
            amps.append(ft[4]/normalization)
            #ax2.errorbar([Er[i]], [phase[4]], color="green", linestyle="None", marker="o", alpha=0.3)
        mean = np.mean(amps)
        std = np.std(amps)
        if not std > abs(mean):
            ax.errorbar([grad_isat_Er[i]], [mean], yerr=[std], #xerr=[Er_err[i]],
                        linestyle="", marker="o", color="black")

    fig.show()

def remove_outliers(time_series, window, threshold):
    arr = time_series.values.astype(float)
    n = len(arr)

    if 2 * window + 1 > n:
        raise ValueError("window too large for data length")

    w_full = 2 * window + 1

    mean_all = uniform_filter1d(arr, size=w_full, mode="reflect")
    mean_sq_all = uniform_filter1d(arr**2, size=w_full, mode="reflect")

    mean_excl = (mean_all * w_full - arr) / (w_full - 1)
    var_excl = (mean_sq_all * w_full - arr**2) / (w_full - 1) - mean_excl**2
    var_excl[var_excl < 0] = 0.0
    std_excl = np.sqrt(var_excl)

    dev = np.abs(arr - mean_excl)

    outlier_mask = dev > threshold * std_excl

    outlier_mask[:window] = False
    outlier_mask[-window:] = False

    removed_count = np.count_nonzero(outlier_mask)
    arr[outlier_mask] = np.nan

    print(f"Removed outliers: {removed_count}")

    return xr.DataArray(arr, coords=time_series.coords, dims=time_series.dims), outlier_mask


if __name__ == "__main__":
    filepath = ("/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/HDF5 Files/"
                "November_2022_HDF5 and NetCDF/flux_nc/")
    print(os.listdir(filepath))
    datasets = [xr.open_dataset(filepath + netcdf) for netcdf in os.listdir(filepath)]

    __ = datasets[0]
    xs = __.coords["x"].values
    shots = __.coords["shot"].values
    times = __.coords["time"].values
    z = 600
    time_window = (10, 20)

    # for i in range(len(datasets))[4:]:
    #     for x in xs[30:]:
    #         for shot in shots:
    #             time_series, times = get_whole_time_series(datasets[i]["isat"], time_window, x, z, shot)
    #             clean_time_series, outlier_mask = remove_outliers(time_series, 500, 6.0)
    #             fig = plt.figure()
    #             ax = fig.add_subplot(111)
    #             outlier_times = times[outlier_mask]
    #             for t in outlier_times:
    #                 ax.axvline(t, color="red", linestyle="--", alpha=0.6, linewidth=0.8)
    #
    #
    #             ax.plot(times, time_series)
    #             ax.plot(times, clean_time_series)
    #             fig.show()
    #             plt.close()

    for data in datasets:
        # make_starfish_plot_vs_Er_grad_isat(data, z, time_window)
        # get_Er(data, z, time_window)
        make_starfish_plot(data, z, time_window)



