"""
List of ranges over which
    1. the plasma is in steady-state
    2. near constant density and temperature gradients are observed
    3. the structure of the Isat fluctuations is similar.
Specific to each experimental run.
"""
import os.path

from lapd_plasma_analysis.fluctuations.analysis import *
from lapd_plasma_analysis.fluctuations.rate_function import get_ng, lookup_ng

keys = ["run", "series", "board", "channel", "z", "xrange", "trange", "grad_id", "in_core"]

specifications_list = [
    #format [run, name, board, channel, z,    (xmin, xmax), (tmin, tmax), sgn(grad t_e / grad n_e), is_in_core]
    #ex.    [19, "Jan22",   2,       1, 860.1,  (-27, -25),      (7, 15),               -1,       True]

    [18, "Mar22", 2, 1, 831.0, (-29, -25), (8, 13), 1, False],
    [18, "Mar22", 2, 1, 831.0, (-25, -20), (7, 15), 1, False],
    [18, "Mar22", 2, 1, 831.0, (13, 19), (7, 15), 1, False],
    [18, "Mar22", 2, 1, 831.0, (20, 24), (10, 15), 1, False],
    [19, "Mar22", 2, 1, 831.0, (-29, -27), (8, 15), 1, False],
    [19, "Mar22", 2, 1, 831.0, (-27, -25), (8, 15), 1, False],
    [19, "Mar22", 2, 1, 831.0, (-25, -22), (8, 15), 1, False],
    [19, "Mar22", 2, 1, 831.0, (19, 21), (8, 15), 1, False],
    [19, "Mar22", 2, 1, 831.0, (21, 23), (8, 15), -1, False],
    [19, "Mar22", 2, 1, 831.0, (23, 25), (8, 15), 0, False],
    [1, "Mar22", 2, 1, 831.0, (-29, -26), (8, 15), 1, False],
    [1, "Mar22", 2, 1, 831.0, (-25, -22), (8, 15), 1, False],
    [1, "Mar22", 2, 1, 831.0, (-11, -8), (8, 15), 1, True],
    [1, "Mar22", 2, 1, 831.0, (1, 8), (6, 15), 1, True],
    [1, "Mar22", 2, 1, 831.0, (10, 16), (7, 15), 0, False],
    [1, "Mar22", 2, 1, 831.0, (18, 22), (8, 15), -1, False],
    [1, "Mar22", 2, 1, 831.0, (24, 28), (8, 15), 1, False],
    [9, "Mar22", 2, 1, 831.0, (-26, -21), (8, 15), 1, False],
    [9, "Mar22", 2, 1, 831.0, (-17, -10), (8, 15), 1, True],
    [9, "Mar22", 2, 1, 831.0, (10, 17), (8, 15), 1, False],
    [9, "Mar22", 2, 1, 831.0, (18, 23), (8, 15), 1, False],
    [10, "Mar22", 2, 1, 831.0, (-29, -27), (8, 15), 0, False],
    [10, "Mar22", 2, 1, 831.0, (-27, -23), (8, 15), 1, False],
    [10, "Mar22", 2, 1, 831.0, (16, 19), (7, 15), 1, False],
    [10, "Mar22", 2, 1, 831.0, (19, 22), (7, 15), 1, False],
    [10, "Mar22", 2, 1, 831.0, (22, 25), (7, 15), -1, False],
    [18, "Nov22", 2, 3, 639.0, (6, 10), (12, 20), 0, True],
    [18, "Nov22", 2, 1, 831.0, (-28, -22), (12, 20), 1, False],
    [19, "Nov22", 2, 3, 639.0, (-26, -20), (12, 20), 1, False],
    [19, "Nov22", 2, 3, 639.0, (13, 18), (12, 20), 1, False],
    [19, "Nov22", 2, 1, 831.0, (-32, -26), (13, 20), 1, False],
    [19, "Nov22", 2, 1, 831.0, (-26, -22), (13, 20), 1, False],
    [19, "Nov22", 2, 1, 831.0, (-21, -12), (8, 20), -1, False],
    [19, "Nov22", 2, 1, 831.0, (9, 13), (8, 20), -1, True],
    [19, "Nov22", 2, 1, 831.0, (19, 24), (12, 20), 1, False],
    [20, "Nov22", 2, 3, 639.0, (-31, -25), (13, 20), 1, False],
    [20, "Nov22", 2, 3, 639.0, (-22, -17), (12, 20), -1, True],
    # [20, "Nov22", 2, 3, 639.0, (-17, -24), (11, 20), -1, True], ## might be bad
    [20, "Nov22", 2, 3, 639.0, (-4, 0), (11, 20), 0, True],
    [20, "Nov22", 2, 3, 639.0, (3, 7), (11, 20), 0, True],
    [20, "Nov22", 2, 3, 639.0, (9, 15), (11, 20), -1, True],
    [20, "Nov22", 2, 3, 639.0, (19, 25), (15, 20), 1, False],
    [20, "Nov22", 2, 1, 831.0, (-32, -28), (12, 20), 1, False],
    [20, "Nov22", 2, 1, 831.0, (-24, -19), (10, 20), -1, True],
    [20, "Nov22", 2, 1, 831.0, (-18, -13), (10, 20), -1, True],
    [20, "Nov22", 2, 1, 831.0, (3, 7), (10, 17), -1, True],
    [20, "Nov22", 2, 1, 831.0, (11, 18), (14, 20), -1, True],
    [20, "Nov22", 2, 1, 831.0, (20, 25), (17, 20), 1, False],
    [21, "Nov22", 2, 3, 639.0, (-32, -27), (11, 20), 1, False],
    [21, "Nov22", 2, 3, 639.0, (-19, -14), (11, 20), -1, True],
    [22, "Jan24", 2, 1, 1118.25, (-21, -17), (19, 30), 1, False],
    [22, "Jan24", 2, 1, 1118.25, (-14, -11), (12, 30), -1, True],
    [22, "Jan24", 2, 1, 1118.25, (13, 19), (18, 30), 0, False],
    [23, "Jan24", 2, 1, 1118.25, (13, 19), (18, 30), 0, False],
    [23, "Jan24", 2, 1, 1118.25, (6, 12), (18, 30), -1, True],
    [23, "Jan24", 2, 1, 1118.25, (-20, -16), (18, 30), 1, False],
    [26, "Jan24", 2, 1, 1118.25, (-26, -20), (16, 28), 1, False],
    [27, "Jan24", 2, 1, 1118.25, (-26, -20), (16, 28), 1, False],
    [27, "Jan24", 2, 1, 1118.25, (19, 23), (20, 31), 0, False],
    [27, "Jan24", 2, 1, 1118.25, (23, 27), (20, 31), 1, False],
    [28, "Jan24", 2, 1, 1118.25, (17, 23), (20, 31), 1, False],
    [22, "Jan24", 3, 1, 766.8, (14, 19), (16, 31), 1, False],
    [22, "Jan24", 3, 1, 766.8, (-21, -16), (16, 31), 1, False],
    [22, "Jan24", 3, 1, 766.8, (4, 8), (16, 31), 0, True],
    [23, "Jan24", 3, 1, 766.8, (3, 7), (15, 33), 0, True],
    [23, "Jan24", 3, 1, 766.8, (13, 18), (15, 33), 1, False],
    [26, "Jan24", 3, 1, 766.8, (16, 22), (15, 33), 1, False],
    [26, "Jan24", 3, 1, 766.8, (-16, -12), (15, 33), 1, True],
    [27, "Jan24", 3, 1, 766.8, (22, 28), (15, 33), 1, False],
    [28, "Jan24", 3, 1, 766.8, (14, 21), (15, 33), 1, False],
    [28, "Jan24", 3, 1, 766.8, (-22, -18), (10, 26), -1, True],
    [28, "Jan24", 3, 1, 766.8, (23, 28), (16, 28), 1, False]
]

specifications_dict = [{keys[i]: spec for i, spec in enumerate(spec)} for spec in specifications_list]

def find_file(specs, hdf5_folder):
    for folder in os.listdir(hdf5_folder):
        if folder.startswith(specs["series"][:2]):
            num_string = str(specs["run"])
            if len(num_string) == 1:
                num_string = "0" + num_string
            for file in os.listdir(hdf5_folder + folder + "/flux_nc/"):
                if file.startswith(num_string):
                    return hdf5_folder + folder + "/flux_nc/" + file

def get_flux_and_Ln(dataset, specs):
    Ln, flux, Ln_err, flux_err = plot_total_flux_vs_Ln(dataset.sel(z=specs["z"], method="nearest"), "density",
                                                       x=specs["xrange"], Ln_range=specs["xrange"],
                                                       time=specs["trange"], bin=specs["trange"])
    return Ln, flux, Ln_err, flux_err

def get_flux(dataset, specs):
    x1, x2 = specs["xrange"]
    spectra, x_positions, freqs = get_radial_spectrogram(dataset["density"].sel(z=specs['z'], method="nearest"),
                                                         x=(x1, x2), bin=specs['trange'],
                                                         z=specs['z'], scaling="psd")

    spectra = np.array(spectra)
    # mask = (x_positions >= x1) & (x_positions <= x2)

    spectrum = np.mean(spectra, axis=0)
    spectrum_err = np.std(spectra, axis=0)

    # spectrum = lowpass(spectrum, freqs)

    m, b, m_err, b_err, cov = linear_fit_profile(dataset["density"].sel(z=specs['z'], method="nearest"), x=(x1, x2),
                                          time=specs['trange'])
    x = 0.5 * (x1 + x2)
    n = m * x + b
    n_err = np.sqrt((x * m_err) ** 2 + (b_err) ** 2 + 2 * x * cov)

    delta_n2 = np.sum(spectrum[1:]*(freqs[1]-freqs[0]))
    delta_n2_err = np.sum(spectrum_err[1:]*(freqs[1]-freqs[0]))

    delta_n = np.sqrt(delta_n2)
    delta_n_err = abs(delta_n2_err/delta_n)

    cov_delta_n_n = np.mean((delta_n - np.mean(delta_n))*(n - np.mean(n)))

    normalized_fluctuations = delta_n/n
    normalized_fluctuations_err = np.sqrt((delta_n_err/n)**2 + (delta_n*n_err/(n**2))**2 - 2*cov_delta_n_n*delta_n/(n**3))

    return normalized_fluctuations, normalized_fluctuations_err

def get_Ln(dataset, specs, sign_conv=False):
    x1, x2 = specs["xrange"]
    m, b, m_err, b_err, cov = linear_fit_profile(dataset["density"].sel(z=specs['z'], method="nearest"), x=(x1, x2),
                       time=specs['trange'])
    x = 0.5*(x1+x2)
    n = m*x + b
    n_err = np.sqrt( (x*m_err)**2 + (b_err)**2 + 2*x*cov )
    Ln_err = np.sqrt( (b*m_err/(m**2))**2 + (b_err/m)**2 -2*b/(m**3)*cov)
    if sign_conv:
        m = m*x/abs(x)
    else:
        m = abs(m)
    return n/m, n, Ln_err, n_err

def get_LT(dataset, specs):
    x1, x2 = specs["xrange"]
    lang_data = get_langmuir_dataset(dataset)
    m, b, m_err, b_err, cov = get_linear_fit_langmuir(lang_data, "T_e", specs['z'],
                                               x=(x1, x2), time=specs['trange'])
    x = 0.5 * (x1 + x2)
    T = m * x + b
    T_err = np.sqrt((x * m_err) ** 2 + (b_err) ** 2 + 2 * x * cov)
    LT_err = np.sqrt((b * m_err / (m ** 2)) ** 2 + (b_err / m) ** 2 - 2 * b / (m ** 3) * cov)
    return abs(T/m), T, LT_err, T_err

def get_pressure(dataset, specs, plot=False):
    x1, x2 = specs["xrange"]
    t1, t2 = specs["trange"] # didn't really check to see if it was ok to average here
    lang_data = get_langmuir_dataset(dataset)
    pressure_data = lang_data["P_ei_from_n_i_OML"]

    pressure_data = pressure_data.sel(x=slice(x1, x2), time=slice(t1, t2))
    pressure_data_mean = pressure_data.mean(dim=["x", "time", "shot"], skipna=True)

    pressure_data_err = pressure_data.std(dim=["x", "time", "shot"], skipna=True)

    z1, z2 = pressure_data_mean.coords["z"].values # assumes two z positions only (valid for current data)
    P1, P2 = pressure_data_mean.values
    P1_err, P2_err = pressure_data_err.values
    Lp = abs(z1-z2 / np.log(P2/P1))
    Lp_err = abs((z1-z2)/np.log(P2/P1)**2)*np.sqrt((P1_err/P1)**2 + (P2_err/P2)**2)

    P = np.float64(pressure_data_mean.sel(z=specs['z'], method="nearest").values)
    P_err =  np.float64(pressure_data_err.sel(z=specs['z'], method="nearest").values)

    dPdz = -abs(P/Lp)
    dPdz_err = np.sqrt((P_err/Lp)**2 + (P*Lp_err/Lp**2)**2)

    if plot:
        plt.plot([z1, z2], [P1, P2])
        plt.show()

    return P, P_err, dPdz, dPdz_err, Lp, Lp_err

def get_parallel_flow(dataset, specs):
    x1, x2 = specs["xrange"]
    t1, t2 = specs["trange"]  # didn't really check to see if it was ok to average here
    lang_data = get_langmuir_dataset(dataset)
    v_par_data = lang_data["v_para"]
    print(v_par_data)
    v_par_data = v_par_data.sel(x=slice(x1, x2), time=slice(t1, t2))

    v_par_mean = v_par_data.mean(dim=["x", "time", "shot"], skipna=True).values
    v_par_err = v_par_data.std(dim=["x", "time", "shot"], skipna=True).values

    return v_par_mean, v_par_err

def test_plot(specs, hdf5_folder, plot_save_folder, plotname, showfig=False, istest=False):
    xmin, xmax = specs["xrange"]
    z = specs['z']
    tmin, tmax = specs["trange"]

    data = xr.open_dataset(find_file(specs, hdf5_folder))
    lang_data = get_langmuir_dataset(data)

    x = (xmin, xmax)
    time = (tmin, tmax)

    fig = plt.figure(figsize=(10, 9), constrained_layout=True)
    ax = fig.add_subplot(221)
    ts, _, __ = get_time_series(data["isat"].sel(z=z, method="nearest"), x=x, time=None, plot=True, axis=ax)
    tsext = [np.min(ts), np.max(ts)]
    ax.plot([tmin, tmin], tsext, color="black")
    ax.plot([tmax, tmax], tsext, color="black")

    ax = fig.add_subplot(222)
    if istest:
        get_profile(data["density"].sel(z=z, method="nearest"), x=None, time=time, plot=True, axis=ax)
    else:
        linear_fit_profile(data["density"].sel(z=z, method="nearest"), x=x, time=time, plot=True, axis=ax)


    # for i in [0,1,2,3,4,5,6,7]:
    #     get_langmuir_profiles(lang_data, "T_e", z, x=None, time=8, shot=i, plot=True)

    # get_langmuir_profiles(lang_data, "T_e", z, x=None, time=time, shot=None, plot=True)
    ax = fig.add_subplot(223)
    try:
        get_linear_fit_langmuir(lang_data, "T_e", z, x=(xmin, xmax), time=time, shot=None, plot=True, axis=ax)
    except:
        get_langmuir_profiles(lang_data, "T_e", z, x=None, time=None, plot=True, axis=ax)

    ax = fig.add_subplot(224)
    _, __, freq = get_radial_spectrogram(data["density"].sel(z=z, method="nearest"), shot=None, bin=time, z=z, scaling="amplitude",
                                         plot=True, axis=ax, plot_save_folder=None)
    ax.plot([xmin, xmin], [min(freq), max(freq)], color='black')
    ax.plot([xmax, xmax], [min(freq), max(freq)], color='black')
    if showfig: fig.show()
    if plotname is not None:
        fig.savefig(plot_save_folder + f"test_range_{plotname}.png")
        print(f"\nFIGURE SAVED: test_range_{plotname}.png\n")

if __name__ == "__main__":

    hdf5_folder = "/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/HDF5 Files/"

    plot_save_folder = "/home/michael/PycharmProjects/LAPD-plasma-analysis/plots/"

    # for i, spec in enumerate(specifications_dict):
    #     if os.path.exists(plot_save_folder + f"test_range_{i}.png"):
    #         continue
    #     try:
    #         test_plot(spec, hdf5_folder, plot_save_folder, str(i))
    #     except:
    #         print(f"\nFAILURE on {i}\n")

    # plot density profile, plot temperature profile, plot time series, plot spectrogram
    # specs = [28, "Jan24", 3, 1, 766.8, (23, 28), (16, 28), 1, False]
    # specs_dict = {keys[i]: spec for i, spec in enumerate(specs)}
    # test_plot(specs_dict, hdf5_folder, plot_save_folder, "TEST", showfig=True, istest=False)

    # 25_line_valves90V_4500A_1kG_He 2024-02-06 12.56.08-013.hdf5
    # /home/michael/Documents/school/Plasma/LAPD Plasma Analysis/HDF5 Files/January_2024_HDF5 and NetCDF/25_line_valves90V_4500A_1kG_He 2024-02-06 12.56.08-013.hdf5
    # 1118.25, 766.8
    # 2, 1     3, 1

    # making some plots

    def gamma_1(dataset, specs):
        Ln, n, Ln_err, n_err = get_Ln(dataset, specs)
        LT, T, LT_err, T_err = get_LT(dataset, specs)
        return n*np.sqrt(T)/(Ln)**2

    gamma1_string = r"$nT_e^{1/2}L_n^{-2}$"

    def gamma_2(dataset, specs):
        Ln, n, Ln_err, n_err = get_Ln(dataset, specs)
        LT, T, LT_err, T_err = get_LT(dataset, specs)
        return n*(T**(3/2))/(Ln**6)

    # def gamma_3(dataset, specs):
    #     Ln, n = get_Ln(dataset, specs)
    #     LT, T = get_LT(dataset, specs)
    #     return n/((T**(1/2))*(Ln**2))

    def gamma_3(dataset, specs):
        Ln, n, Ln_err, n_err = get_Ln(dataset, specs, sign_conv=False)
        return 1/Ln, abs(Ln_err/Ln**2)

    gamma3_string = r"$L_n^{-1}$"

    def encode_style(a, b, c, d):
        return str(a) + " " + str(b) + " " + str(c) + " " + str(d)

    def decode_style(a):
        a, b, c, d = a.split(" ")
        d = int(d)
        return a, b, c, d

    def marker_style(specs):
        if specs["z"] == 639.0:
            color = "blue"
        if specs["z"] == 766.8:
            color = "green"
        if specs["z"] == 831.0:
            color = "goldenrod"
        if specs["z"] == 1118.25:
            color = "red"

        if specs["grad_id"] == 0:
            marker = "o"
        if specs["grad_id"] == 1:
            marker = "^"
        if specs["grad_id"] == -1:
            marker = "v"

        if specs["in_core"]:
            markeredgecolor = 'black'
            markersize = 4
        else:
            markeredgecolor = 'white'
            markersize = 6

        return encode_style(marker, color, markeredgecolor, markersize)

    def make_dataset():
        fluxes = []
        flux_errs = []
        ns = []
        n_errs = []
        Ts = []
        T_errs = []
        Lns = []
        Ln_errs = []
        LTs = []
        LT_errs = []
        zs = []
        Ps = []
        P_errs = []
        dPdzs = []
        dPdz_errs = []
        Lps = []
        Lp_errs = []
        drawing_specs = []
        for specs in tqdm(specifications_dict, desc="making dataset"):
            data = xr.open_dataset(find_file(specs, hdf5_folder))
            flux, flux_err = get_flux(data, specs)
            Ln, n, Ln_err, n_err = get_Ln(data, specs)
            LT, T, LT_err, T_err = get_LT(data, specs)
            P, P_err, dPdz, dPdz_err, Lp, Lp_err = get_pressure(data, specs)
            z = specs["z"]
            if flux < flux_err:
                continue
            if np.isnan(np.array([P, P_err, dPdz, dPdz_err, Lp, Lp_err])).any():
                continue
            drawing_spec = marker_style(specs)
            fluxes.append(flux)
            flux_errs.append(flux_err)
            ns.append(n)
            n_errs.append(n_err)
            Ts.append(T)
            T_errs.append(T_err)
            Lns.append(Ln)
            Ln_errs.append(Ln_err)
            LTs.append(LT)
            LT_errs.append(LT_err)
            zs.append(z)
            Ps.append(P)
            P_errs.append(P_err)
            dPdzs.append(dPdz)
            dPdz_errs.append(dPdz_err)
            Lps.append(Lp)
            Lp_errs.append(Lp_err)
            drawing_specs.append(drawing_spec)
        fluxes = np.array(fluxes)
        flux_errs = np.array(flux_errs)
        ns = np.array(ns)
        n_errs = np.array(n_errs)
        Ts = np.array(Ts)
        T_errs = np.array(T_errs)
        Lns = np.array(Lns)
        Ln_errs = np.array(Ln_errs)
        LTs = np.array(LTs)
        LT_errs = np.array(LT_errs)
        Ps = np.array(Ps)
        P_errs = np.array(P_errs)
        dPdzs = np.array(dPdzs)
        dPdz_errs = np.array(dPdz_errs)
        Lps = np.array(Lps)
        Lp_errs = np.array(Lp_errs)
        zs = np.array(zs)

        np.save("regression_data.npy", np.array([
            fluxes, flux_errs, ns, n_errs, Ts, T_errs, Lns, Ln_errs, LTs, LT_errs, zs,
            Ps, P_errs, dPdzs, dPdz_errs, Lps, Lp_errs
        ]))
        np.save("drawing_specs.npy", drawing_specs)


    #make_dataset()

    fluxes, flux_errs, ns, n_errs, Ts, T_errs, Lns, Ln_errs, LTs, LT_errs, zs, Ps, P_errs, dPdzs, dPdz_errs, Lps, Lp_errs = np.load("regression_data.npy", allow_pickle=True)
    drawing_specs = np.load("drawing_specs.npy", allow_pickle=True)

    def get_v_de(T, T_err, Ln, Ln_err):
        # e = 1.6e-19 #fundamental charge (handled automatically by units of T)
        B = 0.1 #T, representative value in LAPD -- shouldn't vary across experiments? But not sure #todo check
        cov = np.mean((T - np.mean(T))*(Ln - np.mean(Ln)))
        cov = 0
        v_de = T/(B*Ln)
        v_de_err = 1/(B)*np.sqrt( (T_err/Ln)**2 + (T*Ln_err/(Ln**2))**2 -2*T*cov/(Ln**3))
        return v_de, v_de_err

    def get_nu_ei(n, n_err, T, T_err):
        # estimate of Coulomb logarithm
        CL = 16
        cov = np.mean((T - np.mean(T))*(n - np.mean(n)))
        cov = 0
        nu_ei = 2.9e-6*CL*n/(T**(3/2))
        nu_ei_err = 2.9e-6*CL*np.sqrt( (n_err/(T**(3/2)))**2 + (3*n*T_err/(2*T**(5/2)))**2 - 3*cov*n/(T**4))
        return nu_ei, nu_ei_err

    v_de, v_de_err = get_v_de(Ts, T_errs, Lns, Ln_errs)
    nu_ei, nu_ei_err = get_nu_ei(ns, n_errs, Ts, T_errs)

    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(len(fluxes)):
        flux, flux_err = fluxes[i], flux_errs[i]
        v_dep, v_de_errp = v_de[i], v_de_err[i]
        if v_de_errp < v_dep:
            marker, color, markeredgecolor, markersize = decode_style(drawing_specs[i])
            ax.errorbar(v_dep, flux, yerr=flux_err, xerr=v_de_errp,
                    marker=marker, color=color, markeredgecolor=markeredgecolor, markersize=markersize)
    ax.set_xlabel("$v_{de}$ [m/s]")
    ax.set_ylabel(r"$\delta n/n$")
    ax.set_title("$v_{de}$ scaling")
    fig.show()
    fig.savefig(plot_save_folder + "feb22_vde_scaling.png", dpi=300)

    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(len(fluxes)):
        flux, flux_err = fluxes[i], flux_errs[i]
        nu, nu_err = nu_ei[i], nu_ei_err[i]
        if nu < nu_err:
            print("Problem index", i)
            continue
        marker, color, markeredgecolor, markersize = decode_style(drawing_specs[i])
        ax.errorbar(nu, flux, yerr=flux_err, xerr=nu_err,
                    marker=marker, color=color, markeredgecolor=markeredgecolor, markersize=markersize)
    ax.set_xlabel(r"$\nu_{ei}$ [Hz]")
    ax.set_ylabel("$\delta_n/n$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.show()
    fig.savefig(plot_save_folder + "mar16_flux_vs_nu_ei.png", dpi=300)

    # ### testing pressure
    # print("testing pressure")
    # specs = specifications_dict[0]
    # data = xr.open_dataset(find_file(specs, hdf5_folder))
    # print(get_pressure(data, specs, plot=True))

    # ### testing parallel flow
    # print("testing parallel flow")
    # for specs in specifications_dict:
    #     data = xr.open_dataset(find_file(specs, hdf5_folder))
    #     print(get_parallel_flow(data, specs))


    """
        datasets = []
        for specs in specifications_dict:
            datasets.append(xr.open_dataset(find_file(specs, hdf5_folder)))
    
        # plot flux vs Ln
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        norm = plt.Normalize(600, 900)
    
        fluxes = []
        flux_errs = []
        Lns = []
        Ln_errs = []
        LTs = []
        zs = []
        for i in range(len(datasets)):
            print(i)
            Ln, flux, Ln_err, flux_err = get_flux_and_Ln(datasets[i], specifications_dict[i])
            LT = get_LT(datasets[i], specifications_dict[i], hdf5_folder)
            zpos = specifications_dict[i]["z"]
            zs.append(zpos)
            fluxes.append(flux)
            flux_errs.append(flux_err)
            Lns.append(Ln)
            Ln_errs.append(Ln_err)
            LTs.append(LT)
            if Ln <= 50 and Ln_err <= Ln:
                ax.errorbar(Ln, flux, yerr=flux_err, xerr=Ln_err, linestyle="", marker="o",
                        color=cmap(norm(zpos)), ecolor="black")
    
    
        print("start")
    
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("$z$ position (cm)")
        ax.set_xlabel("$L_n (m)$")
        ax.set_ylabel("$\delta n_e / n_e$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        print("end plot")
        fig.show()
        fig.savefig(plot_save_folder + "flux_vs_Ln.png")
        print("end fog")
    
        print("start")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.array(Lns)/np.array(LTs), fluxes, linestyle="", marker="o", color="black")
        ax.set_xlabel("$\eta_e$")
        ax.set_ylabel("$\delta n_e / n_e$")
        ax.set_xscale("log")
        # ax.set_yscale("log")
        print("end plot")
        fig.show()
        fig.savefig(plot_save_folder + "flux_vs_eta.png")
        print("end fog")
    """

    ## REGRESSION PART
    import pymc as pm
    import arviz as az

    fluxes = np.asarray(fluxes)
    flux_errs = np.asarray(flux_errs)
    ns = np.asarray(ns)
    n_errs = np.asarray(n_errs)
    Ts = np.asarray(Ts)
    T_errs = np.asarray(T_errs)
    Lns = np.asarray(Lns)
    Ln_errs = np.asarray(Ln_errs)
    zs = np.asarray(zs)

    dPdzs = abs(dPdzs)

    mask = (
        (fluxes > 0) &
        (ns > 0) &
        (Ts > 0) &
        (Lns > 0)
    )

    fluxes = fluxes[mask]
    flux_errs = flux_errs[mask]
    ns = ns[mask]
    n_errs = n_errs[mask]
    Ts = Ts[mask]
    T_errs = T_errs[mask]
    Lns = Lns[mask]
    Ln_errs = Ln_errs[mask]
    zs = zs[mask]

    Y_obs = np.log(fluxes)
    Xn_obs = np.log(ns)
    XT_obs = np.log(Ts)
    XLn_obs = np.log(Lns)
    Xz_obs = np.log(zs)
    XP_obs = np.log(Ps)
    XdPdz_obs = np.log(dPdzs)
    XLp_obs = np.log(Lps)
    XLT_obs = np.log(LTs)
    Xv_de_obs = np.log(v_de)
    Xnu_ei_obs = np.log(nu_ei)
    Xng_obs = np.log(get_ng(XT_obs))
    eps = 0.1
    XT_obs = np.log(np.clip(Ts, eps, None))
    Xng_obs = np.log(np.clip(get_ng(Ts), eps, None))


    sigma_Y = flux_errs / fluxes
    sigma_Xn = n_errs / ns
    sigma_XT = T_errs / Ts
    sigma_XLn = Ln_errs / Lns
    sigma_XP = P_errs / Ps
    sigma_XdPdz = dPdz_errs / dPdzs
    sigma_XLp = Lp_errs / Lps
    sigma_XLT = LT_errs / LTs
    sigma_Xv_de = v_de_err / v_de
    sigma_Xnu_ei = nu_ei_err / nu_ei

    with pm.Model() as model:

        # set priors
        a = pm.Normal("a", mu=0, sigma=10)
        delta = pm.Normal("delta", mu=0, sigma=5)
        zeta = pm.Normal("zeta", mu=0, sigma=5)
        beta = pm.Normal("beta", mu=0, sigma=5)
        nu = pm.Normal("nu", mu=0, sigma=5)
        sigma_int = pm.HalfNormal("sigma_int", sigma=1)

        #Xn_true = pm.Normal("Xn_true", mu=Xn_obs, sigma=sigma_Xn) #latent vars
        # XT_true = pm.Normal("XT_true", mu=XT_obs, sigma=sigma_XT)
        XT_true = pm.TruncatedNormal("XT_true", mu=XT_obs, sigma=sigma_XT, lower=eps)
        Xnu_ei_true = pm.Normal("Xnu_ei_true", mu=Xnu_ei_obs, sigma=sigma_Xnu_ei)
        XLn_true = pm.Normal("XLn_true", mu=XLn_obs, sigma=sigma_XLn)
        # Xv_de_true = pm.Normal("Xv_de_true", mu=Xv_de_obs, sigma=sigma_Xv_de)
        # XP_true = pm.Normal("XP_true", mu=XP_obs, sigma=sigma_XP)
        # XdPdz_true = pm.Normal("XP_true", mu=XdPdz_obs, sigma=sigma_XdPdz)
        Xng_true = pm.Deterministic("Xng_true", lookup_ng(XT_true))
        XLp_true = pm.Normal("XLp_true", mu=XLp_obs, sigma=sigma_XLp)

        #mu = a + delta * Xn_true + zeta * XT_true + beta * XLn_true + nu*Xz_obs #model
        # mu = a + delta * XLp_true + zeta * Xnu_ei_true + beta * XLn_true + nu * XT_true
        mu = a + delta * XLp_true + zeta * Xnu_ei_true + beta * XLn_true + nu * Xng_true
        # mu = a + zeta * XT_true + beta * XLn_true
        Y_true = pm.Normal("Y_true", mu=mu, sigma=sigma_int)
        Y_like = pm.Normal("Y_like", mu=Y_true, sigma=sigma_Y, observed=Y_obs)

        trace = pm.sample(
            2000,
            tune=2000,
            target_accept=0.99,
            return_inferencedata=True
        )


    print(az.summary(trace, var_names=["a", "delta", "zeta", "beta", "nu", "sigma_int"]))
    # print(az.summary(trace, var_names=["a", "zeta", "beta", "nu", "sigma_int"]))
    # print(az.summary(trace, var_names=["a", "zeta", "beta", "sigma_int"]))

    posterior = trace.posterior

    a_samples = posterior["a"].stack(samples=("chain", "draw")).values
    delta_samples = posterior["delta"].stack(samples=("chain", "draw")).values
    zeta_samples = posterior["zeta"].stack(samples=("chain", "draw")).values
    beta_samples = posterior["beta"].stack(samples=("chain", "draw")).values
    nu_samples = posterior["nu"].stack(samples=("chain", "draw")).values

    a_mean = a_samples.mean()
    delta_mean = delta_samples.mean()
    zeta_mean = zeta_samples.mean()
    beta_mean = beta_samples.mean()
    nu_mean = nu_samples.mean()

    log_flux_pred = (
        a_mean
         + delta_mean * XLp_obs
        + zeta_mean * Xnu_ei_obs
        + beta_mean * XLn_obs
         + nu_mean * Xng_obs
    )

    flux_pred = np.exp(log_flux_pred)

    fig = plt.figure()
    for i in range(len(fluxes)):
        marker, color, markeredgecolor, markersize = decode_style(drawing_specs[i])
        plt.errorbar(flux_pred[i], fluxes[i], yerr=flux_errs[i],
                     marker=marker, color=color, markeredgecolor=markeredgecolor)

    min_val = min(fluxes.min() - flux_errs.max(), flux_pred.min())
    max_val = max(fluxes.max() + flux_errs.max(), flux_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='dotted')

    # plt.xlim(min_val, max_val)
    # plt.ylim(min_val, max_val)

    plt.xscale("log")
    plt.yscale("log")

    plt.ylabel("$\delta_n/n$")
    plt.xlabel("power law")
    fig.show()
    fig.savefig(plot_save_folder + "march17_regression_Lp_nu_ei_Ln_ng.png", dpi=300)

    print("corrcoef")
    print(np.corrcoef([XLp_obs, Xnu_ei_obs, XLn_obs, Xng_obs]))
