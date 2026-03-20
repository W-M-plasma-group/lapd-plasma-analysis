import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

c = 299792458 #m/s
m = 0.510998e6/c**2 #eV s^2/m^2
M = 938.272e6/c**2 # proton mass

sigma_eV = [
    24.6,
    30.0,
    34.0,
    40.0,
    45.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
    150.0,
    200.0,
    300.0,
    500.0,
    700.0,
    1000.0
]

sigma_m2 = [
    0.000000e+0,
    7.100000e-22,
    1.210000e-21,
    1.780000e-21,
    2.120000e-21,
    2.420000e-21,
    2.890000e-21,
    3.130000e-21,
    3.320000e-21,
    3.440000e-21,
    3.510000e-21,
    3.460000e-21,
    3.240000e-21,
    2.900000e-21,
    2.200000e-21,
    1.800000e-21,
    1.400000e-21
]

sigma_eV = np.array(sigma_eV)
sigma_m2 = np.array(sigma_m2)

sigma_log = CubicSpline(np.log(sigma_eV), sigma_m2)

def sigma(E):
    return sigma_log(np.log(E))

E_min = sigma_eV.min()*1.001
E_max = sigma_eV.max()*0.999

def integral(T):
    def integrand(E):
        return sigma(E) * E * np.exp(-E / T)
    val, err = quad(integrand, E_min, E_max)
    return val

def K_iz(T):
    # computes rate constant (I call it a rate function since it's clearly not constant)
    # for electron + He --> 2e + He^+ ionization
    # T is given in eV
    # assumes isotropic Maxwellian distribution function
    return np.array([(np.sqrt(8/(np.pi*m))*t**(-3/2))*integral(t) for t in T])

def uB_over_K_iz(T):
    return np.sqrt(T/M)/K_iz(T)

def get_K_iz(T, T_err):
    # computes uncertainty monte carlo style
    T_samples = np.random.normal(np.random.normal(loc=T, scale=T_err, size=1000))
    K_samples = K_iz(T_samples)
    K = np.nanmean(K_samples)
    K_err = np.nanstd(K_samples)
    K_median = np.nanmedian(K_samples)
    K_low, K_high = np.nanpercentile(K_samples, [25, 75])
    fig = plt.figure()
    plt.hist(np.log(K_samples), bins=50)
    fig.show()
    print(K, K_err)
    print(K_median, (K_high-K_low)/2)
    logK = np.log(K_samples)
    K_log_mean = np.exp(np.mean(logK))
    K_log_std = np.std(logK)
    print(K_log_mean, K_log_mean*np.exp(K_log_std),  K_log_mean*np.exp(-K_log_std))
    return K

def get_ng(T):
    d_eff = 0.1 #m #todo crude pproximation (d_eff is a nonlinear function of n_g via mean free path)
    return (1/d_eff)*uB_over_K_iz(T)

T_grid = np.linspace(0.5, 10, 1000)
ng_grid = get_ng(T_grid)

from scipy.optimize import curve_fit

# Exponential model
def exp_model(T, a, c):
    return a * np.exp(-c / T)

ng_clip = np.clip(ng_grid, 1e-20, None)

y = np.log(ng_clip)
X = 1 / T_grid

coeffs = np.polyfit(X, y, 1)
c_guess = -coeffs[0]
a_guess = np.exp(coeffs[1])

p0 = [a_guess, c_guess]

params, _ = curve_fit(exp_model, T_grid, ng_grid, p0=p0, maxfev=10000)

import pymc.math as pmm
def lookup_ng(T):
    return params[0] * pmm.exp(-params[1] / T)

T = np.linspace(0.501, 9.99, 1000)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(T_grid, ng_grid, color='black', label='Calculated')
ax.plot(T, lookup_ng(T), color='darkred', label='Fit')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r"$T$ (eV)")
ax.set_ylabel(r"$n_g$ ($\text{m}^{-3}$)")
ax.legend()
fig.show()

if __name__=="__main__":
    E = np.linspace(24.61, 999.99, 1000)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(E, sigma(E), color='black', linestyle='dotted')
    ax.plot(sigma_eV, sigma_m2, color='black', linestyle='none', marker='o')
    ax.set_xscale('log')
    ax.set_xlabel("$E$ (eV)")
    ax.set_ylabel(r"$\sigma(E)$ ($\text{m}^2$)")
    fig.show()

    T = np.linspace(0.1, 10, 100)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(T, K_iz(T), color='black')
    # ax.plot(sigma_eV, sigma_m2, color='black', linestyle='none', marker='o')
    # ax.set_xscale('log')
    ax.set_xlabel("$T$ (eV)")
    ax.set_ylabel(r"$K_{iz}$ ($\text{m}^3$/s)")
    fig.show()

    get_ng(4.0, 0.4)



