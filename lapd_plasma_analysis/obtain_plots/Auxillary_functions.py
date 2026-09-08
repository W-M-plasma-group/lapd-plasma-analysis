
import matplotlib
matplotlib.use('TkAgg')
from plasmapy.formulary import Coulomb_logarithm
from scipy.optimize import minimize_scalar

from lapd_plasma_analysis.file_access import *

from lapd_plasma_analysis.langmuir.helper import *
from lapd_plasma_analysis.langmuir.configurations import *
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, peak_widths

import astropy.units as u
from astropy import constants as c
from plasmapy.particles import *
from scipy.ndimage import convolve
from scipy.optimize import curve_fit


def find_sweep_indices(time_array,end_time,search_times,bias,dt):
    '''

    Parameters
    ----------
    time_array - A time sequence of all the time values corresponding to the bias measurements.
    end_time - u.Quantity The time value in ms where the IV sweep ends
    search_times - A mask of the time array starting at the end of the last sweep and ending at the end of the
    sweep of interest.
    bias - 1D array giving the bias values over time for a selected position-shot combination
    dt - u.Quantity timestep in ms

    Returns
    index - int indicating the first index in the time array of the sweep of interest
    last_index - int indicating the last index in the time array of the sweep of interest
    -------

    '''
    # Find the index of the peak of the IV sweep
    last_index = np.searchsorted(time_array, end_time, "right") - 1

    # Make the assumption that the Langmuir probe sweeps at an approximately constant rate and calculate that slope
    check_bias = int(.00008 * len(bias))
    slope = (bias[last_index] - bias[last_index - check_bias]) / (check_bias * dt)
    search_slope = 0

    # Obtain the time of the previous sweep
    first_index = np.argmax(search_times)
    j = 0
    index = first_index

    search_indices = np.where(search_times)[0]

    # There is a dip immediately after the previous sweep that has a large slope as it comes back up
    # however it doesn't indicate the start of the next sweep so the upper bound tries to mitigate that

    # Tolerance for defining "start" of a sweep
    slope_tolerance = 0.2
    # How many steps between checks of the slope
    skip = 10

    while True:
        index = first_index + j
        if (index + skip) >= len(bias):
            break
        search_slope = (bias[index + skip] - bias[index]) / (skip * dt)
        if abs((search_slope - slope) / slope) <= slope_tolerance:
            break
        j += skip

    # Define index as the start of the voltage sweep
    index = first_index + j - skip

    # Check to make sure we have an initial bias less than 0 because otherwise we are not getting a full sweep
    if bias[index] > 0:
        # How many check biases off the end where we are going to start looking
        start = 2
        while bias[index] > 0 and start * check_bias < int(.2*len(bias)):
            slope = (bias[last_index-start*check_bias] - bias[last_index -(start-1) * check_bias]) / (check_bias * dt)
            search_slope = 0
            first_index = np.argmax(search_times)
            slope_tolerance = .3
            h = 0
            # How many steps between checks of the slope
            while True:
                n_index = first_index + h
                if (n_index + skip) >= len(bias):
                    break
                search_slope = (bias[n_index + skip] - bias[n_index]) / (skip * dt)
                if abs((search_slope - slope) / slope) <= slope_tolerance:
                    break
                h += skip

            index = first_index + h - skip
            start += 1


    # Now we want to create a bias threshold so that we don't get a
    # tail which is appearing in some plots for some reason
    voltage_range = bias[last_index] - bias[index]
    voltage_tolerance = voltage_range * .1

    i = index
    for j in range(index,last_index):
        if bias[j] > bias[index]+voltage_tolerance:
            i = j
            break




    return i,last_index

def get_ion_current(sorted_bias, sorted_current, v_f_bias, fit_fraction=0.5):
    '''

    Parameters
    ----------
    sorted_bias - Quantity Array (V)  - of bias values from the minimum to maximum of the bias array
    sorted_current - Quantity Array (A) - of current value sorted by matched bias values from the minimum to
    maximum of the bias array
    v_f_idx - int - The index of the floating potential in the sorted_bias and sorted_current arrays

    Returns
    -------
    ion_current - Quantity (A) Array of the ion current piecewise from a fitted section. Once the fitted section
    crosses 0 the ion current is assumed to be 0
    '''

    v_f_idx = np.where(sorted_bias <= v_f_bias)[0][-1]

    # Use a safer, tunable fraction of the region strictly below V_f
    end_idx = int(v_f_idx * fit_fraction)

    # Strip units for fitting
    bias_vals = sorted_bias[:end_idx + 1].value
    curr_vals = sorted_current[:end_idx + 1].value

    # Fit the line
    slope, intercept = np.polyfit(bias_vals, curr_vals, 1)

    # Ion saturation slope must be positive AND the line must be negative at V_f
    # (by definition I_ion(V_f) = -I_e(V_f) < 0). If either fails, the ion branch
    # is too noisy to trust -> default to a flat ion floor.
    ion_at_vf = slope * v_f_bias.value + intercept
    if slope < 0 or ion_at_vf >= 0:
        slope = 0.0
        intercept = np.median(curr_vals)

    # Calculate full extrapolated line
    full_slope_current = (slope * sorted_bias.value) + intercept

    # Vectorized clamping: keep values <= 0, set positive values to 0
    ion_current_vals = np.where(full_slope_current <= 0, full_slope_current, 0.0)

    # Re-attach Astropy units
    return ion_current_vals * u.A


def get_ion_isat_min(sorted_current,sorted_bias):
    """

    Parameters
    ----------
    sorted_current - Quantity Array (A) - of current value sorted by matched bias values from the minimum to
    maximum of the bias array
    sorted_bias - Quantity Array (V)  - of bias values from the minimum to maximum of the bias array

    Returns
    -------
    ion_isat - Float: value of ion isat in Amps
    ion_isat_index - Int: location of the ion isat within the sorted current and sorted bias array

    Finds the Ion Isat by taking the average of the first avg_volts of the current array
    """

    # Assume the ion saturation current is the minimum of the current array
    ion_isat_index = np.argmin(sorted_current)

    # The ion_isat_index is assumed to be the center of the array - this line determines how many volts around the
    # center to average over to get a more accurate ion saturation current
    avg_volts = 5 * u.V

    # If there is no issues and the first index is less than half the avg_volts number away from the center point then
    # go up avg_volst/2 and go down avg_volts/2 from the ion_isat_index and indicate those as the start and end indices
    if sorted_bias[0] < sorted_bias[ion_isat_index] - avg_volts/2:
        end_index = np.nonzero(sorted_bias < sorted_bias[ion_isat_index] + avg_volts/2)[0][-1]
        start_index = np.nonzero(sorted_bias < sorted_bias[ion_isat_index] - avg_volts/2)[0][-1]

    # If the first index in the bias array is anything bigger than the center - avg_volts/2 the first index is the first
    # index of the bias array and the final index is avg_volts greater than the value at the first index
    else:
        start_index = 0
        end_index = np.nonzero(sorted_bias < sorted_bias[0] + avg_volts)[0][-1]

    # Filter out the current we are interested in and take the mean
    ion_isat_current = sorted_current[start_index:end_index]
    ion_isat = np.mean(ion_isat_current)

    return ion_isat, ion_isat_index

def get_electron_isat_curve_fit(sorted_bias, sorted_current, v_f_index, v_p_index, return_arg = False):
    '''

    Parameters
    ----------
    sorted_bias - Quantity Array (V)  - of bias values from the minimum to maximum of the bias array
    sorted_current - Quantity Array (A) - of current value sorted by matched bias values from the minimum to
    maximum of the bias array
    v_f_index - int - index of the floating potential in sorted bias and sorted current arrays
    v_p_index - int - index of the plasma potential in sorted bias and sorted current arrays
    return_arg - boolean - Does the user want the argument to be returned with the function

    Returns
    -------
    electron_isat - Quantity (A) - value of electron saturation current in Amps
    electron_isat_index - Int - Index of where approximately the electron saturation current could be plotted
    '''

    # We are assuming the electron saturation current must start after the plasma potential
    upper_section_bias = sorted_bias[v_p_index :].value
    upper_section_current = sorted_current[v_p_index :].value

    # Find the temperature
    t_e, exp_int, offset = get_te_v_p_vf(sorted_bias,sorted_current,v_f_index,v_p_index,return_intercept = True)
    slope_exponential = 1/t_e.value

    # Remove the offset from the intercept of the fit
    exp_int = exp_int - np.log(offset)
    exp_const = np.exp(exp_int)

    # Compute DI/DV at every point along the IV curve
    exp_slope_fit = slope_exponential * exp_const * np.exp(slope_exponential * sorted_bias[v_p_index].value) * u.A/u.V


    # Fit a line to the IV current above the plasma potential
    up_slope, up_int = np.polyfit(upper_section_bias, upper_section_current, 1)

    # Taylor expand the exponential to first order and create an array
    exp_array = (exp_slope_fit * (sorted_bias - sorted_bias[v_p_index]) + sorted_current[v_p_index])

    # Create an array for the linear fit
    up_array = up_slope * u.A/u.V * sorted_bias + up_int * u.A

    # Look for the last value before the exponetial section crosses the linear array
    exp_minus_up = exp_array - up_array
    arg_electron_isat = np.nonzero(exp_minus_up < 0 * u.A)[0][-1]
    electron_isat = sorted_current[arg_electron_isat]

    if return_arg:
        return electron_isat, arg_electron_isat

    return electron_isat

# Adapted from PlasmaPy.Langmuir -> Grabs a more accurate V_F with the point slope form
def get_floating_potential(sorted_bias,sorted_current):
    """

    Parameters
    ----------
    sorted_bias - Array of bias value sorted from minimum to maximum
    sorted_current - Array of current value sorted by matched bias values from the minimum to maximum of the bias array

    Returns
    -------
    V_f_bias - Quantity (V): the voltage bias associated with the floating potential
    V_f_current - Quantity (A): the current associated with the floating potential
    arg_v_f - Integer: index within sorted bias (and sorted current) where the floating potential is located

    Searches for the last index where the current is less than zero. Checks to see if the next value is greater than zero,
    then if so, uses point slope form to determine the bias at which the current is equal to zero.
    """

    try:
        total_volatge_len = (sorted_bias[-1] - sorted_bias[0])
    except IndexError:
        return None, None, None

    sign = np.sign(sorted_current)

    # Treat zeros as positive so they count as part of the pos side (Ensures we have a neg to pos crossing)
    sign[sign == 0] = 1

    zero_crossings = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0]

    # The best case scenario is that we can take the last voltage before the current crosses 0
    orig_v_f_idx = zero_crossings[-1]
    arg_v_f = None
    if (sorted_bias[-1] - sorted_bias[orig_v_f_idx]) > 0.15 * total_volatge_len:
        arg_v_f = orig_v_f_idx

    # However, sometimes we have really messy data (far outside core, early in run etc.) so we need to do some prelim
    # filtering
    else:
        # Build an array that has all possible 0 crossings (Additional points are for edge handling)
        edges = np.concatenate(([0], zero_crossings))

        # Find the actual voltage differences between the 0 crossings
        edges_bias = sorted_bias[edges]
        voltage_diff = np.diff(edges_bias)

        # Don't want the first section because that is not actually a zero crossing
        largest_vd_idx = np.argmax(voltage_diff[1:]) + 1
        if voltage_diff[largest_vd_idx] > 0.1 * total_volatge_len:
            arg_v_f = edges[largest_vd_idx]

    # Lock in where the 0 crossing might have actually occured
    v_f_bias = None
    if arg_v_f is not None:
        # Looks for the slope between the first value below and the first value above 0 in current
        slope = (sorted_current[arg_v_f + 1] - sorted_current[arg_v_f]) / (
                    sorted_bias[arg_v_f + 1] - sorted_bias[arg_v_f])

        # Point slope rearrangement of when I = 0 and solving for the bias associated with that
        v_f_bias = -sorted_current[arg_v_f] / slope + sorted_bias[arg_v_f]

    return v_f_bias, 0 * u.A, arg_v_f

def l_get_pressure(temperature, density):
    """
    Parameters
    ----------
    temperature (Quantity): Temperature in eV
    density (Quantity): Density in m^{-3}

    Returns
    -------
    Pressure - Quantity (Pascal): the pressure associated with the corresponding temperature and density values

    Temperature (J) * density (1/m^3)= pressure
    """
    return(temperature.to(u.J, equivalencies=u.temperature_energy()) * density).to(u.Pa)

def get_plasma_potential_spline(sorted_bias, sorted_current, return_spline = False):
    '''

    Parameters
    ----------
    sorted_bias - Quantity Array (V)  - of bias values from the minimum to maximum of the bias array
    sorted_current -  Quantity Array (A) - of current value sorted by matched bias values from the minimum to
    maximum of the bias array
    return_spline - Boolean - This algorithm uses a spline fitting algorithm. This boolean determines if the user
    wants to return that spline for future use

    Returns
    -------
    max_index - Integer - Index of where the first iteration of the plasma potential is
    '''

    # Make sure all bias values are unique
    _, unique_b_mask = np.unique(sorted_bias, return_index=True)
    unique_bias = sorted_bias[unique_b_mask]
    unique_current = sorted_current[unique_b_mask]

    # Create a spline fit for the data to plot on top of the raw data
    spline = UnivariateSpline(unique_bias, unique_current, s = 0.1, k=3)
    current_fit = spline(unique_bias)

    # Take the first derivative of the spline fit
    spline_deriv = spline.derivative(n=1)
    dIdV = spline_deriv(unique_bias)

    # Take the second derivative of the spline fit
    spline_2_deriv = spline.derivative(n=2)
    dI2dV2 = spline_2_deriv(unique_bias)

    # Find Plasma Potential from the maximum of the derivative curve - Chen method (Pace says look at E-sat region
    # and draw a line through that and the region where we are calculating the T_e and where they intersect is the
    # plasma potential
    max_idxs = []
    for i in range(len(dI2dV2) - 1):
        if (np.sign(dI2dV2[i]) == 1 and np.sign(dI2dV2[i + 1]) == -1) or np.sign(dI2dV2[i]) == 0:
            max_idxs.append(i)
    max_idxs = np.array(max_idxs)
    # valid_guesses_mask = dIdV[max_idxs] > (np.max(dIdV[max_idxs]) * .7)
    valid_guesses_mask = dIdV[max_idxs] >= (np.max(dIdV[max_idxs]))
    v_p_idx = max_idxs[valid_guesses_mask][0]
    v_p = unique_bias[v_p_idx]

    fig, ax = plt.subplots(1, 2, figsize = (10,5))
    ax = ax.flatten()

    ax[0].plot(unique_bias, unique_current, color = 'b')
    ax[0].plot(unique_bias, current_fit, color = 'r')
    ax[0].plot(unique_bias[v_p_idx],unique_current[v_p_idx], marker = 'D', markersize = 10, color = 'm')
    ax[0].set_xlabel('Bias (V)')
    ax[0].set_ylabel('Current (A)')

    ax[1].plot(unique_bias, dIdV, label='dIdV')
    ax[1].plot(unique_bias[v_p_idx], dIdV[v_p_idx], marker = 'D',color = 'm', markersize = 10,
             label='Plasma Potential')
    ax[1].set_xlabel('Bias (V)')
    ax[1].set_ylabel('dI/dV (A/V)')
    plt.tight_layout()
    plt.show()

    if return_spline:
        return v_p, v_p_idx, spline
    else:
        return v_p, v_p_idx


def get_ion_density(ion_type,ion_isat,A_p,T_e):
    """
    Parameters
    ----------
    ion_type - string - Indicates the ion you are trying to find the density of, corresponds to an ion in plasmapy.particles
    ion_isat - (Quantity) - Ion saturation current in A
    A_p - (Quantity) - Probe Area in m^2
    T_e - (Quantity) - Electron temperature in eV

    Returns
    -------
    n_i - (Quantity) - Ion density for the inputted ion

    Calculates the ion Density from the formula given in https://davidpace.com/example-of-langmuir-probe-analysis/
    """

    # Gets the mass of the associated ion in kg
    m_i = Particle(ion_type).mass

    # Converts the temperature from eV to J
    T_e_joules = T_e.to(u.J, equivalencies=u.temperature_energy())

    # Computes the ion density
    n_i = -((ion_isat / (c.e.si * A_p * np.exp(-0.5))) * np.sqrt(m_i / T_e_joules))

    return n_i.to(1/(u.m ** 3))

def get_electron_density(electron_isat,A_p,T_e):
    """

    Parameters
    ----------
    electron_isat - (Quantity) - Electron saturation current in A
    A_p - (Quantity) - Probe Area in m^2'
    T_e - (Quantity) - Electron temperature in eV

    Returns
    -------
    n_e - (Quantity) - Electron density in 1/m^3
    """

    # Gets the electron mass
    m_e = Particle("electron").mass

    # Converts the temperature from eV to J
    T_e_joules = T_e.to(u.J, equivalencies=u.temperature_energy())

    # Calculate thermal velocity from the mean value of a Maxwellian distribution
    v_thermal = np.sqrt((8*T_e_joules)/(np.pi * m_e))
    return 4*(electron_isat/(np.pi * A_p * c.e.si * v_thermal)).to(u.m ** -3)

def get_electron_ion_collision_frequency(n_e,ion_type,T_e):
    """

    Parameters
    ----------
    n_e - (Quantity) - Electron density in 1/m^3
    ion_type - string - Indicates the ion you are trying to find the collision frequency of
                        (corresponds to an ion in plasmapy.particles)
    T_e - (Quantity) - Electron temperature in eV
    n_i - (Quantity) - Ion density in 1/m^3

    Returns
    -------
    nu_ei - (Quantity) - Electron-ion collision frequency in 1/s
    """

    # Makes electrons and ions particles in the plasmapy particle library
    p_ion = Particle(ion_type)
    p_electron = Particle("electron")
    epsilon_0 = 8.854187817e-12 * u.F / u.m

    # Computes ion and electron temperature in J
    # Ion temperature is assumed to be 1 eV as reported in Perks et al. 2022
    T_i = 1 * u.eV
    T_i_joule = T_i.to(u.J, equivalencies=u.temperature_energy())
    T_e_joule = T_e.to(u.J, equivalencies=u.temperature_energy())

    # Computes the thermal speed of ions and electrons
    v_ti = np.sqrt(2 * T_i_joule / p_ion.mass)
    v_te = np.sqrt(2 * T_e_joule / p_electron.mass)
    mean_thermal_velocity = np.sqrt(v_ti ** 2 + v_te ** 2)
    z_i = p_ion.charge_number
    # Do Coulomb logarithm calculation from plasmapy.formulary.Coulomb z - mean should be a sum of all the ion contributions
    # 0.5 out in front due to only half of the neutrals in LAPD being ionized.
    z_mean = 0.5 * z_i

    # Calculates the Coulomb Logarithm in PlasmaPy. The HLS full interp takes the hyperbolic Landau Spritzer approach for
    # calculating the Coulomb logarithm and proves to be most accurate according to Gericke because it takes into account
    # both quantum and classical effects.

    coulomb_logarithm = Coulomb_logarithm(T_e,n_e,('e-',ion_type),z_mean=z_mean,method='hls_full_interp')

    lorentz_frequency = (4 * np.pi * n_e * z_i * c.e.si ** 4 * coulomb_logarithm) / (
            (4 * np.pi * epsilon_0) ** 2 * p_electron.mass ** 2*mean_thermal_velocity ** 3)

    # Because we are dealing with Maxwellian distributions we need to use the Maxwellian Frequency
    maxwellian_frequency = 4/(3*np.sqrt(np.pi)) * lorentz_frequency

    return maxwellian_frequency.to(1 / u.s)

def get_te_v_p_vf(sorted_bias,sorted_current,v_f_index,v_p_index, fit_curve = True, return_intercept = False):
    """

    Parameters
    ----------
    sorted_bias
    sorted_current
    v_f_index
    v_p_index
    fit_curve
    return_intercept

    Returns
    -------

    """
    val_sorted_current = sorted_current.to(u.A).value.astype(np.float64)
    offset =   abs(min(val_sorted_current)) + 1e-9
    current_to_adj = val_sorted_current + offset
    #print("Non-positive values:", np.sum(current_to_adj <= 0))
    # print("Non-positive value(s):", current_to_adj[current_to_adj <= 0])


    adjusted_current = np.log(current_to_adj)


    if not fit_curve:
        num = adjusted_current[v_p_index]-adjusted_current[v_f_index]
        denom = sorted_bias[v_p_index]-sorted_bias[v_f_index]
        slope = num/denom
        t_e = (1 / slope).to(u.V).value * u.eV
    else:
        slope, intercept= np.polyfit(sorted_bias[v_f_index:v_p_index].value, adjusted_current[v_f_index:v_p_index], 1)

        t_e = (1 / slope) * u.eV

        if return_intercept:
            return t_e, intercept, offset

    return t_e

def get_t_e_spline (sorted_bias,sorted_current,v_f_bias, plot_title = ''):
    '''

    Parameters
    ----------
    sorted_bias - Quantity Array (V)  - of bias values from the minimum to maximum of the bias array
    sorted_current - Quantity Array (A) - of current value sorted by matched bias values from the minimum to
    maximum of the bias array
    v_f_idx - int - The index of the floating potential in the sorted_bias and sorted_current arrays

    Returns
    -------
    t_e
    '''



    # Test and see what works best for the fit
    off_max_pct_tests = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    fit_colors = [
        "#FF0000",  # Pure Red
        "#8B0000",  # Dark Red
        "#DC143C",  # Crimson
        "#FF1493",  # Deep Pink
        "#C71585",  # Medium Violet Red
        "#9932CC",  # Dark Orchid
        "#800080",  # Purple
        "#4B0082"  # Indigo
    ]

    raw_data_clr = "#00BFFF" # Deep Sky Blue
    tanh_clr = "#00FF00" # Lime green

    fit_labels = ['60% of max',
                  '65% of max',
                  '70% of max',
                  '75% of max',
                  '80% of max',
                  '85% of max',
                  '90% of max',
                  '95% of max']

    if v_f_bias is None:
         return None, None, None, None, None, None, None, None

    if v_f_bias is not None:
        # Remove the ion current from the total current
        ion_current = get_ion_current(sorted_bias, sorted_current, v_f_bias)
        n_sorted_current = sorted_current - ion_current
        val_sorted_current = n_sorted_current.to(u.A).value.astype(np.float64)

        # Shift the current up by an offset so that we don't take the logarithm of a negative value (The slope will be the
        # same regardless of the shift since all values are shifted by the same amount)
        t_e_offset = abs(min(val_sorted_current)) + 1e-9
        current_to_adj = val_sorted_current + t_e_offset
        adjusted_current = np.log(current_to_adj)

        # Make sure we don't have multiple current values for each bias value
        _, unique_b_mask = np.unique(sorted_bias, return_index=True)
        unique_bias = sorted_bias[unique_b_mask]
        unique_adj_current = adjusted_current[unique_b_mask]

        u_b_voltage_len = unique_bias[-1] - unique_bias[0]

        v_f_idx = np.where(unique_bias <= v_f_bias)[0][-1]

        # Restrict the region where we are going to perform the tanh fit
        inxs_of_int = np.where((unique_bias >= (v_f_bias - 0.3 * u_b_voltage_len)) &
                               (unique_bias <= (v_f_bias + 0.5 * u_b_voltage_len)))[0]
        bias_to_fit = unique_bias[inxs_of_int]
        bias_to_fit_values = bias_to_fit.value
        btf_vf_idx = np.where(bias_to_fit <= v_f_bias)[0][-1]

        adj_current_to_fit = adjusted_current[inxs_of_int]

        # Now we can define the tanh function we want to fit for numpy's curve fitting
        def tanh_func(x, A, x0, w, B):
            return A * np.tanh((x - x0) / w) + B

        # For the np.curve_fit we have to provide initial guesses for the parameters that are not the x values so below
        # are general guesses
        # A - Amplitude - The range of the data to be fit
        # x0 - The center of the fit - Chosen to be the spot where the np.gradient is the steepest
        # w - width of the function (steepness of the curve) - (max(bias)-min(bias))/10 Assumes that the transition
        # happens over approximately 10% of the voltage values (Again it is just a guess curve_fit will adjust)
        # B - Offset - Guess is the mean value of the current array of interest

        p0 = [
            (max(adj_current_to_fit) - min(adj_current_to_fit)) / 2,  # A
            bias_to_fit_values[np.argmax(np.gradient(adj_current_to_fit))],  # x0
            (max(bias_to_fit_values) - min(bias_to_fit_values)) / 10,  # w
            np.mean(adj_current_to_fit)  # B
        ]
        # print('A type: ', (max(adj_current_to_fit) - min(adj_current_to_fit)) / 2)
        # print('x0 type: ', bias_to_fit[np.argmax(np.gradient(adj_current_to_fit))].value)
        # print('w type: ', (max(bias_to_fit.value) - min(bias_to_fit.value)) / 10)
        # print('B type: ', np.mean(adj_current_to_fit))

        try:
            # Fit the tanh function
            popt, pcov = curve_fit(tanh_func, bias_to_fit_values, adj_current_to_fit, p0=p0)
        except RuntimeError:
            return None, None, None, None, None, None


        # Return the fitted parameters
        A_fit, x0_fit, w_fit, B_fit = popt

        # Compute fitted curve if needed to plot
        bias_fit = np.linspace(min(bias_to_fit_values), max(bias_to_fit_values), 3000)
        current_fit = tanh_func(bias_fit, *popt)

        # Compute the derivative to do our PINQUED fit
        dlnIdV = A_fit / w_fit * (1 / np.cosh((bias_fit - x0_fit) / w_fit) ** 2)

        # Find the knee from the max curvature of the tanh fit
        # (Proof of equation at https://openstax.org/books/calculus-volume-3/pages/3-3-arc-length-and-curvature)
        # This is specific for tanh functions although a similar function could be created for any other function

        def get_fitted_knee(A, x0, w, B):
            """
            Finds the geometric knee (max curvature) for y = A * tanh((x - x0) / w) + B.
            Finds the knee on the right side of the curve (x > x0).
            """

            def curvature(x):
                # Let u be the inner part of the function
                u = (x - x0) / w

                # First derivative
                y_prime = (A / w) * (1.0 / np.cosh(u) ** 2)

                # Second derivative
                y_double_prime = -2.0 * (A / w ** 2) * np.tanh(u) * (1.0 / np.cosh(u) ** 2)

                # Curvature formula
                k = np.abs(y_double_prime) / (1.0 + y_prime ** 2) ** 1.5
                return k

            # We want to maximize curvature (minimize negative curvature)
            def objective_function(x):
                return -curvature(x)

            # We search for the knee between the center (x0) and an upper bound.
            # 3*w is a very safe upper bound where the curve is almost completely flat.
            result = minimize_scalar(objective_function, bounds=(x0_fit, x0_fit + 3 * w_fit), method='bounded')

            knee_x = result.x
            knee_y = A * np.tanh((knee_x - x0) / w) + B

            return knee_x, knee_y

        # Calculate using your fitted parameters
        knee_x, knee_y = get_fitted_knee(A_fit, x0_fit, w_fit, B_fit)


        # This works well to get the knee but it is not based in physics

        # def get_robust_visual_knee(A_fit, x0_fit, w_fit, B_fit):
        #     """
        #     Finds the visual knee using the Maximum Perpendicular Distance method.
        #     This is invariant to the physical units of X (bias) and Y (current).
        #     """
        #
        #     # 1. Define the search region.
        #     # For a tanh curve, we look from the center (x0) to a flat plateau (x0 + 3*w)
        #     x_start = x0_fit
        #     x_end = x0_fit + 3 * w_fit
        #
        #     y_start = B_fit  # value at center
        #     y_end = A_fit * np.tanh((x_end - x0_fit) / w_fit) + B_fit  # value at plateau
        #
        #     def negative_normalized_distance(x):
        #         # 2. Normalize the current X value between 0 and 1
        #         x_norm = (x - x_start) / (x_end - x_start)
        #
        #         # 3. Calculate Y and normalize it between 0 and 1
        #         y = A_fit * np.tanh((x - x0_fit) / w_fit) + B_fit
        #         y_norm = (y - y_start) / (y_end - y_start)
        #
        #         # 4. Calculate distance from the diagonal line y = x.
        #         # In normalized [0,1] space, the diagonal line is y_norm = x_norm.
        #         # The perpendicular distance is proportional to (y_norm - x_norm).
        #         # We return the negative distance because minimize_scalar finds the minimum.
        #         return -(y_norm - x_norm)
        #
        #     # 5. Run the optimizer
        #     result = minimize_scalar(
        #         negative_normalized_distance,
        #         bounds=(x_start, x_end),
        #         method='bounded'
        #     )
        #
        #     # Calculate final coordinates
        #     knee_x = result.x
        #     knee_y = A_fit * np.tanh((knee_x - x0_fit) / w_fit) + B_fit
        #
        #     return knee_x, knee_y
        #
        # # Usage with your parameters:
        # knee_x, knee_y = get_robust_visual_knee(A_fit, x0_fit, w_fit, B_fit)

        # Find the maximum of the derivative and the bias values where we want to do the actual linear fit
        max_deriv = dlnIdV.max()
        slopes = []
        intercepts = []
        r_squareds = []
        length_arrays = []
        left_edges = []
        right_edges = []

        def check_off_max_pcts(dlnIdV, max_deriv, bias_fit, bias_to_fit_values, adj_current_to_fit, off_max_pct):
            off_max = max_deriv * off_max_pct
            left_edge = np.where(dlnIdV > off_max)[0][0]
            if bias_fit[left_edge] < v_f_bias.value:
                left_edge = np.where(bias_fit <= v_f_bias.value)[0][-1]

            right_edge = np.where(dlnIdV > off_max)[0][-1]

            # Get the indices in the original probe data associated with the left and the right edges
            left_bias_idx = np.where(bias_to_fit_values >= bias_fit[left_edge])[0][0]
            right_bias_idx = np.where(bias_to_fit_values <= bias_fit[right_edge])[0][-1]
            len_array = len(bias_to_fit_values[left_bias_idx:right_bias_idx])

            # Perform a linear fit between the left and the right edge on the probe data to get 1/T_e
            slope, intercept = np.polyfit(bias_to_fit_values[left_bias_idx:right_bias_idx],
                                          adj_current_to_fit[left_bias_idx:right_bias_idx],
                                          1)
            r = np.corrcoef(bias_to_fit_values[left_bias_idx:right_bias_idx],
                            adj_current_to_fit[left_bias_idx:right_bias_idx])[0,1]
            r_squared = r**2

            t_e = 1/slope * u.eV

            return slope, intercept, r_squared, len_array, left_edge, right_edge

        for test_pct in off_max_pct_tests:
            slope, intercept, r_squared, len_array, left_edge, right_edge = (
                check_off_max_pcts(dlnIdV, max_deriv,  bias_fit, bias_to_fit_values, adj_current_to_fit, test_pct))
            slopes.append(slope)
            intercepts.append(intercept)
            r_squareds.append(r_squared)
            length_arrays.append(len_array)
            left_edges.append(left_edge)
            right_edges.append(right_edge)



        # Now to get the Electron saturation current and the Plasma Potential
        slope = slopes[3]
        intercept = intercepts[3]

        knee_idx = np.where(unique_bias.value <= knee_x)[0][-1]

        # Define the region to fit a line to for the Esat current
        esat_bias_region = unique_bias[knee_idx:].value
        esat_adj_curr_region = unique_adj_current[knee_idx:]

        # Create the linear fit for the Esat region
        esat_slope, esat_intercept = np.polyfit(esat_bias_region, esat_adj_curr_region, 1)

        # TODO make the function return None for everything if the esat_slope > 0

        # Find the plasma potential by finding the intersection of the esat and temperature fits
        v_p_value = (esat_intercept - intercept) / (slope - esat_slope)
        v_p = v_p_value * u.V


        if plot_title != '':
            max_len_array_idx = np.argmax(length_arrays)
            max_len_left = left_edges[max_len_array_idx]
            max_len_right = right_edges[max_len_array_idx]
            max_len_left_idx = np.where(bias_to_fit_values >= bias_fit[max_len_left])[0][0]
            max_len_right_idx = np.where(bias_to_fit_values <= bias_fit[max_len_right])[0][-1]


            y_upper_lim = np.max(adj_current_to_fit + 0.5)
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()

            axes[0].scatter(unique_bias, unique_adj_current, color='b', label='Data from Probe')
            axes[0].plot(bias_fit, current_fit, color='r', label='Tanh fit')
            axes[0].axvline(knee_x, color='g', linestyle='--', label='Knee')
            axes[0].plot(v_f_bias, adj_current_to_fit[btf_vf_idx],
                         color='m', label=r'$V_f$', linestyle='None', marker='o')

            axes[1].plot(bias_fit, dlnIdV, color='r', label='Tanh fit Derivative')
            axes[1].axvline(knee_x, color='g', linestyle='--', label='Knee')

            axes[2].scatter(bias_to_fit_values[max_len_left_idx:max_len_right_idx],
                            adj_current_to_fit[max_len_left_idx:max_len_right_idx],
                            color='b', label='Data from Probe')

            for i in range(len(off_max_pct_tests)):
                slope = slopes[i]
                intercept = intercepts[i]
                r_squared = r_squareds[i]
                left_edge = left_edges[i]
                right_edge = right_edges[i]


                left_bias_idx = np.where(bias_to_fit_values >= bias_fit[left_edge])[0][0]
                right_bias_idx = np.where(bias_to_fit_values <= bias_fit[right_edge])[0][-1]

                axes[0].axvline(bias_fit[left_edge], color=fit_colors[i], linestyle='--')
                axes[0].axvline(bias_fit[right_edge], color=fit_colors[i], linestyle='--')
                axes[0].plot(bias_to_fit_values[left_bias_idx:], slope * bias_to_fit_values[left_bias_idx:] + intercept,
                             color = fit_colors[i], label = fit_labels[i] + fr' $T_e$ = {1/slope:.2f}')

                axes[1].axvline(bias_fit[left_edge], color=fit_colors[i], linestyle='--')
                axes[1].axvline(bias_fit[right_edge], color=fit_colors[i], linestyle='--')


                axes[2].plot(bias_to_fit_values[left_bias_idx:right_bias_idx],
                             slope * bias_to_fit_values[left_bias_idx:right_bias_idx] + intercept,
                             color = fit_colors[i], label = fit_labels[i] + fr'$ T_e$ = {1/slope:.2f}, $r^2$ = {r_squared:.2f}')
                axes[2].axvline(bias_fit[left_edge], color=fit_colors[i], linestyle='--')
                axes[2].axvline(bias_fit[right_edge], color=fit_colors[i], linestyle='--')


            slope = slopes[3]
            intercept = intercepts[3]
            left_edge = left_edges[3]
            left_bias_idx = np.where(bias_to_fit_values >= bias_fit[left_edge])[0][0]

            old_knee_pct = 0.3
            old_knee_voltage_idx = np.where(dlnIdV >= (old_knee_pct * max_deriv))[0][-1]
            full_old_knee_idx = np.where(unique_bias.value <= bias_fit[old_knee_voltage_idx])[0][-1]

            # Define the region to fit a line to for the Esat current
            old_esat_bias_region = unique_bias[full_old_knee_idx:].value
            old_esat_adj_curr_region = unique_adj_current[full_old_knee_idx:]

            # Create the linear fit for the Esat region
            old_esat_slope, old_esat_intercept = np.polyfit(old_esat_bias_region, old_esat_adj_curr_region, 1)


            axes[3].scatter(unique_bias, unique_adj_current, color = 'b', label = 'Data from Probe' )
            axes[3].plot(bias_to_fit_values[left_bias_idx:],
                         slope * bias_to_fit_values[left_bias_idx:] + intercept,
                         color = fit_colors[3], label = fr'{fit_labels[3]}, $T_e$ = {1/slope:.2f}')
            axes[3].plot(v_f_bias, adj_current_to_fit[btf_vf_idx],
                         color='m', label=r'$V_f$', linestyle = 'None', marker = 'o')
            axes[3].plot(bias_to_fit_values[left_bias_idx:],
                         esat_slope * bias_to_fit_values[left_bias_idx:] + esat_intercept,
                         color='c', linestyle='--', label='Esat current')
            axes[3].plot(bias_to_fit_values[left_bias_idx:],
                         old_esat_slope * bias_to_fit_values[left_bias_idx:] + old_esat_intercept,
                         color='lime', linestyle='--', label='Old Esat current')
            axes[3].axvline(v_p_value, color='y', linestyle='--', label = r'$V_p$')
            axes[3].axvline(knee_x, color='k', linestyle='--', label = r'$Knee$')
            axes[3].axvline(bias_fit[old_knee_voltage_idx], color = 'orange', linestyle='--', label = 'Old Knee')


            axes[0].set_xlabel('Bias (V)')
            axes[0].set_ylabel('ln(Current)')
            axes[0].set_ylim([-6, y_upper_lim])
            axes[0].legend(loc='lower right', fontsize='small')

            axes[1].set_xlabel('Bias (V)')
            axes[1].set_ylabel('dln(Current)/dV')
            axes[1].legend(loc='upper left',fontsize='small')

            axes[2].set_xlabel('Bias (V)')
            axes[2].set_ylabel('ln(Current)')
            axes[2].legend(loc='lower right', fontsize='small')

            axes[3].set_xlabel('Bias (V)')
            axes[3].set_ylabel('ln(Current)')
            axes[3].set_ylim([-6, y_upper_lim])
            axes[3].legend(loc='upper left')

            fig.suptitle(plot_title)
            fig.tight_layout()
            plt.show()
            plt.close()

        # TODO ensure .75 correct
        slope = slopes[3]
        biases = bias_fit[left_edges[3]:right_edges[3]]
        currents = adj_current_to_fit[left_edges[3]:right_edges[3]]

        intercept = intercepts[3]
        r_squared = r_squareds[3]
        t_e = 1/slope * u.eV


        if r_squared < 0.8:
            return None, None, None, None, None, None, None, None
        else:
            return t_e, intercept,v_p, esat_slope, esat_intercept, t_e_offset,biases,currents







    # # Makes sure we don't have crazy values that will make the splines go crazy
    # exceptions_mask = np.abs(unique_adj_current) < 5 * np.max(np.abs(unique_adj_current))
    # unique_bias = unique_bias[exceptions_mask]
    # unique_adj_current = unique_adj_current[exceptions_mask]
    #
    # # Makes sure we don't have crazy values that will make the splines go crazy
    # exceptions_mask = np.where(unique_adj_current >= 5 * np.max(unique_adj_current))[0]
    # unique_bias = unique_bias[exceptions_mask]
    # unique_adj_current = unique_adj_current[exceptions_mask]
    #
    # bias_vals = unique_bias.value
    #
    # voltage_span = bias_vals.max() - bias_vals.min()
    # points_per_volts = len(bias_vals)/voltage_span
    #
    # smoothing_width = 4.0
    # window_length = int(smoothing_width * points_per_volts)
    #
    # polyorder = 3
    #
    # if window_length % 2 == 0:
    #     window_length += 1
    #
    # fitted_log_current = savgol_filter(unique_adj_current, window_length=window_length, polyorder=polyorder)
    #
    # dx = np.mean(np.diff(bias_vals))
    #
    # dlnIdV = savgol_filter(unique_adj_current, window_length=window_length, polyorder=polyorder, deriv=1, delta=dx)
    #
    # dlnI2dV2 = savgol_filter(unique_adj_current, window_length=window_length, polyorder=polyorder, deriv=2, delta=dx)
    #
    # u_b_v_f_idx = np.where(unique_bias.value >= v_f_bias.value)[0][0]
    # min_idx = u_b_v_f_idx
    #
    # # Boolean mask for low derivative
    # low_mask = np.abs(dlnIdV) < 0.05
    #
    # # Ignore everything before min_idx
    # low_mask[:min_idx] = False
    #
    # # Find edges
    # diff = np.diff(low_mask.astype(int))
    # start_index = np.where(diff == 1)[0] + 1
    # end_index = np.where(diff == -1)[0] + 1
    #
    # # Handle edge cases
    # if low_mask[0]:
    #     start_index = np.r_[0, start_index]
    # if low_mask[-1]:
    #     end_index = np.r_[end_index, len(low_mask)]
    #
    # # Find longest plateau
    # lengths = end_index - start_index
    # try:
    #     longest_idx = np.argmax(lengths)
    #     start_idx = start_index[longest_idx]
    #     end_idx = end_index[longest_idx] - 1
    #
    #     # # Create a spline for the log data
    #     # log_spline = UnivariateSpline(unique_bias, unique_adj_current, s=3, k=3)
    #     #
    #     # spline_bias = np.linspace(min(unique_bias.value), max(unique_bias.value), 10000)
    #     # log_current_fit = log_spline(spline_bias)
    #     #
    #     # # First derivative of the spline of the log data
    #     # log_spline_deriv = log_spline.derivative(n=1)
    #     # dlnIdV = log_spline_deriv(spline_bias)
    #     #
    #     # # Second derivative of the spline of the log data
    #     # log_spline_2_deriv = log_spline.derivative(n=2)
    #     # dlnI2dV2 = log_spline_2_deriv(spline_bias)
    #
    #     # Because everything of interest is going to be within a narrow range, cut down the viewing window to make the
    #     # data easier to see
    #     log_mask_spline = ((bias_vals >= v_f_bias.value - 5) & (bias_vals<= bias_vals[end_idx]))
    #     log_mask_ub = ((unique_bias >= v_f_bias - 5 * u.V) & (unique_bias <= bias_vals[end_idx] * u.V))
    #     log_b_plot = bias_vals[log_mask_spline]
    #     dlnIdV_plot = dlnIdV[log_mask_spline]
    #
    #     # Calculate the electron temperature
    #     slope, t_e_intercept, v_p, esat_slope, esat_intercept = electron_temperature_max(unique_bias, unique_adj_current,
    #                                                                                      bias_vals, fitted_log_current,
    #                                                                                      dlnIdV,
    #                                                                                      v_f_bias,
    #                                                                                      log_mask_spline,
    #                                                                                      log_mask_ub)


    #     if slope is not None:
    #         t_e = 1 / slope * u.eV
    #     else:
    #         t_e = None
    #
    #     return t_e, t_e_intercept, v_p, esat_slope, esat_intercept
    # except ValueError:
    #     return None, None, None, None, None

def electron_temperature_max(unique_bias, unique_adj_current,
                             spline_bias, log_current_fit,
                             dlnIdV,
                             v_f_bias,
                             log_mask_spline,
                             log_mask_ub):
    '''

    Parameters
    ----------
    unique_bias - Unique Bias values from the langmuir probe sweep
    spline_bias - Bias values associated with the spline
    unique_adj_current - Unique ln current values from the langmuir probe sweep
    dlnIdV - derivative of the natural log of current curve
    log_mask_spline - log mask applied to the derivative curve for spline data
    log_mask_ub - log mask applied to the derivative curve for unique bias data

    Returns
    -------
    slope - Slope of the line of best fit of the linear region of the logarithmic curve -> Corresponds to 1/t_e in eV
    intercept - Intercept of the line of best fit of the linear region of the logarithmic curve

    Note: This method only works for higher temperature plasmas -- for lower temperature plasmas see the PINQUED analysis
    code. The primary reason for this is that the lower temperature plasmas don't have a well-defined electron
    saturation region. For further discussion of this see Chen's review of Langmuir probes here
    https://www.seas.ucla.edu/~ffchen/Publs/Chen210R.pdf
    '''

    # How much of the initial data is checked in %
    check = 0.2
    # How many stds do we need to be away from the previous mean before we assume it is a knee
    knee_tol = 2
    # How close to the window maximum value should the maximum values be to register as a valid max value
    max_thresh = 0.5
    # How much up off the peak do you want to calculate the max from
    off_max_thresh = 0.6
    min_thresh = 0.15
    v_from_min = 1

    group_num = 3
    # Ensure that all parameters are compatible for the analysis
    unique_bias = unique_bias[log_mask_ub]
    log_spline = log_current_fit[log_mask_spline]
    spline_bias = spline_bias[log_mask_spline]
    unique_adj_current = unique_adj_current[log_mask_ub]
    dlnIdV = dlnIdV[log_mask_spline]

    len_to_avg_over = int(len(dlnIdV) * check)
    avg_initial_slope = np.mean(dlnIdV[-len_to_avg_over:])
    std_initial_slope = np.std(dlnIdV[-len_to_avg_over:])

    first_idx = -len_to_avg_over - (group_num + 1)
    last_idx = first_idx + group_num
    test_data = dlnIdV[first_idx:last_idx]
    while (np.mean(test_data) - avg_initial_slope < knee_tol * std_initial_slope
                  and first_idx > -len(dlnIdV) + 1):
        first_idx -= group_num
        last_idx -= group_num
        test_data = dlnIdV[first_idx:last_idx]

    neg_knee_idx = (first_idx + last_idx) // 2

    # From the negative knee index we want to slide until we find a max in the dIdV curve

    # Convert negative knee index to positive
    knee_idx = len(dlnIdV) + neg_knee_idx
    v_f_spline_idx = np.where(spline_bias >= v_f_bias.value)[0][0]
    v_f_ub_idx = np.where(unique_bias.value >= v_f_bias.value)[0][0]
    # print('v_f_spline_idx: ', v_f_spline_idx, 'knee_idx: ', knee_idx)
    search_deriv = dlnIdV[v_f_spline_idx:knee_idx]
    # print(len(search_deriv))
    search_bias = spline_bias[v_f_spline_idx:knee_idx]
    search_max = np.max(search_deriv)
    max_thresh = max_thresh * search_max
    # print("search_max:", search_max, "max_thresh:", max_thresh)

    maxs, max_props = find_peaks(
                                 search_deriv,
                                 distance = group_num,
                                 height = max_thresh,
                                 prominence = 0.03 * search_max
                                 )

    if len(maxs) == 0:
        widest_maxs = None
        widest_widths = None
    else:
        # Returns width value, y-value of the peak, left fractional index at specified hight,
        # and right fractional index at specified height
        width_peaks = peak_widths(search_deriv, maxs, rel_height = off_max_thresh)
        widths = width_peaks[0]
        left_ips = width_peaks[2]
        right_ips = width_peaks[3]

        # Take the two widest peaks
        sorted_indices = np.argsort(widths)[::-1]  # descending width
        top_two = sorted_indices[:2] if len(widths) >= 2 else sorted_indices

        widest_peaks = maxs[top_two]
        widest_left_ips = left_ips[top_two]
        widest_right_ips = right_ips[top_two]

        # Select the remaining peak with the highest voltage
        rightmost_idx = np.argmax(widest_peaks)
        max_slope_idx = widest_peaks[rightmost_idx] + v_f_spline_idx
        left_edge = int(np.floor(widest_left_ips[rightmost_idx])) + v_f_spline_idx
        right_edge = int(np.ceil(widest_right_ips[rightmost_idx])) + v_f_spline_idx

        leftmost_idx = None
        max2_idx = None
        left2_edge = None
        right2_edge = None
        if len(sorted_indices) > 1:
            # Find the second peak to plot it
            leftmost_idx = np.argmin(widest_peaks)
            max2_idx = widest_peaks[leftmost_idx] + v_f_spline_idx
            left2_edge = int(np.floor(widest_left_ips[leftmost_idx])) + v_f_spline_idx
            right2_edge = int(np.ceil(widest_right_ips[leftmost_idx])) + v_f_spline_idx

        left_edge_ub = np.where(unique_bias.value < spline_bias[left_edge])[0][-1]
        right_edge_ub = np.where(unique_bias.value < spline_bias[right_edge])[0][-1]

        left2_edge_ub = np.where(unique_bias.value < spline_bias[left2_edge])[0][-1]
        right2_edge_ub = np.where(unique_bias.value < spline_bias[right2_edge])[0][-1]

        knee_idx_ub = np.where(unique_bias.value < spline_bias[knee_idx])[0][-1]
        max_slope_idx_ub = np.where(unique_bias.value < spline_bias[max_slope_idx])[0][-1]

        # Fit the region with a line and return the slope and intercept
        slope, intercept = np.polyfit(unique_bias[left_edge_ub:right_edge_ub],
                                       unique_adj_current[left_edge_ub:right_edge_ub], 1)
        esat_slope, esat_intercept = np.polyfit(unique_bias[knee_idx_ub:],
                                                unique_adj_current[knee_idx_ub:],1)
        r = np.corrcoef(unique_bias[left_edge_ub:right_edge_ub], unique_adj_current[left_edge_ub:right_edge_ub]) [0,1]
        r2 = r ** 2
        # print('r^2: ', r2)
        if np.isclose(slope, esat_slope):
            v_p = np.nan  # Lines nearly parallel
        else:
            v_p = ((esat_intercept - intercept) / (slope - esat_slope)) * u.V

        v_p_idx = np.where(unique_bias < v_p)[0][-1]

        # print('v_p: ', v_p)
    # # Create plots
    # fig, ax = plt.subplots(2,2, figsize = (8,8))
    # ax = ax.flatten()
    # ax[0].plot(unique_bias, unique_adj_current, marker=".", color='b', linestyle='None', label= 'Original data')
    # ax[0].plot(unique_bias[v_f_ub_idx:v_p_idx], slope * unique_bias[v_f_ub_idx:v_p_idx].value + intercept, linestyle="--", color='r', label = 'Temperature Fit')
    # ax[0].plot(unique_bias, esat_slope * unique_bias.value + esat_intercept, linestyle="--", color='c',
    #            label = 'Electron Saturation Fit')
    # ax[0].plot(unique_bias[v_p_idx], unique_adj_current[v_p_idx], marker=".", color='k')
    # ax[0].plot(spline_bias[v_f_spline_idx], log_spline[v_f_spline_idx], marker=".", color='c')
    # ax[0].axvline(spline_bias[left_edge], color='m', linestyle='--', label='Left edge of fit')
    # ax[0].axvline(spline_bias[right_edge], color='y', linestyle='--', label='Right edge of fit')
    # ax[0].plot(spline_bias[max_slope_idx], log_spline[max_slope_idx], marker=".", color='g', label = 'Maximum index')
    # ax[0].set_title('log plot ')
    # ax[0].set_xlabel(r'Voltage (V)')
    # ax[0].set_ylabel(r'$\text{ln}(I)$')
    # ax[0].legend(loc='lower right')
    #
    # ax[1].plot(spline_bias, dlnIdV, marker=".", color='b', label = 'Derivative of spline fit')
    # ax[1].plot(spline_bias[max_slope_idx], dlnIdV[max_slope_idx], marker=".", color='k', label = 'Maximum index')
    # ax[1].plot(spline_bias[v_f_spline_idx], dlnIdV[v_f_spline_idx], marker=".", color='c')
    # ax[1].axvline(spline_bias[left_edge], color='m', linestyle='--', label='Left edge of fit')
    # ax[1].axvline(spline_bias[right_edge], color='y', linestyle='--', label='Right edge of fit')
    # ax[1].axvline(spline_bias[knee_idx], color='r', linestyle='--', label='Knee')
    # ax[1].set_xlabel('Voltage (V)')
    # ax[1].set_ylabel(r'$\frac{\text{dln}(I)}{\text{d}V}$')
    # ax[1].set_title('log plot derivative ')
    # ax[1].legend(loc='lower right')
    #
    # ax[2].plot(unique_bias[left_edge_ub:right_edge_ub], unique_adj_current[left_edge_ub:right_edge_ub], marker=".", color='b',
    #            linestyle = 'None',label = 'Original Data')
    # ax[2].plot(unique_bias[left_edge_ub:right_edge_ub], slope * unique_bias[left_edge_ub:right_edge_ub].value + intercept,
    #            linestyle="--", color='r', label = 'Temperature fit')
    # ax[2].plot(unique_bias[max_slope_idx_ub], unique_adj_current[max_slope_idx_ub], marker=".", color='g', label = 'Maximum index')
    # ax[2].legend(loc='lower right')
    # ax[2].set_xlabel('Voltage (V)')
    # ax[2].set_ylabel(r'$\text{ln}(I)$')
    # ax[2].set_title('Zoomed log plot ')
    #
    # plt.tight_layout()
    # plt.show()
    # plt.close('all')

    if (max_slope_idx < v_f_spline_idx or
            v_p.value > spline_bias[knee_idx] or
            (spline_bias[right_edge]- spline_bias[left_edge] < 2) or
            r2 < 0.85):
        slope = None
        intercept = None
        v_p = None
        esat_slope = None
        esat_intercept = None

    return slope, intercept, v_p, esat_slope, esat_intercept

def nan_summary(ds: xr.Dataset, diagnostic_keys=None):
    """
    Prints a summary of NaN counts and proportions for each variable in an xarray.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to inspect.
    diagnostic_keys : list of str, optional
        If provided, only check these variables.
    """
    if diagnostic_keys is None:
        diagnostic_keys = list(ds.data_vars)

    print("🧪 NaN Summary Report:")
    for var in diagnostic_keys:
        da = ds[var]
        total_values = da.size
        nan_count = da.isnull().sum().item()
        nan_percentage = 100 * nan_count / total_values if total_values > 0 else 0
        print(f"  • {var}: {nan_count:,} NaNs ({nan_percentage:.2f}%) "
              f"out of {total_values:,} total values")

        # Optional: print which dimensions have all-NaN slices
        for dim in da.dims:
            nan_along_dim = da.isnull().all(dim=tuple(d for d in da.dims if d != dim))
            all_nan_indices = nan_along_dim.where(nan_along_dim, drop=True)
            if all_nan_indices.size > 0:
                print(f"     - All-NaN {dim} indices: {all_nan_indices.coords[dim].values}")

def filter_data(mean_data,std_data,first_filter=False):
    """

    Parameters
    ----------
    mean_data : xarray 3D DataArray for a specific probe and averaged over shot
    std_data : xarray 3D DataArray for a specific probe, gives the standard deviation when averaged over shot

    Returns
    -------
    filtered_data : xarray 3D DataArray filtered by standard deviation and nearest neighbors
    """

    # Initial filtering - standard deviation
    std_median = std_data.median()
    std_data = std_data.astype('float32')
    maska = (std_data < (3 * std_median))
    maskb = std_data != 0
    mask1 = maska & maskb

    if first_filter:
        mask2 = xr.zeros_like(mask1, dtype=bool)
    else:
        neighbor_mean = mean_data.rolling(x=3, sweep=5, center=True, min_periods=1).mean()
        compare_neighbor = abs(mean_data - neighbor_mean)
        mask2 = compare_neighbor <= std_median

    valid_mask = mask1 | mask2

    # Filter out loner points

    # Neighborhood size must be odd because we want to have the point we are looking at in the center
    neighborhood_size = 3
    # How many non-NaN neighbors does the point need to have for it to be considered valid
    min_neighbors = 1

    binary_mask = valid_mask.fillna(False).astype(int).values

    neighborhood_matrix  = np.ones((neighborhood_size, neighborhood_size), dtype=int)
    center = neighborhood_size // 2
    neighborhood_matrix[center, center] = 0

    # How many of the neighbors are non zero
    neighbor_counts = convolve(binary_mask[:, 0, :], neighborhood_matrix, mode='constant', cval = 0)

    mask3_2d = neighbor_counts >= min_neighbors

    mask3_3d = np.repeat(mask3_2d[:,np.newaxis,:], valid_mask.sizes['y'], axis=1)
    final_mask3 = xr.DataArray(mask3_3d, coords = valid_mask.coords, dims = valid_mask.dims)

    final_valid_mask = (binary_mask == 1) & final_mask3.values

    filtered_data = mean_data.where(final_valid_mask)

    return filtered_data

def filter_ne_data(filtered_ne_data,min_time,max_time):
    """

    Parameters
    ----------
    filtered_ne_data
    min_time
    max_time

    Returns
    -------

    """
    # filt_plot = filtered_ne_data.plot(
    # x = 'time',
    # y = 'x',
    # vmin = 0,
    # vmax = 3e18,
    # cmap = 'turbo',
    # add_colorbar = True
    # )
    # ax = plt.gca()
    # filt_plot.colorbar.set_label(
    #     f"n_e ({filtered_ne_data.attrs.get('units', 'n_e')})"
    # )
    # ax.set_xlabel(f' time ({filtered_ne_data.attrs.get("time_units")})')
    # ax.set_ylabel(f' x ({filtered_ne_data.attrs.get("x_units")})')
    # plt.show()
    time_mask = (filtered_ne_data['time'] >= min_time) & (filtered_ne_data['time'] <= max_time)
    time_n_e = filtered_ne_data.sel(sweep = time_mask)
    mean_n_e_profile = time_n_e.mean(dim = 'sweep')
    std_n_e_profile = time_n_e.std(dim = 'sweep')
    std_mean_ratio = std_n_e_profile/mean_n_e_profile
    ratio_std = std_mean_ratio.std(dim = 'x')
    ratio_mean = std_mean_ratio.mean(dim = 'x')
    std_away = (std_mean_ratio - ratio_mean)/ ratio_std
    outliers = abs(std_away) > 4
    filtered_ne_mean_profile = mean_n_e_profile.where(outliers == False)
    filtered_ne_std_profile = std_n_e_profile.where(outliers == False)

    return filtered_ne_mean_profile, filtered_ne_std_profile


def find_steady_state(t_e_data_arrays, n_e_data_arrays, ds, probe, run_identifier, prev_start=None, prev_end=None):
    """
    Parameters
    ----------
    t_e_data_arrays
    n_e_data_arrays
    ds : The xarray dataset (needed to redraw the plot)
    probe : The current probe index (needed to redraw the plot)
    run_identifier : String identifier (needed to redraw the plot)
    prev_start : Previously saved left edge (if any)
    prev_end : Previously saved right edge (if any)

    Returns
    -------
    min_time
    max_time
    """
    from lapd_plasma_analysis.obtain_plots.xarray_plots import plot_time_series

    # --- SHOW PREVIOUS BOUNDS ---
    if prev_start is not None and prev_end is not None:
        print(f"\nPrevious Steady State Bounds found: Left = {prev_start}, Right = {prev_end}")

        # Draw the plot
        fig, axes, _ = plot_time_series(ds, probe, run_identifier, return_range=True)

        # Add previous lines
        axes[0].axvline(x=prev_start, color='k', linestyle='--', linewidth=2, label='Prev Left')
        axes[1].axvline(x=prev_start, color='k', linestyle='--', linewidth=2, label='Prev Left')
        axes[0].axvline(x=prev_end, color='k', linestyle=':', linewidth=2, label='Prev Right')
        axes[1].axvline(x=prev_end, color='k', linestyle=':', linewidth=2, label='Prev Right')

        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=14)

        plt.show(block=False)
        plt.pause(0.5)  # Give macOS a half-second to render the window

        input("Press Enter to clear previous bounds and select new ones...")
        plt.close('all')  # Destroy the plot

    # --- LEFT EDGE SELECTION ---
    left_edge = None

    while True:
        # Generate fresh plot
        fig, axes, _ = plot_time_series(ds, probe, run_identifier, return_range=True)

        if left_edge is not None:
            axes[0].axvline(x=left_edge, color='r', linestyle='-', linewidth=2, label='Left Edge')
            axes[1].axvline(x=left_edge, color='r', linestyle='-', linewidth=2, label='Left Edge')
            for ax in axes:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=14)

        plt.show(block=False)
        plt.pause(0.5)

        user_input = input("\nEnter an INTEGER number for the LEFT edge: ")
        try:
            left_edge = int(user_input)
            plt.close('all')  # Close before asking if happy, to redraw with the new line

            # Redraw to show the line you just typed
            fig, axes, _ = plot_time_series(ds, probe, run_identifier, return_range=True)
            axes[0].axvline(x=left_edge, color='r', linestyle='-', linewidth=2, label='Left Edge')
            axes[1].axvline(x=left_edge, color='r', linestyle='-', linewidth=2, label='Left Edge')
            for ax in axes:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=14)
            plt.show(block=False)
            plt.pause(0.5)

            if ask_yes_or_no("Are you happy with this left edge placement (y/n)? "):
                plt.close('all')
                break
            else:
                plt.close('all')

        except ValueError:
            print("Invalid input. Please enter an integer.")
            plt.close('all')
            continue

    # --- RIGHT EDGE SELECTION ---
    right_edge = None

    while True:
        fig, axes, _ = plot_time_series(ds, probe, run_identifier, return_range=True)

        # Always draw the confirmed left edge
        axes[0].axvline(x=left_edge, color='r', linestyle='-', linewidth=2, label='Left Edge')
        axes[1].axvline(x=left_edge, color='r', linestyle='-', linewidth=2, label='Left Edge')

        if right_edge is not None:
            axes[0].axvline(x=right_edge, color='g', linestyle='-', linewidth=2, label='Right Edge')
            axes[1].axvline(x=right_edge, color='g', linestyle='-', linewidth=2, label='Right Edge')

        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=14)

        plt.show(block=False)
        plt.pause(0.5)

        user_input = input("\nEnter an INTEGER number for the RIGHT edge: ")
        try:
            right_edge = int(user_input)
            plt.close('all')

            # Redraw with both lines
            fig, axes, _ = plot_time_series(ds, probe, run_identifier, return_range=True)
            axes[0].axvline(x=left_edge, color='r', linestyle='-', linewidth=2, label='Left Edge')
            axes[1].axvline(x=left_edge, color='r', linestyle='-', linewidth=2, label='Left Edge')
            axes[0].axvline(x=right_edge, color='g', linestyle='-', linewidth=2, label='Right Edge')
            axes[1].axvline(x=right_edge, color='g', linestyle='-', linewidth=2, label='Right Edge')
            for ax in axes:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=14)
            plt.show(block=False)
            plt.pause(0.5)

            happy_with_right = ask_yes_or_no("Are you happy with this right edge placement (y/n)? ")

            if happy_with_right:
                if right_edge <= left_edge:
                    print("Error: The right edge must be strictly greater than the left edge. Try again.")
                    right_edge = None
                    plt.close('all')
                else:
                    plt.close('all')
                    break
            else:
                plt.close('all')

        except ValueError:
            print("Invalid input. Please enter an integer.")
            plt.close('all')
            continue

    # --- DURATION CHECK ---
    duration = right_edge - left_edge
    if duration > 5:
        print(
            f"\nWarning: The selected steady state time period ({duration} ms) is longer than 5 ms and may be too long.")

    return left_edge, right_edge

def find_steady_state_auto(t_e_data_arrays, n_e_data_arrays, middle_guess):
    """
    Parameters
    ----------
    t_e_data_arrays
    n_e_data_arrays
    middle_guess

    Returns
    -------
    min_time
    max_time
    """

    initial_n_e_mean_array = []
    initial_t_e_mean_array = []
    initial_t_e_std_array = []
    initial_n_e_std_array = []
    # The initial window is centered at the user guess -> this tells you how many points away from center we want
    # our standard window we will test everything else against will be
    initial_window = 1
    tolerance = .15

    dt = t_e_data_arrays[0]['time'].diff('sweep').mean().item()
    # print('time array: ', t_e_data_arrays[0]['time'])
    for i in range(len(t_e_data_arrays)):
        t_e_of_int = t_e_data_arrays[i]
        n_e_of_int = n_e_data_arrays[i]
        t_e_mask = ((t_e_of_int['time'] >= float(middle_guess - initial_window * dt)) &
                    (t_e_of_int['time'] <= float(middle_guess + initial_window * dt)))

        n_e_mask = ((n_e_of_int['time'] >= float(middle_guess - initial_window * dt)) &
                    (n_e_of_int['time'] <= float(middle_guess + initial_window * dt)))

        mean_t_e = t_e_of_int.sel(sweep = t_e_of_int['sweep'][t_e_mask]).mean('sweep').mean()
        mean_n_e = n_e_of_int.sel(sweep = n_e_of_int['sweep'][n_e_mask]).mean('sweep').mean()

        std_t_e = t_e_of_int.sel(sweep = t_e_of_int['sweep'][t_e_mask]).std('sweep').mean()
        std_n_e = n_e_of_int.sel(sweep = n_e_of_int['sweep'][n_e_mask]).std('sweep').mean()


        initial_t_e_mean_array.append(mean_t_e)
        initial_n_e_mean_array.append(mean_n_e)
        initial_t_e_std_array.append(std_t_e)
        initial_n_e_std_array.append(std_n_e)

    initial_mean_mean_t_e = float(np.mean(initial_t_e_mean_array))
    initial_mean_mean_n_e = float(np.mean(initial_n_e_mean_array))
    initial_mean_std_t_e = float(np.mean(initial_t_e_std_array))
    initial_mean_std_n_e = float(np.mean(initial_n_e_std_array))

    initial_t_e_std_mean_ratio = initial_mean_std_t_e / initial_mean_mean_t_e
    initial_n_e_std_mean_ratio = initial_mean_std_n_e / initial_mean_mean_n_e

    initial_comparison_standard = initial_t_e_std_mean_ratio + initial_n_e_std_mean_ratio
    up_steps = initial_window + 1

    # Now we will check the upper values to
    while True:
        # print('Up steps:', up_steps)
        # print('condiition: ', middle_guess + up_steps * dt)
        if middle_guess + up_steps * dt >= max(t_e_data_arrays[0]['time'].values):
            break

        u_test_n_e_mean_array = []
        u_test_t_e_mean_array = []
        u_test_t_e_std_array = []
        u_test_n_e_std_array = []
        for j in range(len(t_e_data_arrays)):
            t_e_of_int = t_e_data_arrays[j]
            n_e_of_int = n_e_data_arrays[j]
            t_e_mask = (t_e_of_int['time'] >= float(middle_guess - dt)) & (
                        t_e_of_int['time'] <= float(middle_guess + up_steps * dt))
            n_e_mask = (n_e_of_int['time'] >= float(middle_guess - dt)) & (
                        n_e_of_int['time'] <= float(middle_guess + up_steps * dt))

            u_mean_t_e = t_e_of_int.sel(sweep=t_e_of_int['sweep'][t_e_mask]).mean('sweep').mean()
            u_mean_n_e = n_e_of_int.sel(sweep=n_e_of_int['sweep'][n_e_mask]).mean('sweep').mean()

            u_std_t_e = t_e_of_int.sel(sweep=t_e_of_int['sweep'][t_e_mask]).std('sweep').mean()
            u_std_n_e = n_e_of_int.sel(sweep=n_e_of_int['sweep'][n_e_mask]).std('sweep').mean()

            u_test_t_e_mean_array.append(u_mean_t_e)
            u_test_n_e_mean_array.append(u_mean_n_e)
            u_test_t_e_std_array.append(u_std_t_e)
            u_test_n_e_std_array.append(u_std_n_e)

        u_test_mean_mean_t_e = float(np.mean(u_test_t_e_mean_array))
        # print('test mean of means: ', u_test_mean_mean_t_e)
        u_test_mean_mean_n_e = float(np.mean(u_test_n_e_mean_array))
        u_test_mean_std_t_e = float(np.mean(u_test_t_e_std_array))
        u_test_mean_std_n_e = float(np.mean(u_test_n_e_std_array))

        u_test_t_e_std_mean_ratio = u_test_mean_std_t_e / u_test_mean_mean_t_e
        u_test_n_e_std_mean_ratio = u_test_mean_std_n_e / u_test_mean_mean_n_e
        u_test_comparison_standard = u_test_t_e_std_mean_ratio + u_test_n_e_std_mean_ratio
        # print('regular comparison ', u_comparison_standard)
        # print('test comparison ', u_test_comparison_standard)
        if ((abs(initial_comparison_standard - u_test_comparison_standard)/
            (initial_comparison_standard + u_test_comparison_standard)) >= tolerance):
            up_steps = up_steps - 1
            break
        else:
            up_steps += 1

    # Now for down in index
    down_steps = initial_window + 1


    while True:
        # print('Down steps:', down_steps)
        if middle_guess - down_steps * dt <= min(t_e_data_arrays[0]['time'].values):
            break

        d_test_n_e_mean_array = []
        d_test_t_e_mean_array = []
        d_test_t_e_std_array = []
        d_test_n_e_std_array = []
        for j in range(len(t_e_data_arrays)):
            t_e_of_int = t_e_data_arrays[j]
            n_e_of_int = n_e_data_arrays[j]
            t_e_mask = (t_e_of_int['time'] >= float(middle_guess - down_steps * dt)) & (
                    t_e_of_int['time'] <= float(middle_guess + dt))
            n_e_mask = (n_e_of_int['time'] >= float(middle_guess - down_steps * dt)) & (
                    n_e_of_int['time'] <= float(middle_guess + dt))

            d_mean_t_e = t_e_of_int.sel(sweep=t_e_of_int['sweep'][t_e_mask]).mean('sweep').mean()
            d_mean_n_e = n_e_of_int.sel(sweep=n_e_of_int['sweep'][n_e_mask]).mean('sweep').mean()

            d_std_t_e = t_e_of_int.sel(sweep=t_e_of_int['sweep'][t_e_mask]).std('sweep').mean()
            d_std_n_e = n_e_of_int.sel(sweep=n_e_of_int['sweep'][n_e_mask]).std('sweep').mean()

            d_test_t_e_mean_array.append(d_mean_t_e)
            d_test_n_e_mean_array.append(d_mean_n_e)
            d_test_t_e_std_array.append(d_std_t_e)
            d_test_n_e_std_array.append(d_std_n_e)

        d_test_mean_mean_t_e = float(np.mean(d_test_t_e_mean_array))
        d_test_mean_mean_n_e = float(np.mean(d_test_n_e_mean_array))
        d_test_mean_std_t_e = float(np.mean(d_test_t_e_std_array))
        d_test_mean_std_n_e = float(np.mean(d_test_n_e_std_array))

        d_test_t_e_std_mean_ratio = d_test_mean_std_t_e / d_test_mean_mean_t_e
        d_test_n_e_std_mean_ratio = d_test_mean_std_n_e / d_test_mean_mean_n_e
        d_test_comparison_standard = d_test_t_e_std_mean_ratio + d_test_n_e_std_mean_ratio

        if (abs(initial_comparison_standard - d_test_comparison_standard) / (
                initial_comparison_standard + d_test_comparison_standard)) >= tolerance:
            down_steps = down_steps - 1
            break
        else:
            d_comparison_standard = d_test_comparison_standard
            down_steps += 1

    # print('dt: ', dt)

    min_time = middle_guess - dt * down_steps
    max_time = middle_guess + dt * up_steps
    # print('min_time: ', min_time)
    # print('max_time: ', max_time)

    return min_time, max_time


