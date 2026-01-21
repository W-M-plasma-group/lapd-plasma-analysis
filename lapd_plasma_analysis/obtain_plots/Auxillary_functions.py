# from pty import slave_open

import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const
from plasmapy.diagnostics.langmuir import get_plasma_potential
from plasmapy.formulary import Coulomb_logarithm
from pycparser.c_ast import Return
from scipy.constants import epsilon_0

from lapd_plasma_analysis.file_access import *
from lapd_plasma_analysis.experimental import get_exp_params

from lapd_plasma_analysis.langmuir.helper import *
from lapd_plasma_analysis.langmuir.configurations import *
from lapd_plasma_analysis.langmuir.getIVsweep import get_sweep_voltage, get_sweep_current, get_shot_positions
from lapd_plasma_analysis.langmuir.characterization import make_characteristic_array, isolate_ramps
from lapd_plasma_analysis.langmuir.preview import preview_raw_sweep, preview_characteristics
from lapd_plasma_analysis.langmuir.diagnostics import (langmuir_diagnostics, detect_steady_state_times, get_pressure,
                                                       get_electron_ion_collision_frequencies)
from lapd_plasma_analysis.langmuir.neutrals import get_neutral_density
from lapd_plasma_analysis.langmuir.interferometry import interferometry_calibration
from lapd_plasma_analysis.langmuir.plots import get_title
from lapd_plasma_analysis.langmuir.metadata_for_dataset import get_supplemental_metadata
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


def get_ion_isat_min(sorted_current,sorted_bias):
    """

    Parameters
    ----------
    sorted_current - Array of current value sorted by matched bias values from the minimum to maximum of the bias array

    Returns
    -------
    ion_isat - Float: value of ion isat in Amps
    ion_isat_index - Int: location of the ion isat within the sorted current and sorted bias array

    Finds the Ion Isat by taking the average of the first avg_volts of the current array
    """

    ion_isat_index = np.argmin(sorted_current)

    avg_volts = 5 * u.V

    if sorted_bias[0] < sorted_bias[ion_isat_index] - avg_volts/2:
        end_index = np.nonzero(sorted_bias < sorted_bias[ion_isat_index] + avg_volts/2)[0][-1]
        start_index = np.nonzero(sorted_bias < sorted_bias[ion_isat_index] - avg_volts/2)[0][-1]
    else:
        start_index = 0
        end_index = np.nonzero(sorted_bias < sorted_bias[0] + avg_volts)[0][-1]

    ion_isat_current = sorted_current[start_index:end_index]
    ion_isat = np.mean(ion_isat_current)

    return ion_isat, ion_isat_index

def get_electron_isat_end(sorted_bias,sorted_current,get_V_P=False):
    """

    Parameters
    ----------
    sorted_bias
    sorted_current

    Returns
    -------
    electron_isat - float in Amps corresponding to the electron saturation current
    e_sat_index - int indicating the index of the point that was determined to be the electron saturation current

    This function aims to find the knee in the IV curve for a Langmuir probe by searching for when there is a drastic
    shift in the slope of the IV sweep. It starts sampling from the last element in the current array

    """

    check_tolerance = 3
    confirm_tolerance = check_tolerance
    step = min(30,len(sorted_bias)-2)
    skip = 3
    k = 0
    electron_isat = min(sorted_current)
    # Calculate the average slope of the last step elements in the current and bias arrays
    slope_array = []
    # May take a lot of time -> Test and see if its worth the extra time if not go back to calculating slope as in
    # the slope of the exponential part of the curve calculation
    while electron_isat < .7 * max(sorted_current):
        slope_array = []
        slope_iteration = 1
        for i in range(step):
            try:
                delta_current = sorted_current[-(i + 1)] - sorted_current[-(i + 1 + skip)]
                delta_bias = sorted_bias[-(i + 1)] - sorted_bias[-(i + 1 + skip)]
                if np.isclose(delta_bias.value, 0):
                    continue
                slope_array.append(delta_current / delta_bias)
            except IndexError:
                break  # Avoid crashing if out of bounds
            except ZeroDivisionError:
                continue  # Avoid division by zero

        if len(slope_array) == 0:
            raise ValueError("Not enough data points to compute slope.")
        slope_array = [s for s in slope_array if np.isfinite(s.value)]
        check_slope = u.Quantity(slope_array).mean()

        test_slope = check_slope
        h = 1
        while step * (h+1) <= len(sorted_bias):
            slope_array = []
            for j in range(step):
                try:
                    delta_current = sorted_current[-(h * step + (j + 1))] - sorted_current[-(h * step + (j + 1 + skip))]
                    delta_bias = sorted_bias[-(h * step + (j + 1))] - sorted_bias[-(h * step + (j + 1 + skip))]
                    # Don't add an inf or nan value to the slope array
                    if np.isclose(delta_bias.value, 0):
                        continue
                    slope_array.append(delta_current / delta_bias)
                except IndexError:
                    break  # Avoid crashing if out of bounds
                except ZeroDivisionError:
                    continue  # Avoid division by zero
            h += 1
            slope_array = [s for s in slope_array if np.isfinite(s.value)]
            if len(slope_array) == 0:
                break
            test_slope = u.Quantity(slope_array).mean()

            # Force the electron saturation current to choose an index outside the last data points
            if k == 0:
                last_index_condition = step * (h + 1) >= 0.25 * len(sorted_bias)
            elif k == 1:
                last_index_condition = step * (h + 1) >= 0.1 * len(sorted_bias)
            elif k == 2:
                last_index_condition = True

            # What tolerance are we looking for - the higher threshold indicates the initial dip, the lower threshold
            # confirms that we have the right electron saturation current
            if slope_iteration == 2:
                tolerance = confirm_tolerance
            else:
                tolerance = check_tolerance

            # Checks to see if we have a notable change in the slope - abs because we want a positive slope
            if (abs(test_slope - check_slope) >= tolerance * abs(check_slope) and last_index_condition and
                    slope_iteration == 1):
                pp_index = -(h*step+1)
                slope_iteration += 1
                check_slope_hold = test_slope
            elif (abs(test_slope - check_slope) >= tolerance * abs(check_slope) and last_index_condition and
                    slope_iteration == 2):
                break
            elif ((abs(test_slope - check_slope) <= tolerance * abs(check_slope)) and
                    slope_iteration == 2):
                slope_iteration = 1
                h -= 1
                pp_index = None
                check_slope = check_slope_hold
            else:
                check_slope = test_slope
                pp_index = None

            # Return none if we've searched the entire current array and found nothing
            if step * (h + 1) >= len(sorted_bias) and pp_index is None:
                print("Failed to find an electron Isat")
                return None,None

        # Return the current of the last tested value
        electron_isat = sorted_current[pp_index]
        # if electron_isat > .7*max(sorted_current):
        print('electron_isat =', electron_isat)
        print('test_current =', .7*max(sorted_current))
        #     break
        k += 1
        if k > 2:
            break

    # Allows you to get the index of the upper bound for a search for the plasma potential
    if get_V_P:
        return electron_isat, pp_index

    else:
        return electron_isat, None

def get_electron_isat_max(sorted_current, sorted_bias):
    """

    Parameters
    ----------
    sorted_current - Array of current value sorted by matched bias values from the minimum to maximum of the bias array

    Returns
    -------
    electron_isat - Quantity: Electron Isat value in A
    electron_isat_index - Integer: Electron Isat location within sorted bias (and sorted current) array

    Finds the Electron saturation current by taking the max of the sorted current array
    """

    electron_isat_index = np.argmax(sorted_current)
    avg_volts = 5 * u.V

    if sorted_bias[-1] > sorted_bias[electron_isat_index] + avg_volts/2:
        end_index = np.nonzero(sorted_bias < sorted_bias[electron_isat_index] + avg_volts/2)[0][-1]
        start_index = np.nonzero(sorted_bias < sorted_bias[electron_isat_index] - avg_volts/2)[0][-1]
    else:
        end_index = len(sorted_bias) - 1
        start_index = np.nonzero(sorted_bias < sorted_bias[-1] - avg_volts)[0][-1]

    electron_isat_current = sorted_current[start_index:end_index]
    electron_isat = np.mean(electron_isat_current)

    return electron_isat, electron_isat_index

def get_electron_isat_v_f(sorted_bias, sorted_current,v_f_index, get_V_P = False):
    """

    Parameters
    ----------
    sorted_bias
    sorted_current
    v_f_index - int - floating potential index
    get_V_P - Boolean - Does the user want to the index from the electron saturation current to start searching for the
                        plasma potential?


    Returns
    -------
    electron_isat - Quantity in Amps corresponding to the electron saturation current
    e_sat_index - int indicating the index of the point that was determined to be the electron saturation current
                    Only returned if get_V_P is True.

    This function aims to find the knee in the IV curve for a Langmuir probe by searching for when there is a drastic
    shift in the slope of the IV sweep. It starts sampling from the last element in the current array

    """

    original_tolerance = .7
    # Confirm_tolerance tells what % of check slope you want the confirmation loop to look for
    confirm_tolerance = .7
    step = min(30, len(sorted_bias) - 2)
    skip = 3
    k = 0
    electron_isat = min(sorted_current)
    threshold_current = 0.7 * max(sorted_current)
    # Calculate the average slope of the last step elements in the current and bias array
    # May take a lot of time -> Test and see if its worth the extra time if not go back to calculating slope as in
    # the slope of the exponential part of the curve calculation
    for k in range(3):
        check_tolerance = original_tolerance * (1 if k == 0 else 0.8 if k == 1 else (.8-.2*(k-1))/.8)
        slope_array = []
        slope_iteration = 1
        for i in range(step):
            try:
                delta_current = sorted_current[v_f_index+i] - sorted_current[v_f_index + skip + i]
                delta_bias = sorted_bias[v_f_index + i] - sorted_bias[v_f_index + skip + i]
                if np.isclose(delta_bias.value, 0):
                    continue
                slope_array.append(delta_current / delta_bias)
            except (IndexError, ZeroDivisionError):
                continue

        if len(slope_array) == 0:
            raise ValueError("Not enough data points to compute slope.")
        slope_array = [s for s in slope_array if np.isfinite(s.value)]
        check_slope = u.Quantity(slope_array).mean()

        test_slope = check_slope
        h = 1
        pp_index = None
        while v_f_index + step * (h + 1) <= len(sorted_bias):
            slope_array = []
            for j in range(step):
                try:
                    delta_current = sorted_current[h * step + j + v_f_index] - sorted_current[h * step + j + skip + v_f_index]
                    delta_bias = sorted_bias[h * step + j + v_f_index] - sorted_bias[h * step + j + skip + v_f_index]
                    # Don't add an inf or nan value to the slope array
                    if np.isclose(delta_bias.value, 0):
                        continue
                    slope_array.append(delta_current / delta_bias)
                except (IndexError, ZeroDivisionError):
                    continue
            h += 1
            slope_array = [s for s in slope_array if np.isfinite(s.value)]
            if len(slope_array) == 0:
                break
            test_slope = u.Quantity(slope_array).mean()



            # What tolerance are we looking for - the higher threshold indicates the initial dip, the lower threshold
            # confirms that we have the right electron saturation current
            if slope_iteration == 2:
                tolerance = confirm_tolerance * check_tolerance
            else:
                tolerance = check_tolerance

            # Checks to see if we have a notable change in the slope - abs because we want a positive slope
            if (abs((test_slope - check_slope)/u.Quantity([check_slope,test_slope]).mean()) >= tolerance and
                    slope_iteration == 1 and sorted_current[v_f_index + step * h] > threshold_current):
                pp_index = v_f_index + step * h
                slope_iteration += 1
                check_slope_hold = test_slope
            elif (abs((test_slope - check_slope)/u.Quantity([check_slope,test_slope]).mean()) >= tolerance and
                  slope_iteration == 2):
                electron_isat = sorted_current[pp_index]
                break
            elif (abs((test_slope - check_slope)/u.Quantity([check_slope,test_slope]).mean()) <= tolerance and
                  slope_iteration == 2):
                slope_iteration = 1
                h -= 1
                pp_index = None
                check_slope = check_slope_hold
            else:
                check_slope = test_slope
                pp_index = None

            # Return none if we've searched the entire current array and found nothing
            if step * (h + 1) >= len(sorted_bias) and pp_index is None:
                print("Failed to find an electron Isat")
                return None

    if pp_index is None:
        # print("Failed to find an electron Isat")
        return None,None

    # Allows you to get the index of the upper bound for a search for the plasma potential
    if get_V_P:
        return electron_isat, pp_index

    else:
        return electron_isat, None

def get_electron_isat_curve_fit(sorted_bias, sorted_current, v_f_index, v_p_index, return_arg = False):

    upper_section_bias = sorted_bias[v_p_index :].value
    upper_section_current = sorted_current[v_p_index :].value

    t_e, exp_int, offset = get_te_v_p_vf(sorted_bias,sorted_current,v_f_index,v_p_index,return_intercept = True)
    slope_exponential = 1/t_e.value
    exp_int = exp_int - np.log(offset)

    exp_const = np.exp(exp_int)
    exp_slope_fit = slope_exponential * exp_const * np.exp(slope_exponential * sorted_bias[v_p_index].value) * u.A/u.V


    up_slope, up_int = np.polyfit(upper_section_bias, upper_section_current, 1)

    # print("Up slope is: ", up_slope)
    # print("Up intercept is: ", up_int)

    exp_array = (exp_slope_fit * (sorted_bias - sorted_bias[v_p_index]) + sorted_current[v_p_index])

    up_array = up_slope * u.A/u.V * sorted_bias + up_int * u.A

    exp_minus_up = exp_array - up_array

    arg_electron_isat = np.nonzero(exp_minus_up < 0 * u.A)[0][-1]
    electron_isat = sorted_current[arg_electron_isat]

    # plt.plot(sorted_bias, up_array, label = 'upper section fit', color = 'c')
    # plt.plot(sorted_bias, exp_array, label = 'exponential fit', color = 'm')
    # plt.xlabel("Voltage [V]")
    # plt.ylabel("Current [A]")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # print('electron saturation bias = ', sorted_bias[arg_electron_isat])

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
    arg_v_f - Integer: index within sorted bias (and sorted current) where the plasma potential is located

    Searches for the last index where the current is less than zero. Checks to see if the next value is greater than zero,
    then if so, uses point slope form to determine the bias at which the current is equal to zero.
    """

    arg_v_f = np.nonzero(sorted_current < 0 * u.A)[0][-1]
    slope = (sorted_current[arg_v_f+1]-sorted_current[arg_v_f])/(sorted_bias[arg_v_f+1]-sorted_bias[arg_v_f])
    v_f_bias = -sorted_current[arg_v_f] / slope + sorted_bias[arg_v_f]

    return v_f_bias, 0 * u.A, arg_v_f

def get_plasma_potential(sorted_bias,sorted_current,I_esat_index):
    """

    Parameters
    ----------
    sorted_bias - Array of bias value sorted from minimum to maximum
    sorted_current - Array of current value sorted by matched bias values from the minimum to maximum of the bias array
    I_esat_index - Index of the electron saturation current - negative from the end

    Returns
    -------
    V_p_bias - Quantity (V): the voltage bias associated with the plasma potential
    V_p_current - Quantity (A): the current associated with the plasma potential

    Performs a slope sweep from the point where I = 0 to the Electron saturation current and picks the value with the
    highest slope
    """
    h = 0
    step = 10
    skip = 50
    mean_slope_array = []
    index_array = []
    while True:


        start_index = I_esat_index - (h + skip)
        end_index = I_esat_index - h

        # print('start index, end index ',start_index, end_index)
        if start_index < 0 or end_index >= len(sorted_bias):
            break

        bias_slice = sorted_bias[start_index:end_index]
        current_slice = sorted_current[start_index:end_index]
        slope_array = []
        # Take the average value of the slope between skip points by doing pointwise slope calculations
        # Good if there is a general trend in the data - otherwise not so much -> Slow but works well
        for i in range(len(bias_slice)-1):
            delta_current = current_slice[i+1]-current_slice[i]
            delta_bias = bias_slice[i+1]-bias_slice[i]
            if np.isclose(delta_bias.value, 0):
                continue
            slope_array.append((delta_current / delta_bias).to(u.A/u.V).value)
            # print('slope_array: ', slope_array)
        if slope_array:
            mean_slope_array.append(np.mean(slope_array))
            # print('mean slope array', mean_slope_array)
            index_array.append(start_index + skip // 2)
            # print('index_array', index_array)
        h += step
    if not mean_slope_array:
        return None, None, None
    # Finds the value of the greatest slope
    max_index = np.argmax(mean_slope_array)
    # print('max_index = ',max_index)
    v_p_index = index_array[max_index]

    # Returns the current and voltage associated with the plasma potential
    # Plasma potential is assumed to be the first value in the largest slope
    return sorted_bias[v_p_index], sorted_current[v_p_index], v_p_index

def get_plasma_potential_slope(sorted_bias,sorted_current,v_f_index,electron_esat_index):
    """

    Parameters
    ----------
    sorted_bias - Array of bias quantities (V) sorted from minimum to maximum
    sorted_current - Array of current quantities (A) sorted by matched bias values from the minimum to maximum of the bias array
    v_f_index - Index of the floating potential (where the current is 0) in the sorted bias and sorted current arrays
    electron_esat_index - Index of the electron saturation current in the sorted bias and sorted current arrays

    Returns
    -------
    v_p - Quantity (V): the voltage bias associated with the plasma potential
    v_p_index - Integer index of the plasma potential

    Starts building an array of slopes at the floating potential and ends it at the electron saturation. Returned value
    is the maximum value of that slope array and the index returned is the middle of the slope search array
    """

    min_value = 0.2 * sorted_current[electron_esat_index]
    min_index = np.argmin(np.abs(sorted_current-min_value))


    m_sorted_bias = sorted_bias[min_index : electron_esat_index + 1]
    m_sorted_current = sorted_current[min_index : electron_esat_index + 1]
    # print('min_index: ', min_index)


    # We look at 30 steps between the floating potential and the electron saturation current - TODO Only getting 1-2 values in final slope array
    step = 19
    skip = 3
    start_index = 0
    end_index = step - 1

    slope_array = []
    index_array = []
    while end_index < len(m_sorted_bias):
        mid_slope_array =[]

        for i in range(start_index, min(end_index - skip + 1, len(m_sorted_current)-skip), skip):
            # We do end_index - skip + 1 so it includes the last index
            slope_num = m_sorted_current[i + skip] - m_sorted_current[i]
            slope_den = m_sorted_bias[i + skip] - m_sorted_bias[i]
            if np.isclose(slope_den.value, 0):
                continue
            slope = slope_num / slope_den
            mid_slope_array.append(slope)

        if mid_slope_array:
            q_mid_slope_array = u.Quantity(mid_slope_array)
            mean_slope = q_mid_slope_array.mean()
            if q_mid_slope_array.std() <= 0.65 * mean_slope:
                slope_array.append(mean_slope)
                index_array.append((start_index + end_index) // 2)
        start_index = end_index
        end_index = end_index + step - 1

    # print('slope array: ', slope_array)
    if not slope_array:
        return None, None

    index_of_int = int(np.argmax([slope.value for slope in slope_array]))
    v_p = m_sorted_bias[index_array[index_of_int]]
    v_p_index = index_array[index_of_int] + v_f_index

    return v_p.to(u.V), v_p_index

def l_get_pressure(temperature, density):
    """
    Parameters
    ----------
    temperature (Quantity): Temperature in eV
    density (Quantity): Density in m^{-3}

    Returns
    -------
    Pressure - Quantity (Pascal): the pressure associated with the corresponding temperature and density values
    """
    return(temperature.to(u.J, equivalencies=u.temperature_energy()) * density).to(u.Pa)

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
    m_i = Particle(ion_type).mass
    # e * T_e = T_e Joules
    T_e_joules = T_e.to(u.J, equivalencies=u.temperature_energy())
    n_i = -((ion_isat / (c.e.si * A_p * np.exp(-0.5))) * np.sqrt(m_i / T_e_joules))
    return n_i.to(1/(u.m ** 3))

def get_ion_density_chi(ion_type,ion_isat,A_p,T_e,chi):
    m_i = Particle(ion_type).mass

    n_i = -(ion_isat/(chi * c.e.si * A_p) * (m_i/(c.e.si * T_e)) ** 0.5).value
    return n_i * 1/(u.m ** 3)

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
    m_e = Particle("electron").mass

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
    p_ion = Particle(ion_type)
    p_electron = Particle("electron")
    epsilon_0 = 8.854187817e-12 * u.F / u.m

    # Ion temperature is assumed to be 1 eV as reported in Perks et al. 2022
    T_i = 1 * u.eV
    T_i_joule = T_i.to(u.J, equivalencies=u.temperature_energy())
    T_e_joule = T_e.to(u.J, equivalencies=u.temperature_energy())
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

def get_te(sorted_bias,sorted_current,v_f_index):
    # Convert to numpy array with units handled
    current_vals = sorted_current[v_f_index :]
    bias_vals = sorted_bias[v_f_index :]

    bias_vals = u.Quantity(bias_vals,u.V)
    current_vals = u.Quantity(current_vals,u.A)


    # Mask: keep only values where current is non-negative
    mask = current_vals > 0 * u.A
    adjusted_bias = bias_vals[mask]
    adjusted_current = np.log(current_vals[mask].to(u.A).value)


    # Now
    # May want to add the addition of *10^-9 or so after the absolute value - but there are significant outliers
    # Need to fit a line to the straight region of the curve.

    # Build an array of skip slopes and calculate the mean and standard deviation if the standard deviation is bigger
    # than the tolerance (which should be derived from the mean) then that should be the slope we use.
    # Plot the calculated slope on the log plot using point slope form from the first point of the median slope value

    skip = 24
    elem_in_array = 4
    found_slope = False
    j = 0  # set the start index
    while not found_slope and j + (elem_in_array + 1) * skip < len(adjusted_current):
        slopes_to_use = []
        for h in range(0, elem_in_array):
            slope=((adjusted_current[j + skip * h] -
                    adjusted_current[j + skip * (h + 1)]) /
                    (adjusted_bias[j + skip * h] -
                    adjusted_bias[j + skip * (h + 1)]))
            slope = u.Quantity(slope, 1/u.V)
            slopes_to_use.append(slope)
        mean = u.Quantity(slopes_to_use).mean().to(1/u.V)
        standard_dev = u.Quantity(slopes_to_use).std()
        tolerance = 0.3 * abs(mean)
        # print('tolerance:', tolerance)
        # print('standard deviation:', standard_dev)
        if standard_dev < tolerance:
            slope = mean
            found_slope = True
        else:
            j += skip
    if not found_slope:
        return None

    return  abs(1/slope).to(u.V).value * u.eV

def get_te_speed(sorted_bias,sorted_current,v_f_index,v_p_index,get_indices = False):
    current_vals = sorted_current[v_f_index: v_p_index]
    bias_vals = sorted_bias[v_f_index: v_p_index]
    mask = current_vals > 0 * u.A
    adjusted_bias = bias_vals[mask]
    adjusted_current = np.log(current_vals[mask].to(u.A).value)
    if len(adjusted_bias) < 2:
        if get_indices:
            return None, None, None
        else:
            return None

    remove_indices = 0.01
    # Check to make sure we have the proper amount of indices to slice
    if 2 * remove_indices * len(sorted_bias) > len(adjusted_bias):
        remove_indices = remove_indices/2
        if 2 * remove_indices * len(sorted_bias) > len(adjusted_bias):
            remove_indices = 0

    indices_to_remove = int(remove_indices * len(sorted_bias))
    adjusted_current = adjusted_current[indices_to_remove: -(indices_to_remove + 1)]
    adjusted_bias = adjusted_bias[indices_to_remove:-(indices_to_remove + 1)]
    slope_array = []
    skip = max(int(0.05 * len(adjusted_bias)), 1)
    if skip == 1:
        if get_indices:
            return None, None, None
        else:
            return None
    index = 0
    while index + skip < len(adjusted_bias):
        slope_num = adjusted_current[index + skip] - adjusted_current[index]
        slope_denom = adjusted_bias[index + skip] - adjusted_bias[index]
        if slope_denom == 0 * u.V:
            index += skip
            continue

        slope = (slope_num / slope_denom)
        slope_array.append(slope)
        index += skip
        # (print('index = ', index))
    if not slope_array:
        if get_indices:
            return None, None, None
        else:
            return None
    mean_slope = u.Quantity(slope_array).mean()
    if mean_slope < 0:
        if get_indices:
            return None, None, None
        else:
            return None

    t_e = (1 / mean_slope).to(u.V).value * u.eV
    if get_indices:
        beginning_index = v_f_index + indices_to_remove
        end_index = v_p_index - indices_to_remove + 1
        return t_e, beginning_index, end_index

    return t_e

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






def dataset_detect_steady_states(ds, ramp_times):

    y_positions = ds["y"].size
    x_positions = ds["x"].size
    box_size = 1
    tolerance = .1
    # Check to make sure we are not looking at more outer squares than are actually available
    if y_positions > 1:
        y_range = list(range(-box_size, box_size + 1))
    else:
        y_range = [0]
    if x_positions > 1:
        x_range = list(range(-box_size, box_size + 1))
    else:
        x_range = [0]
    # Get the mean ion saturation current across all shots for a square of positions around the center
    sweep_means_iisat = []
    for i in y_range:
        for j in x_range:
            # Grab the mean along the shot dimension for the
            sweep_means_iisat.append(ds["ion_isat"].sel(x=j, y=i, method='nearest').isel(probe=0).mean(dim='shot'))

    means_list = []
    # Create a mean of each individual sweep across all elements in the box. Should return a list of floats that give
    # the mean of each sweep for the entire box of positions.
    for k in range(len(sweep_means_iisat[0])):
        sweep_holder = []
        for h in range(len(sweep_means_iisat)):
            sweep_holder.append(sweep_means_iisat[h][k])
        means_list.append(np.mean(sweep_holder))

    start_indices = []
    end_indices = []
    g = 0
    while g < len(means_list) - 1:
        compare_iisat = means_list[g]
        if abs((compare_iisat - means_list[g + 1])/np.mean([compare_iisat,means_list[g + 1]])) < tolerance:
            start_current = compare_iisat
            test_index = g
            l = 2
            while (l + g < len(means_list) and
                    abs((start_current - means_list[g + l])/np.mean([start_current,means_list[g + l]])) < tolerance):
                l += 1
            if l > 2:
                start_indices.append(test_index)
                end_indices.append(g + l - 1)
                g += l
            else:
                g += 1
        else:
            g += 1

    start_times = [ramp_times[idx] for idx in start_indices]
    end_times = [ramp_times[idx] for idx in end_indices]
    return start_indices, end_indices, start_times, end_times



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





def find_steady_state(t_e_data_arrays, n_e_data_arrays, middle_guess):
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



