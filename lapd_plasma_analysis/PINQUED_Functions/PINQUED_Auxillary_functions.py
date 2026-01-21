import numpy as np
from astropy import units as u
from astropy import constants as c
from plasmapy.particles import *


def p_get_floating_potential(sorted_bias,sorted_current):
    """

    Parameters
    ----------
    sorted_bias - Array of bias values in V sorted from minimum to maximum
    sorted_current - Array of current values in A sorted by matched bias values from the minimum to
    maximum of the bias array

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

def p_get_ion_isat_min(sorted_current,sorted_bias):
    """

    Parameters
    ----------
    sorted_current - Array of current values in A sorted by matched bias values from the minimum to
    maximum of the bias array

    Returns
    -------
    ion_isat - Quantiry (A): value of ion saturation current in Amps
    ion_isat_index - Int: location of the minimum current within the sorted current and sorted bias array

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

def get_ion_density(ion_type,ion_isat,A_p,T_e):
    """
    Parameters
    ----------
    ion_type - (String) - Indicates the ion you are trying to find the density of.
    each ion corresponds to an ion in plasmapy.particles
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