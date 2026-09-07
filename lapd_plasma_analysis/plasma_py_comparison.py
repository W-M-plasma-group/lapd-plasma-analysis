# All of these functions are pulled directly from the plasmapy langmuir analysis and adapted to fit the format the data
# is put in main_luke

import astropy.units as u
import copy
import numpy as np


from astropy.constants import si as const
from astropy.visualization import quantity_support
from scipy.optimize import curve_fit
from warnings import warn


from plasmapy.particles import Particle
from plasmapy.utils.decorators import validate_quantities

def _pp_langmuir_futurewarning() -> None:
   warn(
       "The plasmapy.diagnostics.langmuir module will be deprecated in favor of "
       "the plasmapy.analysis.swept_langmuir sub-package and phased out over "
       "2021.  The plasmapy.analysis package was released in v0.5.0.",
       FutureWarning,
   )

def _pp_fit_func_lin_inverse(x, x0, y0, T0):
   r"""Linear fitting function with inverse slope parameter for use in fitting
   of the electron current growth region.
   """


   return y0 + (x - x0) / T0

def _pp_fit_func_double_lin_inverse(x, x0, y0, T0, Delta_T):
   r"""Piecewise linear fitting function with inverse slope parameters and
   an offset for use in fitting a bi-Maxwellian electron current growth
   region. (x0, y0) denotes the location of the knee of the transition,
   with T0 and T0 + Delta_T being the cold and hot temperatures, respectively.
   """


   def hot_T_func(x):
       return y0 + (x - x0) / (T0 + Delta_T)


   def cold_T_func(x):
       return y0 + (x - x0) / T0


   return np.piecewise(x, x < x0, [hot_T_func, cold_T_func])

def pp_get_plasma_potential(sorted_bias,sorted_current, return_arg=False):
   r"""Implement the simplest but crudest method for obtaining an estimate of
   the plasma potential from the probe characteristic.


   Parameters
   ----------
   sorted_bias : array of bias values sorted in order of ascending bias value
   sorted_current : array of current values sorted in order of ascending corresponding bias value


   return_arg : `bool`, optional
       Controls whether or not the argument of the plasma potential within the
       characteristic array should be returned instead of the value of the
       voltage. Default is False.


   Returns
   -------
   V_P : `~astropy.units.Quantity`
       Estimate of the plasma potential in units convertible to V.


   Notes
   -----
   The method used in the function takes the maximum gradient of the probe
   current as the 'knee' of the transition from exponential increase into the
   electron the saturation region.


   """


   # _pp_langmuir_futurewarning()


   # if not isinstance(probe_characteristic, Characteristic):
   #     raise TypeError(
   #         "For 'probe_characteristic' expected type "
   #         f"{Characteristic.__module__}.{Characteristic.__qualname__} "
   #         f"and got {type(probe_characteristic)}."
   #     )


   # Sort the characteristic prior to differentiation
   """
   probe_characteristic.sort()


   # Acquiring first derivative
   dIdV = np.gradient(
       probe_characteristic.current.to(u.A).value,
       probe_characteristic.bias.to(u.V).value,
   )


   arg_V_P = np.argmax(dIdV)
   """
   # Below: Leo replacement of the above
   # Not robust because bias is not equally spaced; does not calculate region of true steepest gradient
   current = sorted_current.to(u.A)
   bias = sorted_bias.to(u.V)
   counts, bins = np.histogram(current.value)
   min_bin_index = np.argmin(counts)   # counts[1:-1]
   arg_V_P = np.where(current.value > (bins[min_bin_index] + bins[min_bin_index + 1]) / 2)[0][0]
   # End Leo replacement


   if return_arg:
       return bias[arg_V_P], arg_V_P
   return bias[arg_V_P]


def pp_get_floating_potential(sorted_bias, sorted_current, return_arg=False):
   r"""Implement the simplest but crudest method for obtaining an estimate of
   the floating potential from the probe characteristic.


   Parameters
   ----------
   sorted_bias : array of bias values sorted in order of ascending bias value
   sorted_current : array of current values sorted in order of ascending corresponding bias value


   return_arg : `bool`, optional
       Controls whether or not the argument of the floating potential within
       the characteristic array should be returned instead of the value of the
       voltage. Default is False.


   Returns
   -------
   V_F : `~astropy.units.Quantity`
       Estimate of the floating potential in units convertible to V.


   Notes
   -----
   The method used in this function takes the probe current closest to zero
   Amperes as the floating potential.


   """


   _pp_langmuir_futurewarning()


   # if not isinstance(probe_characteristic, Characteristic):
   #     raise TypeError(
   #         "For 'probe_characteristic' expected type "
   #         f"{Characteristic.__module__}.{Characteristic.__qualname__} "
   #         f"and got {type(probe_characteristic)}"
   #     )


   try:
       arg_V_F = np.nonzero(sorted_current < 0)[0][-1]  # Leo version
   except IndexError:
       arg_V_F = np.argmin(np.abs(sorted_current))  # original


   if return_arg:
       return sorted_bias[arg_V_F], arg_V_F


   return sorted_bias[arg_V_F]

def pp_extract_exponential_section(sorted_bias, sorted_current, T_e=None, ion_current=None):
   r"""Extract the section of exponential electron current growth from the
   probe characteristic.


   Parameters
   ----------
   sorted_bias : array of bias values sorted in order of ascending bias value
   sorted_current : array of current values sorted in order of ascending corresponding bias value


   T_e : `~astropy.units.Quantity`, optional
       If given, the electron temperature can improve the accuracy of the
       bounds of the exponential region.


   ion_current : `~plasmapy.diagnostics.langmuir.Characteristic`, optional
       If given, the ion current will be subtracted from the probe
       characteristic to yield a better estimate of the electron current in
       the exponential region.


   Returns
   -------
   exponential_section : `~plasmapy.diagnostics.langmuir.Characteristic`
       The exponential electron current growth section.


   Notes
   -----
   This function extracts the region of exponential electron growth from the
   probe characteristic under the assumption that this bias region is bounded
   by the floating and plasma potentials. Additionally, an improvement in
   accuracy can be made when the electron temperature is supplied.
   """


   _pp_langmuir_futurewarning()


   # if not isinstance(probe_characteristic, Characteristic):
   #     raise TypeError(
   #         "For 'probe_characteristic' expected type "
   #         f"{Characteristic.__module__}.{Characteristic.__qualname__} "
   #         f"and got {type(probe_characteristic)}."
   #     )


   V_F = pp_get_floating_potential(sorted_bias, sorted_current)


   V_P = pp_get_plasma_potential(sorted_bias, sorted_current)


   # LEO DEBUG
   """
   print(f"V_F: {V_F}, V_P: {V_P}")
   # """
   # LEO


   if T_e is not None:
       # If a bi-Maxwellian electron temperature is supplied grab the first
       # (cold) temperature
       if np.array(T_e).size > 1:
           T_e = np.min(T_e)


       _filter = (sorted_bias > V_F + 0.2 * 1.5 * T_e / const.e) & (  # LEO MODIFICATION
           sorted_bias < V_P - 0.2 * T_e / const.e
       )
   else:
       _filter = (sorted_bias > V_F) & (sorted_bias < V_P)


   exponential_section_b = sorted_bias[_filter]
   exponential_section_c = sorted_current[_filter]


   # if ion_current is not None:
   #     exponential_section = exponential_section - ion_current[_filter]


   return exponential_section_b, exponential_section_c

def pp_get_electron_temperature(
   exponential_section_b, exponential_section_c,
   bimaxwellian=False,
   visualize=False,
   return_fit=False,
   return_hot_fraction=False,
):
   r"""Obtain the Maxwellian or bi-Maxwellian electron temperature using the
   exponential fit method.


   Parameters
   ----------
   exponential_section_b : Bias values sorted in ascending order filtered to only include the
                            exponential region of the data.
   exponential_section_c : Current values sorted by corresponding bias in ascending order filtered to only include the
                            exponential region of the data.
exponential_sec


   bimaxwellian : `bool`, optional
       If `True` the exponential section will be fit assuming bi-Maxwellian
       electron populations, as opposed to Maxwellian. Default is False.


   visualize : `bool`, optional
       If `True` a plot of the exponential fit is shown. Default is `False`.


   return_fit: `bool`, optional
       If `True` the parameters of the fit will be returned in addition to the
       electron temperature. Default is `False`.


   return_hot_fraction: float, optional
       If `True` the total fraction of hot electrons will be returned if the
       population is bi-Maxwellian. Default is `False`.


   Returns
   -------
   T_e : `~astropy.units.Quantity`, (ndarray)
       The estimated electron temperature in eV. In case of a bi-Maxwellian
       plasma an array containing two Quantities is returned.


   Notes
   -----
   In the electron growth region of the probe characteristic the electron
   current grows exponentially with bias voltage:


   .. math::
       I_e = I_{es} \textrm{exp} \left(
       -\frac{e\left(V_P - V \right)}{T_e} \right).


   In log space the current in this region should be a straight line if the
   plasma electrons are fully Maxwellian, or exhibit a knee in a
   bi-Maxwellian case. The slope is inversely proportional to the
   temperature of the respective electron population:


   .. math::
       \textrm{log} \left(I_e \right ) \propto \frac{1}{T_e}.


   """


   _pp_langmuir_futurewarning()


   # if not isinstance(exponential_section, Characteristic):
   #     raise TypeError(
   #         "For 'probe_characteristic' expected type "
   #         f"{Characteristic.__module__}.{Characteristic.__qualname__} "
   #         f"and got {type(exponential_section)}."
   #     )


   # Remove values in the section with a current equal to or smaller than
   # zero.
   exponential_section_mask = exponential_section_c.to(u.A).value > 0
   exponential_section_c = exponential_section_c[exponential_section_mask]
   exponential_section_b = exponential_section_b[exponential_section_mask]


   initial_guess = None  # for fitting


   bounds = (-np.inf, np.inf)


   # Instantiate the correct fitting equation, initial values and bounds.
   if bimaxwellian:
       max_exp_bias = np.max(exponential_section_b)
       min_exp_bias = np.min(exponential_section_b)
       x0 = min_exp_bias + 2 / 3 * (max_exp_bias - min_exp_bias)


       initial_guess = [x0.to(u.V).value, 0.6, 2, 1]


       bounds = ([-np.inf, -np.inf, 0, 0], np.inf)


       fit_func = _pp_fit_func_double_lin_inverse
   else:
       fit_func = _pp_fit_func_lin_inverse


   # Perform the actual fit of the data
   fit, covariance_matrix = curve_fit(     # LEO REVISION; used to be fit, _ = curve_fit(
       fit_func,
       exponential_section_b.to(u.V).value,
       np.log(exponential_section_c.to(u.A).value),
       p0=initial_guess,
       bounds=bounds,
   )


   # LEO ADDITION
   # print(f"T_e est. = {fit[2]:.1e}, inv slope error = {np.sqrt(covariance_matrix[2][2])}")
   # END LEO ADDITION


   hot_fraction = None


   # Obtain the plasma parameters from the fit
   if not bimaxwellian:
       T0 = fit[2]


       T_e = T0 * u.eV
   else:
       x0, y0 = fit[0], fit[1]
       T0, Delta_T = [fit[2], fit[3]]


       # In order to obtain the energetic electron fraction the fits of the
       # cold and hot populations are extrapolated to the plasma potential
       # (ie. the maximum bias of the exponential section). The logarithmic
       # difference between these currents equates to the density difference.


       k1 = _pp_fit_func_lin_inverse(
           np.max(exponential_section_b.to(u.V).value), *[x0, y0, T0]
       )


       k2 = _pp_fit_func_lin_inverse(
           np.max(exponential_section_b.to(u.V).value), *[x0, y0, T0 + Delta_T]
       )


       # Compute the total hot (energetic) fraction
       hot_fraction = 1 / (1 + np.exp(k1 - k2))


       # If bi-Maxwellian, return main temperature first
       T_e = np.array([T0, T0 + Delta_T]) * u.eV


   if visualize:  # coverage: ignore
       import matplotlib.pyplot as plt


       with quantity_support():
           plt.figure()


           plt.scatter(
               exponential_section_b.to(u.V),
               np.log(exponential_section_c.to(u.A).value),
               color="k",
               marker=".",
               label="Exponential section",
           )


           if bimaxwellian:
               plt.scatter(x0, y0, marker="o", c="g")
               plt.plot(
                   exponential_section_b.to(u.V),
                   _pp_fit_func_lin_inverse(
                       exponential_section_b.to(u.V).value,
                       fit[0],
                       fit[1],
                       fit[2] + fit[3],
                   ),
                   c="g",
                   linestyle="--",
                   label="Bimaxwellian exponential section fit",
               )


           plt.plot(
               exponential_section_b.to(u.V),
               fit_func(exponential_section_b.to(u.V).value, *fit),
               label="Exponential fit",
               c="g",
           )


           plt.ylabel("Logarithmic current")
           plt.title("Exponential fit")
           plt.legend(loc="best")
           plt.tight_layout()


   # LEO ADDITION  # TODO extremely hardcoded!
   # print(np.sqrt(covariance_matrix[2][2]))
   # print(np.min([T0, 10]))
   temperature_problem = (np.sqrt(covariance_matrix[2][2]) / np.min([T0, 10]) > 0.05)
   # END LEO ADDITION


   k = [T_e]


   if return_hot_fraction:
       k.append(hot_fraction)


   if return_fit:
       k.append(fit)


   return k, temperature_problem

def pp_get_ion_isat(sorted_current):
    """

    Parameters
    ----------
    sorted_current- Array of current Quantities in A sorted by their corresponding bias values

    Returns
    -------
    I_isat- Quantity value of the Ion saturation current in A

    """

    I_isat = np.min(sorted_current)
    return I_isat

def pp_get_electron_isat(sorted_bias,sorted_current):
    """

    Parameters
    ----------
    sorted_bias - Array of Bias quantities in V sorted from minimum to maximum
    sorted_current - Array of current Quantities in A sorted by their corresponding bias values

    Returns
    -------
    E_isat - Quantity value of the Electron saturation current in A

    """
    _, arg_v_p = pp_get_plasma_potential(sorted_bias, sorted_current, return_arg=True)

    return sorted_current[arg_v_p]
