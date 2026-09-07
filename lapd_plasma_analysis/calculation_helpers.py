'''
This is designed as a helper and catch all for repeated calculations like for sound spedd, etc.
'''

import numpy as np
import plasmapy.particles as particles
import astropy.units as u
import astropy.constants as const
import xarray as xr


def _to_ev_quantity(val, name="temperature"):
    """
    Ensures input is returned as an Astropy Quantity in eV.
    Safely handles Quantities, xarray DataArrays, and raw scalars without double-multiplying units.
    """

    if val is None:
        return np.nan

    # Handle xarray DataArrays
    if isinstance(val, xr.DataArray):
        data = val.data
        if isinstance(data, u.Quantity):
            return data.to(u.eV, equivalencies=u.temperature_energy())
        else:
            return data * u.eV

    # Handle Astropy Quantities
    elif isinstance(val, u.Quantity):
        return val.to(u.eV, equivalencies=u.temperature_energy())

    # Handle raw numbers
    elif isinstance(val, (int, float, np.number, np.ndarray)):
        return val * u.eV

    else:
        raise TypeError(f"{name} must be a DataArray, Astropy Quantity, or numeric scalar.")


def get_ion_mass(ion_type = None, ion_mass = None):
    """
    Parameters
    ----------
ion_type : str, optional
        Ion string compatible with PlasmaPy (e.g., 'H+', 'He-4+').
    ion_mass : astropy.units.Quantity, optional
        Ion mass with units convertible to kg.

    Returns
    -------

    """

    if ion_type is None and ion_mass is None:
        raise ValueError("Please provide either 'ion_type' or 'ion_mass'.")
    elif ion_type is not None:
        m_i = particles.Particle(ion_type).mass.to(u.kg)
    else:
        try:
            m_i = ion_mass.to(u.kg)
        except (AttributeError, u.UnitConversionError):
            raise TypeError("ion_mass must be an Astropy Quantity convertible to kg.")
    return m_i


def sound_speed_calculation(t_e, t_i = None, ion_type = None, ion_mass = None, gamma_i = 0, z = 1):
    """
    Calculates ion sound speed using temperatures in eV based on Hutchinson (1988) Eq. 6:
    c_s = sqrt( e * (Z * T_e + gamma_i * T_i) / m_i)

    Parameters
    ----------
    t_e : xarray.DataArray
        Electron temperature in eV.
    t_i : xarray.DataArray or float, optional
        Ion temperature in eV. Defaults to T_i = T_e if gamma_i > 0 and t_i is None.
    ion_type : str, optional
        Ion string compatible with PlasmaPy (e.g., 'H+', 'He-4+').
    ion_mass : astropy.units.Quantity, optional
        Ion mass with units convertible to kg.
    gamma_i : float, default 0
        Ion adiabatic index (0, 1, 5/3, or 3).
    z : int, default 1
        Electron ionization state / charge state.

    Returns
    -------
    c_s : xarray.DataArray
        Sound speed in m/s with coordinates matching t_e.
    """
    print('gamma_i = ', gamma_i)
    print('z = ', z)
    print('t_i = ', t_i)

    # Make relevant quantities actually quantities
    m_i = get_ion_mass(ion_type, ion_mass)
    m_i = m_i.to(u.kg)

    t_e_q = _to_ev_quantity(t_e, 't_e')

    if t_i is not None:
        t_i_q = _to_ev_quantity(t_i, 't_i')
    elif gamma_i != 0:
        t_i_q = t_e_q
    else:
        t_i_q = 0.0 * u.eV

    # Calculate sound speed natively as a Quantity
    energy_sum = (z * t_e_q) + (gamma_i * t_i_q)
    c_s_quantity = np.sqrt(energy_sum / m_i).to(u.m / u.s)

    # If input was an xarray.DataArray, wrap the Quantity array inside xarray
    if isinstance(t_e, xr.DataArray):
        c_s = xr.DataArray(
            c_s_quantity,
            coords=t_e.coords if hasattr(t_e, 'coords') else None,
            dims=t_e.dims if hasattr(t_e, 'dims') else None,
            attrs=t_e.attrs.copy() if hasattr(t_e, 'attrs') else {}
        )
        c_s.attrs['units'] = str(c_s_quantity.unit)
        c_s.attrs['long_name'] = 'Ion Sound Speed'
        c_s.name = 'c_s'
        return c_s
    return c_s_quantity

# def sound_speed_calculation_t_e(t_e, t_i = None, ion_type = None, ion_mass = None, gamma_i = 0, z = 1):
#     """
#     Calculates ion sound speed using temperatures in eV based on Hutchinson (1988) Eq. 6:
#     c_s = sqrt((Z * T_e + gamma_i * T_i) / m_i)
#
#     Parameters
#     ----------
#     t_e : float, int, or astropy.units.Quantity
#         Electron temperature (assumed eV if unitless).
#     t_i : float, int, or astropy.units.Quantity, optional
#         Ion temperature (assumed eV if unitless). Defaults to T_i = T_e if gamma_i > 0.
#     ion_type : str, optional
#         Ion string compatible with PlasmaPy (e.g., 'H+', 'He-4+').
#     ion_mass : astropy.units.Quantity, optional
#         Ion mass with units convertible to kg
#     gamma_i : float, default 0
#         Ion adiabatic index (0, 1, 5/3, or 3).
#     z : int, default 1
#         Electron ionization state / charge state.
#
#     Returns
#     -------
#     c_s : astropy.units.Quantity
#         Sound speed as an Astropy Quantity with units of m/s.
#     """
#
#     if ion_type is None and ion_mass is None:
#         raise ValueError("Please provide either 'ion_type' or 'ion_mass'.")
#     elif ion_type is not None:
#         m_i = particles.Particle(ion_type).mass.to(u.kg)
#     else:
#         try:
#             m_i = ion_mass.to(u.kg)
#         except (AttributeError, u.UnitConversionError):
#             raise TypeError("ion_mass must be an Astropy Quantity convertible to kg.")
#
#     # Helper function to convert numeric inputs into eV Quantities cleanly
#     def _to_ev_quantity(val, name):
#         if isinstance(val, u.Quantity):
#             return val.to(u.eV, equivalencies=u.temperature_energy())
#         elif isinstance(val, (int, float, np.number)):
#             return val * u.eV
#         else:
#             raise TypeError(f"{name} must be an int, float, or Astropy Quantity.")
#
#
#     t_e_quantity = _to_ev_quantity(t_e, 't_e')
#     if t_i is not None:
#         t_i_quantity = _to_ev_quantity(t_i, 't_i')
#     elif gamma_i != 0:
#         t_i_quantity = t_e_quantity
#     else:
#         t_i_quantity = 0.0 * u.eV
#
#     energy_sum_ev = (z * t_e_quantity) + (gamma_i * t_i_quantity)
#     c_s_w_units = np.sqrt(energy_sum_ev / m_i).to(u.m / u.s)
#
#     return c_s_w_units