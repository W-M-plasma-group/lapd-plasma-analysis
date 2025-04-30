import astropy.constants as const

from lapd_plasma_analysis.langmuir.helper import *


def get_neutral_density(gas_pressure):

    # TODO We assume that neutrals are evenly distributed throughout LAPD and follow the ideal gas law.
    #   But they may not be evenly spread

    # LAPD parameters from MATLAB code
    neutral_temperature = 300. * u.K
    try:
        neutral_pressure = gas_pressure.to(u.Pa)
        print("tried")
    except: #todo removed u.UnitConversionError
        neutral_pressure = value_safe(gas_pressure)*133.322*u.Pa
        print("excepted")

    print(neutral_pressure)

    # Correction factor for electron density measurements from ion gauge
    correction_factor = 0.18  # from MATLAB

    # Ideal gas law: p = nKT, so n = p / KT / (correction_factor)
    neutral_density = neutral_pressure / (neutral_temperature * const.k_B) / correction_factor
    return neutral_density.to(u.m ** -3)


def get_neutral_ratio(electron_density, steady_state_times, operation="median"):

    # RADIAL/AREAL CODE #
    # _________________ #

    steady_state_electron_density = core_steady_state(electron_density, steady_state_times=steady_state_times,
                                                      operation=operation,
                                                      dims_to_keep=["probe", "face", "x", "y", "shot"])

    if electron_density.sizes['x'] == 1 and electron_density.sizes['y'] == 1:  # scalar (0D) data
        raise ValueError("Finding neutral gas ratio for zero-dimensional position data is unsupported")
    elif electron_density.sizes['x'] == 1 or electron_density.sizes['y'] == 1:  # radial (1D) data
        radial_dimension = 'x' if electron_density.sizes['y'] == 1 else 'y'
        # Interpolate nan values for n_e before averaging to form radial profile?

        linear_electron_density = steady_state_electron_density.squeeze()  # squeeze out length-1 dimension

        # Find average electron density at each absolute radial position (average symmetrically about zero position)
        # Change to xarray concat -> average along new dimension to ensure average of real and nan not divided by 2
        radial_electron_density = (linear_electron_density + linear_electron_density.assign_coords(
            {radial_dimension: -1 * linear_electron_density.coords[radial_dimension]})) / 2

        # debug
        radial_electron_density.where(radial_electron_density.coords[radial_dimension] >= 0, drop=True).plot()
        plt.show()
        #

    elif electron_density.sizes['x'] > 1 and electron_density.sizes['y'] > 1:  # areal (2D) data
        print("Areal neutral analysis is not yet implemented")

    return
