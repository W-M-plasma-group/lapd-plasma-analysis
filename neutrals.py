import astropy.units as u
import xarray as xr


def plasma_neutrals(electron_density, COMPLETE_THIS, experimental_parameters, steady_state_start, steady_state_end):
    # SETUP CODE #
    # __________ #

    # LAPD parameters from MATLAB code
    neutral_temperature = 300. * u.K
    neutral_pressure = experimental_parameters['Fill pressure'].to(u.Pa)
    length = 16.5 * u.m

    # Correction factor for electron density measurements from ion gauge
    correction_factor = 0.18  # from MATLAB

    neutral_density = neutral_pressure / (neutral_temperature * u.k) / correction_factor  # u.k is Boltzmann constant

    # RADIAL/AREAL CODE #
    # _________________ #

    both, plateau = xr.ufuncs.logical_and, electron_density.coords['plateau']  # rename variables for comprehensibility

    steady_state_mask = both(plateau >= steady_state_start, plateau <= steady_state_end)
    steady_state_electron_density = electron_density.where(steady_state_mask, drop=True).mean('plateau')

    if electron_density.sizes['x'] == 1 and electron_density.sizes['y'] == 1:  # scalar (0D) data
        raise ValueError("Finding neutral gas ratio for zero-dimensional position data is unsupported")
    elif electron_density.sizes['x'] == 1 or electron_density.sizes['y'] == 1:  # radial (1D) data
        radial_dimension = 'x' if electron_density.sizes['y'] == 1 else 'y'
        # Interpolate nan values for n_e before averaging to form radial profile?

        linear_electron_density = steady_state_electron_density.squeeze()  # squeeze out length-1 dimension

        # Find average electron density at each absolute radial position (average symmetrically about zero position)
        radial_electron_density = (linear_electron_density + linear_electron_density.assign_coords(
            {radial_dimension: -1 * linear_electron_density.coords[radial_dimension]})) / 2

        # debug
        radial_electron_density.where(radial_electron_density.coords[radial_dimension] >= 0, drop=True).plot()
        #

    elif electron_density.sizes['x'] > 1 and electron_density.sizes['y'] > 1:  # areal (2D) data
        pass  # not yet implemented

    return
