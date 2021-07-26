import numpy as np
import astropy.units as u

from hdf5reader import *


def interferometry_calibration(density_xarray, temperature_xarray, interferometry_filename, bias, current,
                               steady_state_start_plateau, steady_state_end_plateau):

    interferometry_file = open_hdf5(interferometry_filename)
    interferometry_data_raw = item_at_path(interferometry_file,
                                           '/MSI/Interferometer array/Interferometer [0]/Interferometer trace/')
    interferometry_data_array = np.array(interferometry_data_raw)
    interferometry_data = to_real_units_interferometry(interferometry_data_array)

    # debug
    print("Interferometry data shape:", interferometry_data.shape)
    print(interferometry_data)
    #

    return None, None


def to_real_units_interferometry(interferometry_data_array):
    scale_factor = 8. * 10 ** 13 / (u.cm ** 2)
    return interferometry_data_array * scale_factor
