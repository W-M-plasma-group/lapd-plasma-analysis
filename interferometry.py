import numpy as np
import astropy.units as u

from hdf5reader import *


def interferometry_calibration(density_xarray, temperature_xarray, interferometry_filename, bias, current,
                               steady_state_start_plateau, steady_state_end_plateau):

    interferometry_file = open_hdf5(interferometry_filename)
    interferometry_data_raw = item_at_path(interferometry_file,
                                           '/MSI/Interferometer array/Interferometer [0]/Interferometer trace/')
    interferometry_data_array = np.array(interferometry_data_raw)

    print("Interferometry data shape:", interferometry_data_array.shape)

    interferometry_means_abstract = np.mean(interferometry_data_array, axis=0)
    interferometry_time_abstract = np.arange(len(interferometry_means_abstract))

    interferometry_data, interferometry_time = to_real_units_interferometry(interferometry_means_abstract,
                                                                            interferometry_time_abstract)

    # debug
    # print(interferometry_data, "\n", interferometry_time)
    #

    return None, None


def to_real_units_interferometry(interferometry_data_array, interferometry_time_array):
    area_factor = 8. * 10 ** 13 / (u.cm ** 2)           # from MATLAB code
    time_factor = ((4.88 * 10 ** -5) * u.s).to(u.ms)    # from MATLAB code
    return interferometry_data_array * area_factor, interferometry_time_array * time_factor
