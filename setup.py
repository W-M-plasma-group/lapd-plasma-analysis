import numpy as np

from hdf5reader import *


def setup_lapd(filename):

    print("Setting up HDF5 file...")
    setup_file = open_hdf5(filename)

    # print(list(setup_file['/MSI/Gas pressure/'].attrs.items()))
    gas_pressure = np.array([list(row) for row in get_gas_pressure(setup_file)])
    # print(gas_pressure.shape)
    mean_fill_pressure = np.mean(gas_pressure[..., 4])

    magnetic_field = get_magnetic_field(setup_file)
    print(magnetic_field[:])
    return mean_fill_pressure


def get_gas_pressure(file):
    return item_at_path(file, '/MSI/Gas pressure/Gas pressure summary/')


def get_magnetic_field(file):
    return item_at_path(file, '/MSI/Magnetic field/Magnetic field summary/')