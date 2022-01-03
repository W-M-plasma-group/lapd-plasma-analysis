import numpy as np
import astropy.units as u

from hdf5reader import *


def setup_lapd(filename):

    print("Setting up HDF5 file...")
    setup_file = open_hdf5(filename)
    # Note: relevant data is accessed using hard-coded indices
    # TODO: access using numpy named fields?
    
    # Gas fill pressure
    gas_pressure = np.array([list(row) for row in get_gas_pressure(setup_file)])
    mean_fill_pressure = np.mean(gas_pressure[..., 4])
    
    # Peak magnetic field 
    magnetic_field = np.array([list(row) for row in get_magnetic_field(setup_file)])
    mean_peak_field = np.mean(magnetic_field[..., 3])

    # Cathode heater discharge current
    discharge = np.array([list(row) for row in get_discharge(setup_file)])
    mean_discharge = np.mean(discharge[..., 4])
    # TODO Future work: plotting the discharge current could give a really helpful
    #     visualization of the plasma heating over time

    # Return experimental parameters, first exact, then rounded, using MATLAB-code-given units
    return ({"Fill pressure": mean_fill_pressure * u.Torr,
             "Peak field": mean_peak_field * u.gauss,
             "Discharge": mean_discharge * u.A},
            {"Fill pressure": np.round(mean_fill_pressure * u.Torr, 7),
             "Peak field": np.round(mean_peak_field * u.gauss),
             "Discharge": np.round(mean_discharge * u.A)})


def get_gas_pressure(file):
    return item_at_path(file, '/MSI/Gas pressure/Gas pressure summary/')


def get_magnetic_field(file):
    return item_at_path(file, '/MSI/Magnetic field/Magnetic field summary/')


def get_discharge(file):
    return item_at_path(file, '/MSI/Discharge/Discharge summary/')
