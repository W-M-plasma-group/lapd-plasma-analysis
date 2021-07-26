import xarray as xr
import os


def read_netcdf(filename):
    return xr.open_dataset(filename)


def write_netcdf(dataset, filename):
    dataset.to_netcdf(path=filename)


def search_netcdf():
    netcdf_files = []
    for path, dirs, files in os.walk(os.getcwd()):
        for filename in files:
            if filename.endswith(".nc"):
                netcdf_files.append(os.path.join(path, filename))
    return netcdf_files
