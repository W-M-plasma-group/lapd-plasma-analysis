import os
import warnings

import xarray as xr


def read_netcdf(filename):
    if check_netcdf(filename):
        return open_netcdf(filename)
    else:
        netcdf_files = search_netcdf()
        print("The file at the specified location", repr(filename), "does not exist.",
              "The following NetCDF files were found in the current working directory:")
        print(*(str(i + 1) + ":\t " + str(netcdf_files[i]) for i in range(len(netcdf_files))), sep="\n")
        file_choice = int(input("Type a number to select the corresponding file, "
                                "or 0 to create a new diagnostic dataset from the given HDF5 file path: "))
        if file_choice == 0:
            return None
        dataset = xr.open_dataset(netcdf_files[file_choice - 1])
    return dataset


def check_netcdf(filename):
    try:
        xr.open_dataset(filename)
    except FileNotFoundError:
        return False
    return True


def open_netcdf(filename):
    print("Opening NetCDF dataset file...")
    return xr.open_dataset(filename)


def write_netcdf(dataset, filename):
    print("Saving diagnostic dataset...")
    dataset.to_netcdf(path=filename)


def search_netcdf():
    netcdf_files = []
    for path, dirs, files in os.walk(os.getcwd()):
        for filename in files:
            if filename.endswith(".nc"):
                netcdf_files.append(os.path.join(path, filename))
    return netcdf_files


def ensure_netcdf_directory(directory_path):
    if not os.path.isdir(directory_path):  # directory does not exist and should be created
        if os.path.split(directory_path)[0] != "" and not os.path.isabs(directory_path):
            warnings.warn("The NetCDF subfolder pathname " + repr(directory_path) + " has a leading head path "
                          "and is not an absolute filename. It will be treated as a series of nested directories.")
        os.makedirs(directory_path)  # create the given directory if it does not yet already exist
    return directory_path


def netcdf_path(netcdf_filename, netcdf_subfolder, bimaxwellian):
    # if not diagnostic_filename.endswith(".nc"):
    if not os.path.splitext(netcdf_filename)[1] == ".nc":  # checks if file has .nc extension
        raise ValueError("Name for NetCDF save file " + repr(netcdf_filename) + " should end with a .nc file extension")
    else:  # valid filename
        return os.path.join(netcdf_subfolder, netcdf_filename)
