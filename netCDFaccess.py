import os
import warnings

import xarray as xr


# Return a dataset opened from a NetCDF file, or None if no file was found
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


# Ensure that the file path contains an xarray Dataset
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
    save_mode = 'a' if check_netcdf(filename) else 'w'
    dataset.to_netcdf(path=filename, mode=save_mode)


# Search the current working directory and all subfolders for .nc (NetCDF) files
def search_netcdf():
    netcdf_files = []
    for path, dirs, files in os.walk(os.getcwd()):
        for filename in files:
            if filename.endswith(".nc"):
                netcdf_files.append(os.path.join(path, filename))
    return netcdf_files


# Check that the specified NetCDF subdirectory exists, and if not, create it
def ensure_netcdf_directory(directory_path):
    if not os.path.isdir(directory_path):  # directory does not exist and should be created
        if os.path.split(directory_path)[0] != "" and not os.path.isabs(directory_path):
            warnings.warn("The NetCDF subfolder pathname " + repr(directory_path) + " has a leading head path "
                          "and is not an absolute filename. It will be treated as a series of nested directories.")
        os.makedirs(directory_path)  # create the given directory if it does not yet already exist
    return directory_path


# Generate the netcdf file path to save/open diagnostic data
def netcdf_path(netcdf_name, netcdf_subfolder, bimaxwellian):
    name, extension = os.path.splitext(netcdf_name)
    # TODO remove
    # if extension != "":  # checks if netcdf name has an extension
    #      warnings.warn("Name for NetCDF save file " + repr(netcdf_name) + " should not end with a file extension "
    #                    "(file extension will be added automatically). Trimming extension from filename.")
    # full_netcdf_filename = name + ("_bimaxwellian" if bimaxwellian else "") + ".nc"
    bimaxwellian_filename = (name + "_bimax") if bimaxwellian else name
    full_netcdf_path = os.path.join(netcdf_subfolder, bimaxwellian_filename + ".nc")
    # print("Diagnostic dataset filename:", repr(full_netcdf_path))
    return full_netcdf_path
