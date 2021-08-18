import xarray as xr
import os


def read_netcdf(filename):
    if check_netcdf(filename):
        return open_netcdf(filename)
    else:
        netcdf_files = search_netcdf()
        print("The file at the specified location '", filename, "' does not exist. ",
              "The following NetCDF files were found in the current working directory:\n",
              [str(i + 1) + ":" + str(netcdf_files[i]) for i in range(len(netcdf_files))], sep="")
        file_choice = int(input("Type a number to select the corresponding file, "
                                "or 0 to create a new diagnostic dataset from the given HDF5 file path.\n"))
        if file_choice == 0:
            return False
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
