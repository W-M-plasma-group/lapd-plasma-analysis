import xarray as xr
import os


def read_netcdf(filename):
    try:
        dataset = xr.open_dataset(filename)
    except FileNotFoundError:
        netcdf_files = search_netcdf()
        print("The file at the specified location '", filename, "' does not exist.",
              "The following NetCDF files were found in the current working directory:\n",
              [str(i + 1) + ":" + str(netcdf_files[i]) for i in range(len(netcdf_files))])
        file_choice = int(input("Type a number to select the corresponding file.")) - 1
        dataset = xr.open_dataset(netcdf_files[file_choice])
    return dataset


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
