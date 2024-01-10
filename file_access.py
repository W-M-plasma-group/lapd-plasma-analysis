import os
import warnings

import xarray as xr


def choose_multiple_list(choices, name, null_action=None):
    if len(choices) > 52:
        warnings.warn("More than 52 " + name + "s found. Only the first 52 are displayed.")
    print(*["  " + num_to_chr(i) + ": " + choices[i] for i in range(len(choices[:52]))], sep="\n")
    prompt = "Input a string of letters to select the corresponding " + name + "s"
    if null_action is not None:
        prompt += ", \n\tor the empty string to " + null_action
    prompt += ": "
    selection_str = input(prompt)

    if selection_str == "" and null_action is not None:
        return []
    if not selection_str.isalpha():
        raise ValueError("Selection " + repr(selection_str) + " is not only letters")

    return [chr_to_num(letter) for letter in selection_str]


def chr_to_num(car):
    code = ord(car)
    if 97 <= code <= 122:
        return code - 97
    elif 65 <= code <= 90:
        return code - 65 + 26
    else:
        raise ValueError("Cannot convert char " + repr(car) + " to number")


def num_to_chr(num):
    if 0 <= num <= 25:
        return chr(num + 97)
    elif 26 <= num <= 52:
        return chr(num - 26 + 65)
    else:
        raise ValueError("Cannot convert number " + str(num) + " to char")


# Ensure that the file path contains a Dataset
def check_netcdf(filename):
    try:
        xr.open_dataset(filename)
    except FileNotFoundError:
        return False
    return True


def open_netcdf(filename):
    return xr.open_dataset(filename)


def write_netcdf(dataset, path):
    save_mode = 'a' if check_netcdf(path) else 'w'
    dataset.to_netcdf(path=path, mode=save_mode)


def search_folder(directory, ext, limit=None) -> list[str]:
    r"""Searches the given directory and all subdirectories for files of a desired extension,
    stopping when it reaches 'limit' number of file paths."""
    ext = ext if ext.startswith(".") else "." + ext
    paths_found = []
    for path, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ext):
                paths_found.append(os.path.join(path, filename))
                if type(limit) == int and len(paths_found) >= limit:
                    return paths_found
    return paths_found


def ensure_directory(directory_path: str):
    r"""
    Ensures that the path to the directory of saved NetCDF files is properly formatted
    and creates the directory if it does not currently exist.
    :return:
    """
    head, tail = os.path.split(directory_path)
    if tail != "":
        name, ext = os.path.splitext(tail)
        if ext == "":
            warnings.warn("Adding final '/' to directory path " + repr(directory_path))
            directory_path += "/"
        else:
            raise ValueError("The path " + repr(directory_path) + " is not a directory path")
    if not os.path.isabs(directory_path):
        raise ValueError("Path must be absolute")
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
    return directory_path


def make_path(folder: str, name: str, ext: str) -> str:
    r"""
    Generates an absolute file path from a folder, a filename (not a path), and an extension.
    :return:
    """
    # path, extension = os.path.splitext(name)
    # full_netcdf_path = os.path.join(netcdf_folder, bimaxwellian_filename + ".nc")
    extension = ext if ext.startswith(".") else "." + ext
    return os.path.join(folder, name + extension)
