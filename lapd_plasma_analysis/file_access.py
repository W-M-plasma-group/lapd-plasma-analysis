"""
Provide functions to access files, to load NetCDF files, and to allow user selection of options from a list.
"""

import os
import warnings
import xarray as xr


def choose_multiple_from_list(choices, name, null_action=None):
    r"""Allows user to choose multiple items from a list via console input.

    Prompts user to select any object (described by `name`) from a list `choices`
    of such objects. This function can handle 52 choices, since each choice corresponds
    to an upper-or-lower-case letter. For example, inputting the string `'abc'` would
    select the first, second, and third options in `choices`.

    Parameters
    __________
    choices : `list` of `str`
        List of options that may be selected.
    name : `str`
        A name for the thing being chosen (e.g. 'HDF5 file' if `choices` is a list
        of HDF5 file paths).

    null_action : `str`, optional
        Parameter used to convey to the user the consequence of providing no
        input when prompted (e.g. `'skip to Mach probe analysis'`).

    Returns
    _______
    `list`
        A list of `int`, the indices of the selected items in `choices`. This list
        is empty if no items are selected.

    Raises
    ______

    `ValueError`
        Gives an error if the user inputs a string containing symbols other than
        letters.

    """

    if len(choices) > 52:
        warnings.warn("More than 52 " + name + "s found. Only the first 52 are displayed.")
    print(*["  " + num_to_chr(i) + ": " + choices[i] for i in range(len(choices[:52]))], sep="\n")
    prompt = "Input a string of letters to select the corresponding " + name + "s (e.g. 'abc')"
    if null_action is not None:
        prompt += ", \n\tor the empty string to " + null_action
    prompt += ": "
    selection_str = input(prompt)

    if selection_str == "" and null_action is not None:
        return []
    if not selection_str.isalpha():
        raise ValueError("Selection " + repr(selection_str) + " is not only letters")

    return [chr_to_num(letter) for letter in selection_str]


def ask_yes_or_no(prompt):
    """Prompts the user to answer a yes-or-no question.

    Asks user to input 'y' or 'n' in response to a prompt, then returns the corresponding boolean value
    (`True` if `'y'`, `False` if `'n'`). Repeats prompt until given valid input.

    Parameters
    ----------
    prompt : `str`
        Message to user posed as a yes-or-no question. Consider ending with ' (y/n) ',
        including spaces.

    Returns
    -------
    `bool`
        True or False value representing user's choice of 'y' or 'n'.

    """

    response = ""
    while response not in ("y", "n"):
        response = input(prompt).lower()
    return response == "y"


def chr_to_num(car):
    r"""Converts a letter to an integer.

    Auxiliary function to `lapd_plasma_analysis.file_access.choose_multiple_list`,
    used to reconvert the user's letter input to an integer, based off the letter's
    position in the alphabet.

    Parameters
    ----------
    car : `str`
        A letter in the alphabet

    Returns
    -------
    `int`
        The integer corresponding to the letter's position in the alphabet.

    Raises
    ______
    `ValueError`
        If the provided string is not a letter in the alphabet

    """
    code = ord(car)
    if 97 <= code <= 122:
        return code - 97
    elif 65 <= code <= 90:
        return code - 65 + 26
    else:
        raise ValueError("Cannot convert char " + repr(car) + " to number")


def num_to_chr(num):
    r"""Converts an integer to a letter in the alphabet.

    Auxiliary function to `lapd_plasma_analysis.file_access.choose_multiple_list`,
    used to convert an index of a list (`int`) to a letter in the alphabet, for
    user selection.

    Parameters
    ----------
    num : `int`
        The integer to be converted

    Returns
    -------
    `str`
        The corresponding letter in the alphabet

    Raises
    ______
    `ValueError`
        If the provided number is outside the range of allowable values. Ensure
        0 <= `num` <= 52 and that `num` is an `int`.

    """
    if 0 <= num <= 25:
        return chr(num + 97)
    elif 26 <= num <= 52:
        return chr(num - 26 + 65)
    else:
        raise ValueError("Cannot convert number " + str(num) + " to char")


def check_netcdf(file_path):
    """
    Checks if the given path leads to a valid NetCDF file.

    Parameters
    ----------
    file_path : `str`
        Path to a file. This function checks if this file can be opened as a NetCDF file.
        (This should end with `'.nc'` )

    Returns
    -------
    `bool`
        `True` or `False` value indicating if `file_path` indicates a usable NetCDF file.
        `False` only if attempt to access file yields `FileNotFoundError`.

    """
    try:
        xr.open_dataset(file_path)
    except FileNotFoundError:
        return False
    return True


def open_netcdf(filename):
    return xr.open_dataset(filename)


def write_netcdf(dataset, path):
    save_mode = 'a' if check_netcdf(path) else 'w'
    dataset.to_netcdf(path=path, mode=save_mode)


def search_folder(directory, ext, limit=None) -> list[str]:
    r"""Searches the given directory for files matching given extension.

    Stops when the number of collected files reaches `limit` if `limit != None`. It
    otherwise continues until every file in the provided directory has been checked.
    In many cases, `limit` should be <= 52 since this function works in conjunction
    with `lapd_plasma_analysis.file_access.choose_multiple_list`, which handles a
    maximum of 52 choices.

    Parameters
    ----------
    directory : `str`
        Directory in which to search for files of extension `ext`. Should end with `'/'`

    ext : `str`
        The file extension to search for. (e.g. `'nc'` if looking for NetCDF files)

    limit : `int`
        The maximum number of files to return. If `None` (default), all files that match
        the provided extension are returned.

    Returns
    -------
    `list`
        A list of file paths (`str`) matching the given extension.

    """
    ext = ext if ext.startswith(".") else "." + ext
    paths_found = []
    for path, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ext):
                paths_found.append(os.path.join(path, filename))
                if isinstance(limit, int) and len(paths_found) >= limit:
                    return paths_found
    return paths_found


def ensure_directory(directory_path: str):
    """
    Ensures that the path to the directory of saved NetCDF files is properly formatted
    and creates the directory if it does not currently exist.
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


def make_path(folder, name, ext):
    """
    Generates an absolute file path from a folder, a filename (not a path), and an extension.

    Parameters
    ----------
    folder : `str`
        Path indicating parent folder of file.
    name : `str`
        Name of file, with no extension.
    ext : `str`
        Extension of file, e.g. ".pdf"; leading period is optional.
    """
    # path, extension = os.path.splitext(name)
    # full_netcdf_path = os.path.join(netcdf_folder, bimaxwellian_filename + ".nc")
    extension = ext if ext.startswith(".") else "." + ext
    return os.path.join(folder, name + extension)
