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
    print(*["  " + num_to_chr(i) + ": " + str(choices[i]) for i in range(len(choices[:52]))], sep="\n")
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

def int_choose_multiple_from_list(choices, name, null_action=None, return_idxs = False, lim_length = None):
    prompt = "Input a list of integers corresponding to " + name + ("s  \n Note: they must be separated by commas "
                                                                            "(e.g. '1,2,3,4,5,etc.)'")

    print(*["  " + str(i) + ": " + str(choices[i]) for i in range(len(choices))], sep="\n")
    if null_action is not None:
        prompt += ", \n\tor two empty strings in a row to " + null_action
    prompt += ": "
    proper_input = False
    num_empty = 0
    selected_options_idxs = []
    loop_i = 0

    while not proper_input:
        bad_selections = 0

        if loop_i > 0:
            print(*["  " + str(i) + ": " + str(choices[i]) for i in range(len(choices))], sep="\n")
            print("Selected options: ", selected_options_idxs)
            remove_indices = ask_yes_or_no('Remove any indices from selection? (y/n) ')
        else:
            remove_indices = False

        if remove_indices:
            idxs_to_remove = input('Input the index of the option you want to remove '
                                   '(Any invalid entries will reset the loop, must be comma separated): ')
            try:
                idxs_to_remove = idxs_to_remove.split(',')
                for idx in idxs_to_remove:
                    idx = idx.strip()
                    try:
                        idx = int(idx)
                        if idx not in selected_options_idxs:
                            print(f'{idx} is not in the selected options and thus will not be removed.')
                        else:
                            selected_options_idxs.remove(idx)
                    except ValueError:
                        print(f'{idx} is not an integer in the selected options and thus will not be removed.')

            except ValueError:
                print('Invalid Input: No options removed')

        if loop_i > 0:
            print(*["  " + str(i) + ": " + str(choices[i]) for i in range(len(choices))], sep="\n")
        selection_str = input(prompt)

        if selection_str.strip() == "":
            num_empty += 1
            print('\n One empty string selected, exit the loop by inputting another empty string.')

        elif selection_str.strip() == "" and num_empty > 0:
            print('\n You have decided to exit the loop by inputting another empty string. Returning the empty list.')
            return []

        elif selection_str.strip() != "" and num_empty > 0:
            num_empty = 0

        try:
            selections = selection_str.split(',')
            if lim_length is not None:
                if len(selections) > lim_length:
                    print(f'\n Too many options selected, you are restricted to {lim_length} selectons')

            for selection in selections:
                selection = selection.strip()
                try:
                    i_selection = int(selection)
                    if i_selection < 0 or i_selection > len(choices) - 1:
                        print(
                            f'\n {selection} is not an integer between 0 and {len(choices) - 1} corresponding to a valid '
                            f'option. Please try again.')
                        bad_selections += 1
                    else:
                        if i_selection not in selected_options_idxs:
                            selected_options_idxs.append(i_selection)

                except ValueError:
                    print(f'\n {selection} is not a valid integer')
                    bad_selections += 1

            if len(selected_options_idxs) > 0 and bad_selections == 0:
                break
            loop_i += 1

        except ValueError:
            print("Please enter a comma-separated list of integers.")



    selected_options = []
    for idx in selected_options_idxs:
        choice = choices[idx]
        selected_options.append(choice)
        print(f"\n {choice}")


    if return_idxs:
        return selected_options_idxs
    else:
        return selected_options


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

def default_fig_params():
    default_fig_height = 4.8
    default_fig_width = 6.4
    return default_fig_height, default_fig_width

def xarray_gradient_strings():
    dens_grad_regions_str = 'dens_grad_regions'
    dens_grad_slopes_str = 'dens_grad_slopes'
    dens_grad_intercepts_str = 'dens_grad_intercepts'
    temp_grad_regions_str = 'temp_grad_regions'
    temp_grad_slopes_str = 'temp_grad_slopes'
    temp_grad_intercepts_str = 'temp_grad_intercepts'
    return (dens_grad_regions_str, dens_grad_slopes_str, dens_grad_intercepts_str,
            temp_grad_regions_str, temp_grad_slopes_str, temp_grad_intercepts_str)

def allow_only_ints(prompt, min_condition = None, max_condition = None, accept_empty = True):
    '''

    Parameters
    ----------
    prompt: 'str'
        String indicating what the integers being used should be referring to.
    min_condition: `int`, optional
        Minimum integer value to allow.
    max_condition: `int`, optional
        Maximum integer value to allow.
    accept_empty: `bool`, optional
        If True, returns an empty list if the user presses Enter.
        If False, forces the user to input at least one valid integer.

    Returns
    -------
    int_list: `list`
        List of integers indicating which integers to allow.
    '''


    while True:
        skip_text = "(or press Enter to skip)" if accept_empty else "(cannot be blank)"
        user_input = input(
            f"{prompt}, separated by commas {skip_text}. \n"
            f"Between min value {min_condition: .1f} and max value {max_condition: .1f}: \n")

        # Success Condition 1: User just presses Enter
        if not user_input.strip():
            if accept_empty:
                int_list = []
                break  # Exits the while loop entirely, returning []
            else:
                print("Error: This value cannot be left blank. Please enter at least one integer.")
                continue  # Jumps back to the top of the while loop

        # Success Condition 2: User enters valid integers
        try:
            # Attempt to split and convert to integers
            int_list = [int(val.strip()) for val in user_input.split(',')]

            if min_condition is not None and any(val < min_condition for val in int_list):
                print(f"Error: All values must be greater than or equal to {min_condition}.")
                continue  # Skips the rest of the loop and asks again


            if max_condition is not None and any(val > max_condition for val in int_list):
                print(f"Error: All values must be less than or equal to {max_condition}.")
                continue  # Skips the rest of the loop and asks again
            break  # If successful, exit the while loop

        # Failure Condition: User enters letters, decimals, or gibberish
        except ValueError:
            # Print an error message. Because there is no 'break' here,
            # the loop jumps back up to the 'input()' prompt.
            print(
                "Error: Invalid input. Please enter ONLY integers separated by commas, or press Enter to skip.")

    return int_list


def get_hdf5_filename(exp_name, run_number, file_list):
    # Format the run number to always be two digits (e.g., 5 becomes "05")
    run_str = str(run_number)

    for filename in file_list:

        # 1. March 2022: Starts with "Mar22_" + the run number
        if exp_name == "March_2022" and filename.startswith(f"Mar22_{run_str}_"):
            return filename

        # 2. Jan 2024: Starts with the run number AND contains "2024" in the filename
        elif exp_name == "January_2024" and filename.startswith(f"{run_str}_") and "2024" in filename:
            return filename

        # 3. Nov 2022: Starts with the run number but DOES NOT contain "2024"
        elif exp_name == "November_2022" and filename.startswith(f"{run_str}_") and "2024" not in filename:
            return filename

    # Fallback if the file is genuinely missing
    print(f"Warning: Could not find a matching file for {exp_name} run {run_number}.")
    return None