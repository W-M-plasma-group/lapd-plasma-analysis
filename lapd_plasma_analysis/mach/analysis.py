import numpy as np
import xarray as xr

from lapd_plasma_analysis.experimental import get_exp_params, get_config_id
from lapd_plasma_analysis.file_access import search_folder, choose_multiple_from_list, ensure_directory

from lapd_plasma_analysis.langmuir.plots import get_exp_run_string
from lapd_plasma_analysis.langmuir.analysis import save_datasets_nc
from lapd_plasma_analysis.langmuir.configurations import get_ion

from lapd_plasma_analysis.mach.getMachIsat import get_mach_isat
from lapd_plasma_analysis.mach.metadata_for_dataset import get_supplemental_metadata
from lapd_plasma_analysis.mach.velocity import get_mach_numbers, get_velocity
from lapd_plasma_analysis.mach.configurations import get_mach_config


def get_mach_datasets(mach_nc_folder, hdf5_folder, lang_datasets, hdf5_selected_paths, mach_mode):
    """

    Parameters
    ----------
    mach_nc_folder
    hdf5_folder
    lang_datasets
    hdf5_selected_paths
    mach_mode

    Returns
    -------

    """

    mach_nc_folder = ensure_directory(mach_nc_folder)
    print_mach_file_choices(hdf5_folder, mach_nc_folder, mach_mode)

    if mach_mode == "skip":
        return []

    mach_completed = False
    mach_datasets = []

    if hdf5_selected_paths is None:
        # the Langmuir data retrieved from NetCDF files; use either Mach HDF5s or NetCDFs
        print("\nChoose Mach data files to match with corresponding Lang files in the **same order**. \n"
              "Langmuir dataset are: ")
        for lang_dataset in lang_datasets:
            print("  * " + get_exp_run_string(lang_dataset.attrs))

        print("\nThe following NetCDF files were found in the Mach NetCDF folder (specified in main.py): ")
        mach_nc_paths = sorted(search_folder(mach_nc_folder, 'nc', limit=52))
        mach_nc_paths_to_open_ints = choose_multiple_from_list(mach_nc_paths, "Mach data NetCDF file",
                                                               null_action="select HDF5 files or skip Mach calculations")

        if len(mach_nc_paths_to_open_ints) > 0:
            mach_datasets = [xr.open_dataset(mach_nc_paths[choice]) for choice in mach_nc_paths_to_open_ints]
            mach_completed = True
        else:
            print("\nThe following HDF5 files were found in the HDF5 folder (specified in main.py): ")
            hdf5_paths = sorted(search_folder(hdf5_folder, "hdf5", limit=52))
            hdf5_chosen_ints = choose_multiple_from_list(hdf5_paths, "HDF5 file", null_action="skip Mach calculations")
            hdf5_selected_paths = [hdf5_paths[choice] for choice in hdf5_chosen_ints]

    if not mach_completed:
        # Fresh Langmuir datasets were made, so get Mach data from same HDF5 files
        # OR we have selected our desired HDF5 files to extract Mach data
        for hdf5_path in hdf5_selected_paths:
            exp_params_dict = get_exp_params(hdf5_path)  # list of experimental parameters
            ion_type = get_ion(exp_params_dict['Run name'])
            exp_params_dict = exp_params_dict | {"Ion type": ion_type}
            config_id = get_config_id(exp_params_dict['Exp name'])

            mach_configs = get_mach_config(hdf5_path, config_id)
            mach_isat = get_mach_isat(hdf5_path, mach_configs)

            mach_datasets += [get_mach_numbers(mach_isat).assign_attrs(exp_params_dict)]

        for i in range(len(mach_datasets)):
            mach_datasets[i] = mach_datasets[i].assign_attrs(get_supplemental_metadata(mach_datasets[i]))

        # Save Mach datasets as separate NetCDF files
        save_datasets_nc(mach_datasets, mach_nc_folder, "mach_")

    return mach_datasets


# TODO
# Raise issue of uTorr/ other misc units not working with SI prefixes?
# uTorr = u.def_unit("uTorr", 1e-6 * u.Torr)
# lapd_plot_units = (uTorr, u.gauss, u.kA)


def get_velocity_datasets(lang_datasets, mach_datasets, mach_mode):
    """

    Parameters
    ----------
    lang_datasets
    mach_datasets
    mach_mode

    Returns
    -------

    """
    if mach_mode == "skip":
        return lang_datasets

    datasets = lang_datasets.copy()
    for i in range(len(datasets)):
        for mach_variable_name in mach_datasets[i]:  # e.g. M_para
            velocity_variable_name = "v" + mach_variable_name[1:]
            if mach_mode == "append" and velocity_variable_name in datasets[i] \
                    and ~np.isnan(datasets[i][velocity_variable_name]).all():
                # The Mach mode is "append" and this dataset already has non-NaN values for this velocity variable,
                #    so skip calculating velocity for it
                continue
            else:
                # TODO this will fail with bimaxwellian; would it be best to use T_e_avg or T_e_cold in that case?
                velocity_ds = get_velocity(mach_datasets[i],
                                           datasets[i]['T_e'].isel(face=0).expand_dims({"face": datasets[i].face}),
                                           get_ion(datasets[i].attrs['Run name'])).assign_attrs(datasets[i].attrs)
                datasets[i] = datasets[i].assign(velocity_ds).transpose("probe", "face", "x", "y", "shot", "time")

    return datasets


def print_mach_file_choices(hdf5_folder, mach_nc_folder, mach_mode):
    mach_mode_actions = ({"skip": "skipped",
                          "append": "added to datasets without existing Mach data only",
                          "overwrite": "recalculated for all datasets"})
    print(f"Mach velocity mode is {repr(mach_mode)}. "
          f"Mach and velocity data will be {mach_mode_actions[mach_mode]}.")

    print("Current HDF5 directory path:           \t", repr(hdf5_folder),
          "\nCurrent Mach NetCDF directory path:  \t", repr(mach_nc_folder),
          "\nThese can be changed in main.py.")
    input("Enter any key to continue: ")
