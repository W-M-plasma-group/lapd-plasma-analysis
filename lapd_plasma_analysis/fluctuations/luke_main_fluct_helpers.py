import os
import xarray as xr

from lapd_plasma_analysis.obtain_plots.xarray_plots import *
from lapd_plasma_analysis.mach.luke_main_mach_helpers import *

def generate_colors_valid_ds_ri(fluct_datasets, fluct_files, langmuir_nc_folder, mach_folder,
                                make_presentable = False):
    print('Mach folder: ', mach_folder)
    """
    Parameters
    ----------
    fluct_datasets
    fluct_files
    langmuir_nc_folder
    mach_folder
    make_presentable

    Returns
    -------
    valid_runs: List of dictionaries containing a fluctuations dataset, a swept Langmuir probe dataset,
     their corresponding paths and identifying strings colors and marks

    """
    # Match fluctuations data with Langmuir datasets
    valid_runs = joint_langmuir_flucts_ds(fluct_datasets, fluct_files, langmuir_nc_folder)
    for run in valid_runs:
        lang_pathname = run['lang_pathname']
        mach_filename, mach_filepath = connect_mach_nc_to_lang_nc(lang_pathname, mach_folder)
        if os.path.exists(mach_filepath):
            with xr.open_dataset(mach_filepath) as ds:
                mach_ds = ds.load()

            run['mach_ds'] = mach_ds
            run['mach_filename'] = mach_filename


    # Generate Colors and marks
    lang_ds_list = [run['lang_ds'] for run in valid_runs]
    clors, marks = determine_colors(lang_ds_list)

    if make_presentable:
        run_identifiers = generic_run_identifiers(lang_ds_list)
    else:
        run_identifiers = [f_run_identifier(ds) for ds in lang_ds_list]

    # Add these properties to our bundled dictionary
    for i, run in enumerate(valid_runs):
        run['color'] = clors[i]
        run['marker'] = marks[i]
        run['run_id'] = run_identifiers[i]

    return valid_runs


def joint_langmuir_flucts_ds(fluct_datasets, fluct_files, langmuir_nc_folder):
    """
    Parameters
    ----------
    fluct_datasets: List of xarray.Dataset objects corresponding to selected fluctuation datasets
    fluct_files: List of strings corresponding to selected fluctuation files
    langmuir_nc_folder: Path to folder containing processed swept langmuir probe datasets

    Returns
    -------
    valid_runs: List of dictionaries containing a fluctuations dataset, a swept Langmuir probe dataset,
    and their corresponding paths

    A function to determine how many matched datasets there are between the selected fluctuations datasets and processed
    datasets from the swept Langmuir probes
    """

    valid_runs = []  # We will store a dictionary for each valid run to keep data bundled

    for fluct_ds, clr_filename in zip(fluct_datasets, fluct_files):
        if 'updated' in clr_filename:
            updated_lang_folder = langmuir_nc_folder + 'updated/'
            base_name = clr_filename.replace('_updated', '')
            lang_pathname = os.path.join(updated_lang_folder, f"{base_name}_tanh_updated.nc")
        else:
            lang_pathname = os.path.join(langmuir_nc_folder, f"{clr_filename}_tanh.nc")

        if os.path.exists(lang_pathname):
            with xr.open_dataset(lang_pathname) as ds:
                lang_ds = ds.load()

            valid_runs.append({
                'fluct_ds': fluct_ds,
                'filename': clr_filename,
                'lang_ds': lang_ds,
                'lang_pathname': lang_pathname
            })
        else:
            print(f'Skipping - Langmuir dataset not found: {clr_filename}')

    return valid_runs

