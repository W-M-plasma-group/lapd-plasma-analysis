import os
import numpy as np

def connect_lang_nc_to_hdf5(lang_nc_filename, lang_nc_folder, hdf5_folder):
    jan2024 = False
    mar2022 = False
    nov2022 = False
    if 'mar2022' in lang_nc_filename.lower():
        mar2022 = True
        config_id = 1

    elif 'nov2022' in lang_nc_filename.lower():
        nov2022 = True
        config_id = 2
    elif 'jan2024' in lang_nc_filename.lower():
        jan2024 = True
        config_id = 3
    else:
        print('Files not yet supported for Mach Analysis \n'
              '(to fix go to luke_main_mach_helpers)')
        config_id = None

    if jan2024 or mar2022 or nov2022:
        run_num = lang_nc_filename.split('_')[1]
    else:
        run_num = ''

    hdf5list = [f for f in os.listdir(hdf5_folder) if f.endswith('.hdf5')]
    matched_filename = ''
    for filename in hdf5list:
        split_filename = filename.split('_')
        len_split_filename = len(split_filename)
        if jan2024 and len_split_filename > 4:
            if split_filename[0] == run_num:
                matched_filename = filename
        if mar2022 and split_filename[0].lower() == 'mar22':
            if split_filename[1] == run_num:
                matched_filename = filename
        if nov2022 and len_split_filename == 4:
            if split_filename[0] == run_num:
                matched_filename = filename
    print('matched_filename: ', matched_filename)

    return matched_filename, os.path.join(hdf5_folder, matched_filename), config_id

def connect_lang_nc_to_mach_nc(mach_nc_filename, lang_nc_folder):
    run_num = mach_nc_filename.split('_')[1]
    if 'updated' in mach_nc_filename.lower():
        search_folder = os.path.join(lang_nc_folder, 'updated/')
    else:
        search_folder = lang_nc_folder

    lang_nc_list = [f for f in os.listdir(search_folder) if f.endswith('.nc')]
    matched_filename = None
    matched_filepath = None
    for lang_nc in lang_nc_list:
        if lang_nc.split('_')[1] == run_num:
            matched_filename = lang_nc
            matched_filepath = os.path.join(search_folder, matched_filename)
            print('Langmuir matched filename: ', matched_filename)

    return matched_filename, matched_filepath


def connect_mach_nc_to_lang_nc(lang_nc_filename, mach_nc_folder):
    print('Function mach folder: ', mach_nc_folder)
    run_num = lang_nc_filename.split('_')[2]
    print('run number : ', run_num)

    search_folder = mach_nc_folder
    if 'updated' in lang_nc_filename.lower():
        mach_nc_list = [f for f in os.listdir(search_folder) if f.endswith('updated_mach.nc')]
    else:
        mach_nc_list = [f for f in os.listdir(search_folder) if f.endswith('tanh_mach.nc')]

    print('mach_nc_list: ', mach_nc_list)

    matched_filename = None
    matched_filepath = None
    for mach_nc in mach_nc_list:
        if mach_nc.split('_')[1] == run_num:
            matched_filename = mach_nc
            matched_filepath = os.path.join(search_folder, matched_filename)
            print('Mach matched filename: ', matched_filename)

    return matched_filename, matched_filepath


def process_mach_data(mach_ds, probe_idx, var_name, ss_start, ss_end):
    """
    Averages Mach probe data across shots and steady-state sweeps, safely
    matching the temporal dimension size of target_data.
    """
    # 1. Select probe and average across shots if present
    target_data = mach_ds[var_name].sel(probe=probe_idx)
    if 'shot' in target_data.dims:
        target_data = target_data.mean('shot')

    # 2. Identify temporal dimension on target_data ('sweep' or 'time')
    temp_dim = None
    for dim in ['sweep', 'time']:
        if dim in target_data.dims:
            temp_dim = dim
            break

    if temp_dim is not None:
        dim_size = target_data.sizes[temp_dim]

        # Look for a 1D coordinate on target_data that matches the dimension length
        coord_for_mask = None
        for coord_name in ['time', 'sweep']:
            if coord_name in target_data.coords:
                coord = target_data[coord_name]
                if coord.ndim == 1 and coord.size == dim_size:
                    coord_for_mask = coord
                    break

        # Fallback to the dimension index itself if no 1D coordinate matches
        if coord_for_mask is None:
            coord_for_mask = target_data[temp_dim]

        # Generate boolean mask matching dim_size (e.g., length 42)
        time_mask = (coord_for_mask >= ss_start) & (coord_for_mask <= ss_end)

        # Average across steady-state window using positional indices
        if time_mask.any():
            valid_indices = np.where(time_mask.values)[0]
            var_ss = target_data.isel({temp_dim: valid_indices}).mean(temp_dim)
        else:
            var_ss = target_data.mean(temp_dim)
    else:
        var_ss = target_data

    # 3. Extract numpy arrays
    v_vals = var_ss.squeeze().values
    v_x_vals = target_data['x'].values

    return v_vals, v_x_vals







