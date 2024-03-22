from file_access import *
from helper import *
from configurations import *
from experimental import get_exp_params
from getIVsweep import get_isweep_vsweep
from characterization import characterize_sweep_array
from preview import preview_raw_sweep, preview_characteristics
from diagnostics import (langmuir_diagnostics, detect_steady_state_ramps, get_pressure,
                         get_electron_ion_collision_frequency)
from interferometry import interferometry_calibration
from plots import get_title


def setup_datasets(langmuir_nc_folder, hdf5_folder, interferometry_folder, isweep_choice, bimaxwellian):
    netcdf_folder = ensure_directory(langmuir_nc_folder)  # Create folder to save NetCDF files if not yet existing

    # Ask user to choose either NetCDF files or HDF5 files, then create datasets from them
    datasets, generate_new_itfm = load_datasets(hdf5_folder, netcdf_folder, interferometry_folder, isweep_choice,
                                                bimaxwellian)

    # Get ramp indices for beginning and end of steady state period in plasma; TODO hardcoded
    if "january" in hdf5_folder.lower():
        steady_state_plateaus_runs = [(16, 24) for dataset in datasets]
    else:
        steady_state_plateaus_runs = [detect_steady_state_ramps(dataset['n_e'], core_radius) for dataset in datasets]

    # Only calibrate electron densities using interferometry data if new diagnostics were generated from HDF5 files
    if generate_new_itfm:
        datasets = interferometry_calibrate_datasets(datasets, interferometry_folder, steady_state_plateaus_runs)

    # Calculate pressure
    for i in range(len(datasets)):
        datasets[i] = datasets[i].assign({"P_e": get_pressure(datasets[i])})

    # NEW: calculate collision frequency
    for i in range(len(datasets)):
        datasets[i] = datasets[i].assign_attrs({'nu_ei': get_electron_ion_collision_frequency(
            datasets[i][{"isweep": 0, "x": 27, "y": 0, "shot": 0, "time": 16}], ion_type="H+")})  # TODO very hardcoded!
        # TODO make assign() because it will be a dataset!

    # Save diagnostics datasets to folder
    save_datasets(datasets, netcdf_folder, bimaxwellian)

    # Get possible diagnostics and their full names, e.g. "n_e" and "Electron density"
    diagnostic_name_dict = {key: get_title(key) for key in set.intersection(*[set(dataset) for dataset in datasets])}

    return datasets, steady_state_plateaus_runs, diagnostic_name_dict


def load_datasets(hdf5_folder, lang_nc_folder, interferometry_folder, isweep_choice, bimaxwellian):

    if not bool(interferometry_folder):
        print("\nInterferometry calibration is OFF. "
              "Interferometry-calibrated electron density ('n_e_cal') is not available.")

    print("Current HDF5 directory path:           \t", repr(hdf5_folder),
          "\nCurrent NetCDF directory path:         \t", repr(lang_nc_folder),
          "\nCurrent interferometry directory path: \t", repr(interferometry_folder),
          "\nLinear combinations of isat sources:   \t", repr(isweep_choice),
          "\nThese can be changed in main.py.")
    input("Enter any key to continue: ")

    print("\nThe following NetCDF files were found in the NetCDF folder (specified in main.py): ")
    nc_paths = sorted(search_folder(lang_nc_folder, 'nc', limit=52))
    nc_paths_to_open_ints = choose_multiple_list(nc_paths, "NetCDF file",
                                                 null_action="perform diagnostics on HDF5 files")

    if len(nc_paths_to_open_ints) > 0:
        datasets = [xr.open_dataset(nc_paths[choice]) for choice in nc_paths_to_open_ints]
    else:
        print("\nThe following HDF5 files were found in the HDF5 folder (specified in main.py): ")
        hdf5_paths = sorted(search_folder(hdf5_folder, "hdf5", limit=52))
        hdf5_chosen_ints = choose_multiple_list(hdf5_paths, "HDF5 file")
        hdf5_chosen_list = [hdf5_paths[choice] for choice in hdf5_chosen_ints]

        sweep_view_mode = (len(hdf5_chosen_list) == 1) and ask_yes_or_no("Use raw sweep preview mode? (y/n) ")
        chara_view_mode = (len(hdf5_chosen_list) == 1) and ask_yes_or_no("Use characteristic preview mode? (y/n) ")

        datasets = []
        for hdf5_path in hdf5_chosen_list:

            print("\nOpening file", repr(hdf5_path), "...")

            exp_params_dict = get_exp_params(hdf5_path)  # list of experimental parameters
            ion_type = get_ion(exp_params_dict['Run name'])
            config_id = get_config_id(exp_params_dict['Exp name'])
            vsweep_board_channel = get_vsweep_bc(config_id)

            langmuir_probes = get_langmuir_config(hdf5_path, config_id)  # TODO print face strings in outputs
            voltage_gain = get_voltage_gain(config_id)
            orientation = get_orientation(config_id)

            # get current and bias data from Langmuir probe
            bias, currents, positions, dt, ports = get_isweep_vsweep(hdf5_path, vsweep_board_channel,
                                                                     langmuir_probes, voltage_gain, orientation)

            if sweep_view_mode:
                preview_raw_sweep(bias, currents, positions, ports, exp_params_dict, dt)

            # organize sweep bias and current into an array of Characteristic objects
            characteristics, ramp_times = characterize_sweep_array(bias, currents, dt)

            # cleanup 1
            del bias, currents

            if chara_view_mode:
                preview_characteristics(characteristics, positions, ports, ramp_times, exp_params_dict,
                                        diagnostics=True, areas=langmuir_probes['area'], ion=ion_type,
                                        bimaxwellian=bimaxwellian)

            # perform langmuir diagnostics on each dataset
            diagnostics_dataset = langmuir_diagnostics(characteristics, positions, ramp_times, ports,
                                                       langmuir_probes['area'], ion_type, bimaxwellian=bimaxwellian)

            # cleanup 2
            del characteristics, positions

            diagnostics_dataset = diagnostics_dataset.assign_attrs(exp_params_dict)
            datasets.append(diagnostics_dataset)

    return datasets, len(nc_paths_to_open_ints) == 0


def interferometry_calibrate_datasets(datasets, interferometry_folder, steady_state_ramps):

    for i in range(len(datasets)):
        calibrated_electron_density = interferometry_calibration(datasets[i]['n_e'].copy(),
                                                                 datasets[i].attrs,          # exp params
                                                                 interferometry_folder,
                                                                 steady_state_ramps[i],
                                                                 core_radius=core_radius)
        datasets[i] = datasets[i].assign({"n_e_cal": calibrated_electron_density})
        """
        except (IndexError, ValueError, TypeError, AttributeError, KeyError) as e:
            print(f"Error in calibrating electron density: \n{str(e)}")
            calibrated_electron_density = datasets[i]['n_e'].copy()
        """

        datasets[i] = datasets[i].assign_attrs({"Interferometry calibrated": True})

    return datasets


def save_datasets(datasets: list[xr.Dataset], lang_nc_folder, bimaxwellian):
    for i in range(len(datasets)):
        ds_save_path = make_path(lang_nc_folder, datasets[i].attrs['Run name'] + ("_bimax" if bimaxwellian else "")
                                 + "_lang", "nc")
        write_netcdf(datasets[i], ds_save_path)
