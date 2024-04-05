from langmuir.file_access import *
from langmuir.helper import *
from langmuir.configurations import *
from langmuir.experimental import get_exp_params
from langmuir.getIVsweep import get_isweep_vsweep
from langmuir.characterization import characterize_sweep_array
from langmuir.preview import preview_raw_sweep, preview_characteristics
from langmuir.diagnostics import (langmuir_diagnostics, detect_steady_state_ramps, get_pressure,
                                  get_electron_ion_collision_frequencies)
from langmuir.interferometry import interferometry_calibration
from langmuir.plots import get_title


def get_langmuir_datasets(langmuir_nc_folder, hdf5_folder, interferometry_folder, interferometry_mode, isweep_choice,
                          core_radius, bimaxwellian) -> (list[xr.Dataset], list[tuple], dict, list[np.ndarray]):

    print("\n===== Langmuir probe analysis =====")
    netcdf_folder = ensure_directory(langmuir_nc_folder)  # Create folder to save NetCDF files if not yet existing

    # Display user file parameters
    print_file_choices(hdf5_folder, langmuir_nc_folder, interferometry_folder, interferometry_mode, isweep_choice)

    # Ask user to choose either NetCDF files or HDF5 files, then create datasets from them
    datasets = load_datasets(hdf5_folder, netcdf_folder, bimaxwellian)

    steady_state_plateaus_runs = [detect_steady_state_ramps(dataset, core_radius) for dataset in datasets]

    # Calibrate electron densities using interferometry data, depending on interferometry mode
    for density_diagnostic in ('n_e', 'n_i', 'n_i_OML'):
        datasets = interferometry_calibrate_datasets(datasets, density_diagnostic, interferometry_folder,
                                                     interferometry_mode, core_radius, steady_state_plateaus_runs)

    # Calculate pressures
    for i in range(len(datasets)):
        # TODO check behavior with advisor: use avg or cold T_e for bimaxwellian plasmas?
        electron_temperature = datasets[i]['T_e_avg'] if 'T_e_avg' in datasets[i] else datasets[i]['T_e']
        datasets[i] = datasets[i].assign({"P_e": get_pressure(datasets[i]['n_e'], electron_temperature)})
        if not np.isnan(datasets[i]['n_e_cal']).all():
            datasets[i] = datasets[i].assign({"P_e_cal": get_pressure(datasets[i]['n_e_cal'], electron_temperature)})

    # Calculate collision frequency
    for i in range(len(datasets)):
        electron_ion_collision_frequencies_da = datasets[i]['n_e'].copy().rename("nu_ei")   # DataArray
        electron_ion_collision_frequencies = get_electron_ion_collision_frequencies(        # Quantity array
            datasets[i],
            ion_type=get_ion(datasets[i].attrs['Run name']))
        electron_ion_collision_frequencies_da[...] = electron_ion_collision_frequencies
        electron_ion_collision_frequencies_da = electron_ion_collision_frequencies_da.assign_attrs(
            {"units": str(electron_ion_collision_frequencies.unit)})
        datasets[i] = datasets[i].assign({'nu_ei': electron_ion_collision_frequencies_da})

    # Final save diagnostics datasets to folder (after earlier save point in load_datasets function)
    save_datasets(datasets, netcdf_folder, bimaxwellian)

    # Get possible diagnostics and their full names, e.g. "n_e" and "Electron density"
    diagnostic_name_dict = {key: get_title(key) for key in set.intersection(*[set(dataset) for dataset in datasets])}
    # Ask users for list of diagnostics to plot
    print("The following diagnostics are available to plot: ")
    diagnostics_sort_indices = np.argsort(list(diagnostic_name_dict.keys()))
    diagnostics_to_plot_ints = choose_multiple_list(
        np.array(list(diagnostic_name_dict.values()))[diagnostics_sort_indices],
        "diagnostic", null_action="skip")
    diagnostics_to_plot_list = [np.array(list(diagnostic_name_dict.keys()))[diagnostics_sort_indices][choice]
                                for choice in diagnostics_to_plot_ints]

    return datasets, steady_state_plateaus_runs, diagnostic_name_dict, diagnostics_to_plot_list


def print_file_choices(hdf5_folder, lang_nc_folder, interferometry_folder, interferometry_mode, isweep_choice):
    interferometry_mode_actions = ({"skip": "skipped",
                                    "append": "added to uncalibrated datasets only",
                                    "overwrite": "recalculated for all datasets"})
    print(f"Interferometry mode is {repr(interferometry_mode)}. "
          f"Calibrated density data will be {interferometry_mode_actions[interferometry_mode]}.")

    print("Current HDF5 directory path:           \t", repr(hdf5_folder),
          "\nCurrent NetCDF directory path:         \t", repr(lang_nc_folder),
          "\nCurrent interferometry directory path: \t", repr(interferometry_folder),
          "\nLinear combinations of isat sources:   \t", repr(isweep_choice),
          "\nThese can be changed in main.py.")
    input("Enter any key to continue: ")


def load_datasets(hdf5_folder, lang_nc_folder, bimaxwellian):

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

            # Intermediate save point in case diagnostics are interrupted later
            save_datasets([diagnostics_dataset], lang_nc_folder, bimaxwellian)
            datasets.append(diagnostics_dataset)

    return datasets


def interferometry_calibrate_datasets(datasets, density_diagnostic, interferometry_folder, interferometry_mode,
                                      core_radius, steady_state_ramps):
    # datasets[i].attrs = exp params
    calibrated_density_diagnostic = density_diagnostic + '_cal'
    print(f"{get_title(density_diagnostic)} mean steady-state interferometry calibration factors")

    for i in range(len(datasets)):
        if interferometry_mode == "overwrite" or (interferometry_mode == "append"
                                                  and np.isnan(datasets[i][calibrated_density_diagnostic]).all()):
            print("\t", end="")  # indent interferometry scale factor
            calibrated_electron_density = interferometry_calibration(datasets[i][density_diagnostic].copy(),
                                                                     datasets[i].attrs,
                                                                     interferometry_folder,
                                                                     steady_state_ramps[i],
                                                                     core_radius)
            datasets[i] = datasets[i].assign({calibrated_density_diagnostic: calibrated_electron_density})

            """
            except (IndexError, ValueError, TypeError, AttributeError, KeyError) as e:
                print(f"Error in calibrating electron density: \n{str(e)}")
                calibrated_electron_density = datasets[i]['n_e'].copy()
            """

            # datasets[i] = datasets[i].assign_attrs({"Interferometry calibrated": True})  # TODO necessary?/improve

    return datasets


def save_datasets(datasets: list[xr.Dataset], lang_nc_folder, bimaxwellian):
    for i in range(len(datasets)):
        ds_save_name = (datasets[i].attrs['Exp name'][:3] + datasets[i].attrs['Exp name'][-2:] + "_"
                        + datasets[i].attrs['Run name'] + ("_bimax" if bimaxwellian else "") + "_lang")
        ds_save_path = make_path(lang_nc_folder, ds_save_name, "nc")
        write_netcdf(datasets[i], ds_save_path)
